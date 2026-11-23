#include "Tracking.h"

#include <string>

#include <algorithm>
#include <cmath>
#include <iostream>

#include <yaml-cpp/yaml.h>

#include "FeatureFactory.h"
#include "afvslam_log.hpp"

namespace AF_VSLAM
{

void Tracking::load_camera_parameters(const std::string& calibration_yaml, const std::string& settings_yaml)
{
    const YAML::Node settings = YAML::LoadFile(settings_yaml);
    const YAML::Node calibration = YAML::LoadFile(calibration_yaml);

    // Find the camera the settings select
    const std::string cam_name = settings["cam_mono"].as<std::string>();
    YAML::Node cam{};
    for (const YAML::Node& camera : calibration["cameras"]) {
        if (camera["cam_name"].as<std::string>() == cam_name) {
            cam = camera;
            break;
        }
    }
    if (!cam) {
        AF_ERROR("[Tracking] camera '" + cam_name + "' (settings cam_mono) not found in " + calibration_yaml);
        exit(-1);
    }

    K = (cv::Mat_<float>(3, 3) << cam["focal_length"][0].as<float>(), 0.0f, cam["principal_point"][0].as<float>(),
            0.0f, cam["focal_length"][1].as<float>(), cam["principal_point"][1].as<float>(),
            0.0f, 0.0f, 1.0f);

    image_width_ = cam["image_dimension"][0].as<int>();
    image_height_ = cam["image_dimension"][1].as<int>();

    // Optional: a property of the depth source, not of the lens. Absent for every dataset today,
    // so the default reproduces the previous hardcoded value exactly (issue #3).
    if(cam["depth_noise_coeff"])
        depth_noise_coeff_ = cam["depth_noise_coeff"].as<float>();

    if(fix_image_size_){
        // Tracking.ImageWidth/Height set: exactly that size (per-axis scaling of K is valid for the pinhole model:
        // the distortion acts in normalised coordinates). Else: Tracking.NominalImageArea with the aspect ratio kept.
        int new_width = params.image_width, new_height = params.image_height;
        if(new_width <= 0 || new_height <= 0){
            const float ratio = float(image_width_) / float(image_height_);
            new_height = static_cast<int>(std::sqrt(params.nominal_image_area / ratio));
            new_width = static_cast<int>(float(new_height) * ratio);
        }
        const float scale_height = float(new_height) / float(image_height_);
        const float scale_width = float(new_width) / float(image_width_);
        if(new_width % 32 != 0 || new_height % 32 != 0)
            AF_WARN("[Tracking] frames resized to " << new_width << "x" << new_height << " are cropped to "
                    << (new_width / 32) * 32 << "x" << (new_height / 32) * 32 << " (multiples of 32)");
        AF_INFO("[Tracking] frames " << image_width_ << "x" << image_height_ << " -> " << new_width << "x" << new_height
                << (params.image_width > 0 && params.image_height > 0 ? " (Tracking.ImageWidth/Height)"
                                                                       : " (Tracking.NominalImageArea, aspect ratio kept)"));

        image_width_ = new_width;
        image_height_ = new_height;

        K.at<float>(0,0) *= scale_width;
        K.at<float>(1,1) *= scale_height;
        K.at<float>(0,2) *= scale_width;
        K.at<float>(1,2) *= scale_height;
    }

    // Distortion coefficients (zeros when the calibration declares none)
    distortion_coefficients = cv::Mat::zeros(4, 1, CV_32F);
    if (cam["distortion_type"] && cam["distortion_coefficients"]) {
        std::vector<float> dist_coeffs = cam["distortion_coefficients"].as<std::vector<float>>();
        distortion_coefficients = cv::Mat(int(dist_coeffs.size()), 1, CV_32F, dist_coeffs.data()).clone();
    }

    // Camera frequency (Hz)
    fps_ = cam["fps"].as<float>();
    if(fps_ <= 0.0f)
        fps_ = DEFAULT_FPS;

    AF_CONFIG_BEGIN("Camera Parameters");
    AF_CONFIG_FIELD("cam_name:           ", cam["cam_name"].as<std::string>());
    AF_CONFIG_FIELD("cam_type:           ", cam["cam_type"].as<std::string>());
    AF_CONFIG_FIELD("cam_model:          ", cam["cam_model"].as<std::string>());
    if (cam["distortion_type"] && cam["distortion_coefficients"])
        AF_CONFIG_FIELD("distortion_type:    ", cam["distortion_type"].as<std::string>());
    AF_CONFIG_FIELD("fx:                 ", K.at<float>(0,0));
    AF_CONFIG_FIELD("fy:                 ", K.at<float>(1,1));
    AF_CONFIG_FIELD("cx:                 ", K.at<float>(0,2));
    AF_CONFIG_FIELD("cy:                 ", K.at<float>(1,2));
    if (cam["distortion_type"] && cam["distortion_coefficients"])
        AF_CONFIG_FIELD("distortion_coefficients: ", distortion_coefficients.t());
    AF_CONFIG_FIELD("fps:                ", fps_);
    AF_CONFIG_END();
}

std::shared_ptr<FeatureExtractor> Tracking::get_feature_extractor(const int max_num_features, const FeatureType feature_type)
{
    auto settings = std::make_shared<FeatureExtractorSettings>(feature_type, max_num_features);
    return get_feature(feature_type).create_extractor(settings);
}

// The keypoints per frame of `feature_type` at `features_fraction` of its own maximum: 1 gives the
// maximum, 0 no keypoints
int Tracking::features_budget(const FeatureType feature_type, const float features_fraction)
{
    return static_cast<int>(std::lround(features_fraction * get_feature(feature_type).get_max_features()));
}

// Gives every extractor the budget of Tracking.FeaturesFraction when it changed.
void Tracking::apply_features_fraction()
{
    const float features_fraction = std::clamp(params.features_fraction.load(), 0.0f, 1.0f);
    if(features_fraction == features_fraction_in_use_)
        return;
    features_fraction_in_use_ = features_fraction;

    for(const FeatureType ft : feature_types_)
    {
        const int max_num_features = features_budget(ft, features_fraction);
        feature_extractor_left_.at(ft)->set_budget(max_num_features);
        AF_INFO("[Tracking] " << get_feature(ft).get_feature_name() << ": " << max_num_features
                << " keypoints per frame (features fraction " << features_fraction << ")");
    }
}

bool Tracking::run_tracking_stage(const std::function<bool()>& stage)
{
    try {
        return stage();
    }
    catch(const TrackingLostException& e) {
        AF_WARN("Tracking lost — " << e.what());
        return false;
    }
    // Any other exception in a tracking stage (an OpenCV assertion in optical flow or matching, an out-of-range
    // lookup) fails the stage, so the frame goes LOST and relocalizes, instead of ending the tracking thread
    catch(const std::exception& e) {
        AF_WARN("Tracking stage failed (" << e.what() << ") — treating the frame as lost");
        return false;
    }
}

} // namespace AF_VSLAM
