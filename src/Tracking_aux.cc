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

float Tracking::load_camera_parameters(const std::string& calibration_yaml, const std::string& cam_name)
{
    const YAML::Node calibration = YAML::LoadFile(calibration_yaml);

    // The camera the settings select (cam_mono)
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
    if (cam["cam_model"] && cam["cam_model"].as<std::string>() != "pinhole")
        AF_WARN("[Tracking] cam_model '" << cam["cam_model"].as<std::string>() << "' of camera '" << cam_name
                << "': treated as pinhole");

    K = (cv::Mat_<float>(3, 3) << cam["focal_length"][0].as<float>(), 0.0f, cam["principal_point"][0].as<float>(),
            0.0f, cam["focal_length"][1].as<float>(), cam["principal_point"][1].as<float>(),
            0.0f, 0.0f, 1.0f);

    image_width_ = cam["image_dimension"][0].as<int>();
    image_height_ = cam["image_dimension"][1].as<int>();

    // Optional: a property of the depth source, not of the lens (issue #3)
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
        constexpr int multiple = Image::SIZE_MULTIPLE;
        if(new_width % multiple != 0 || new_height % multiple != 0)
            AF_WARN("[Tracking] frames resized to " << new_width << "x" << new_height << " are cropped to "
                    << (new_width / multiple) * multiple << "x" << (new_height / multiple) * multiple
                    << " (multiples of " << multiple << ")");
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

    // Distortion: radtan4/radtan5 (OpenCV k1 k2 p1 p2 [k3]) or equid4 (Kannala-Brandt k1..k4); none = zeros
    distortion_coefficients = cv::Mat::zeros(4, 1, CV_32F);
    distortion_model_ = DistortionModel::radtan;
    std::string distortion_type = "none";
    if (cam["distortion_type"] && cam["distortion_coefficients"]) {
        distortion_type = cam["distortion_type"].as<std::string>();
        const std::vector<float> dist_coeffs = cam["distortion_coefficients"].as<std::vector<float>>();
        const size_t expected = distortion_type == "radtan4" ? 4 : distortion_type == "radtan5" ? 5
                              : distortion_type == "equid4" ? 4 : 0;
        if (expected == 0 || dist_coeffs.size() != expected) {
            AF_ERROR("[Tracking] distortion_type '" << distortion_type << "' with " << dist_coeffs.size()
                     << " coefficients (camera '" << cam_name << "'): expected radtan4 (4), radtan5 (5) or equid4 (4)");
            exit(-1);
        }
        if (distortion_type == "equid4")
            distortion_model_ = DistortionModel::equidistant;
        distortion_coefficients = cv::Mat(dist_coeffs, true).reshape(1, static_cast<int>(dist_coeffs.size()));
    }

    // Camera rate (Hz); absent or <= 0: DEFAULT_FPS
    float fps = cam["fps"] ? cam["fps"].as<float>() : 0.0f;
    if(fps <= 0.0f)
        fps = DEFAULT_FPS;

    AF_CONFIG_BEGIN("Camera Parameters");
    AF_CONFIG_FIELD("cam_name:           ", cam_name);
    if (cam["cam_type"])
        AF_CONFIG_FIELD("cam_type:           ", cam["cam_type"].as<std::string>());
    if (cam["cam_model"])
        AF_CONFIG_FIELD("cam_model:          ", cam["cam_model"].as<std::string>());
    AF_CONFIG_FIELD("distortion_type:    ", distortion_type);
    AF_CONFIG_FIELD("fx:                 ", K.at<float>(0,0));
    AF_CONFIG_FIELD("fy:                 ", K.at<float>(1,1));
    AF_CONFIG_FIELD("cx:                 ", K.at<float>(0,2));
    AF_CONFIG_FIELD("cy:                 ", K.at<float>(1,2));
    if (distortion_type != "none")
        AF_CONFIG_FIELD("distortion_coefficients: ", distortion_coefficients.t());
    AF_CONFIG_FIELD("fps:                ", fps);
    AF_CONFIG_END();
    return fps;
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
        feature_extractors_.at(ft)->set_budget(max_num_features);
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

}
