/**
 * Auxiliary Tracking members kept out of Tracking.cc so the per-frame flow
 * there reads uninterrupted (companion to include/Tracking_aux.h).
 */
#include "Tracking.h"

#include <cmath>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include "FeatureFactory.h"
#include "FrameDrawer.h"

#include <iostream>

#include "afvslam_log.hpp"

namespace AF_VSLAM
{

// Low-rate heartbeat so post-mortems can see the inlier/map trend leading
// into a tracking loss, not just the loss line itself.
void Tracking::log_heartbeat()
{
#ifdef PROFILING_EXHAUSTIVE
    constexpr int HEARTBEAT_PERIOD_FRAMES = 100;
    if(current_frame_.frame_id % HEARTBEAT_PERIOD_FRAMES != 0)
        return;
    AF_INFO("track: heartbeat | frame=" << current_frame_.frame_id
            << " inliers=" << num_inlier_matches_
            << " localPts=" << local_points_.size()
            << " KFs=" << map_->keyframes_in_map()
            << " mapPts=" << map_->map_points_in_map());
    std::cout.flush(); // stdout is fully buffered under the runner's redirect
#endif
}

// Post-mortem dump on a pose-optimization collapse: per-match reprojection
// residuals AT THE SEED POSE (before any optimization),
// with each map point's provenance — enough to test offline whether the match
// set is bimodal (two coherent populations with no common pose) and which
// population (old vs freshly-created points, image region, depth) is
// inconsistent. No-op when no exp_folder is set.
void Tracking::dump_pose_collapse(const mat4f& seed_pose)
{
    if(FrameDrawer::exp_folder.empty())
        return;

    const mat3f Rcw_seed = seed_pose.block<3,3>(0,0);
    const vec3f tcw_seed = seed_pose.block<3,1>(0,3);
    const float fx = mK.at<float>(0,0), fy = mK.at<float>(1,1);
    const float cx = mK.at<float>(0,2), cy = mK.at<float>(1,2);
    std::ofstream dump(FrameDrawer::exp_folder + "/collapse_frame_"
                       + std::to_string(current_frame_.frame_id) + ".csv");
    dump << "ft,kpIdx,u_kp,v_kp,u_proj,v_proj,z_cam,invDepth_meas,ptId,firstKFid,nObs\n";
    for (const auto& [ft, num_keypoints] : current_frame_.N)
    {
        const auto& keypoints = current_frame_.keypoints.at(ft);
        const auto& inv_depth = current_frame_.inv_depth.at(ft);
        for(int i = 0; i < num_keypoints; i++)
        {
            const Pt& map_point = current_frame_.pts.at(ft)[i];
            if(!map_point)
                continue;
            const vec3f Xc = Rcw_seed * map_point->get_world_pos() + tcw_seed;
            if(Xc(2) <= 0.0f)
                continue;
            dump << int(ft) << "," << i << ","
                 << keypoints[i].pt.x << "," << keypoints[i].pt.y << ","
                 << (fx * Xc(0) / Xc(2) + cx) << "," << (fy * Xc(1) / Xc(2) + cy) << ","
                 << Xc(2) << "," << inv_depth[i] << ","
                 << map_point->ptId << "," << map_point->mnFirstKFid << ","
                 << map_point->number_of_observations() << "\n";
        }
    }
}

// Cumulative per-stage timing histograms, printed through the AF_PROFILE sink.
void Tracking::log_profile()
{
#ifdef PROFILING_EXHAUSTIVE
    AF_PROFILE_BEGIN("Tracking Profiling");
    AF_PROFILE_FIELD(resize_times_,      "Resize Image");
    AF_PROFILE_FIELD(frame_times_,       "Frame Creation");
    AF_PROFILE_FIELD(tracking_times_,    "Tracking");
    AF_PROFILE_FIELD(track_ref_times_,   "  Track Ref");
    AF_PROFILE_FIELD(pose_opt_times_,    "  Pose Optimization");
    AF_PROFILE_FIELD(local_map_times_,   "  Track Local Map");
    AF_PROFILE_FIELD(grab_image_times_, "Grab Image");
    AF_PROFILE_END();
#endif
}

std::shared_ptr<FeatureExtractor> Tracking::get_feature_extractor(const int scale_num_features,
                                                                  const std::string& feature_settings_yaml,
                                                                  const FeatureType feature_type)
{
    auto settings = std::make_shared<FeatureExtractorSettings>(feature_type, feature_settings_yaml);
    settings->maxNumFeatures *= scale_num_features;
    return get_feature(feature_type).createExtractor(settings);
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
}

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

    mK = (cv::Mat_<float>(3, 3) << cam["focal_length"][0].as<float>(), 0.0f, cam["principal_point"][0].as<float>(),
            0.0f, cam["focal_length"][1].as<float>(), cam["principal_point"][1].as<float>(),
            0.0f, 0.0f, 1.0f);

    image_width_ = cam["image_dimension"][0].as<int>();
    image_height_ = cam["image_dimension"][1].as<int>();

    if(fix_image_size_){
        const float ratio = float(image_width_) / float(image_height_);
        const int new_height = static_cast<int>(std::sqrt(NOMINAL_IMAGE_AREA / ratio));
        const int new_width = static_cast<int>(float(new_height) * ratio);
        const float scale_height = float(new_height) / float(image_height_);
        const float scale_width = float(new_width) / float(image_width_);

        image_width_ = new_width;
        image_height_ = new_height;

        mK.at<float>(0,0) *= scale_width;
        mK.at<float>(1,1) *= scale_height;
        mK.at<float>(0,2) *= scale_width;
        mK.at<float>(1,2) *= scale_height;
    }

    // Distortion coefficients (zeros when the calibration declares none)
    mDistCoef = cv::Mat::zeros(4, 1, CV_32F);
    if (cam["distortion_type"] && cam["distortion_coefficients"]) {
        const std::vector<float> dist_coeffs = cam["distortion_coefficients"].as<std::vector<float>>();
        mDistCoef = cv::Mat(dist_coeffs.size(), 1, CV_32F, const_cast<float*>(dist_coeffs.data())).clone();
    }

    // Camera frequency (Hz)
    fps_ = cam["fps"].as<float>();
    if(fps_ <= 0.0f)
        fps_ = DEFAULT_FPS;

    // Color order (assignment to the member — a local here once shadowed it, #14)
    is_rgb_ = cam["cam_type"].as<std::string>() != "bgr";

    AF_CONFIG_BEGIN("Camera Parameters");
    AF_CONFIG_FIELD("cam_name:           ", cam["cam_name"].as<std::string>());
    AF_CONFIG_FIELD("cam_type:           ", cam["cam_type"].as<std::string>());
    AF_CONFIG_FIELD("cam_model:          ", cam["cam_model"].as<std::string>());
    if (cam["distortion_type"] && cam["distortion_coefficients"])
        AF_CONFIG_FIELD("distortion_type:    ", cam["distortion_type"].as<std::string>());
    AF_CONFIG_FIELD("fx:                 ", mK.at<float>(0,0));
    AF_CONFIG_FIELD("fy:                 ", mK.at<float>(1,1));
    AF_CONFIG_FIELD("cx:                 ", mK.at<float>(0,2));
    AF_CONFIG_FIELD("cy:                 ", mK.at<float>(1,2));
    if (cam["distortion_type"] && cam["distortion_coefficients"])
        AF_CONFIG_FIELD("distortion_coefficients: ", mDistCoef.t());
    AF_CONFIG_FIELD("fps:                ", fps_);
    if(is_rgb_)
        AF_CONFIG_FIELD("color order:        ", "RGB (ignored if grayscale)");
    else
        AF_CONFIG_FIELD("color order:        ", "BGR (ignored if grayscale)");
    AF_CONFIG_END();
}

} // namespace AF_VSLAM
