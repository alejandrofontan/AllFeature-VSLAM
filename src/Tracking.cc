#include "Tracking.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>

#include "Converter.h"
#include "FeatureFactory.h"
#include "Initializer.h"
#include "Map.h"
#include "Optimizer.h"
#include "PnPsolver.h"
#include "Report.h"
#include "Utils.h"
#include "afvslam_log.hpp"
#include "onnx/OrtSession.h"
#include "SettingsReader.h"

namespace AF_VSLAM
{

TrackingParameters Tracking::params{};

void Tracking::LoadParameters(const cv::FileStorage &fSettings)
{
    auto read_if_present = [&fSettings](const char* key, auto& field)
    {
        read_setting_if_present(fSettings, key, field);
    };

    int sequential = params.sequential ? 1 : 0;   // cv::FileStorage has no bool reader
    read_if_present("Tracking.Sequential", sequential);
    params.sequential = (sequential != 0);

    read_if_present("Tracking.InitMinKeypoints", params.init_min_keypoints);
    read_if_present("Tracking.InitSigma", params.init_sigma);
    read_if_present("Tracking.InitMinMatches", params.init_min_matches);
    read_if_present("Tracking.InitRansacIterations", params.init_ransac_iterations);
    read_if_present("Tracking.InitMinMedianDisparity", params.init_min_median_disparity);
    read_if_present("Tracking.InitGbaIterations", params.init_gba_iterations);
    read_if_present("Tracking.InitMinTrackedPoints", params.init_min_tracked_points);
    read_if_present("Tracking.InitMinDepthSamples", params.init_min_depth_samples);
    float features_fraction = params.features_fraction.load();
    read_if_present("Tracking.FeaturesFraction", features_fraction);
    params.features_fraction.store(features_fraction);
    read_if_present("Tracking.NominalImageArea", params.nominal_image_area);
    read_if_present("Tracking.ImageWidth", params.image_width);
    read_if_present("Tracking.ImageHeight", params.image_height);
    std::string frame_tracking = params.frame_tracking == FrameTracking::optical_flow ? "optical_flow" : "reference_keyframe";
    read_if_present("Tracking.FrameTracking", frame_tracking);
    read_if_present("Tracking.FlowAssociationRadius", params.flow_association_radius);
    read_if_present("Tracking.FlowMaxBackwardError", params.flow_max_backward_error);
    read_if_present("Tracking.FlowMinMatches", params.flow_min_matches);
    read_if_present("Tracking.FlowWindow", params.flow_window);
    read_if_present("Tracking.FlowPyramidLevels", params.flow_pyramid_levels);
    if(frame_tracking == "reference_keyframe")
        params.frame_tracking = FrameTracking::reference_keyframe;
    else if(frame_tracking == "optical_flow")
        params.frame_tracking = FrameTracking::optical_flow;
    else
    {
        AF_ERROR("[Tracking] Tracking.FrameTracking must be \"reference_keyframe\" or \"optical_flow\", got \""
                 << frame_tracking << "\"");
        exit(-1);
    }
    read_if_present("Tracking.TrackRefMinMatches", params.track_ref_min_matches);
    read_if_present("Tracking.TrackRefMinInliers", params.track_ref_min_inliers);
    read_if_present("Tracking.PoseRescueMinMatches", params.pose_rescue_min_matches);
    read_if_present("Tracking.TrackLocalMapMinInliers", params.track_local_map_min_inliers);
    read_if_present("Tracking.TrackLocalMapMinInliersAfterReloc", params.track_local_map_min_inliers_after_reloc);
    read_if_present("Tracking.MaxLocalKeyframes", params.max_local_keyframes);
    read_if_present("Tracking.BestCovisibleKeyframes", params.best_covisible_keyframes);
    read_if_present("Tracking.ViewingCosLimit", params.viewing_cos_limit);
    read_if_present("Tracking.ProjectionGateSigma2", params.projection_gate_sigma2);
    read_if_present("Tracking.KeyframeMinInformation", params.keyframe_min_information);
    int log_keyframe_information = params.log_keyframe_information ? 1 : 0;   // cv::FileStorage has no bool reader
    read_if_present("Tracking.LogKeyframeInformation", log_keyframe_information);
    params.log_keyframe_information = (log_keyframe_information != 0);
    read_if_present("Tracking.EmergencyInlierDropRatio", params.emergency_inlier_drop_ratio);
    read_if_present("Tracking.InliersHistorySize", params.inliers_history_size);
    read_if_present("Tracking.EmergencyKeyframeCooldown", params.emergency_keyframe_cooldown);
    read_if_present("Tracking.RelocEmbargoBypassInliers", params.reloc_embargo_bypass_inliers);
    read_if_present("Tracking.RelocEmbargoMinKeyframes", params.reloc_embargo_min_keyframes);
    read_if_present("Tracking.RelocMinMatches", params.reloc_min_matches);
    read_if_present("Tracking.RelocInliersHigh", params.reloc_inliers_high);
    read_if_present("Tracking.RelocInliersMedium", params.reloc_inliers_medium);
    read_if_present("Tracking.RelocInliersLow", params.reloc_inliers_low);
    read_if_present("Tracking.RelocRansacProbability", params.reloc_ransac_probability);
    read_if_present("Tracking.RelocRansacMinInliers", params.reloc_ransac_min_inliers);
    read_if_present("Tracking.RelocRansacMaxIterations", params.reloc_ransac_max_iterations);
    read_if_present("Tracking.RelocRansacEpsilon", params.reloc_ransac_epsilon);
}

Tracking::Tracking(std::shared_ptr<PlaceRecognition> place_recognition,
                   std::shared_ptr<FrameDrawer> frame_drawer, std::shared_ptr<MapDrawer> map_drawer,
                   std::shared_ptr<Map> map,
                   const std::string& calibration_yaml, const std::string& settings_yaml,
                   const std::vector<FeatureType>& feature_types,
                   const bool fix_image_size):
    feature_types_(feature_types), place_recognition_(std::move(place_recognition)),
    frame_drawer_(frame_drawer), map_drawer_(map_drawer), map_(map), fix_image_size_(fix_image_size)
{
    load_camera_parameters(calibration_yaml, settings_yaml);

    max_frames_ = static_cast<size_t>(fps_);

    features_fraction_in_use_ = std::clamp(params.features_fraction.load(), 0.0f, 1.0f);
    for (const FeatureType ft : feature_types)
        feature_extractor_left_[ft] = get_feature_extractor(features_budget(ft, features_fraction_in_use_), ft);

    const int frame_width = fix_image_size_ ? (image_width_ / 32) * 32 : image_width_;
    const int frame_height = fix_image_size_ ? (image_height_ / 32) * 32 : image_height_;
    for (const auto& [ft, extractor] : feature_extractor_left_)
        extractor->prepare(frame_width, frame_height);

    matcher_ = std::make_shared<FeatureMatcher>(feature_types, "Tracking");
    if(params.frame_tracking == FrameTracking::optical_flow)
        AF_INFO("[Tracking] frame-to-frame stage: optical_flow (KLT on the last frame's tracked points, association radius "
                << params.flow_association_radius << " px, min " << params.flow_min_matches
                << " matches; fallback: the reference-keyframe match)");
    else
        AF_INFO("[Tracking] frame-to-frame stage: reference_keyframe (a descriptor match per frame)");
}

void Tracking::release_map_references()
{
    ref_keyframe_.reset();
    local_keyframes_.clear();
    local_points_.clear();
    current_frame_ = Frame();
    last_frame_ = Frame();
    initial_frame_ = Frame();
}

mat4f Tracking::grab_image(Image &im, const double timestamp)
{
    // Report: the thread's total for this frame (its frame is set below, before the extraction)
    AF_REPORT_SCOPE();

    // A degenerate initialization asked for a reset: before this frame takes an id, so ids restart with it
    if(reset_after_failed_initialization_)
    {
        reset_after_failed_initialization_ = false;
        reset();
    }

    // This thread's model runs go ahead of Local Mapping's and Loop Closing's on a shared session
    // (OrtSession, Threading). Set per frame rather than once: this is the thread's entry point,
    // and the flag is a thread-local bool.
    onnx::set_run_priority(true);

    // A Features Fraction moved in the viewer takes effect from this frame
    apply_features_fraction();

    const auto t_start = std::chrono::steady_clock::now();

    // Convert image to grayscale and resize
    im.get_gray_image();
    if(fix_image_size_)
        im.fix_image_size(image_width_, image_height_);

    // The calibration is derived from the image AS RESIZED/CROPPED above: image_width_ and
    // image_height_ are the requested size and can be larger (Image::fix_image_size crops to
    // multiples of 32), so the dimensions have to come from the image itself.
    if(!calibration_)
        calibration_ = std::make_shared<const Calibration>(K, distortion_coefficients,
                                                           im.grayImg.cols, im.grayImg.rows,
                                                           FRAME_GRID_COLS, FRAME_GRID_ROWS,
                                                           depth_noise_coeff_);

    last_grey_image_ = grey_image_;                   // the previous frame's grey, for optical-flow tracking
    grey_image_ = im.grayImg.clone();                 // the caller may reuse its Image
    color_image_ = im.img;
    mask_image_ = im.mask;

    // Create the frame (feature extraction). The report context is set first, with the id the frame is about
    // to take, so that what the extraction reports is stamped with its own frame
    AF_REPORT_CONTEXT("tracking", Frame::next_id);
    current_frame_ = Frame(im, timestamp, feature_extractor_left_, place_recognition_, calibration_);

    const mat4f Tcw = track();

    // Median excludes the current frame (updated below)
    if(viewer_)
        viewer_->set_grab_image_time_median(map_median(grab_image_times_));

    if(state_ == TrackingState::OK)
        grab_image_times_[int(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count())]++;

    return Tcw;
}

mat4f Tracking::track()
{
    AF_REPORT_CONTEXT("tracking", current_frame_.frame_id);

    // A degenerate initialization asked for a reset (create_initial_map): run it here, before the map lock is
    // taken. reset() waits for Local Mapping and Loop Closing, and a global-BA correction holding Local Mapping
    // stopped while it waits for map_update_mutex_ deadlocked against a reset run under that lock.
    if(state_ == TrackingState::NO_IMAGES_YET)
        state_ = TrackingState::NOT_INITIALIZED;

    last_processed_state_ = state_;

    // Get Map Mutex -> Map cannot be changed
    std::unique_lock<std::mutex> lock(map_->map_update_mutex_);

    // No map yet: the frame goes to two-view initialization, which finishes the
    // frame itself (drawer update, first stored relative pose once a map exists).
    if(state_ == TrackingState::NOT_INITIALIZED)
    {
        monocular_initialization();
        if(params.sequential)
        {
            // The two founding keyframes must be fully mapped before the next frame
            lock.unlock();
            wait_for_local_mapper(/*drain_queue=*/true);
        }
        return current_frame_.get_Tcw();
    }

    // System is initialized: track the frame.
    bool ok{false};
    if(state_ == TrackingState::OK)
    {
        // Local Mapping might have changed some MapPoints tracked in last frame
        check_replaced_in_last_frame();
        if(params.frame_tracking == FrameTracking::optical_flow)   // false: not lost, fall back to the descriptor match
            ok = run_tracking_stage([this] { return track_with_optical_flow(); });
        if(!ok)
            ok = run_tracking_stage([this] { return track_reference_keyframe(); });
    }
    else
        ok = run_tracking_stage([this] { return relocalize(); });

    current_frame_.ref_keyframe = ref_keyframe_;

    // If we have an initial estimation of the camera pose and matching, track the local map.
    if(ok)
        ok = run_tracking_stage([this] { return track_local_map(); });

    state_ = ok ? TrackingState::OK : TrackingState::LOST;

    frame_drawer_->update(this);

    // If tracking was good, check if we insert a keyframe
    if(ok)
    {
        ++num_tracked_frames_;

        map_drawer_->set_current_camera_pose(current_frame_.get_Tcw());

        // Check if we need to insert a new keyframe
        if(need_new_keyframe())
            create_new_keyframe();

        // We allow points with high innovation (considered outliers by the Huber function)
        // to pass to the new keyframe, so that bundle adjustment will finally decide
        // if they are outliers or not. We don't want the next frame to estimate its pose
        // with those points, so we discard them in the frame.
        current_frame_.drop_outlier_points();
    }

    if(!current_frame_.ref_keyframe)
        current_frame_.ref_keyframe = ref_keyframe_;

    // Both read current_frame_, so they run before it is moved from
    store_relative_pose();
    const mat4f Tcw = current_frame_.get_Tcw();

    // Moved, not copied: grab_image rebuilds current_frame_ from scratch next frame, and nothing
    // reads it in between. The copy this replaced was ~250 KB plus one atomic refcount increment
    // per matched map point, every frame.
    last_frame_ = std::move(current_frame_);

    // Sequential (deterministic) mode: every frame ends with Local Mapping idle and the
    // map fully updated, so the next frame always sees the same map state (issue #16).
    if (params.sequential)
    {
        lock.unlock();
        wait_for_local_mapper(/*drain_queue=*/true);
    }
    // An emergency keyframe was just inserted: block until Local Mapping has taken it
    else if (run_.emergency_keyframe)
    {
        lock.unlock();
        wait_for_local_mapper(/*drain_queue=*/false);
        run_.emergency_keyframe = false;
    }

    return Tcw;
}

void Tracking::store_relative_pose()
{
    if(state_ == TrackingState::OK)
        run_.last_frame_relative_pose = current_frame_.get_Tcw() * current_frame_.ref_keyframe->get_Twc();
}

void Tracking::monocular_initialization()
{
    attempt_monocular_initialization();
    frame_drawer_->update(this);
    if(state_ == TrackingState::OK)
        store_relative_pose();
}

void Tracking::attempt_monocular_initialization()
{
    // Every gate below pools over all feature types.
    const auto total_keypoints = [this](const Frame& frame) {
        size_t n = 0;
        for (const auto& ft : feature_types_)
            n += frame.features.at(ft).keypoints.size();
        return n;
    };

    // The previous attempt's matches index the previous reference / current frames: never let them reach the
    // FrameDrawer with this frame's keypoints (it draws matches_per_feature_ while NOT_INITIALIZED). They are
    // refilled below when this attempt matches.
    matches_per_feature_.clear();

    if(!initializer_)
    {
        // Set Reference Frame
        if(total_keypoints(current_frame_) > static_cast<size_t>(params.init_min_keypoints))
        {
            initial_frame_ = current_frame_;
            last_frame_ = current_frame_;
            initializer_ = std::make_shared<Initializer>(current_frame_, params.init_sigma, params.init_ransac_iterations);
            return;
        }
    }
    else
    {
        // Try to initialize
        if(total_keypoints(current_frame_) <= static_cast<size_t>(params.init_min_keypoints))
        {
            initializer_ = nullptr;
            return;
        }

        // Find correspondences
        const auto matched_pairs = matcher_->match_frames_for_initialization(initial_frame_, current_frame_, feature_types_);

        // Fill matches_per_feature_ (used later in create_initial_map) and
        // init_matches_ (flat over all feature types, the structure the initializer uses).
        init_matches_.clear();
        size_t offset2 = 0;
        for (const auto& ft : feature_types_) {
            const size_t offset1 = init_matches_.size();
            init_matches_.resize(offset1 + initial_frame_.features.at(ft).keypoints.size(), -1);
            auto& matches_ft = matches_per_feature_[ft];
            matches_ft.assign(initial_frame_.features.at(ft).keypoints.size(), -1);
            for (const auto& m : matched_pairs.at(ft)) {
                matches_ft[m.first] = m.second;
                init_matches_[offset1 + m.first] = m.second + offset2;
            }
            offset2 += current_frame_.features.at(ft).keypoints.size();
        }

        // Check if there are enough correspondences (pooled over all feature types)
        size_t num_matches = 0;
        for (const auto& ft : feature_types_)
            num_matches += matched_pairs.at(ft).size();
        if(num_matches < static_cast<size_t>(params.init_min_matches))
        {
            initializer_ = nullptr;
            return;
        }

        // Disparity gate: with (near-)zero baseline — e.g. a static camera — two-view
        // initialization can only produce ill-conditioned geometry, so don't attempt it.
        // Keep the reference frame: disparity only grows once the camera starts moving.
        {
            std::vector<float> disparities;
            disparities.reserve(num_matches);
            for (const auto& ft : feature_types_) {
                const auto& keypoints1 = initial_frame_.features.at(ft).keypoints;
                const auto& keypoints2 = current_frame_.features.at(ft).keypoints;
                for (const auto& m : matched_pairs.at(ft))
                    disparities.push_back(static_cast<float>(cv::norm(keypoints2[m.second].pt - keypoints1[m.first].pt)));
            }
            const auto mid = disparities.begin() + disparities.size() / 2;
            std::nth_element(disparities.begin(), mid, disparities.end());
            if (*mid < params.init_min_median_disparity)
                return;
        }

        mat3f Rcw{}; // Current Camera Rotation
        vec3f tcw{}; // Current Camera Translation
        std::vector<bool> triangulated;
        if(initializer_->initialize(current_frame_, init_matches_, Rcw, tcw, init_points3d_, triangulated))
        {
            // Discard matches the initializer could not triangulate
            for (size_t j = 0; j < init_matches_.size(); j++)
                if (init_matches_[j] >= 0 && !triangulated[j])
                    init_matches_[j] = -1;

            // Set Frame Poses
            initial_frame_.set_pose(mat4f::Identity());
            mat4f Tcw{mat4f::Identity()};
            Tcw.block<3,3>(0,0) = Rcw;
            Tcw.block<3,1>(0,3) = tcw;
            current_frame_.set_pose(Tcw);
            create_initial_map();
        }
    }
}

void Tracking::create_initial_map()
{
    // Create the two founding keyframes
    Keyframe keyframe_ini = std::make_shared<KeyFrame>(initial_frame_, map_, place_recognition_);
    Keyframe keyframe_cur = std::make_shared<KeyFrame>(current_frame_, map_, place_recognition_);

    map_->add_keyframe(keyframe_ini);
    map_->add_keyframe(keyframe_cur);

    // Triangulated matches become map points observed by both keyframes
    // (create_monocular_map_point registers them in the map)
    int num_triangulated = 0;
    size_t flat_index = 0; // runs over init_matches_, flattened across feature types
    for (const auto& ft : feature_types_) {
        const auto& matches = matches_per_feature_.at(ft);
        for (size_t i = 0; i < matches.size(); i++, flat_index++) {
            if (init_matches_[flat_index] < 0)
                continue;

            Pt map_point = map_->create_monocular_map_point(init_points3d_[flat_index], keyframe_cur,
                                                            KeypointIndex(matches[i]),
                                                            keyframe_ini, KeypointIndex(i), ft);
            if (!map_point)
                continue;   // a slot was taken between the emptiness check and the write
            current_frame_.pts.at(ft)[matches[i]] = map_point;
            current_frame_.outliers.at(ft)[matches[i]] = false;
            num_triangulated++;
        }
    }

    // Depth-verified metric scale, decided from the raw triangulations (median of
    // sensor-depth / triangulated-depth over the geometrically verified matches;
    // keyframe_ini sits at the origin, so a point's z IS its depth there). Decided
    // BEFORE global BA so that depth-completed matches can join the map and be
    // refined by the BA — whose RGB-D inverse-depth edges then anchor the metric
    // scale during the optimization itself.
    std::vector<float> depth_ratios;
    for (const auto& ft : feature_types_) {
        const auto& inv_depth_kf = keyframe_ini->features.at(ft).inv_depth;
        const std::vector<Pt> map_points = keyframe_ini->get_map_point_matches(ft);
        for (size_t i = 0; i < map_points.size(); i++)
        {
            if (!map_points[i])
                continue;
            const float sensor_inv_depth = inv_depth_kf[i];
            if (sensor_inv_depth <= 0.0f)
                continue;
            const float triangulated_depth = map_points[i]->get_world_pos()(2);
            if (triangulated_depth <= 0.0f)
                continue;
            depth_ratios.push_back((1.0f / sensor_inv_depth) / triangulated_depth);
        }
    }
    const bool metric_scale = depth_ratios.size() >= static_cast<size_t>(params.init_min_depth_samples);

    int num_depth_completed = 0;
    if (metric_scale)
    {
        const auto mid = depth_ratios.begin() + depth_ratios.size() / 2;
        std::nth_element(depth_ratios.begin(), mid, depth_ratios.end());
        const float scale = *mid;

        // Rescale the baseline and the triangulated points to metric now, so that
        // metric depth back-projections below are scale-consistent with them
        mat4f Tc2w = keyframe_cur->get_Tcw();
        Tc2w.block<3,1>(0,3) *= scale;
        keyframe_cur->set_pose(Tc2w);
        for (const auto& ft : feature_types_)
            for (const Pt& map_point : keyframe_ini->get_map_point_matches(ft))
                if (map_point)
                    map_point->set_world_pos(map_point->get_world_pos() * scale);

        // Matched pairs the initializer could not triangulate (low parallax,
        // degenerate geometry): back-project from sensor depth — reference frame
        // preferred (origin), current frame as fallback — and keep BOTH
        // observations, exactly like a triangulated point. Added before BA, so
        // they are refined together with everything else.
        const float cx = calibration_->cx, cy = calibration_->cy;
        const float invfx = calibration_->invfx, invfy = calibration_->invfy;
        const mat4f Twc_cur = keyframe_cur->get_Twc();
        const mat3f Rwc_cur = Twc_cur.block<3,3>(0,0);
        const vec3f twc_cur = Twc_cur.block<3,1>(0,3);

        flat_index = 0;
        for (const auto& ft : feature_types_) {
            const auto& matches = matches_per_feature_.at(ft);
            const auto& inv_depth_ini = keyframe_ini->features.at(ft).inv_depth;
            const auto& inv_depth_cur = keyframe_cur->features.at(ft).inv_depth;
            const auto& keypoints_ini = keyframe_ini->features.at(ft).keypoints;
            const auto& keypoints_cur = keyframe_cur->features.at(ft).keypoints;
            for (size_t i = 0; i < matches.size(); i++, flat_index++) {
                if (matches[i] < 0 || init_matches_[flat_index] >= 0)
                    continue; // no match, or already triangulated
                const int j = matches[i];

                vec3f world_pos;
                if (inv_depth_ini[i] > 0.0f)
                {
                    const cv::KeyPoint& kp = keypoints_ini[i]; // keyframe_ini is the origin
                    world_pos = vec3f{(kp.pt.x - cx) * invfx, (kp.pt.y - cy) * invfy, 1.0f} / inv_depth_ini[i];
                }
                else if (inv_depth_cur[j] > 0.0f)
                {
                    const cv::KeyPoint& kp = keypoints_cur[j];
                    const vec3f xn{(kp.pt.x - cx) * invfx, (kp.pt.y - cy) * invfy, 1.0f};
                    world_pos = Rwc_cur * (xn / inv_depth_cur[j]) + twc_cur;
                }
                else
                    continue; // matched, but neither frame has depth here

                Pt map_point = map_->create_monocular_map_point(world_pos, keyframe_cur, KeypointIndex(j),
                                                                keyframe_ini, KeypointIndex(i), ft);
                if (!map_point)
                    continue;   // a slot was taken between the emptiness check and the write
                current_frame_.pts.at(ft)[j] = map_point;
                current_frame_.outliers.at(ft)[j] = false;
                num_depth_completed++;
            }
        }
    }

    keyframe_ini->update_connections();
    keyframe_cur->update_connections();

    Optimizer::global_bundle_adjustment(map_, params.init_gba_iterations);

    const std::optional<float> median_depth = keyframe_ini->compute_scene_median_depth();
    const int tracked_map_points = keyframe_cur->count_tracked_map_points(1);

    if (!median_depth || *median_depth < 0 || tracked_map_points < params.init_min_tracked_points)
    {
        AF_WARN("create_initial_map: degenerate initialization (median_depth=" << median_depth.value_or(-1.0f)
                << ", tracked map points=" << tracked_map_points << " < " << params.init_min_tracked_points
                << ") — resetting...");
        reset_after_failed_initialization_ = true;   // run at the start of the next frame, outside the map lock
        return;
    }

    // Monocular fallback (no usable depth): normalize to the arbitrary
    // "median scene depth = 1" convention, from the BA-refined map as before
    if (!metric_scale)
    {
        const float inv_median_depth = 1.0f / *median_depth;
        mat4f Tc2w = keyframe_cur->get_Tcw();
        Tc2w.block<3,1>(0,3) *= inv_median_depth;
        keyframe_cur->set_pose(Tc2w);
        for (const auto& ft : feature_types_)
            for (const Pt& map_point : keyframe_ini->get_map_point_matches(ft))
                if (map_point)
                    map_point->set_world_pos(map_point->get_world_pos() * inv_median_depth);
    }

    // Unmatched keypoints of the current keyframe with valid depth: single-
    // observation points added AFTER the BA (nothing to refine against yet), same
    // any-inv_depth>0 trust policy as LocalMapping's depth-seeded pass. Metric map
    // only. The reference frame's unmatched keypoints are deliberately left out:
    // without an association they would duplicate the same surfaces.
    int num_depth_only = 0;
    if (metric_scale)
    {
        const float cx = calibration_->cx, cy = calibration_->cy;
        const float invfx = calibration_->invfx, invfy = calibration_->invfy;
        const mat4f Twc_cur = keyframe_cur->get_Twc();
        const mat3f Rwc_cur = Twc_cur.block<3,3>(0,0);
        const vec3f twc_cur = Twc_cur.block<3,1>(0,3);

        for (const auto& ft : feature_types_) {
            const auto& inv_depth_cur = keyframe_cur->features.at(ft).inv_depth;
            const auto& keypoints_cur = keyframe_cur->features.at(ft).keypoints;
            for (size_t j = 0; j < inv_depth_cur.size(); j++) {
                if (inv_depth_cur[j] <= 0.0f || keyframe_cur->get_map_point(j, ft))
                    continue;
                const cv::KeyPoint& kp = keypoints_cur[j];
                const vec3f xn{(kp.pt.x - cx) * invfx, (kp.pt.y - cy) * invfy, 1.0f};
                Pt map_point = map_->create_map_point(Rwc_cur * (xn / inv_depth_cur[j]) + twc_cur,
                                                      keyframe_cur, KeypointIndex(j), ft);
                if (!map_point)
                    continue;   // a slot was taken between the emptiness check and the write
                current_frame_.pts.at(ft)[j] = map_point;
                current_frame_.outliers.at(ft)[j] = false;
                num_depth_only++;
            }
        }
    }

    AF_INFO("create_initial_map: " << map_->map_points_in_map() << " points ("
            << num_triangulated << " triangulated, " << num_depth_completed
            << " depth-completed matches, " << num_depth_only << " depth-only), scale: "
            << (metric_scale ? "metric (depth-verified)" : "monocular convention"));

    local_mapper_->insert_keyframe(keyframe_ini);
    local_mapper_->insert_keyframe(keyframe_cur);

    current_frame_.set_pose(keyframe_cur->get_Tcw());

    local_keyframes_.push_back(keyframe_cur);
    local_keyframes_.push_back(keyframe_ini);
    local_points_ = map_->get_all_map_points();
    ref_keyframe_ = keyframe_cur;
    current_frame_.ref_keyframe = keyframe_cur;

    last_frame_ = Frame(current_frame_);

    map_drawer_->set_current_camera_pose(keyframe_cur->get_Tcw());
    map_->keyframe_origins_.push_back(keyframe_ini);

    state_ = TrackingState::OK;
}

void Tracking::check_replaced_in_last_frame()
{
    for (auto& [ft, pts] : last_frame_.pts)
        for (Pt& map_point : pts)
            if (map_point)
                if (const Pt replacement = map_point->get_replaced())
                    map_point = replacement;
}


bool Tracking::track_reference_keyframe()
{
    AF_REPORT_SCOPE();

    // 1. Match: reference-keyframe descriptors, joint geometric filter; each match takes the keyframe's map point
    std::map<FeatureType, std::vector<Pt>> map_point_matches;
    const std::map<FeatureType, int> num_matches_per_feature =
        matcher_->match_keyframe_to_frame(ref_keyframe_, current_frame_, map_point_matches, feature_types_);

    int num_matches = 0;
    for (auto& [ft, matches] : map_point_matches)
    {
        current_frame_.outliers.at(ft) = std::vector<bool>(matches.size(), false);
        current_frame_.pts.at(ft) = std::move(matches);
        num_matches += num_matches_per_feature.at(ft);
    }

    if(num_matches < params.track_ref_min_matches)
    {
        std::ostringstream reason;
        reason << "track_reference_keyframe: insufficient matches to reference keyframe (num_matches="
               << num_matches << " < " << params.track_ref_min_matches << ")"
               << " | frame=" << current_frame_.frame_id << " ref_keyframe=KF" << ref_keyframe_->keyId;
        throw TrackingLostException(reason.str());
    }

    // 2. Pose: optimized from the last frame's pose; too few inliers -> lost
    const int num_inliers = optimize_pose_from_last_frame(num_matches, "track_reference_keyframe");
    if(num_inliers < params.track_ref_min_inliers)
    {
        std::ostringstream reason;
        reason << "track_reference_keyframe: insufficient inlier matches after pose optimization (num_inliers="
               << num_inliers << " < " << params.track_ref_min_inliers << ")"
               << " | frame=" << current_frame_.frame_id << " raw_matches=" << num_matches;
        throw TrackingLostException(reason.str());
    }

    discard_pose_outliers();
    return true;
}

int Tracking::optimize_pose_from_last_frame(const int num_matches, const char* stage)
{
    // Seed: the last frame's Tlr on its reference keyframe's current pose (absorbs BA and loop corrections); no motion model
    const mat4f seed_pose = run_.last_frame_relative_pose * last_frame_.ref_keyframe->get_Tcw();
    current_frame_.set_pose(seed_pose);
    int num_inliers = Optimizer::pose_optimization(&current_frame_);

    // Rescue: a collapse despite many matches retries without the depth channel (RGB-D only: mono would repeat itself)
    const auto has_matched_depth = [this]
    {
        for (const auto& [ft, pts] : current_frame_.pts)
        {
            const std::vector<float>& inv_depth = current_frame_.features.at(ft).inv_depth;
            for (size_t i = 0; i < pts.size() && i < inv_depth.size(); i++)
                if (pts[i] && inv_depth[i] > 0.0f)
                    return true;
        }
        return false;
    };
    const bool rescue = num_inliers < params.track_ref_min_inliers
                        && num_matches >= params.pose_rescue_min_matches && has_matched_depth();
    if(rescue)
    {
        AF_WARN(stage << ": pose optimization collapsed (" << num_inliers
                << " inliers of " << num_matches << " raw matches) — retrying without depth channel"
                << " | frame=" << current_frame_.frame_id);
        current_frame_.set_pose(seed_pose);
        num_inliers = Optimizer::pose_optimization(&current_frame_, /*use_depth_channel=*/false);
    }
    AF_REPORT("pose.rescue", "", rescue);

#ifdef ALLFEATURE_REPORT
    // Seed -> optimized pose: the motion the constant-position seed missed (map units; metric in RGB-D)
    const mat4f seed_to_optimized = current_frame_.get_Tcw() * seed_pose.inverse();
    const float seed_error = seed_to_optimized.block<3,1>(0,3).norm();
    const float seed_error_deg = Eigen::AngleAxisf(mat3f(seed_to_optimized.block<3,3>(0,0))).angle() * 180.0f / float(M_PI);
    AF_REPORT("pose.seed_error", "", seed_error);
    AF_REPORT("pose.seed_error_deg", "", seed_error_deg);
#endif
    report_pose_inliers();

    return num_inliers;
}

void Tracking::discard_pose_outliers()
{
    // Outliers lose their match and are marked seen, so search_local_points neither re-matches nor re-counts them
    for (const auto& [ft, pts] : current_frame_.pts)
        for (size_t i = 0; i < pts.size(); i++)
            if (pts[i] && current_frame_.outliers.at(ft)[i])
                pts[i]->last_frame_seen = current_frame_.frame_id;
    current_frame_.drop_outlier_points();   // the flags too: a slot re-matched later must not start as an outlier
}

bool Tracking::track_with_optical_flow()
{
    AF_REPORT_SCOPE();

    // A failed attempt leaves no match behind: the reference-keyframe match starts from a clean frame
    const auto fall_back = [this]([[maybe_unused]] const char* why, [[maybe_unused]] const int count)
    {
        for (auto& [ft, pts] : current_frame_.pts)
        {
            std::fill(pts.begin(), pts.end(), nullptr);
            std::fill(current_frame_.outliers.at(ft).begin(), current_frame_.outliers.at(ft).end(), false);
        }
        AF_REPORT("flow.fallback", "", 1);
        AF_DEBUG("track_with_optical_flow: " << why << " (" << count << ") - falling back to the reference-keyframe match"
                 << " | frame=" << current_frame_.frame_id);
        return false;
    };
    if(last_grey_image_.empty() || last_grey_image_.size() != grey_image_.size())
        return fall_back("no previous frame of the same size", 0);

    // 1. Flow: the last frame's tracked points, KLT forward-backward, snapped to this frame's keypoints
    const FeatureMatcher::FlowMatchCounts counts = matcher_->match_frame_to_frame_by_flow(
        last_frame_, last_grey_image_, current_frame_, grey_image_, params.flow_window, params.flow_pyramid_levels,
        params.flow_max_backward_error, params.flow_association_radius);
    if(counts.matched < params.flow_min_matches)   // matched <= survived <= tracked: the first short count is the reason
    {
        if(counts.tracked < params.flow_min_matches)
            return fall_back("too few tracked points in the last frame", counts.tracked);
        if(counts.survived < params.flow_min_matches)
            return fall_back("too few flow tracks survived the forward-backward check", counts.survived);
        return fall_back("too few flow tracks landed on a detected keypoint", counts.matched);
    }

    // 2. Pose: as track_reference_keyframe; too few inliers -> fall back, never lost
    const int num_inliers = optimize_pose_from_last_frame(counts.matched, "track_with_optical_flow");
    if(num_inliers < params.track_ref_min_inliers)
        return fall_back("too few inliers after pose optimization", num_inliers);

    discard_pose_outliers();
    AF_REPORT("flow.fallback", "", 0);
    return true;
}

bool Tracking::track_local_map()
{
    AF_REPORT_SCOPE();

    // 1. Local map: keyframes observing the frame's points, expanded; their points
    update_local_keyframes();
    update_local_points();

    // 2. Search: unmatched local points projected into the frame, matched in the gate
    const std::vector<Pt> visible_points = search_local_points();

    // 3. Pose: re-optimized on every match; too few inliers -> lost (stricter shortly after a relocalization)
    run_.num_inliers = Optimizer::pose_optimization(&current_frame_);
    AF_REPORT("local.keyframes", "", local_keyframes_.size());
    AF_REPORT("local.points", "", local_points_.size());
    AF_REPORT("local.visible", "", visible_points.size());
    report_pose_inliers();

    if(shortly_after_relocalization() && run_.num_inliers < params.track_local_map_min_inliers_after_reloc)
    {
        std::ostringstream reason;
        reason << "track_local_map: insufficient inliers shortly after relocalization (inliers="
               << run_.num_inliers << " < " << params.track_local_map_min_inliers_after_reloc << ")"
               << " | frame=" << current_frame_.frame_id
               << " frames_since_reloc=" << (current_frame_.frame_id - *run_.last_reloc_frame_id)
               << " max_frames=" << max_frames_;
        throw TrackingLostException(reason.str());
    }

    if(run_.num_inliers < params.track_local_map_min_inliers)
    {
        std::ostringstream reason;
        reason << "track_local_map: insufficient inliers against local map (inliers="
               << run_.num_inliers << " < " << params.track_local_map_min_inliers << ")"
               << " | frame=" << current_frame_.frame_id << " local_points=" << local_points_.size();
        throw TrackingLostException(reason.str());
    }

    // Found/visible counts (MapPointCulling's found ratio), only from a frame that tracked
    for (const Pt& pt : visible_points)
        pt->increase_visible();
    for (const auto& [ft, pts] : current_frame_.pts)
        for(size_t i = 0; i < pts.size(); i++)
            if(pts[i] && !current_frame_.outliers.at(ft)[i])
                pts[i]->increase_found();

    return true;
}

void Tracking::report_pose_inliers() const
{
#ifdef ALLFEATURE_REPORT
    for (const auto& [ft, pts] : current_frame_.pts)
    {
        int num_inliers_ft = 0;
        for (size_t i = 0; i < pts.size(); i++)
            num_inliers_ft += (pts[i] && !current_frame_.outliers.at(ft)[i]) ? 1 : 0;
        AF_REPORT("pose.inliers", get_feature(ft).get_feature_name(), num_inliers_ft);
    }
#endif
}

void Tracking::update_local_keyframes()
{
    // Votes: each tracked point's observers (bad points leave the frame)
    std::map<KeyframeId, std::pair<Keyframe, int>> votes;   // keyframe, points shared with the frame
    for (auto& [ft, pts] : current_frame_.pts) {
        for(auto& pt : pts){
            if(!pt)
                continue;
            if(pt->is_bad()){
                pt = nullptr;
                continue;
            }
            for (const auto& [key_id, observer] : pt->get_observers()) {
                auto& [keyframe, num_shared] = votes[key_id];
                keyframe = observer;
                num_shared++;
            }
        }
    }
    if(votes.empty())
        return;

    // Every voter joins the local map; the top voter becomes the reference keyframe
    int max_shared_points = 0;
    Keyframe keyframe_most_shared{};
    std::set<KeyframeId> seen_keyframe_ids;
    local_keyframes_.clear();
    local_keyframes_.reserve(LOCAL_KEYFRAMES_RESERVE_SCALE * votes.size());
    for (const auto& [key_id, vote] : votes) {
        const auto& [keyframe, num_shared] = vote;
        if(keyframe->is_bad())
            continue;
        if(num_shared > max_shared_points){
            max_shared_points = num_shared;
            keyframe_most_shared = keyframe;
        }
        local_keyframes_.push_back(keyframe);
        seen_keyframe_ids.insert(key_id);
    }

    if(keyframe_most_shared){
        ref_keyframe_ = keyframe_most_shared;
        current_frame_.ref_keyframe = ref_keyframe_;
    }

    // Expansion: per local keyframe its best new covisible neighbour, a child and its parent, up to MaxLocalKeyframes
    for(size_t i = 0; i < local_keyframes_.size(); i++){
        if(local_keyframes_.size() > static_cast<size_t>(params.max_local_keyframes))
            break;
        const Keyframe keyframe = local_keyframes_[i];   // a copy: push_back may reallocate

        for(const Keyframe& neighbor : keyframe->get_best_covisibility_keyframes(params.best_covisible_keyframes)){
            if(!neighbor->is_bad() && seen_keyframe_ids.insert(neighbor->keyId).second){
                local_keyframes_.push_back(neighbor);
                break;
            }
        }

        for(const Keyframe& child : keyframe->get_children()){
            if(!child->is_bad() && seen_keyframe_ids.insert(child->keyId).second){
                local_keyframes_.push_back(child);
                break;
            }
        }

        const Keyframe parent = keyframe->get_parent();
        if(parent && !parent->is_bad() && seen_keyframe_ids.insert(parent->keyId).second)
            local_keyframes_.push_back(parent);
    }
}

void Tracking::update_local_points()
{
    // The local keyframes' points, each once (stamped with this frame; a bad one is stamped and skipped)
    local_points_.clear();
    const FrameId frame_id = current_frame_.frame_id;
    for(const Keyframe& keyframe : local_keyframes_){
        for(const FeatureType ft : feature_types_){
            for(const Pt& pt : keyframe->get_map_point_matches(ft)){
                if(!pt || pt->local_map_frame == frame_id)
                    continue;
                pt->local_map_frame = frame_id;
                if(!pt->is_bad())
                    local_points_.push_back(pt);
            }
        }
    }
}

std::vector<Pt> Tracking::search_local_points()
{
    // Already-matched points are visible and are not candidates (bad ones leave the frame)
    std::vector<Pt> visible_points;
    for (auto& [ft, pts] : current_frame_.pts) {
        for(auto& pt : pts){
            if(pt && !pt->is_bad()){
                visible_points.push_back(pt);
                pt->last_frame_seen = current_frame_.frame_id;
            }
            else
                pt = nullptr;
        }
    }

    // Candidates: the other local points in the frustum, with their projections from this frame's pose
    std::vector<Pt> candidates;
    std::vector<cv::Point2f> projections;
    candidates.reserve(local_points_.size());
    projections.reserve(local_points_.size());
    for(const Pt& pt : local_points_){
        if(pt->last_frame_seen == current_frame_.frame_id || pt->is_bad())
            continue;
        if(const std::optional<cv::Point2f> uv = current_frame_.project_if_visible(pt, params.viewing_cos_limit)){
            candidates.push_back(pt);
            projections.push_back(*uv);
        }
    }

    if(!candidates.empty())
        matcher_->match_map_points_to_frame(current_frame_, candidates, projections, params.projection_gate_sigma2);

    visible_points.insert(visible_points.end(), candidates.begin(), candidates.end());
    return visible_points;
}

bool Tracking::need_new_keyframe()
{
    AF_REPORT_SCOPE();

    // Recent inliers: the emergency trigger's reference (median of the last InliersHistorySize tracked frames,
    // once half of them exist)
    int median_recent_inliers = -1;
    if (run_.recent_inliers_history.size() >= static_cast<size_t>(params.inliers_history_size) / 2) {
        std::vector<int> history(run_.recent_inliers_history.begin(), run_.recent_inliers_history.end());
        const auto mid = history.begin() + history.size() / 2;
        std::nth_element(history.begin(), mid, history.end());
        median_recent_inliers = *mid;
    }
    run_.recent_inliers_history.push_back(run_.num_inliers);
    if (run_.recent_inliers_history.size() > static_cast<size_t>(params.inliers_history_size))
        run_.recent_inliers_history.pop_front();
    AF_REPORT("inliers.median_recent", "", median_recent_inliers);

    const float tau = LocalMapping::params.keyframe_culling_max_unexplained.load();
    keyframe_information_->record_thresholds(tau, params.keyframe_min_information);
    std::optional<KeyframeInformationValue> information{};

    // Every decision: the decision history (placecell Recorder), the report, the log line
    const auto decide = [&](const bool insert, const std::string& reason, const bool emergency = false,
                            [[maybe_unused]] const bool blocked = false) -> bool
    {
        const std::optional<float> unexplained =
            information ? std::optional<float>(information->unexplained) : std::nullopt;
        keyframe_information_->record_decision(current_frame_.frame_id, insert, unexplained, reason);
        AF_REPORT("keyframe.inserted", "", insert);
        AF_REPORT("keyframe.emergency", "", emergency);
        AF_REPORT("keyframe.blocked", "", blocked);   // novel, but Local Mapping busy
        if(insert || params.log_keyframe_information)
            log_keyframe_decision(insert, reason, emergency, information, tau, median_recent_inliers);
        return insert;
    };

    // Gates: no keyframe while Local Mapping is stopped (loop correction) or under the relocalization embargo
    if(local_mapper_->is_stopped() || local_mapper_->is_stop_requested())
        return decide(false, "local mapping stopped");
    if(shortly_after_relocalization()
       && map_->keyframes_in_map() > static_cast<size_t>(params.reloc_embargo_min_keyframes)
       && run_.num_inliers < params.reloc_embargo_bypass_inliers)
        return decide(false, "relocalization embargo");

    // Information: the view's unexplained information given the local keyframes (the culler's kernel and tau)
    information = keyframe_information_->information(current_frame_, local_keyframes_,
                                                     LocalMapping::params.keyframe_culling_centred);
    if(information)
        AF_REPORT("keyframe.information", "", information->unexplained);

    // Bands: below KeyframeMinInformation redundant, above tau novel; a label only while minInfo < tau
    if(information && information->unexplained < params.keyframe_min_information)
        return decide(false, "redundant");
    if(!information || information->unexplained <= tau)
        return decide(false, "not novel");

    // Insert: novel and Local Mapping accepts it (sequential mode: always idle here, the busy flag is not read)
    if(params.sequential || local_mapper_->accepts_keyframes())
        return decide(true, "novel");

    // Busy: only on an emergency inlier drop against the recent median, with a cooldown
    if(median_recent_inliers > 0
       && run_.num_inliers < params.emergency_inlier_drop_ratio * static_cast<float>(median_recent_inliers)
       && current_frame_.frame_id >= run_.last_emergency_keyframe_id + static_cast<FrameId>(params.emergency_keyframe_cooldown))
    {
        run_.last_emergency_keyframe_id = current_frame_.frame_id;
        run_.emergency_keyframe = true;
        return decide(true, "novel+emergency", /*emergency=*/true, /*blocked=*/true);
    }
    return decide(false, "novel, local mapping busy", /*emergency=*/false, /*blocked=*/true);
}

void Tracking::log_keyframe_decision(const bool insert, const std::string& reason, const bool emergency,
                                     const std::optional<KeyframeInformationValue>& information, const float tau,
                                     const int median_recent_inliers) const
{
    std::ostringstream line;
    line << (insert ? "need_new_keyframe: keyframe (" : "need_new_keyframe: skip (") << reason << ")"
         << " | frame=" << current_frame_.frame_id;
    if(information)
    {
        KeyframeId best_explainer_key_id = 0;
        for(const Keyframe& keyframe : local_keyframes_)
            if(keyframe && keyframe->frame_id == information->best_explainer)
                best_explainer_key_id = keyframe->keyId;
        line << " info=" << std::fixed << std::setprecision(3) << information->unexplained
             << " tau=" << tau << " min_info=" << params.keyframe_min_information
             << " explainers=" << information->explainers
             << " best=KF" << best_explainer_key_id << "(" << information->best_similarity << ")";
    }
    else
        line << " info=n/a";
    if(params.log_keyframe_information || emergency)
        line << " inliers=" << run_.num_inliers << " median_recent_inliers=" << median_recent_inliers
             << " keyframes=" << map_->keyframes_in_map();
    if(emergency)
        AF_WARN(line.str());
    else
        AF_INFO(line.str());
}

void Tracking::create_new_keyframe()
{
    // Lock Local Mapping against stopping while the keyframe is inserted; fails (no insertion) if a loop closure stopped it
    if(!local_mapper_->set_insertion_lock(true))
    {
        run_.emergency_keyframe = false;   // nothing inserted: nothing to wait for
        return;
    }

    const Keyframe keyframe = std::make_shared<KeyFrame>(current_frame_, map_, place_recognition_);

    ref_keyframe_ = keyframe;
    current_frame_.ref_keyframe = keyframe;

    local_mapper_->insert_keyframe(keyframe);
    local_mapper_->set_insertion_lock(false);
}

void Tracking::wait_for_local_mapper(const bool drain_queue) const
{
    // A stopped (loop correction) or finished Local Mapping processes nothing this frame could wait for
    while(((drain_queue && local_mapper_->has_new_keyframes()) || !local_mapper_->accepts_keyframes())
          && !local_mapper_->is_stopped() && !local_mapper_->is_finished())
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

bool Tracking::relocalize()
{
    // Without an active VPR backend there is no keyframe database to query —
    // recovery from a tracking loss is impossible.
    if(!place_recognition_->is_active())
        return false;

    AF_REPORT_SCOPE();

    const FeatureType feature_type = place_recognition_->verification_feature();
    current_frame_.compute_global_descriptor();

    const std::vector<Keyframe> candidates = place_recognition_->detect_relocalization_candidates(current_frame_);
    AF_REPORT("reloc.candidates", "", candidates.size());
    if(candidates.empty())
    {
        AF_REPORT("reloc.success", "", 0);
        return false;
    }
    const size_t num_candidates = candidates.size();

    // Match each candidate against the frame; enough matches arm a PnP solver for it
    std::vector<std::unique_ptr<PnPsolver>> solvers(num_candidates);
    std::vector<std::map<FeatureType, std::vector<Pt>>> matches_per_candidate(num_candidates);
    std::vector<bool> discarded(num_candidates, false);
    int num_active = 0;
    for(size_t i = 0; i < num_candidates; i++)
    {
        Keyframe candidate = candidates[i];
        if(candidate->is_bad())
        {
            discarded[i] = true;
            continue;
        }
        std::map<FeatureType, int> num_matches =
            matcher_->match_keyframe_to_frame(candidate, current_frame_, matches_per_candidate[i],
                                              std::vector<FeatureType>{feature_type});
        if(num_matches.at(feature_type) < params.reloc_min_matches)
        {
            discarded[i] = true;
            continue;
        }
        solvers[i] = std::make_unique<PnPsolver>(current_frame_, matches_per_candidate[i][feature_type], feature_type);
        solvers[i]->set_ransac_parameters(params.reloc_ransac_probability, params.reloc_ransac_min_inliers, params.reloc_ransac_max_iterations,
                                        RANSAC_MIN_SET, params.reloc_ransac_epsilon, ErrorModel::params.chi2_mono);
        num_active++;
    }
    AF_REPORT("reloc.candidates_armed", "", num_active);   // enough matches for a PnP solver

    // Round-robin P4P RANSAC over the surviving candidates until one pose
    // hypothesis is supported by enough inliers
    while(num_active > 0)
    {
        for(size_t i = 0; i < num_candidates; i++)
        {
            if(discarded[i])
                continue;

            std::vector<bool> inliers;
            int num_inliers = 0;
            bool ransac_exhausted = false;
            const cv::Mat Tcw = solvers[i]->iterate(RANSAC_ITERATIONS_PER_ROUND, ransac_exhausted, inliers, num_inliers);

            if(ransac_exhausted)
            {
                discarded[i] = true;
                num_active--;
            }

            if(Tcw.empty())
                continue;
            current_frame_.set_pose(Converter::to_matrix4f(Tcw));

            if(accept_relocalization_hypothesis(candidates[i], matches_per_candidate[i].at(feature_type),
                                                inliers, feature_type))
            {
                run_.last_reloc_frame_id = current_frame_.frame_id;
                AF_REPORT("reloc.success", "", 1);
                return true;
            }
        }
    }
    AF_REPORT("reloc.success", "", 0);
    return false;
}

bool Tracking::accept_relocalization_hypothesis(Keyframe candidate, const std::vector<Pt>& matches,
                                                const std::vector<bool>& inliers, const FeatureType feature_type)
{
    // One report scope per pose hypothesis: which candidate, the inliers at each step, what
    // each projection search added, and whether it was accepted
    AF_REPORT_SCOPE();
    AF_REPORT("reloc.keyframe_frame_id", "", candidate->frame_id);
    AF_REPORT("reloc.pnp_inliers", "", std::count(inliers.begin(), inliers.end(), true));

    // Seed the frame with the hypothesis' inlier matches
    std::set<Pt> found;
    for(size_t j = 0; j < inliers.size(); j++)
    {
        if(inliers[j])
        {
            current_frame_.pts.at(feature_type)[j] = matches[j];
            found.insert(matches[j]);
        }
        else
            current_frame_.pts.at(feature_type)[j] = nullptr;
    }

    // Pose optimization, then the outliers it flagged lose their match (their keypoints are free for the searches)
    const auto optimize = [&]
    {
        const int num_inliers = Optimizer::pose_optimization(&current_frame_);
        for(size_t j = 0; j < current_frame_.features.at(feature_type).count(); j++)
            if(current_frame_.outliers.at(feature_type)[j])
                current_frame_.pts.at(feature_type)[j] = nullptr;
        return num_inliers;
    };

    int num_good = optimize();
    AF_REPORT("reloc.inliers_optimized", "", num_good);
    if(num_good < params.reloc_inliers_low)
    {
        AF_REPORT("reloc.accepted", "", 0);
        return false;
    }

    // More matches for the hypothesis, now that it has a pose: project the candidate keyframe's
    // map points that are not matched yet into the frame, and match them to the frame's free
    // keypoints -- the same pose-guided step normal tracking uses for the local map
    // (search_local_points). The viewing-angle test is disabled (-1): relocalization must accept
    // a point seen from any direction. Returns the number of keypoints that gained a map point.
    const auto match_more_points = [&](const std::set<Pt>& already_found)
    {
        std::vector<Pt> points;
        std::vector<cv::Point2f> projections;
        for(const Pt& pt : candidate->get_map_point_matches(feature_type))
        {
            if(!pt || pt->is_bad() || already_found.count(pt))
                continue;
            if(const std::optional<cv::Point2f> uv = current_frame_.project_if_visible(pt, -1.0f))
            {
                points.push_back(pt);
                projections.push_back(*uv);
            }
        }
        if(points.empty())
            return 0;

        const std::vector<Pt>& frame_points = current_frame_.pts.at(feature_type);
        const auto num_matched = [&frame_points] {
            return std::count_if(frame_points.begin(), frame_points.end(), [](const Pt& pt) { return bool(pt); });
        };
        const auto before = num_matched();
        matcher_->match_map_points_to_frame(current_frame_, points, projections, params.projection_gate_sigma2);
        return static_cast<int>(num_matched() - before);
    };

    // Few inliers: look for more matches with the hypothesis pose and optimize again
    if(num_good < params.reloc_inliers_high)
    {
        const int additional = match_more_points(found);
        AF_REPORT("reloc.first_search_added", "", additional);
        if(additional + num_good >= params.reloc_inliers_high)
        {
            num_good = optimize();
            AF_REPORT("reloc.inliers_after_first_search", "", num_good);

            // Many inliers but still not enough: the pose has moved, so the projections have
            // too -- search once more and run a final optimization
            if(num_good > params.reloc_inliers_medium && num_good < params.reloc_inliers_high)
            {
                found.clear();
                for(int j = 0; j < static_cast<int>(current_frame_.features.at(feature_type).count()); j++)
                    if(current_frame_.pts.at(feature_type)[j])
                        found.insert(current_frame_.pts.at(feature_type)[j]);
                const int second_additional = match_more_points(found);
                AF_REPORT("reloc.second_search_added", "", second_additional);
                if(num_good + second_additional >= params.reloc_inliers_high)
                {
                    num_good = optimize();
                    AF_REPORT("reloc.inliers_after_second_search", "", num_good);
                }
            }
        }
    }

    AF_REPORT("reloc.accepted", "", num_good >= params.reloc_inliers_high);
    if(num_good < params.reloc_inliers_high)
        return false;

    AF_INFO("relocalize: succeeded | frame=" << current_frame_.frame_id
            << " feature=" << feature_name(feature_type)
            << " matchedKeyframe=" << candidate->keyId
            << " inliers=" << num_good << " requiredInliers=" << params.reloc_inliers_high);
    std::cout.flush(); // AF_INFO's stdout is fully buffered under the runner's redirect

    return true;
}

void Tracking::request_relocalization()
{
    if(state_ == TrackingState::OK)
        state_ = TrackingState::LOST;
}

void Tracking::reset()
{
    // Each step is logged and flushed before it runs, so a hang shows the step that started last
    const auto step = [](const char* what)
    {
        AF_INFO("reset: " << what);
        std::cout.flush();
    };
    AF_REPORT("tracking.reset", "", 1);   // frame ids restart: the report's frames repeat after this

    step("stopping viewer...");
    if(viewer_)
    {
        viewer_->request_stop();
        while(!viewer_->is_stopped() && !viewer_->isFinished())
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }

    step("resetting local mapping...");
    local_mapper_->request_reset();

    step("resetting loop closing...");
    loop_closing_->request_reset();

    step("clearing keyframe database and map...");
    place_recognition_->clear();
    keyframe_information_->clear();
    map_->clear();

    KeyFrame::next_id = 0;
    Frame::next_id = 0;
    MapPoint::next_id = 0;
    state_ = TrackingState::NO_IMAGES_YET;
    initializer_ = nullptr;

    // Everything pointing into the wiped map (ids restart, so a stale keyframe would alias a new one), and the run
    release_map_references();
    matches_per_feature_.clear();
    init_matches_.clear();
    last_grey_image_.release();
    grey_image_.release();
    num_tracked_frames_ = FOUNDING_FRAMES;
    run_ = {};
    grab_image_times_.clear();

    if(viewer_)
        viewer_->release();

    step("done");
}

}
