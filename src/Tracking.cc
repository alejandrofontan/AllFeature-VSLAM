/**
 * Tracking thread: the per-frame front-end of AllFeature-VSLAM (monocular, with
 * an optional RGB-D depth channel). Receives images (grab_image), initializes
 * the map via two-view monocular initialization, tracks each frame against the
 * reference keyframe and the local map, decides keyframe insertion, and
 * recovers from losses via relocalization. Auxiliary members (profilers,
 * diagnostics, calibration loading) live in Tracking_aux.{h,cc}.
 */
#include "Tracking.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include <opencv2/core/core.hpp>

#include "Converter.h"
#include "Initializer.h"
#include "Map.h"
#include "Optimizer.h"
#include "PnPsolver.h"
#include "Tracking_aux.h"
#include "Utils.h"
#include "afvslam_log.hpp"

namespace AF_VSLAM
{

TrackingParameters Tracking::params{};

void Tracking::LoadParameters(const cv::FileStorage &fSettings)
{
    auto read_if_present = [&fSettings](const char* key, auto& field)
    {
        const cv::FileNode node = fSettings[key];
        if(!node.empty())
            node >> field;
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
    read_if_present("Tracking.InitExtractorFeaturesScale", params.init_extractor_features_scale);
    read_if_present("Tracking.TrackRefMinMatches", params.track_ref_min_matches);
    read_if_present("Tracking.TrackRefMinInliers", params.track_ref_min_inliers);
    read_if_present("Tracking.TrackLocalMapMinInliers", params.track_local_map_min_inliers);
    read_if_present("Tracking.TrackLocalMapMinInliersAfterReloc", params.track_local_map_min_inliers_after_reloc);
    read_if_present("Tracking.MaxLocalKeyframes", params.max_local_keyframes);
    read_if_present("Tracking.BestCovisibleKeyframes", params.best_covisible_keyframes);
    read_if_present("Tracking.ViewingCosLimit", params.viewing_cos_limit);
    read_if_present("Tracking.KeyframeMinInformation", params.keyframe_min_information);
    int log_keyframe_information = params.log_keyframe_information ? 1 : 0;   // cv::FileStorage has no bool reader
    read_if_present("Tracking.LogKeyframeInformation", log_keyframe_information);
    params.log_keyframe_information = (log_keyframe_information != 0);
    read_if_present("Tracking.MinMedianFlow", params.min_median_flow);
    read_if_present("Tracking.MinSharedPointsForFlow", params.min_shared_points_for_flow);
    read_if_present("Tracking.RefMatchesRatio", params.ref_matches_ratio);
    read_if_present("Tracking.MinInliersForKeyframe", params.min_inliers_for_keyframe);
    read_if_present("Tracking.MinObservationsHigh", params.min_observations_high);
    read_if_present("Tracking.MinObservationsLow", params.min_observations_low);
    read_if_present("Tracking.YoungMapKeyframes", params.young_map_keyframes);
    read_if_present("Tracking.MinRefOverlap", params.min_ref_overlap);
    read_if_present("Tracking.EmergencyInlierDropRatio", params.emergency_inlier_drop_ratio);
    read_if_present("Tracking.InliersHistorySize", params.inliers_history_size);
    read_if_present("Tracking.EmergencyKeyframeCooldown", params.emergency_keyframe_cooldown);
    read_if_present("Tracking.RelocMinMatches", params.reloc_min_matches);
    read_if_present("Tracking.RelocInliersHigh", params.reloc_inliers_high);
    read_if_present("Tracking.RelocInliersMedium", params.reloc_inliers_medium);
    read_if_present("Tracking.RelocInliersLow", params.reloc_inliers_low);
    read_if_present("Tracking.RelocSearchRadiusCoarse", params.reloc_search_radius_coarse);
    read_if_present("Tracking.RelocSearchRadiusNarrow", params.reloc_search_radius_narrow);
    read_if_present("Tracking.RelocRansacProbability", params.reloc_ransac_probability);
    read_if_present("Tracking.RelocRansacMinInliers", params.reloc_ransac_min_inliers);
    read_if_present("Tracking.RelocRansacMaxIterations", params.reloc_ransac_max_iterations);
    read_if_present("Tracking.RelocRansacEpsilon", params.reloc_ransac_epsilon);
}

Tracking::Tracking(std::shared_ptr<PlaceRecognition> place_recognition,
                   std::shared_ptr<FrameDrawer> frame_drawer, std::shared_ptr<MapDrawer> map_drawer,
                   std::shared_ptr<Map> map,
                   const std::string& calibration_yaml, const std::string& settings_yaml,
                   const std::map<FeatureType, std::string>& feature_settings_yaml_file,
                   const std::vector<FeatureType>& feature_types,
                   const bool fix_image_size):
    feature_types_(feature_types), place_recognition_(std::move(place_recognition)),
    frame_drawer_(frame_drawer), map_drawer_(map_drawer), map_(map), fix_image_size_(fix_image_size)
{
    load_camera_parameters(calibration_yaml, settings_yaml);

    // Frame window for the post-relocalization embargo/strictness (~1 s at the camera rate)
    max_frames_ = static_cast<size_t>(fps_);

    // One extractor pair per feature type: normal, and a denser one for initialization
    for (const FeatureType ft : feature_types){
        feature_extractor_left_[ft] = get_feature_extractor(1, feature_settings_yaml_file.at(ft), ft);
        init_feature_extractor_[ft] = get_feature_extractor(params.init_extractor_features_scale, feature_settings_yaml_file.at(ft), ft);
    }

    matcher_ = std::make_shared<FeatureMatcher>(image_width_, image_height_, feature_types, "Tracking");
}

mat4f Tracking::grab_image(Image &im, const double timestamp)
{
    GrabProfiler profiler{};

    // Convert image to grayscale and resize
    im.get_gray_image(is_rgb_);
    if(fix_image_size_)
        im.fix_image_size(image_width_, image_height_);

    gray_image_ = im.grayImg;
    mask_image_ = im.mask;
    image_name_ = im.imageName;
    profiler.resize_done(resize_times_);

    // Create the frame (feature extraction); initialization uses the denser extractor set
    const auto& extractors = (state_ == TrackingState::NOT_INITIALIZED || state_ == TrackingState::NO_IMAGES_YET)
                           ? init_feature_extractor_ : feature_extractor_left_;
    current_frame_ = Frame(im, timestamp, extractors, place_recognition_, mK, mDistCoef, mbf, mThDepth);
    profiler.frame_created(frame_times_);

    track();
    profiler.tracking_done(tracking_times_, state_ == TrackingState::OK);

    // Median excludes the current frame (updated below), matching the original order.
    if(viewer_)
        viewer_->set_grab_image_time_median(map_median(grab_image_times_));

    log_profile();

    if(state_ == TrackingState::OK)
        grab_image_times_[profiler.total_ms()]++;

    return current_frame_.Tcw;
}

void Tracking::track()
{
    if(state_ == TrackingState::NO_IMAGES_YET)
        state_ = TrackingState::NOT_INITIALIZED;

    last_processed_state_ = state_;

    TrackProfiler profiler{};

    // Get Map Mutex -> Map cannot be changed
    std::unique_lock<std::mutex> lock(map_->map_update_mutex_);
    profiler.lock_acquired();

    // No map yet: the frame goes to two-view initialization, which finishes the
    // frame itself (drawer update, first stored relative pose once a map exists).
    if(state_ == TrackingState::NOT_INITIALIZED)
    {
        monocular_initialization();
        if(params.sequential)
        {
            // The two founding keyframes must be fully mapped before the next frame
            lock.unlock();
            wait_for_idle_local_mapper();
        }
        return;
    }

    // System is initialized: track the frame.
    bool ok{false};
    if(state_ == TrackingState::OK)
    {
        // Local Mapping might have changed some MapPoints tracked in last frame
        check_replaced_in_last_frame();
        ok = run_tracking_stage([this] { return track_reference_keyframe(); });
    }
    else
        ok = relocalize();

    current_frame_.ref_keyframe = ref_keyframe_;
    profiler.track_ref_done();

    // If we have an initial estimation of the camera pose and matching, track the local map.
    if(ok)
        ok = run_tracking_stage([this] { return track_local_map(); });
    profiler.local_map_done(local_map_times_);

    state_ = ok ? TrackingState::OK : TrackingState::LOST;

    frame_drawer_->update(this);

    // If tracking was good, check if we insert a keyframe
    if(ok)
    {
        ++num_tracked_frames_;
        log_heartbeat();

        map_drawer_->set_current_camera_pose(current_frame_.Tcw);

        // Clean VO matches: drop points no map observation backs
        current_frame_.drop_unobserved_points();

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

    last_frame_ = Frame(current_frame_);

    store_relative_pose();

    // Sequential (deterministic) mode: every frame ends with Local Mapping idle and the
    // map fully updated, so the next frame always sees the same map state (issue #16).
    if (params.sequential)
    {
        lock.unlock();
        wait_for_idle_local_mapper();
    }
    // An emergency keyframe was just inserted: block until Local Mapping has processed
    // it (the trigger is already logged by need_new_keyframe, and in profiling builds the
    // wait shows up as emergencyWait in the slow-frame report below).
    else if (emergency_keyframe_)
    {
        profiler.emergency_wait_begin();
        lock.unlock();
        while(!local_mapper_->accepts_keyframes())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        emergency_keyframe_ = false;
        profiler.emergency_wait_end();
    }

    profiler.report(current_frame_.frame_id);
}

void Tracking::store_relative_pose()
{
    if(current_frame_.Tcw(3,3) == 1.0f)
        last_frame_relative_pose_ = current_frame_.Tcw * current_frame_.ref_keyframe->get_pose_inverse();
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
            n += frame.keypoints.at(ft).size();
        return n;
    };

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
            init_matches_.resize(offset1 + initial_frame_.keypoints.at(ft).size(), -1);
            auto& matches_ft = matches_per_feature_[ft];
            matches_ft.assign(initial_frame_.keypoints.at(ft).size(), -1);
            for (const auto& m : matched_pairs.at(ft)) {
                matches_ft[m.first] = m.second;
                init_matches_[offset1 + m.first] = m.second + offset2;
            }
            offset2 += current_frame_.keypoints.at(ft).size();
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
                const auto& keypoints1 = initial_frame_.keypoints.at(ft);
                const auto& keypoints2 = current_frame_.keypoints.at(ft);
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

    keyframe_ini->compute_global_descriptor();
    keyframe_cur->compute_global_descriptor();

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

            Pt map_point = keyframe_cur->create_monocular_map_point(init_points3d_[flat_index], KeypointIndex(matches[i]),
                                                                 keyframe_ini, KeypointIndex(i), ft);
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
        const auto& inv_depth_kf = keyframe_ini->inv_depth.at(ft);
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
        mat4f Tc2w = keyframe_cur->get_pose();
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
        const float fx = mK.at<float>(0,0), fy = mK.at<float>(1,1);
        const float cx = mK.at<float>(0,2), cy = mK.at<float>(1,2);
        const float invfx = 1.0f / fx, invfy = 1.0f / fy;
        const mat4f Twc_cur = keyframe_cur->get_pose_inverse();
        const mat3f Rwc_cur = Twc_cur.block<3,3>(0,0);
        const vec3f twc_cur = Twc_cur.block<3,1>(0,3);

        flat_index = 0;
        for (const auto& ft : feature_types_) {
            const auto& matches = matches_per_feature_.at(ft);
            const auto& inv_depth_ini = keyframe_ini->inv_depth.at(ft);
            const auto& inv_depth_cur = keyframe_cur->inv_depth.at(ft);
            const auto& keypoints_ini = keyframe_ini->keypoints.at(ft);
            const auto& keypoints_cur = keyframe_cur->keypoints.at(ft);
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

                Pt map_point = keyframe_cur->create_monocular_map_point(world_pos, KeypointIndex(j),
                                                                        keyframe_ini, KeypointIndex(i), ft);
                current_frame_.pts.at(ft)[j] = map_point;
                current_frame_.outliers.at(ft)[j] = false;
                num_depth_completed++;
            }
        }
    }

    keyframe_ini->update_connections();
    keyframe_cur->update_connections();

    Optimizer::global_bundle_adjustment(map_, params.init_gba_iterations);

    const float median_depth = keyframe_ini->compute_scene_median_depth(2);
    const int tracked_map_points = keyframe_cur->tracked_map_points(1);

    if (median_depth < 0 || tracked_map_points < params.init_min_tracked_points)
    {
        AF_WARN("create_initial_map: degenerate initialization (median_depth=" << median_depth
                << ", tracked map points=" << tracked_map_points << " < " << params.init_min_tracked_points
                << ") — resetting...");
        reset();
        return;
    }

    // Monocular fallback (no usable depth): normalize to the arbitrary
    // "median scene depth = 1" convention, from the BA-refined map as before
    if (!metric_scale)
    {
        const float inv_median_depth = 1.0f / median_depth;
        mat4f Tc2w = keyframe_cur->get_pose();
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
        const float fx = mK.at<float>(0,0), fy = mK.at<float>(1,1);
        const float cx = mK.at<float>(0,2), cy = mK.at<float>(1,2);
        const float invfx = 1.0f / fx, invfy = 1.0f / fy;
        const mat4f Twc_cur = keyframe_cur->get_pose_inverse();
        const mat3f Rwc_cur = Twc_cur.block<3,3>(0,0);
        const vec3f twc_cur = Twc_cur.block<3,1>(0,3);

        for (const auto& ft : feature_types_) {
            const auto& inv_depth_cur = keyframe_cur->inv_depth.at(ft);
            const auto& keypoints_cur = keyframe_cur->keypoints.at(ft);
            for (size_t j = 0; j < inv_depth_cur.size(); j++) {
                if (inv_depth_cur[j] <= 0.0f || keyframe_cur->get_map_point(j, ft))
                    continue;
                const cv::KeyPoint& kp = keypoints_cur[j];
                const vec3f xn{(kp.pt.x - cx) * invfx, (kp.pt.y - cy) * invfy, 1.0f};
                Pt map_point = keyframe_cur->create_map_point(Rwc_cur * (xn / inv_depth_cur[j]) + twc_cur,
                                                              KeypointIndex(j), ft);
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

    current_frame_.set_pose(keyframe_cur->get_pose());
    last_keyframe_id_ = current_frame_.frame_id;
    last_keyframe_ = keyframe_cur;

    local_keyframes_.push_back(keyframe_cur);
    local_keyframes_.push_back(keyframe_ini);
    local_points_ = map_->get_all_map_points();
    ref_keyframe_ = keyframe_cur;
    current_frame_.ref_keyframe = keyframe_cur;

    last_frame_ = Frame(current_frame_);

    map_->set_reference_map_points(local_points_);
    map_drawer_->set_current_camera_pose(keyframe_cur->get_pose());
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
    StageTimer timer{};

    // Feature matching against the reference keyframe (global descriptor matching)
    std::map<FeatureType, std::vector<Pt>> map_point_matches;
    const std::map<FeatureType, int> num_matches_per_feature =
        matcher_->match_keyframe_to_frame(ref_keyframe_, current_frame_, map_point_matches, current_frame_.featureTypes);

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
               << " | frame=" << current_frame_.frame_id << " ref_keyframe_=" << ref_keyframe_->keyId;
        throw TrackingLostException(reason.str());
    }
    timer.record(track_ref_times_);

    // Optimize pose — seeded from the last frame's pose, recomputed on the fly from its
    // stored RELATIVE pose and the reference keyframe's CURRENT pose, so BA/loop-closure
    // corrections since last frame are absorbed into the seed by construction (no stale
    // stored pose to refresh). Constant-position beyond that: deliberately no motion
    // prior — a prediction is only as good as its assumption, and a violated one (abrupt
    // motion change) turns anything built on it into a failure cascade. Divergence
    // protection comes from the optimizer itself (g2o LM monotone-acceptance fix) plus
    // the depth-free rescue below.
    // Invariant: last_frame_relative_pose_ belongs to last_frame_ — every path here
    // had state_ == OK last frame, which stored it.
    const mat4f seed_pose = last_frame_relative_pose_ * last_frame_.ref_keyframe->get_pose();
    current_frame_.set_pose(seed_pose);
    Optimizer::pose_optimization(&current_frame_);
    int num_map_inliers = current_frame_.count_inlier_map_points();

    // Divergence rescue: a collapse to (almost) zero inliers despite plentiful raw matches
    // means the optimizer left the basin, not that the matches are bad. Re-seed and
    // re-optimize once with the RGB-D depth channel disabled — pure 2D reprojection, the
    // configuration the 4-pass scheme was originally tuned for.
    if(num_map_inliers < params.track_ref_min_inliers
       && num_matches >= 3 * params.track_ref_min_matches)
    {
        AF_WARN("track_reference_keyframe: pose optimization collapsed (" << num_map_inliers
                << " inliers of " << num_matches << " raw matches) — retrying without depth channel"
                << " | frame=" << current_frame_.frame_id);

        for (auto& [ft, outlier_flags] : current_frame_.outliers)
            std::fill(outlier_flags.begin(), outlier_flags.end(), false);
        current_frame_.set_pose(seed_pose);
        Optimizer::pose_optimization(&current_frame_, /*useDepthChannel=*/false);
        num_map_inliers = current_frame_.count_inlier_map_points();
    }

    // Discard outliers: remove the match, and mark the map point so search_local_points
    // neither revisits nor double-counts it this frame.
    for (auto& [ft, num_keypoints] : current_frame_.N)
    {
        for(int i = 0; i < num_keypoints; i++)
        {
            const Pt& map_point = current_frame_.pts.at(ft)[i];
            if(map_point && current_frame_.outliers.at(ft)[i])
            {
                map_point->mbTrackInView = false;
                map_point->idLastFrameSeen = current_frame_.frame_id;
                current_frame_.outliers.at(ft)[i] = false;
                current_frame_.pts.at(ft)[i] = nullptr;
            }
        }
    }
    timer.record(pose_opt_times_);

    if(num_map_inliers < params.track_ref_min_inliers)
    {
        std::ostringstream reason;
        reason << "track_reference_keyframe: insufficient inlier matches after pose optimization (num_map_inliers="
               << num_map_inliers << " < " << params.track_ref_min_inliers << ")"
               << " | frame=" << current_frame_.frame_id << " raw_matches=" << num_matches;
        throw TrackingLostException(reason.str());
    }

    return true;
}

bool Tracking::track_local_map()
{
    // We have a pose estimate and some map points tracked in the frame: retrieve
    // the local map, match its points into the frame, and refine the pose.
    update_local_map();
    search_local_points();
    Optimizer::pose_optimization(&current_frame_);

    // Found-ratio statistics: every non-outlier match counts as "found"
    // (feeds MapPointCulling's found/visible ratio).
    for (auto& [ft, pts] : current_frame_.pts)
        for(size_t i = 0; i < pts.size(); i++)
            if(pts[i] && !current_frame_.outliers.at(ft)[i])
                pts[i]->increase_found();

    num_inlier_matches_ = current_frame_.count_inlier_map_points();

    // Decide if tracking succeeded — more restrictive shortly after a relocalization
    if(current_frame_.frame_id < last_reloc_frame_id_ + max_frames_
       && num_inlier_matches_ < params.track_local_map_min_inliers_after_reloc)
    {
        std::ostringstream reason;
        reason << "track_local_map: insufficient inliers shortly after relocalization (inliers="
               << num_inlier_matches_ << " < " << params.track_local_map_min_inliers_after_reloc << ")"
               << " | frame=" << current_frame_.frame_id
               << " framesSinceReloc=" << (current_frame_.frame_id - last_reloc_frame_id_)
               << " max_frames_=" << max_frames_;
        throw TrackingLostException(reason.str());
    }

    if(num_inlier_matches_ < params.track_local_map_min_inliers)
    {
        std::ostringstream reason;
        reason << "track_local_map: insufficient inliers against local map (inliers="
               << num_inlier_matches_ << " < " << params.track_local_map_min_inliers << ")"
               << " | frame=" << current_frame_.frame_id << " localPts=" << local_points_.size();
        throw TrackingLostException(reason.str());
    }

    return true;
}


void Tracking::update_local_map()
{
    // Publish the current local points to the map for visualization
    map_->set_reference_map_points(local_points_);

    update_local_keyframes();
    update_local_points();
}

void Tracking::update_local_keyframes()
{
    // Each map point of the current frame votes for the keyframes observing it
    // (bad points are dropped from the frame along the way)
    std::map<KeyframeId, int> shared_points_per_keyframe;
    std::map<KeyframeId, Keyframe> keyframe_by_id;
    for (auto& [ft, pts] : current_frame_.pts) {
        for(auto& pt : pts){
            if(!pt)
                continue;
            if(pt->is_bad()){
                pt = nullptr;
                continue;
            }
            // By-value snapshot: get_observations copies under the point's mutex
            const std::map<KeyframeId, Obs> observations = pt->get_observations();
            for (const auto& [key_id, obs] : observations) {
                shared_points_per_keyframe[key_id]++;
                keyframe_by_id[key_id] = obs->projKeyframe;
            }
        }
    }
    if(shared_points_per_keyframe.empty())
        return;

    // Every keyframe observing a current map point joins the local map; the one
    // sharing the most points becomes the reference keyframe.
    int max_shared_points = 0;
    Keyframe keyframe_most_shared{};
    std::set<KeyframeId> seen_keyframe_ids;
    local_keyframes_.clear();
    local_keyframes_.reserve(LOCAL_KEYFRAMES_RESERVE_SCALE * shared_points_per_keyframe.size());
    for (const auto& [key_id, keyframe] : keyframe_by_id) {
        if(keyframe->is_bad())
            continue;

        const int num_shared = shared_points_per_keyframe.at(key_id);
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

    // Expand with neighbors of the included keyframes: per keyframe, ONE best
    // covisible neighbor, ONE child, and the parent. Indexed loop because the
    // vector grows while being traversed (a range-for reference would dangle on
    // reallocation). Semantics match stock ORB-SLAM2, including the outer-loop
    // break after the first parent insertion.
    for(size_t i = 0; i < local_keyframes_.size(); i++){
        if(local_keyframes_.size() > static_cast<size_t>(params.max_local_keyframes))
            break;
        const Keyframe keyframe = local_keyframes_[i]; // copy: push_back may reallocate

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
        if(parent && !parent->is_bad() && seen_keyframe_ids.insert(parent->keyId).second){
            local_keyframes_.push_back(parent);
            break;
        }
    }
}

void Tracking::update_local_points()
{
    // local_points_ = union of the map points of all local keyframes, deduplicated.
    // Dedup before the is_bad check: points are shared across many keyframes, and
    // is_bad locks the point's mutex — check it once per unique point.
    local_points_.clear();
    std::set<PtId> seen_point_ids;
    for(const Keyframe& keyframe : local_keyframes_){
        for(const FeatureType ft : feature_types_){
            // By-value snapshot: get_map_point_matches copies under the keyframe's mutex
            const std::vector<Pt> pts = keyframe->get_map_point_matches(ft);
            for(const Pt& pt : pts){
                if(!pt)
                    continue;
                if(!seen_point_ids.insert(pt->ptId).second)
                    continue; // already collected (or already seen and rejected as bad)
                if(!pt->is_bad())
                    local_points_.push_back(pt);
            }
        }
    }
}

void Tracking::search_local_points()
{
    // Mark the frame's already-matched points (visible, seen this frame, not a
    // search candidate) so the frustum loop below skips them; drop bad ones.
    for (auto& [ft, pts] : current_frame_.pts) {
        for(auto& pt : pts){
            if(pt && !pt->is_bad()){
                pt->increase_visible();
                pt->idLastFrameSeen = current_frame_.frame_id;
                pt->mbTrackInView = false;
            }
            else
                pt = nullptr;
        }
    }

    // Project the local points into the frame and check their visibility
    // (is_in_frustum fills the MapPoint variables the matcher reads)
    int num_to_match = 0;
    for(const Pt& pt : local_points_){
        if(pt->idLastFrameSeen == current_frame_.frame_id)
            continue;
        if(pt->is_bad())
            continue;

        if(current_frame_.is_in_frustum(pt, params.viewing_cos_limit)){
            pt->increase_visible();
            num_to_match++;
        }
    }

    if(num_to_match > 0)
        matcher_->match_map_points_to_frame(current_frame_, local_points_);
}

bool Tracking::need_new_keyframe()
{
    // If Local Mapping is frozen by a loop closure do not insert keyframes
    if(local_mapper_->is_stopped() || local_mapper_->is_stop_requested())
        return false;

    const size_t num_keyframes_in_map = map_->keyframes_in_map();

    // Do not insert keyframes if not enough frames have passed from the last relocalization —
    // unless tracking is already demonstrably strong again (>=2x the track_local_map "high"
    // threshold): at driving speed the full embargo freezes the reference keyframe for
    // ~a second of travel, decaying its matches until tracking is lost AGAIN right after
    // a successful relocalization (observed echo losses 20-22 frames after reloc; #9).
    if(current_frame_.frame_id < last_reloc_frame_id_ + max_frames_
       && num_keyframes_in_map > max_frames_
       && num_inlier_matches_ < 2 * params.track_local_map_min_inliers_after_reloc)
        return false;

    // Tracked map points in the reference keyframe (observation bar relaxed while the map is young)
    const int min_observations = (num_keyframes_in_map <= static_cast<size_t>(params.young_map_keyframes)) ? params.min_observations_low
                                                                               : params.min_observations_high;
    const int num_ref_matches = ref_keyframe_->tracked_map_points(min_observations);

    // Tracking-health triggers. (A close-point condition — insert when too few nearby
    // RGB-D points are tracked but many could be created — belongs here once depth
    // populates per-point close/far classification; see CLAUDE.md depth table row 8.)
    const bool weak_tracking = num_inlier_matches_ < num_ref_matches * params.ref_matches_ratio
                               && num_inlier_matches_ > params.min_inliers_for_keyframe;

    bool forced_by_cadence{false};
#ifdef ALLFEATURE_MAX_KEYFRAMES
    forced_by_cadence = (current_frame_.frame_id % ALLFEATURE_MAX_KEYFRAMES) == 0;
#endif

    const float overlap = current_frame_.get_overlap();
    const bool low_overlap = overlap < params.min_ref_overlap;
    const bool forced = forced_by_cadence;

    // Median inlier count over the recent tracked frames (reference for the
    // emergency trigger below) — computed before pushing the current frame's
    // count, so the current frame is compared against its predecessors.
    int median_recent_inliers = -1;
    if (recent_inliers_history_.size() >= static_cast<size_t>(params.inliers_history_size) / 2) {
        std::vector<int> history(recent_inliers_history_.begin(), recent_inliers_history_.end());
        const auto mid = history.begin() + history.size() / 2;
        std::nth_element(history.begin(), mid, history.end());
        median_recent_inliers = *mid;
    }
    recent_inliers_history_.push_back(num_inlier_matches_);
    if (recent_inliers_history_.size() > static_cast<size_t>(params.inliers_history_size))
        recent_inliers_history_.pop_front();

    // Information the current view would add to the local map: its unexplained
    // information v in [0,1] given the local keyframes, on the same global-descriptor
    // kernel (and centring) keyframe culling marginalises (placecell). Costs one MegaLoc
    // embedding per tracked frame; the embedding is cached in current_frame_ so a
    // keyframe made from this frame is not embedded twice. Three bands, sharing the
    // culler's budget tau (LocalMapping.KeyframeCullingMaxUnexplained):
    //   v <  keyframe_min_information : redundant — the local map already explains the view;
    //                                   no keyframe (replaces the pixel-flow stationarity gate:
    //                                   a static camera is the extreme case of this band)
    //   v >  tau                      : novel — insert even if tracking is fine, so the alive
    //                                   keyframes keep every view seen within tau (the same
    //                                   invariant the culler maintains from the other end;
    //                                   by the Schur identity a keyframe inserted here has
    //                                   unique information v > tau, so it is not a cull
    //                                   candidate on its own account)
    //   in between                    : the tracking-health triggers above decide.
    // Without an information measure (vpr: none) only the tracking triggers decide.
    const float tau = LocalMapping::params.keyframe_culling_max_unexplained.load();
    const std::optional<KeyframeInformation> information =
        place_recognition_->keyframe_information(current_frame_, local_keyframes_, LocalMapping::params.keyframe_culling_centred);
    last_keyframe_information_ = information ? std::optional<float>(information->unexplained) : std::nullopt;
    const bool redundant = information && information->unexplained < params.keyframe_min_information;
    const bool novel = information && information->unexplained > tau;

    // Thresholds in force, for the backend's decision history (placecell Recorder):
    // recorded as a step wherever they change (Viewer slider), ignored otherwise.
    place_recognition_->record_keyframe_thresholds(tau, params.keyframe_min_information);

    // Decision sink: the backend's decision history (placecell Recorder, drawn as
    // insertion markers on the information plot), the insertion line (always) and the
    // per-frame diagnostic line (Tracking.LogKeyframeInformation), all carrying the
    // information value.
    const auto decide = [&](const bool insert, const std::string& reason) -> bool
    {
        place_recognition_->record_keyframe_decision(current_frame_.frame_id, insert, last_keyframe_information_, reason);
        if(insert || params.log_keyframe_information)
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
                     << " tau=" << tau << " minInfo=" << params.keyframe_min_information
                     << " explainers=" << information->explainers
                     << " best=KF" << best_explainer_key_id << "(" << information->best_similarity << ")";
            }
            else
                line << " info=n/a";
            if(params.log_keyframe_information)
            {
                // Pixel flow to the last frame: kept as a diagnostic only, no longer a gate
                const std::optional<float> median_flow = median_flow_from_last_frame();
                line << " flow=" << (median_flow ? std::to_string(*median_flow) : std::string("n/a"))
                     << " inliers=" << num_inlier_matches_ << " refMatches=" << num_ref_matches
                     << " medianRecentInliers=" << median_recent_inliers
                     << std::setprecision(2) << " overlap=" << overlap
                     << " weak=" << weak_tracking << " lowOverlap=" << low_overlap
                     << " novel=" << novel << " redundant=" << redundant << " forced=" << forced
                     << " KFs=" << num_keyframes_in_map;
            }
            AF_INFO(line.str());
        }
        return insert;
    };

    // Redundancy band. Forced keyframes bypass it.
    if(!forced && redundant)
        return decide(false, "redundant");

    if(!(weak_tracking || forced || low_overlap || novel))
        return decide(false, "no trigger");

    std::string reason;
    for(const auto& [active, name] : {std::pair{forced, "forced"}, std::pair{novel, "novel"},
                                      std::pair{weak_tracking, "weak_tracking"}, std::pair{low_overlap, "low_overlap"}})
        if(active)
            reason += (reason.empty() ? "" : "+") + std::string(name);

    // Sequential mode: the end-of-frame wait guarantees Local Mapping is idle, so do
    // not consult the live busy flag — run() toggles it once per idle cycle, and
    // sampling that flicker would make the insertion decision timing-dependent.
    if(params.sequential)
        return decide(true, reason);

    if(local_mapper_->accepts_keyframes())
        return decide(true, reason);

    // Local Mapping is busy. Emergency keyframe: only on a genuine drop against the
    // *recent frames'* own inlier level (not ref_keyframe_->tracked_map_points(),
    // which inflates after every insertion as LocalMapping triangulates new points
    // into the keyframe, re-arming the trigger indefinitely), and with a refire
    // cooldown so a persistent low-inlier state can't chain insertions.
    if(median_recent_inliers > 0
       && num_inlier_matches_ < params.emergency_inlier_drop_ratio * static_cast<float>(median_recent_inliers)
       && current_frame_.frame_id >= last_emergency_keyframe_id_ + static_cast<FrameId>(params.emergency_keyframe_cooldown))
    {
        AF_WARN("need_new_keyframe: emergency keyframe (inliers=" << num_inlier_matches_
                << " < " << params.emergency_inlier_drop_ratio << "*medianRecentInliers=" << median_recent_inliers
                << ", info=" << (information ? std::to_string(information->unexplained) : std::string("n/a"))
                << ") | frame=" << current_frame_.frame_id);
        last_emergency_keyframe_id_ = current_frame_.frame_id;
        emergency_keyframe_ = true;
        return decide(true, reason + "+emergency");
    }
    return decide(false, reason + ", local mapping busy");
}

void Tracking::create_new_keyframe()
{
    // Lock Local Mapping against stopping while the keyframe is inserted;
    // fails (and skips insertion) if a loop closure already stopped it.
    if(!local_mapper_->set_insertion_lock(true))
        return;

    const Keyframe keyframe = std::make_shared<KeyFrame>(current_frame_, map_, place_recognition_);

    ref_keyframe_ = keyframe;
    current_frame_.ref_keyframe = keyframe;

    local_mapper_->insert_keyframe(keyframe);
    local_mapper_->set_insertion_lock(false);

    last_keyframe_ = keyframe;
    last_keyframe_id_ = current_frame_.frame_id;
}

void Tracking::wait_for_idle_local_mapper() const
{
    while(local_mapper_->has_new_keyframes() || !local_mapper_->accepts_keyframes())
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

std::optional<float> Tracking::median_flow_from_last_frame() const
{
    // Collect pixel positions of last frame's map points, then measure how far the
    // same points moved in the current frame. Scale-free (pure 2D), cheap (two
    // linear passes), and needs no extra bookkeeping in the matchers.
    std::unordered_map<const MapPoint*, cv::Point2f> last_positions;
    for (const auto& [ft, points] : last_frame_.pts) {
        const auto& keypoints = last_frame_.keypoints.at(ft);
        for (size_t i = 0; i < points.size(); i++)
            if (points[i])
                last_positions[points[i].get()] = keypoints[i].pt;
    }

    std::vector<float> flows;
    for (const auto& [ft, points] : current_frame_.pts) {
        const auto& keypoints = current_frame_.keypoints.at(ft);
        for (size_t i = 0; i < points.size(); i++) {
            if (!points[i])
                continue;
            const auto it = last_positions.find(points[i].get());
            if (it != last_positions.end())
                flows.push_back(static_cast<float>(cv::norm(keypoints[i].pt - it->second)));
        }
    }

    if (flows.size() < static_cast<size_t>(params.min_shared_points_for_flow))
        return std::nullopt;

    const auto mid = flows.begin() + flows.size() / 2;
    std::nth_element(flows.begin(), mid, flows.end());
    return *mid;
}

bool Tracking::relocalize()
{
    // Without an active VPR backend there is no keyframe database to query —
    // recovery from a tracking loss is impossible.
    if(!place_recognition_->is_active())
        return false;

    const FeatureType feature_type = place_recognition_->verification_feature();
    current_frame_.compute_global_descriptor();

    const std::vector<Keyframe> candidates = place_recognition_->detect_relocalization_candidates(current_frame_);
    if(candidates.empty())
        return false;
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
                                        RANSAC_MIN_SET, params.reloc_ransac_epsilon, RANSAC_TH2);
        num_active++;
    }

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
            current_frame_.Tcw = Converter::to_matrix4f(Tcw);

            if(accept_relocalization_hypothesis(candidates[i], matches_per_candidate[i].at(feature_type),
                                                inliers, feature_type))
            {
                last_reloc_frame_id_ = current_frame_.frame_id;
                return true;
            }
        }
    }
    return false;
}

bool Tracking::accept_relocalization_hypothesis(Keyframe candidate, const std::vector<Pt>& matches,
                                                const std::vector<bool>& inliers, const FeatureType feature_type)
{
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

    int num_good = Optimizer::pose_optimization(&current_frame_);
    if(num_good < params.reloc_inliers_low)
        return false;

    for(int j = 0; j < current_frame_.N.at(feature_type); j++)
        if(current_frame_.outliers.at(feature_type)[j])
            current_frame_.pts.at(feature_type)[j] = nullptr;

    // Few inliers: search by projection in a coarse window and optimize again
    if(num_good < params.reloc_inliers_high)
    {
        const int additional = matcher_->search_by_projection(current_frame_, candidate, found,
                                                            params.reloc_search_radius_coarse, true, feature_type);
        if(additional + num_good >= params.reloc_inliers_high)
        {
            num_good = Optimizer::pose_optimization(&current_frame_);

            // Many inliers but still not enough: the pose is already close, so search
            // once more in a narrower window and run a final optimization
            if(num_good > params.reloc_inliers_medium && num_good < params.reloc_inliers_high)
            {
                found.clear();
                for(int j = 0; j < current_frame_.N.at(feature_type); j++)
                    if(current_frame_.pts.at(feature_type)[j])
                        found.insert(current_frame_.pts.at(feature_type)[j]);
                const int narrow_additional = matcher_->search_by_projection(current_frame_, candidate, found,
                                                                           params.reloc_search_radius_narrow, false, feature_type);
                if(num_good + narrow_additional >= params.reloc_inliers_high)
                {
                    num_good = Optimizer::pose_optimization(&current_frame_);
                    for(int j = 0; j < current_frame_.N.at(feature_type); j++)
                        if(current_frame_.outliers.at(feature_type)[j])
                            current_frame_.pts.at(feature_type)[j] = nullptr;
                }
            }
        }
    }

    if(num_good < params.reloc_inliers_high)
        return false;

    AF_INFO("relocalize: succeeded | frame=" << current_frame_.frame_id
            << " feature=" << feature_name(feature_type)
            << " matchedKeyframe=" << candidate->keyId
            << " inliers=" << num_good << " requiredInliers=" << params.reloc_inliers_high);
    std::cout.flush(); // AF_INFO's stdout is fully buffered under the runner's redirect

    return true;
}

void Tracking::reset()
{
    // Each step logs before it runs and flushes (stdout is fully buffered under
    // the runner's redirect), so a hang shows the last started step.
    AF_INFO("reset: stopping viewer...");
    std::cout.flush();
    if(viewer_)
    {
        viewer_->request_stop();
        while(!viewer_->is_stopped())
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }

    AF_INFO("reset: resetting local mapping...");
    std::cout.flush();
    local_mapper_->request_reset();

    AF_INFO("reset: resetting loop closing...");
    std::cout.flush();
    loop_closing_->request_reset();

    AF_INFO("reset: clearing keyframe database and map...");
    std::cout.flush();
    place_recognition_->clear();
    map_->clear(); // erases all map points and keyframes

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    state_ = TrackingState::NO_IMAGES_YET;
    initializer_ = nullptr;

    // Per-run tracking state: frame ids restart at 0, so id-anchored state from
    // the previous run (reloc embargo, emergency cooldown) must not leak into the
    // next one and suppress keyframes there.
    last_reloc_frame_id_ = 0;
    last_emergency_keyframe_id_ = 0;
    emergency_keyframe_ = false;
    num_inlier_matches_ = 0;
    recent_inliers_history_.clear();

    resize_times_.clear();
    frame_times_.clear();
    tracking_times_.clear();
    track_ref_times_.clear();
    pose_opt_times_.clear();
    local_map_times_.clear();
    grab_image_times_.clear();

    if(viewer_)
        viewer_->release();

    AF_INFO("reset: done");
    std::cout.flush();
}

} //namespace AF_VSLAM
