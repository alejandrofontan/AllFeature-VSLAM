


#include "Tracking.h"
#include "Tracking_aux.h"

#include <opencv2/core/core.hpp>
//#include<opencv2/features2d/features2d.hpp>

#include "FrameDrawer.h"
#include "Converter.h"
#include "Map.h"
#include "Initializer.h"
#include "afvslam_log.hpp"

#include "Optimizer.h"
#include "PnPsolver.h"
#include "Utils.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include <mutex>

#include <yaml-cpp/yaml.h>

using namespace std;

namespace AF_VSLAM
{

TrackingParameters Tracking::params{};

void Tracking::LoadParameters(const cv::FileStorage &fSettings)
{
    auto readIfPresent = [&fSettings](const char* key, auto& field)
    {
        const cv::FileNode node = fSettings[key];
        if(!node.empty())
            node >> field;
    };

    readIfPresent("Tracking.InitMinKeypoints", params.init_min_keypoints);
    readIfPresent("Tracking.InitSigma", params.init_sigma);
    readIfPresent("Tracking.InitMinMatches", params.init_min_matches);
    readIfPresent("Tracking.InitRansacIterations", params.init_ransac_iterations);
    readIfPresent("Tracking.InitMinMedianDisparity", params.init_min_median_disparity);
    readIfPresent("Tracking.InitGbaIterations", params.init_gba_iterations);
    readIfPresent("Tracking.InitMinTrackedPoints", params.init_min_tracked_points);
    readIfPresent("Tracking.InitMinDepthSamples", params.init_min_depth_samples);
}

Tracking::Tracking(shared_ptr<Vocabulary> vocabulary,
                   std::shared_ptr<FrameDrawer> frame_drawer, std::shared_ptr<MapDrawer> map_drawer,
                   shared_ptr<Map> map, shared_ptr<KeyFrameDatabase> pKFDB,
                   const string &strCalibrationPath, const string &strSettingPath,
                   const std::map<FeatureType, string>& feature_settings_yaml_file,
                   const vector<FeatureType>& featureTypes,
                   const bool fix_image_size):
    state_(NO_IMAGES_YET), feature_types_(featureTypes), mbVO(false), vocabulary(vocabulary),
    keyframe_db_(pKFDB),
    frame_drawer_(frame_drawer), map_drawer_(map_drawer), map_(map), last_reloc_frame_id_(0), fix_image_size_(fix_image_size)
{
    // Load camera parameters from settings yaml file
    Tracking::loadCameraParameters(strCalibrationPath, strSettingPath);

    // Frame window for the post-relocalization embargo/strictness (~1 s at the camera rate)
    max_frames_ = size_t(fps);

    // Load feature parameters from settings yaml file
    for (auto& ft: featureTypes){
        feature_extractor_left_[ft] = Tracking::getFeatureExtractor(1, feature_settings_yaml_file.at(ft), ft);
        init_feature_extractor_[ft] = Tracking::getFeatureExtractor(scaleNumFeaturesMonocular , feature_settings_yaml_file.at(ft), ft);
    }

    matcher_ = std::make_shared<FeatureMatcher>(image_width_, image_height_, featureTypes, "Tracking");
}

mat4f Tracking::grab_image(Image &im, const double timestamp)
{
    GrabProfiler profiler{};

    // Convert image to grayscale and resize
    im.GetGrayImage(is_rgb_);
    if(fix_image_size_)
        im.FixImageSize(image_width_, image_height_);

    gray_image_ = im.grayImg;
    mask_image_ = im.mask;
    image_name_ = im.imageName;
    profiler.resize_done(resize_times_);

    // Create the frame (feature extraction); initialization uses the denser extractor set
    const auto& extractors = (state_ == NOT_INITIALIZED || state_ == NO_IMAGES_YET)
                           ? init_feature_extractor_ : feature_extractor_left_;
    current_frame_ = Frame(im, timestamp, extractors, vocabulary, mK, mDistCoef, mbf, mThDepth);
    profiler.frame_created(frame_times_);

    track();
    profiler.tracking_done(tracking_times_, state_ == OK);

    // Median excludes the current frame (updated below), matching the original order.
    if(viewer_)
        viewer_->set_grab_image_time_median(map_median(grab_image_times_));

    log_profile();

    if(state_ == OK)
        grab_image_times_[profiler.total_ms()]++;

    return current_frame_.Tcw;
}

void Tracking::track()
{
    if(state_ == NO_IMAGES_YET)
        state_ = NOT_INITIALIZED;

    last_processed_state_ = state_;

    TrackProfiler profiler{};

    // Get Map Mutex -> Map cannot be changed
    std::unique_lock<std::mutex> lock(map_->map_update_mutex_);
    profiler.lock_acquired();

    // No map yet: the frame goes to two-view initialization, which finishes the
    // frame itself (drawer update, first stored relative pose once a map exists).
    if(state_ == NOT_INITIALIZED)
    {
        monocular_initialization();
        return;
    }

    // System is initialized: track the frame.
    bool ok{false};
    if(state_ == OK)
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

    state_ = ok ? OK : LOST;

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

    // An emergency keyframe was just inserted: block until Local Mapping has processed
    // it (the trigger is already logged by need_new_keyframe, and in profiling builds the
    // wait shows up as emergencyWait in the slow-frame report below).
    if (emergency_keyframe_)
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
    if(state_ == OK)
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

        // Fill matches_per_feature_ (used later in create_initial_map_monocular) and
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
            create_initial_map_monocular();
        }
    }
}

void Tracking::create_initial_map_monocular()
{
    // Create KeyFrames
    Keyframe keyframe_ini = make_shared<KeyFrame>(initial_frame_, map_, keyframe_db_);
    Keyframe keyframe_cur = make_shared<KeyFrame>(current_frame_, map_, keyframe_db_);

    keyframe_ini->compute_global_descriptor();
    keyframe_cur->compute_global_descriptor();

    // Insert KFs in the map
    map_->add_keyframe(keyframe_ini);
    map_->add_keyframe(keyframe_cur);

    // Create MapPoints and associate to keyframes
    size_t flat_index = 0; // runs over init_matches_, flattened across feature types
    for (const auto& ft : feature_types_) {
        const auto& matches = matches_per_feature_.at(ft);
        for (size_t i = 0; i < matches.size(); i++, flat_index++) {
            if (init_matches_[flat_index] < 0)
                continue;

            Pt map_point = keyframe_cur->create_monocular_map_point(init_points3d_[flat_index], KeypointIndex(matches[i]),
                                                                 keyframe_ini, KeypointIndex(i), ft);
            // Fill current frame structure
            current_frame_.pts.at(ft)[matches[i]] = map_point;
            current_frame_.outliers.at(ft)[matches[i]] = false;

            map_->add_map_point(map_point);
        }
    }

    // Update Connections
    keyframe_ini->update_connections();
    keyframe_cur->update_connections();

    // Bundle Adjustment
    AF_INFO("New Map created with " << map_->map_points_in_map() << " points");
    Optimizer::global_bundle_adjustment(map_, params.init_gba_iterations);

    // Set the initial map's scale: prefer a depth-verified scale over the arbitrary
    // monocular "median depth = 1" convention, when enough points have valid sensor depth.
    const float median_depth = keyframe_ini->compute_scene_median_depth(2);
    const int tracked_map_points = keyframe_cur->tracked_map_points(1);

    if (median_depth < 0 || tracked_map_points < params.init_min_tracked_points)
    {
        AF_WARN("Wrong initialization (median_depth=" << median_depth << ", tracked map points="
                << tracked_map_points << " < " << params.init_min_tracked_points << ") — resetting...");
        reset();
        return;
    }

    vector<float> depth_ratios;
    for (const auto& ft : feature_types_) {
        const auto& inv_depth_kf = keyframe_ini->inv_depth.at(ft);
        const vector<Pt> map_points = keyframe_ini->get_map_point_matches(ft);
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

    float inv_median_depth = 1.0f / median_depth;
    if (depth_ratios.size() >= static_cast<size_t>(params.init_min_depth_samples))
    {
        const auto mid = depth_ratios.begin() + depth_ratios.size() / 2;
        std::nth_element(depth_ratios.begin(), mid, depth_ratios.end());
        inv_median_depth = *mid;
    }

    // Scale initial baseline
    mat4f Tc2w = keyframe_cur->get_pose();
    Tc2w.block<3,1>(0,3) *= inv_median_depth;
    keyframe_cur->set_pose(Tc2w);

    // Scale points
    for (const auto& ft : feature_types_) {
        for (const Pt& map_point : keyframe_ini->get_map_point_matches(ft))
            if (map_point)
                map_point->set_world_pos(map_point->get_world_pos() * inv_median_depth);
    }

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

    state_ = OK;
}

void Tracking::check_replaced_in_last_frame()
{
    for (auto& [ft, pts] : last_frame_.pts) {
        for(int i = 0; i<last_frame_.N.at(ft); i++)
        {
            Pt pMP = last_frame_.pts.at(ft)[i];

            if(pMP)
            {
                Pt pRep = pMP->GetReplaced();
                if(pRep)
                {
                    last_frame_.pts.at(ft)[i] = pRep;
                }
            }
        }
    }
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

    if(num_matches < TRACK_REFERENCE_KEYFRAME_MIN_MATCHES_HIGH)
    {
        std::ostringstream reason;
        reason << "track_reference_keyframe: insufficient matches to reference keyframe (num_matches="
               << num_matches << " < " << TRACK_REFERENCE_KEYFRAME_MIN_MATCHES_HIGH << ")"
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
    Optimizer::PoseOptimization(&current_frame_);
    int num_map_inliers = current_frame_.count_inlier_map_points();

    // Divergence rescue: a collapse to (almost) zero inliers despite plentiful raw matches
    // means the optimizer left the basin, not that the matches are bad. Re-seed and
    // re-optimize once with the RGB-D depth channel disabled — pure 2D reprojection, the
    // configuration the 4-pass scheme was originally tuned for.
    if(num_map_inliers < TRACK_REFERENCE_KEYFRAME_MIN_MATCHES_LOW
       && num_matches >= 3 * TRACK_REFERENCE_KEYFRAME_MIN_MATCHES_HIGH)
    {
        AF_WARN("track_reference_keyframe: pose optimization collapsed (" << num_map_inliers
                << " inliers of " << num_matches << " raw matches) — retrying without depth channel"
                << " | frame=" << current_frame_.frame_id);
        dump_pose_collapse(seed_pose);

        for (auto& [ft, outlier_flags] : current_frame_.outliers)
            std::fill(outlier_flags.begin(), outlier_flags.end(), false);
        current_frame_.set_pose(seed_pose);
        Optimizer::PoseOptimization(&current_frame_, /*useDepthChannel=*/false);
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

    if(num_map_inliers < TRACK_REFERENCE_KEYFRAME_MIN_MATCHES_LOW)
    {
        std::ostringstream reason;
        reason << "track_reference_keyframe: insufficient inlier matches after pose optimization (num_map_inliers="
               << num_map_inliers << " < " << TRACK_REFERENCE_KEYFRAME_MIN_MATCHES_LOW << ")"
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
    Optimizer::PoseOptimization(&current_frame_);

    // Found-ratio statistics: every non-outlier match counts as "found"
    // (feeds MapPointCulling's found/visible ratio).
    for (auto& [ft, pts] : current_frame_.pts)
        for(size_t i = 0; i < pts.size(); i++)
            if(pts[i] && !current_frame_.outliers.at(ft)[i])
                pts[i]->IncreaseFound();

    num_inlier_matches_ = current_frame_.count_inlier_map_points();

    // Decide if tracking succeeded — more restrictive shortly after a relocalization
    if(current_frame_.frame_id < last_reloc_frame_id_ + max_frames_
       && num_inlier_matches_ < TRACK_LOCAL_MAP_MIN_INLIERS_HIGH)
    {
        std::ostringstream reason;
        reason << "track_local_map: insufficient inliers shortly after relocalization (inliers="
               << num_inlier_matches_ << " < " << TRACK_LOCAL_MAP_MIN_INLIERS_HIGH << ")"
               << " | frame=" << current_frame_.frame_id
               << " framesSinceReloc=" << (current_frame_.frame_id - last_reloc_frame_id_)
               << " max_frames_=" << max_frames_;
        throw TrackingLostException(reason.str());
    }

    if(num_inlier_matches_ < TRACK_LOCAL_MAP_MIN_INLIERS_LOW)
    {
        std::ostringstream reason;
        reason << "track_local_map: insufficient inliers against local map (inliers="
               << num_inlier_matches_ << " < " << TRACK_LOCAL_MAP_MIN_INLIERS_LOW << ")"
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
            // By-value snapshot: GetObservations copies under the point's mutex
            const std::map<KeyframeId, Obs> observations = pt->GetObservations();
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
        if(local_keyframes_.size() > MAX_LOCAL_KEYFRAMES)
            break;
        const Keyframe keyframe = local_keyframes_[i]; // copy: push_back may reallocate

        for(const Keyframe& neighbor : keyframe->GetBestCovisibilityKeyFrames(BEST_COVISIBLE_KEYFRAMES)){
            if(!neighbor->is_bad() && seen_keyframe_ids.insert(neighbor->keyId).second){
                local_keyframes_.push_back(neighbor);
                break;
            }
        }

        for(const Keyframe& child : keyframe->GetChilds()){
            if(!child->is_bad() && seen_keyframe_ids.insert(child->keyId).second){
                local_keyframes_.push_back(child);
                break;
            }
        }

        const Keyframe parent = keyframe->GetParent();
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
                pt->IncreaseVisible();
                pt->idLastFrameSeen = current_frame_.frame_id;
                pt->mbTrackInView = false;
            }
            else
                pt = nullptr;
        }
    }

    // Project the local points into the frame and check their visibility
    // (isInFrustum fills the MapPoint variables the matcher reads)
    int num_to_match = 0;
    for(const Pt& pt : local_points_){
        if(pt->idLastFrameSeen == current_frame_.frame_id)
            continue;
        if(pt->is_bad())
            continue;

        if(current_frame_.isInFrustum(pt, VIEWING_COS_LIMIT)){
            pt->IncreaseVisible();
            num_to_match++;
        }
    }

    if(num_to_match > 0)
        matcher_->match_map_points_to_frame(current_frame_, local_points_);
}

    float Tracking::MedianFlowFromLastFrame() const
    {
        // Collect pixel positions of last frame's map points, then measure how far the
        // same points moved in the current frame. Scale-free (pure 2D), cheap (two
        // linear passes), and needs no extra bookkeeping in the matchers.
        std::unordered_map<const MapPoint*, cv::Point2f> lastPositions;
        for (const auto& [ft, ptsFt] : last_frame_.pts) {
            const auto& kpsFt = last_frame_.keypoints.at(ft);
            for (size_t i = 0; i < ptsFt.size(); i++)
                if (ptsFt[i])
                    lastPositions[ptsFt[i].get()] = kpsFt[i].pt;
        }

        std::vector<float> flows;
        for (const auto& [ft, ptsFt] : current_frame_.pts) {
            const auto& kpsFt = current_frame_.keypoints.at(ft);
            for (size_t i = 0; i < ptsFt.size(); i++) {
                if (!ptsFt[i])
                    continue;
                auto it = lastPositions.find(ptsFt[i].get());
                if (it != lastPositions.end())
                    flows.push_back(static_cast<float>(cv::norm(kpsFt[i].pt - it->second)));
            }
        }

        if (flows.size() < static_cast<size_t>(minSharedPtsForFlow))
            return -1.0f;

        auto mid = flows.begin() + flows.size() / 2;
        std::nth_element(flows.begin(), mid, flows.end());
        return *mid;
    }

    bool Tracking::need_new_keyframe()
    {
        // If Local Mapping is freezed by a Loop Closure do not insert keyframes
        if(local_mapper_->isStopped() || local_mapper_->stopRequested())
            return false;

        const size_t numKeyframesInMap = map_->keyframes_in_map();

        // Do not insert keyframes if not enough frames have passed from last relocalisation —
        // unless tracking is already demonstrably strong again (>=2x the track_local_map "high"
        // threshold): at driving speed the full embargo freezes the reference keyframe for
        // ~a second of travel, decaying its matches until tracking is lost AGAIN right after
        // a successful relocalization (observed echo losses 20-22 frames after reloc; #9).
        if((current_frame_.frame_id < last_reloc_frame_id_ + max_frames_) && (numKeyframesInMap > max_frames_)
           && num_inlier_matches_ < 2 * TRACK_LOCAL_MAP_MIN_INLIERS_HIGH)
            return false;

        // Tracked MapPoints in the reference keyframe
        int nMinObs = nMinObs_high;
        if(numKeyframesInMap <= size_t(minNKFs))
            nMinObs = nMinObs_low;
        int nRefMatches = ref_keyframe_->tracked_map_points(nMinObs);

        // Local Mapping accept keyframes?
        bool localMappingIdle = local_mapper_->accepts_keyframes();

        // Check how many "close" points are being tracked and how many could be potentially created.
        int nNonTrackedClose = 0;
        int nTrackedClose= 0;

        bool bNeedToInsertClose = (nTrackedClose < minTrackedClose) && (nNonTrackedClose > minNonTrackedClose);

        // Thresholds
        const bool c1 = ((num_inlier_matches_ < nRefMatches * refRatio_high_needNewKey || bNeedToInsertClose) && num_inlier_matches_ > minMatchesInliers);

        bool c2{false};
        #ifdef ALLFEATURE_EVALUATION
        c2 = ((int( current_frame_.frame_id) % ALLFEATURE_EVALUATION) == 0);
        #endif

        bool c3{false};
        #ifdef ALLFEATURE_MAX_KEYFRAMES
        c3 = ((current_frame_.frame_id % ALLFEATURE_MAX_KEYFRAMES) == 0);
        #endif

        float overlap = current_frame_.GetOverlap();
        bool c4 = (overlap < 0.7f);

        // Median inlier count over the recent tracked frames (reference for the
        // emergency trigger below) — computed before pushing the current frame's
        // count, so the current frame is compared against its predecessors.
        int medianRecentInliers = -1;
        if (recentInliersHistory.size() >= inliersHistorySize / 2) {
            std::vector<int> history(recentInliersHistory.begin(), recentInliersHistory.end());
            auto mid = history.begin() + history.size() / 2;
            std::nth_element(history.begin(), mid, history.end());
            medianRecentInliers = *mid;
        }
        recentInliersHistory.push_back(num_inlier_matches_);
        if (recentInliersHistory.size() > inliersHistorySize)
            recentInliersHistory.pop_front();

        // Stationarity gate: a static camera adds no viewpoint information — new
        // keyframes would only feed zero-baseline triangulation, which poisons the map
        // with ill-conditioned points (see CLAUDE.md, Stop-Induced Keyframe Runaway
        // Investigation). Evaluation-forced keyframes (c2/c3) bypass the gate.
        const float medianFlow = MedianFlowFromLastFrame();
        if (!c2 && !c3 && medianFlow >= 0.0f && medianFlow < minMedianFlow_needNewKey)
            return false;

        //if(c4)
        if(c1 || c2 || c3 || c4)
        {
            if(localMappingIdle)
            {
                return true;
            }
            else
            {
                // if(c2 || c3 || c4){
                //     std::cout << "\nEmergency keyframe triggered by evaluation condition at frame " << current_frame_.frame_id << std::endl;
                //     emergency_keyframe_ = true;
                //     return true;
                // }
                // Emergency keyframe: only on a genuine drop against the *recent frames'*
                // own inlier level (not ref_keyframe_->tracked_map_points(), which inflates
                // after every insertion as LocalMapping triangulates new points into the
                // keyframe, re-arming the trigger indefinitely), and with a refire
                // cooldown so a persistent low-inlier state can't chain insertions.
                if(medianRecentInliers > 0
                   && num_inlier_matches_ < 0.5f * static_cast<float>(medianRecentInliers)
                   && current_frame_.frame_id >= lastEmergencyKFId + static_cast<FrameId>(emergencyKFCooldown)){
                    AF_WARN("need_new_keyframe: emergency keyframe (num_inlier_matches_=" << num_inlier_matches_
                            << " < 0.5*medianRecentInliers=" << medianRecentInliers
                            << ", medianFlow=" << medianFlow << ") | frame=" << current_frame_.frame_id);
                    lastEmergencyKFId = current_frame_.frame_id;
                    emergency_keyframe_ = true;
                    return true;
                }
                return false;
            }
        }
        else
            return false;
    }

    void Tracking::create_new_keyframe(){
        if(!local_mapper_->SetNotStop(true))
            return;

        Keyframe keyframe = make_shared<KeyFrame>(current_frame_,map_,keyframe_db_);

        ref_keyframe_ = keyframe;
        current_frame_.ref_keyframe = keyframe;

        local_mapper_->insert_keyframe(keyframe);
        local_mapper_->SetNotStop(false);
        last_keyframe_id_ = current_frame_.frame_id;
        last_keyframe_ = keyframe;
    }

bool Tracking::relocalize()
{
    // Without an active VPR backend there is no keyframe database to query —
    // recovery from a tracking loss is impossible.
    if(!vocabulary->is_active())
        return false;

    const FeatureType featureType = vocabulary->featureType;

    // Compute the global descriptor (BoW vector)
    current_frame_.compute_global_descriptor();

    // relocalize is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<Keyframe> vpCandidateKFs = keyframe_db_->DetectRelocalizationCandidates(&current_frame_);

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we set up a PnP solver

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<std::map<FeatureType, vector<Pt>>> vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        Keyframe pKF = vpCandidateKFs[i];
        if(pKF->is_bad())
            vbDiscarded[i] = true;
        else
        {
            std::map<FeatureType, int> nmatches_ft = matcher_->match_keyframe_to_frame(pKF, current_frame_, vvpMapPointMatches[i], std::vector<FeatureType>{featureType});
            if(nmatches_ft[featureType] < minNmatches)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(current_frame_,vvpMapPointMatches[i][featureType], featureType);
                pSolver->SetRansacParameters(ransac_probability,ransac_minInliers,ransac_maxIterations,ransac_minSet,ransac_epsilon,ransac_th2);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw_tmp = pSolver->iterate(numItpSolver,bNoMore,vbInliers,nInliers);
            mat4f Tcw{mat4f::Zero()};
            if(!Tcw_tmp.empty())
                Tcw = Converter::toMatrix4f(Tcw_tmp);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(Tcw(3,3) == 1.0f)
            {
                current_frame_.Tcw = Tcw;
                set<Pt> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        current_frame_.pts.at(featureType)[j]=vvpMapPointMatches[i][featureType][j];
                        sFound.insert(vvpMapPointMatches[i][featureType][j]);
                    }
                    else
                        current_frame_.pts.at(featureType)[j]=nullptr;
                }

                int nGood = Optimizer::PoseOptimization(&current_frame_);

                if(nGood < nGood_low)
                    continue;

                for(int io =0; io<current_frame_.N.at(featureType); io++)
                    if(current_frame_.outliers.at(featureType)[io])
                        current_frame_.pts.at(featureType)[io]=static_cast<Pt>(nullptr);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood < nGood_high)
                {
                    int nadditional = matcher_->SearchByProjection(current_frame_,vpCandidateKFs[i],sFound,radiusTh_high_reloc, true, featureType);

                    if(nadditional+nGood >= nGood_high)
                    {
                        nGood = Optimizer::PoseOptimization(&current_frame_);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood > nGood_medium && nGood < nGood_high)
                        {
                            sFound.clear();
                            for(int ip =0; ip<current_frame_.N.at(featureType); ip++)
                                if(current_frame_.pts.at(featureType)[ip])
                                    sFound.insert(current_frame_.pts.at(featureType)[ip]);
                            nadditional =matcher_->SearchByProjection(current_frame_,vpCandidateKFs[i],sFound,radiusTh_low_reloc,false, featureType);

                            // Final optimization
                            if(nGood+nadditional >= nGood_high)
                            {
                                nGood = Optimizer::PoseOptimization(&current_frame_);

                                for(int io =0; io < current_frame_.N.at(featureType); io++)
                                    if(current_frame_.outliers.at(featureType)[io])
                                        current_frame_.pts.at(featureType)[io]=nullptr;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood >= nGood_high)
                {
                    AF_INFO("relocalize: succeeded | frame=" << current_frame_.frame_id
                            << " feature=" << featureName(featureType)
                            << " matchedKeyframe=" << vpCandidateKFs[i]->keyId
                            << " inliers=" << nGood << " requiredInliers=" << nGood_high);
                    std::cout.flush(); // AF_INFO writes to std::cout, which — unlike std::cerr's
                                        // implicit unitbuf flush behind AF_WARN — is fully buffered
                                        // once stdout is redirected to a file (as VSLAM-LAB's runner
                                        // does), so without this the line can sit unflushed for a while.
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        last_reloc_frame_id_ = current_frame_.frame_id;
        return true;
    }

}

void Tracking::reset()
{

    cout << "System Reseting" << endl;
    if(viewer_)
    {
        viewer_->RequestStop();
        while(!viewer_->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    local_mapper_->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    loop_closing_->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    keyframe_db_->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    map_->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    state_ = NO_IMAGES_YET;

    if(initializer_)
    {
        initializer_ = nullptr;
    }

    resize_times_.clear();
    frame_times_.clear();
    tracking_times_.clear();
    track_ref_times_.clear();
    pose_opt_times_.clear();
    local_map_times_.clear();
    grab_image_times_.clear();

    if(viewer_)
        viewer_->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::loadCameraParameters(const string &strCalibrationPath, const string &strSettingPath){

    YAML::Node settings = YAML::LoadFile(strSettingPath);
    YAML::Node calibration = YAML::LoadFile(strCalibrationPath);
    const YAML::Node& cameras = calibration["cameras"];

    std::string cam_name;
    cam_name = settings["cam_mono"].as<std::string>();
    YAML::Node cam{};
    for (size_t i{0}; i < cameras.size(); ++i){
        if (cameras[i]["cam_name"].as<std::string>() == cam_name){
            cam = cameras[i];
            break;
        }
    }

    mK = (cv::Mat_<float>(3, 3) << cam["focal_length"][0].as<float>(), 0.0f, cam["principal_point"][0].as<float>(),
            0.0f, cam["focal_length"][1].as<float>(), cam["principal_point"][1].as<float>(),
            0.0f, 0.0f, 1.0f);

    image_width_ = cam["image_dimension"][0].as<int>();
    image_height_ = cam["image_dimension"][1].as<int>();

    if(fix_image_size_){
        float ratio = float(image_width_) / float(image_height_);
        int new_h = (int) sqrt(307200.f / ratio);
        int new_w = (int) (float(new_h) * ratio);
        float conv_ratio_h = float(new_h)/float(image_height_);
        float conv_ratio_w = float(new_w)/float(image_width_);

        image_width_ = new_w;
        image_height_ = new_h;

        mK.at<float>(0,0) *= conv_ratio_w;
        mK.at<float>(1,1) *= conv_ratio_h;
        mK.at<float>(0,2) *= conv_ratio_w;
        mK.at<float>(1,2) *= conv_ratio_h;
    }

    // Distortion coefficients
    mDistCoef = cv::Mat::zeros(4,1,CV_32F);
    if (cam["distortion_type"] && cam["distortion_coefficients"]) {
        std::vector<float> dist_coeffs_vec = cam["distortion_coefficients"].as<std::vector<float>>();
        mDistCoef = cv::Mat(dist_coeffs_vec.size(), 1, CV_32F, dist_coeffs_vec.data()).clone();
    }

    // Camera frequence (hz)
    fps = cam["fps"].as<float>();
    if(fps == 0)
        fps = fps0;

    // RGB order
    bool is_rgb_ = cam["cam_type"].as<std::string>() != "bgr";

    // Load settings file
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    // Print camera parameters
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
    AF_CONFIG_FIELD("fps:                ", cam["fps"].as<float>());
    if(is_rgb_)        AF_CONFIG_FIELD("color order:        ", "RGB (ignored if grayscale)");
    else            AF_CONFIG_FIELD("color order:        ", "BGR (ignored if grayscale)");
    AF_CONFIG_END();
}

shared_ptr<FeatureExtractor> Tracking::getFeatureExtractor(const int& scaleNumFeaturesMonocular_,
                                                           const string &featureSettingsYamlFile,
                                                            const FeatureType& featureType){

    shared_ptr<FeatureExtractorSettings> extractorSettings = make_shared<FeatureExtractorSettings>(featureType, featureSettingsYamlFile);
    extractorSettings->maxNumFeatures *= scaleNumFeaturesMonocular_;

    const AF_VSLAM::Feature& ft = get_feature(featureType);
    return ft.createExtractor(extractorSettings);
}

void Tracking::getGrayImage(cv::Mat& im , const bool& rgb){
    if(im.channels() == 3)
    {
        if(rgb)
            cvtColor(im,im,CV_RGB2GRAY);
        else
            cvtColor(im,im,CV_BGR2GRAY);
    }
    else if(im.channels() == 4)
    {
        if(rgb)
            cvtColor(im,im,CV_RGBA2GRAY);
        else
            cvtColor(im,im,CV_BGRA2GRAY);
    }
}

} //namespace ORB_SLAM
