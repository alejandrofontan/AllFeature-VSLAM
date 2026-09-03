
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <sstream>

#include "afvslam_log.hpp"
#include "BruteForceMatcher.h"
#include "FeatureMatcher.h"
#include "Feature_superpoint256.h"

#include <opencv2/core/utility.hpp>

#include <PoseLib/robust.h>

using namespace std;

namespace AF_VSLAM
{

std::map<FeatureType, Descriptor_Distance_Type> FeatureMatcher::TH_LOW = {};
std::map<FeatureType, Descriptor_Distance_Type> FeatureMatcher::descDistTh_high_reloc = {};
std::map<FeatureType, Descriptor_Distance_Type> FeatureMatcher::descDistTh_low_reloc = {};

VerbosityLevel FeatureMatcher::verbosity{MEDIUM};

float FeatureMatcher::radiusScale{1.15f};

// Initializes all feature matching backends: SiftMatchGPU, LightGlue, and SuperPoint-LightGlue (TensorRT).
FeatureMatcher::FeatureMatcher(const int& image_width, const int& image_height,
    const std::vector<FeatureType>& feature_types,
    std::string name, float nnratio, bool checkOri):
    name(name), mfNNratio(nnratio), mbCheckOrientation(checkOri), image_width(image_width), image_height(image_height),
    feature_types(feature_types)
{
    AF_INFO("Initializing FeatureMatcher: " + name);
    {
        // Diagnostic (issue #13 P2): record which parallel backend this OpenCV build runs,
        // since descriptor-matching cost depends on it and conda builds vary.
        const std::string info = cv::getBuildInformation();
        const size_t p = info.find("Parallel framework");
        AF_INFO("OpenCV threading: getNumThreads=" << cv::getNumThreads() << " | "
                << (p == std::string::npos ? std::string("Parallel framework: (not reported)")
                                           : info.substr(p, info.find('\n', p) - p)));
    }
    for(const auto& feat_type : feature_types) {
        if (feat_type == FEAT_SIFT128) {
            AF_INFO("Initializing SiftMatchGPU...");
            sift_match_gpu_ = SiftMatchGPU();
            sift_match_gpu_.SetLanguage(SiftMatchGPU::SIFTMATCH_CUDA);
            if (sift_match_gpu_.VerifyContextGL() == 0) {
                AF_ERROR("SiftMatchGPU initialization failed!");
            }
            sift_match_gpu_.Allocate(FeatureMatcher::max_supported, 1);
            AF_INFO("SiftMatchGPU initialized.");
        }

        if (feat_type == FEAT_SUPERPOINT256) {
            AF_INFO("Initializing SuperPoint-LightGlue...");
            const std::string& config_path = Superpoint256::getConfigYaml();
            const std::string& model_dir = Superpoint256::getModelsDir();
            Configs configs(config_path, model_dir);
            matcher_lightglue_superpoint = std::make_shared<SuperPointLightGlue>(configs.superpoint_lightglue_config);

            AF_CONFIG_BEGIN("SuperPoint-LightGlue config");
            AF_CONFIG_FIELD("onnx_file:           ", configs.superpoint_lightglue_config.onnx_file);
            AF_CONFIG_FIELD("engine_file:         ", configs.superpoint_lightglue_config.engine_file);
            AF_CONFIG_FIELD("input_tensor_names:  ", configs.superpoint_lightglue_config.input_tensor_names.size());
            AF_CONFIG_FIELD("output_tensor_names: ", configs.superpoint_lightglue_config.output_tensor_names.size());
            AF_CONFIG_END();

            // Build reuses a cached .engine file if present, or compiles from ONNX on first run
            if (!matcher_lightglue_superpoint->build()) {
                throw std::runtime_error(
                        "SuperPoint-LightGlue: failed to build/load TensorRT engine.\n"
                        "  config: " + config_path + "\n"
                        "  weights: " + model_dir);
            }

            // Warm up the model by running a dummy inference to avoid first-run overhead during actual matching
            Eigen::Matrix<double, 258, Eigen::Dynamic> dummy = Eigen::Matrix<double, 258, Eigen::Dynamic>::Zero(258, 1);
            std::vector<cv::DMatch> dummy_matches;
            matcher_lightglue_superpoint->matching_points(dummy, dummy, dummy_matches);

            AF_INFO("SuperPoint-LightGlue initialized and warmed up from: " + model_dir);
        }
        if (feat_type == FEAT_ALIKED128 || feat_type == FEAT_SIFT128) {
            AF_INFO("Initializing LightGlue...");
            // Select CUDA if available, fall back to CPU
            torch_device = std::make_shared<torch::Device>(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
            matcher_lightglue = std::make_shared<matcher::LightGlue>();
            matcher_lightglue->to(*torch_device);
            AF_INFO("LightGlue initialized.");
        }
    }
}

std::vector<cv::DMatch> FeatureMatcher::swap_match_direction(const std::vector<cv::DMatch>& in)
{
    std::vector<cv::DMatch> out = in;
    for (auto& m : out)
        std::swap(m.queryIdx, m.trainIdx);
    return out;
}

// Matches features from a keyframe to a frame across all requested feature types,
// then associates the surviving matches to 3D map points.
//
// The matching proceeds in three stages:
//   1. For each feature type, brute-force NN descriptor matching is run between
//      the keyframe and the frame. All matches are pooled into a single list with
//      globally offset indices.
//   2. Outliers are jointly filtered across all feature types with PoseLib
//      LO-RANSAC on the fundamental matrix (homography fallback for degenerate scenes).
//   3. Each inlier match is resolved to a map point via the keyframe's map point
//      associations. Only valid, non-bad map points are written to map_pts_matches.
//
// The matched keypoint pairs are also cached in both the keyframe and the frame
// for downstream use (e.g. visualization, loop closing).
//
// Returns the number of valid map point matches per feature type.
// Used in: Tracking::track_reference_keyframe, Tracking::relocalize
map<FeatureType, int> FeatureMatcher::match_keyframe_to_frame(Keyframe& keyframe, Frame& frame,
    map<FeatureType, vector<Pt>>& map_pts_matches, const vector<FeatureType>& feat_types)
{
    vector<cv::KeyPoint> kps1, kps2;
    vector<int> kps1_indexes, kps2_indexes;
    vector<cv::DMatch> all_matches;
    vector<FeatureType> used_feature_types;
    map<FeatureType, vector<Pt>> map_pts_kf;

    // Slow-call diagnosis (hiccup profiling): per-feature matching and filter timings
    using MatchClock = std::chrono::steady_clock;
    const auto matchMs = [](MatchClock::time_point a, MatchClock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::map<FeatureType, double> msMatchByFt;

    for (const auto& ft : feat_types) {
        // Skip feature types not present in both keyframe and frame
        auto it1 = keyframe->descriptors.find(ft);
        auto it2 = frame.descriptors.find(ft);

        if (it1 == keyframe->descriptors.end() || it2 == frame.descriptors.end())
            continue;

        // Get map points associated to the keyframe for this feature type
        map_pts_kf[ft] = keyframe->get_map_point_matches(ft);

        // Brute-force NN matching between keyframe and frame descriptors
        const auto& v1 = keyframe->keypoints.at(ft);
        const auto& v2 = frame.keypoints.at(ft);
        const auto tMatch0 = MatchClock::now();
        vector<cv::DMatch> matches = match_descriptors(
            it1->second, it2->second,
            v1, v2, ft);
        msMatchByFt[ft] = matchMs(tMatch0, MatchClock::now());

        // Offset match indices to account for previously appended feature types
        const size_t size_kpts1 = kps1.size();
        const size_t size_kpts2 = kps2.size();
        for (auto& m : matches) {
            m.queryIdx += size_kpts1;
            m.trainIdx += size_kpts2;
        }

        // Accumulate matches and keypoints across feature types
        all_matches.insert(all_matches.end(), matches.begin(), matches.end());
        kps1.insert(kps1.end(), v1.begin(), v1.end());
        kps2.insert(kps2.end(), v2.begin(), v2.end());

        // Track per-feature-type original keypoint indices for later map point lookup
        const size_t base1 = kps1_indexes.size();
        kps1_indexes.resize(base1 + v1.size());
        std::iota(kps1_indexes.begin() + base1, kps1_indexes.end(), 0);

        const size_t base2 = kps2_indexes.size();
        kps2_indexes.resize(base2 + v2.size());
        std::iota(kps2_indexes.begin() + base2, kps2_indexes.end(), 0);

        used_feature_types.insert(used_feature_types.end(), v1.size(), ft);

        // Initialize output map point matches for this feature type
        map_pts_matches[ft] = vector<Pt>(frame.N.at(ft), nullptr);
    }

    // Filter outliers robustly (PoseLib LO-RANSAC) across all feature types jointly
    const auto tFilter0 = MatchClock::now();
    auto robust_matches = filter_matches_by_fundamental(all_matches, kps1, kps2);
    const double msFilter = matchMs(tFilter0, MatchClock::now());

    double msMatchTotal = 0.0;
    for (const auto& [ft, ms] : msMatchByFt) msMatchTotal += ms;
    if (msMatchTotal + msFilter > 100.0) {
        std::ostringstream perFt;
        for (const auto& [ft, ms] : msMatchByFt)
            perFt << " ft" << int(ft) << "=" << int(ms) << "ms(n=" << keyframe->keypoints.at(ft).size() << ")";
        AF_WARN("match_keyframe_to_frame slow: match=" << int(msMatchTotal) << "ms [" << perFt.str()
                << " ], filter=" << int(msFilter) << "ms (nMatches=" << all_matches.size() << ")");
    }

    // Associate surviving matches to map points
    auto& it1 = keyframe->cache_matched_pairs_feat_type[frame.frame_id];
    auto& it2 = frame.cache_matched_pairs_feat_type[keyframe->frame_id];

    // One entry per requested feature type, zero for types with no match: callers index this
    // with .at(ft) (a type with zero matches used to be a missing key -> std::out_of_range)
    map<FeatureType, int> matches_count;
    for (const auto& ft : feat_types)
        matches_count[ft] = 0;
    for (const auto& m : robust_matches) {
        const int query_idx = kps1_indexes[m.queryIdx];
        const int train_idx = kps2_indexes[m.trainIdx];
        const FeatureType feat_type = used_feature_types[m.queryIdx];
        const Pt pt = map_pts_kf[feat_type][query_idx];
        if (!pt || pt->is_bad())
            continue;
        map_pts_matches[feat_type][train_idx] = pt;
        matches_count[feat_type]++;

        it1[feat_type].push_back(cv::DMatch(m.queryIdx, m.trainIdx, m.distance));
        it2[feat_type].push_back(cv::DMatch(m.trainIdx, m.queryIdx, m.distance));
    }

    // Cache matched pairs in both keyframe and frame for downstream use
    frame.cache_matched_pairs.insert_or_assign(keyframe->frame_id, swap_match_direction(robust_matches));
    keyframe->cache_matched_pairs.insert_or_assign(frame.frame_id, std::move(robust_matches));

    return matches_count;
}

// For each feature type, matches unmatched frame keypoints against projected map points
// by descriptor similarity, then verifies each match via a spatial projection check.
// Returns the number of new map point assignments made to the frame.
// Used in: Tracking::track_local_map
int FeatureMatcher::match_map_points_to_frame(Frame& frame, const vector<Pt>& map_pts)
{
    map<FeatureType, vector<size_t>> pt_idx, frame_idx;
    map<FeatureType, cv::Mat> desc_pts, desc_frame;

    // Collect valid map points grouped by feature type. Two passes — indices first, then
    // rows into a preallocated matrix: cv::Mat::push_back reallocates and re-copies the
    // whole matrix as it grows, which at local-map size dominated this collection step.
    for (size_t idx = 0; idx < map_pts.size(); idx++) {
        const auto& pt = map_pts[idx];
        if (!pt || pt->is_bad())
            continue;

        pt_idx[pt->featureType].push_back(idx);
    }
    for (const auto& [ft, idxs] : pt_idx) {
        const cv::Mat first = map_pts[idxs[0]]->get_descriptor();
        cv::Mat& d = desc_pts[ft];
        d.create(static_cast<int>(idxs.size()), first.cols, first.type());
        for (size_t k = 0; k < idxs.size(); k++)
            map_pts[idxs[k]]->get_descriptor().copyTo(d.row(static_cast<int>(k)));
    }

    int num_matches = 0;
    for (const auto& [ft, N]: frame.N) {
        // Collect unmatched frame keypoints grouped by feature type (same two-pass shape)
        const auto& pts_ft  = frame.pts.at(ft);
        const auto& desc_ft = frame.descriptors.at(ft);
        frame_idx[ft].reserve(N);
        for (size_t i = 0; i < static_cast<size_t>(N); i++) {
            // Skip keypoints already assigned to an observed map point
            if (pts_ft[i] && pts_ft[i]->number_of_observations() > 0)
                continue;

            frame_idx[ft].push_back(i);
        }
        if (!frame_idx[ft].empty()) {
            cv::Mat& d = desc_frame[ft];
            d.create(static_cast<int>(frame_idx[ft].size()), desc_ft.cols, desc_ft.type());
            for (size_t k = 0; k < frame_idx[ft].size(); k++)
                desc_ft.row(static_cast<int>(frame_idx[ft][k])).copyTo(d.row(static_cast<int>(k)));
        }

        // Skip feature types with no descriptors on either side
        auto it1 = desc_frame.find(ft);
        auto it2 = desc_pts.find(ft);
        if (it1 == desc_frame.end() || it2 == desc_pts.end())
            continue;

        // Match frame keypoint descriptors against map point descriptors
        vector<cv::DMatch> matches = match_descriptors_only(it1->second, it2->second, ft);

        // Validate each match: accept only if the map point projects near the matched keypoint
        const auto& f_idx = frame_idx.at(ft);
        const auto& p_idx = pt_idx.at(ft);
        for (const auto& m : matches) {
            const size_t query_idx = f_idx[m.queryIdx];
            const size_t train_idx = p_idx[m.trainIdx];
            const Pt pt = map_pts[train_idx];
            const vector<size_t> area_indices = frame.get_features_in_area(pt->track_proj_x, pt->track_proj_y, projection_match_radius_th, ft);
            if (area_indices.empty())
                continue;

            if (std::find(area_indices.begin(), area_indices.end(), query_idx) != area_indices.end()) {
                frame.pts.at(ft)[query_idx] = pt;
                num_matches++;
            }
        }
    }
    return num_matches;
}

// Matches two keyframes across all feature types to find candidate pairs for triangulation.
// Reuses cached matches from a prior tracking step if available; otherwise runs parallel
// descriptor matching followed by LMEDS fundamental matrix filtering.
// Outputs only pairs where neither keyframe has an existing 3D map point for those keypoints.
// Used in: LocalMapping::CreateNewMapPoints
map<FeatureType, vector<pair<size_t,size_t>>> FeatureMatcher::match_keyframes(const Keyframe& keyframe1, const Keyframe& keyframe2,
                                const std::vector<FeatureType>& feat_types){

    // Reuse cached matches if keyframe1 already has matches stored for keyframe2
    auto cache_it = keyframe1->cache_matched_pairs.find(keyframe2->frame_id);
    const bool cached = (cache_it != keyframe1->cache_matched_pairs.end() && !cache_it->second.empty());

    // No cache — run parallel descriptor matching across all feature types
    map<FeatureType, vector<cv::DMatch>> matches_by_type;
    if (!cached) {
        matches_by_type = match_descriptors_parallel(feat_types,
            keyframe1->descriptors, keyframe2->descriptors, keyframe1->keypoints, keyframe2->keypoints);
    }

    // Offset match indices to account for previously appended feature types
    vector<cv::KeyPoint> kps1, kps2;
    vector<size_t> kps1_indexes, kps2_indexes;
    vector<cv::DMatch> all_matches;
    vector<FeatureType> used_feature_types;
    for (const auto& ft : feat_types) {

        // Ensure both frames contain the requested feature type
        auto it1 = keyframe1->descriptors.find(ft);
        auto it2 = keyframe2->descriptors.find(ft);
        if (it1 == keyframe1->descriptors.end() || it2 == keyframe2->descriptors.end())
            continue;

        // Accumulate keypoints and track per-feature-type original indices for later lookup
        auto const& v1 = keyframe1->keypoints.at(ft);
        auto const& v2 = keyframe2->keypoints.at(ft);

        if (!cached) {
            auto matches = std::move(matches_by_type[ft]);
            const size_t size_kpts1 = kps1.size();
            const size_t size_kpts2 = kps2.size();
            for (auto& m : matches) {
                m.queryIdx += size_kpts1;
                m.trainIdx += size_kpts2;
            }
            all_matches.insert(all_matches.end(),
                std::make_move_iterator(matches.begin()),
                std::make_move_iterator(matches.end()));

            kps1.insert(kps1.end(), v1.begin(), v1.end());
            kps2.insert(kps2.end(), v2.begin(), v2.end());
        }

        const size_t base1 = kps1_indexes.size();
        kps1_indexes.resize(base1 + v1.size());
        std::iota(kps1_indexes.begin() + base1, kps1_indexes.end(), 0);
        const size_t base2 = kps2_indexes.size();
        kps2_indexes.resize(base2 + v2.size());
        std::iota(kps2_indexes.begin() + base2, kps2_indexes.end(), 0);
        used_feature_types.insert(used_feature_types.end(), v1.size(), ft);
    }

    // Filter outliers jointly across all feature types, or retrieve from cache
    vector<cv::DMatch> computed_matches;
    if (!cached) {
        if (all_matches.size() < min_match_keyframes)
            return map<FeatureType, vector<pair<size_t,size_t>>>(); // Not enough matches to reliably estimate fundamental matrix

        computed_matches = filter_matches_by_fundamental(all_matches, kps1, kps2);
    }
    const vector<cv::DMatch>& filtered_matches = cached ? cache_it->second : computed_matches;

    auto& it1 = keyframe1->cache_matched_pairs_feat_type[keyframe2->frame_id];
    auto& it2 = keyframe2->cache_matched_pairs_feat_type[keyframe1->frame_id];
    map<FeatureType, vector<pair<size_t,size_t>>> matched_pairs;
    for (const auto& m : filtered_matches) {
        const size_t query_idx = kps1_indexes[m.queryIdx];
        const size_t train_idx = kps2_indexes[m.trainIdx];
        const FeatureType feat_type = used_feature_types[m.queryIdx];

        // Only triangulate points that don't already have a 3D MapPoint
        matched_pairs[feat_type].emplace_back(query_idx, train_idx);
        if (!cached){
            it1[feat_type].push_back(cv::DMatch(m.queryIdx, m.trainIdx, m.distance));
            it2[feat_type].push_back(cv::DMatch(m.trainIdx, m.queryIdx, m.distance));
        }
    }

    if(!cached){
        keyframe2->cache_matched_pairs.insert_or_assign(keyframe1->frame_id, swap_match_direction(computed_matches));
        keyframe1->cache_matched_pairs.insert_or_assign(keyframe2->frame_id, std::move(computed_matches));
    }
    return matched_pairs;
}

void FeatureMatcher::match_keyframes_for_triangulation(const Keyframe& keyframe1, const Keyframe& keyframe2,
                                std::map<FeatureType, vector<pair<size_t,size_t>>>& matched_pairs,
                                const std::vector<FeatureType>& feat_types){


    std::map<FeatureType, vector<pair<size_t,size_t>>> matched_pairs_ = match_keyframes(keyframe1, keyframe2, feat_types);

    // Only triangulate points that don't already have a 3D MapPoint
    for (const auto& [ft, matches] : matched_pairs_) {
        for (const auto& m : matches) {
            if (!keyframe1->get_map_point(m.first, ft) && !keyframe2->get_map_point(m.second, ft))
                matched_pairs[ft].emplace_back(m.first, m.second);
        }
    }
}

void FeatureMatcher::match_keyframes_for_compute_sim3(const Keyframe& keyframe1, const Keyframe& keyframe2,
                                std::map<FeatureType, vector<pair<size_t,size_t>>>& matched_pairs,
                                const std::vector<FeatureType>& feat_types){

    std::map<FeatureType, vector<pair<size_t,size_t>>> matched_pairs_ = match_keyframes(keyframe1, keyframe2, feat_types);

    // Only match valid points
    for (const auto& [ft, matches] : matched_pairs_) {
        for (const auto& m : matches) {
            auto pMP1 = keyframe1->get_map_point(m.first, ft);
            auto pMP2 = keyframe2->get_map_point(m.second, ft);
            if (pMP1 && pMP2 && (!pMP1->is_bad() || !pMP2->is_bad()))
                matched_pairs[ft].emplace_back(m.first, m.second);
        }
    }
}

// Projects each map point into the keyframe and attempts to fuse it with an existing keypoint.
// For each map point, applies visibility checks (depth, image bounds, scale, viewing angle),
// then finds the closest keypoint by descriptor distance within radius_th.
// If a match is found: replaces the weaker map point if one already exists, or adds a new observation.
// Returns the number of fused map points.
// Used in: LocalMapping::SearchInNeighbors
int FeatureMatcher::fuse_map_points_to_keyframe(const Keyframe& keyframe, const vector<Pt>& map_pts, const float& radius_th, const FeatureType& feat_type)
{
    const mat3f Rcw = keyframe->get_rotation();
    const vec3f tcw = keyframe->get_translation();
    const float fx = keyframe->fx;
    const float fy = keyframe->fy;
    const float cx = keyframe->cx;
    const float cy = keyframe->cy;

    const Descriptor_Distance_Type th_low = TH_LOW[feat_type];

    const vec3f cam_center = keyframe->get_camera_center();

    int n_fused = 0;
    const auto& kps_ft  = keyframe->keypoints.at(feat_type);
    const auto& desc_ft = keyframe->descriptors.at(feat_type);

    // Phase A (issue #13 P45): the per-point search — projection, visibility checks, and
    // best-descriptor lookup — reads only immutable keyframe data (keypoints, descriptors,
    // grid; pose snapshotted above) and the point's own state, so it parallelizes. All map
    // MUTATION happens in phase B below, serially in the original order. One bounded
    // deviation from strict serial order: a candidate's descriptor search sees its pre-call
    // descriptor even if an earlier fuse in this same call updated it via replace() (rare
    // shared-point corner — same information-freshness class as the concurrent cross-thread
    // descriptor updates that already exist).
    struct FuseCandidate { int best_idx{-1}; Descriptor_Distance_Type best_dist{}; };
    std::vector<FuseCandidate> fuse_cand(map_pts.size());

    #pragma omp parallel for schedule(dynamic, 64)
    for (int iPt = 0; iPt < static_cast<int>(map_pts.size()); iPt++) {
        const auto& pt = map_pts[iPt];
        if (!pt)
            continue;

        if (pt->is_bad() || pt->is_in_keyframe(keyframe))
            continue;

        const vec3f pos_world = pt->get_world_pos();

        // Project map point into keyframe
        const vec3f pos_cam = Rcw * pos_world + tcw;

        // Depth must be positive
        if (pos_cam(2) < 0.0f)
            continue;

        const float invz = 1.0f / pos_cam(2);

        // Compute projected image coordinates
        const float u = fx * pos_cam(0) * invz + cx;
        const float v = fy * pos_cam(1) * invz + cy;

        // Point must be inside the image
        if (!keyframe->is_in_image(u, v))
            continue;

        const float max_distance = pt->get_max_distance_invariance();
        const float min_distance = pt->get_min_distance_invariance();
        const vec3f dir_to_pt = pos_world - cam_center;
        const float dist3D = dir_to_pt.norm();

        // Depth must be inside the scale pyramid of the image
        if (dist3D < min_distance || dist3D > max_distance)
            continue;

        // Viewing angle must be less than 60 deg
        const vec3f normal = pt->get_normal();

        if (dir_to_pt.dot(normal) < cos_viewing_angle_th * dist3D)
            continue;

        // Search in a radius
        const vector<size_t> area_indices = keyframe->get_features_in_area(u, v, radius_th, feat_type);
        if (area_indices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat ref_descriptor = pt->get_descriptor();
        Descriptor_Distance_Type best_dist{highest_possible_distance};
        int best_idx{-1};
        for (const size_t idx : area_indices) {
            // Skip reprojection outliers
            const float ex = u - kps_ft[idx].pt.x;
            const float ey = v - kps_ft[idx].pt.y;
            const float e2 = ex * ex + ey * ey;

            if (e2 * keyframe->get_keypt_1Dinf(KeypointIndex(idx), feat_type) > chi2_perc)
                continue;

            const cv::Mat &descriptor = desc_ft.row(idx);
            const Descriptor_Distance_Type desc_dist = descriptor_distance(ref_descriptor, descriptor, pt->featureType);

            // Keep only the closest descriptor match
            if (desc_dist < best_dist) {
                best_dist = desc_dist;
                best_idx = idx;
            }
        }

        if (best_dist <= th_low) {
            fuse_cand[iPt].best_idx = best_idx;
            fuse_cand[iPt].best_dist = best_dist;
        }
    }

    // Phase B — serial fuse in original order. The is_bad/is_in_keyframe re-check is what
    // makes this equivalent to the former single loop: a replace() earlier in THIS loop can
    // turn a later candidate bad or add it to the keyframe (points shared between the two
    // keyframes appear both as candidates and as existing points) — the serial version saw
    // that through its iteration order, so the state checks must be evaluated here, not in
    // phase A.
    for (size_t iPt = 0; iPt < map_pts.size(); iPt++) {
        const int best_idx = fuse_cand[iPt].best_idx;
        if (best_idx < 0)
            continue;
        const auto& pt = map_pts[iPt];
        if (pt->is_bad() || pt->is_in_keyframe(keyframe))
            continue;

        // Fuse: replace weaker map point or add new observation
        Pt existing_pt = keyframe->get_map_point(best_idx, feat_type);
        if (existing_pt) {
            if (!existing_pt->is_bad()) {
                if (existing_pt->number_of_observations() > pt->number_of_observations())
                    pt->replace(existing_pt);
                else
                    existing_pt->replace(pt);
            }
        }
        else {
            pt->add_observation(keyframe, best_idx);
            keyframe->add_map_point(pt, best_idx);
        }
        n_fused++;
    }

    return n_fused;
}

// Matches keypoints between two frames across all feature types for monocular initialization.
// Descriptors are matched in parallel per feature type, then all matches are pooled and filtered
// jointly via LMEDS fundamental matrix estimation. Matched index pairs are written into
// matched_pairs per feature type.
// Used in: Tracking::monocular_initialization
map<FeatureType, vector<pair<size_t, size_t>>> FeatureMatcher::match_frames_for_initialization(const Frame& F1, const Frame& F2,
    const vector<FeatureType>& feat_types)
{
    map<FeatureType, vector<cv::DMatch>> matches_by_type;
    matches_by_type = match_descriptors_parallel(feat_types, F1.descriptors, F2.descriptors, F1.keypoints, F2.keypoints);

    // Offset match indices to account for previously appended feature types
    vector<cv::KeyPoint> kps1, kps2;
    vector<size_t> kps1_indexes, kps2_indexes;
    vector<cv::DMatch> all_matches;
    vector<FeatureType> used_feature_types;
    for (const auto& ft : feat_types) {
        // Ensure both frames contain the requested feature type
        auto it1 = F1.descriptors.find(ft);
        auto it2 = F2.descriptors.find(ft);
        if (it1 == F1.descriptors.end() || it2 == F2.descriptors.end())
            continue;

        // Accumulate keypoints and track per-feature-type original indices for later lookup
        auto const& v1 = F1.keypoints.at(ft);
        auto const& v2 = F2.keypoints.at(ft);

        auto matches = std::move(matches_by_type[ft]);
        const size_t size_kpts1 = kps1.size();
        const size_t size_kpts2 = kps2.size();
        for (auto& m : matches) {
            m.queryIdx += size_kpts1;
            m.trainIdx += size_kpts2;
        }
        all_matches.insert(all_matches.end(),
            std::make_move_iterator(matches.begin()),
            std::make_move_iterator(matches.end()));

        kps1.insert(kps1.end(), v1.begin(), v1.end());
        kps2.insert(kps2.end(), v2.begin(), v2.end());

        const size_t base1 = kps1_indexes.size();
        kps1_indexes.resize(base1 + v1.size());
        std::iota(kps1_indexes.begin() + base1, kps1_indexes.end(), 0);
        const size_t base2 = kps2_indexes.size();
        kps2_indexes.resize(base2 + v2.size());
        std::iota(kps2_indexes.begin() + base2, kps2_indexes.end(), 0);
        used_feature_types.insert(used_feature_types.end(), v1.size(), ft);
    }

    // Filter outliers jointly across all feature types using fundamental matrix (LMEDS)
    auto filtered_matches = filter_matches_by_fundamental(all_matches, kps1, kps2);

    std::map<FeatureType, std::vector<pair<size_t, size_t>>> matched_pairs;
    for (const auto& m : filtered_matches) {
        const size_t query_idx = kps1_indexes[m.queryIdx];
        const size_t train_idx = kps2_indexes[m.trainIdx];
        const FeatureType feat_type = used_feature_types[m.queryIdx];
        matched_pairs[feat_type].emplace_back(query_idx, train_idx);
    }

    return matched_pairs;
}

/// @brief Projects candidate map points into @p keyframe using the Sim3 @p Scw and validates
///        each projection against cached descriptor matches.
///        Accepted matches are written into @p pts_matched at the matched keypoint index.
/// @param keyframe      Target keyframe into which map points are projected.
/// @param Scw           Sim3 transform from world to camera (scale × rotation + translation).
/// @param pts           Candidate map points to project and validate.
/// @param pts_matched   Output vector (indexed by keypoint); filled with accepted map points.
/// @param pt_to_keyframe_id  Maps each map-point ID to the keyframe it was originally observed in.
/// @return Number of new map point assignments written into @p pts_matched.
/// @note Used in: LoopClosing::ComputeSim3
int FeatureMatcher::search_by_projection_for_compute_sim3(const Keyframe& keyframe, const mat4f& Scw,
    const vector<Pt>& pts, vector<Pt>& pts_matched,
    const map<PtId, Keyframe>& pt_to_keyframe_id)
{
    const float fx = keyframe->fx;
    const float fy = keyframe->fy;
    const float cx = keyframe->cx;
    const float cy = keyframe->cy;

    // Decompose Sim3 into rotation and translation
    const mat3f sRcw = Scw.block<3,3>(0,0);
    const float  scw  = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    const mat3f Rcw  = sRcw / scw;
    const vec3f tcw  = Scw.block<3,1>(0,3);

    // Build the set of already-matched map points to skip
    unordered_set<Pt> pts_already_found(pts_matched.begin(), pts_matched.end());
    pts_already_found.erase(nullptr);

    int num_matches = 0;

    for (const Pt& pt : pts) {
        if (pt->is_bad() || pts_already_found.count(pt))
            continue;

        // Look up cached match index for this map point
        const Keyframe& ref_kf = pt_to_keyframe_id.at(pt->ptId);
        const auto cache_it = keyframe->cache_matched_pairs_feat_type.find(ref_kf->frame_id);
        if (cache_it == keyframe->cache_matched_pairs_feat_type.end())
            continue;

        const int ref_idx = pt->GetIndexInKeyFrame(ref_kf);
        int bestIdx{-1};
        for (const auto& m : cache_it->second[pt->featureType]) {
            if (m.trainIdx == ref_idx) {
                bestIdx = m.queryIdx;
                break;
            }
        }
        if (bestIdx == -1)
            continue;

        // Project map point into the keyframe
        const vec3f p3Dc = Rcw * pt->get_world_pos() + tcw;
        if (p3Dc(2) < 0.0f)
            continue;

        const float invz = 1.0f / p3Dc(2);
        const float u = fx * p3Dc(0) * invz + cx;
        const float v = fy * p3Dc(1) * invz + cy;

        if (!keyframe->is_in_image(u, v))
            continue;

        // Accept match only if bestIdx falls within the projected search radius
        const vector<size_t> area_indices = keyframe->get_features_in_area(u, v, radiusTh_factor, pt->featureType);
        if (std::find(area_indices.begin(), area_indices.end(), static_cast<size_t>(bestIdx)) != area_indices.end()) {
            pts_matched[bestIdx] = pt;
            num_matches++;
        }
    }
    return num_matches;
}

// Sim3 overload: projects map points into a keyframe using a Sim3 transform and fuses them.
// Used in: LoopClosing
int FeatureMatcher::fuse_map_points_to_keyframe(Keyframe& keyframe, const mat4f& Scw,
    const vector<Pt> &map_pts, const float& radius_th, vector<Pt> &replace_pts, const FeatureType& feat_type)
{

    // Get Calibration Parameters for later projection
    const float &fx = keyframe->fx;
    const float &fy = keyframe->fy;
    const float &cx = keyframe->cx;
    const float &cy = keyframe->cy;

    // Decompose Scw
    const mat3f sRcw = Scw.block<3,3>(0,0);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    const mat3f Rcw = sRcw / scw;
    const vec3f tcw = Scw.block<3,1>(0,3);
    const vec3f Ow = -Rcw.transpose() * tcw;

    // Set of MapPoints already found in the KeyFrame
    const set<Pt> spAlreadyFound = keyframe->get_map_points(feat_type);

    int nFused=0;

    const size_t nPoints = map_pts.size();

    // For each candidate MapPoint project and match
    for(size_t iMP=0; iMP<nPoints; iMP++)
    {
        const Pt pMP = map_pts[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->is_bad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        const vec3f p3Dw = pMP->get_world_pos();

        // Transform into Camera Coords.
        const vec3f p3Dc = Rcw * p3Dw + tcw;

        // Depth must be positive
        if(p3Dc(2) < 0.0f)
            continue;

        // Project into Image
        const float invz = 1.0f / p3Dc(2);
        const float x = p3Dc(0) * invz;
        const float y = p3Dc(1) * invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!keyframe->is_in_image(u,v))
            continue;

        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->get_max_distance_invariance();
        const float minDistance = pMP->get_min_distance_invariance();
        const vec3f PO = p3Dw-Ow;
        const float dist3D = PO.norm();

        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        const vec3f Pn = pMP->get_normal();

        if(PO.dot(Pn) < 0.5f * dist3D)
            continue;

        // Search in a radius
        const float predictedSize = pMP->PredictSize(dist3D);
        const float radius = radiusScale * radius_th * predictedSize;

        const vector<size_t> vIndices = keyframe->get_features_in_area(u,v,radius, feat_type);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat refDescriptor = pMP->get_descriptor();
        Descriptor_Distance_Type bestDist{highest_possible_distance};
        int bestIdx{-1};

        for(const size_t idx : vIndices)
        {
            //const float keyPtSize = pKF->GetKeyPtSize(KeypointIndex (idx), featType);
            //if((keyPtSize < predictedSize / pKF->sizeTolerance) || (keyPtSize > predictedSize * pKF->sizeTolerance))
            //    continue;

            const cv::Mat &descriptor = keyframe->descriptors.at(feat_type).row(idx);
            const Descriptor_Distance_Type descDist = descriptor_distance(refDescriptor,descriptor,pMP->featureType);

            if(descDist < bestDist)
            {
                bestDist = descDist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist <= TH_LOW[feat_type])
        {
            const Pt pMPinKF = keyframe->get_map_point(bestIdx, feat_type);
            if(pMPinKF)
            {
                if(!pMPinKF->is_bad())
                    replace_pts[iMP] = pMPinKF;
            }
            else
            {
                pMP->add_observation(keyframe,bestIdx);
                keyframe->add_map_point(pMP,bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

// Projects map points from a keyframe into a current frame for relocalization.
// Applies rotation histogram filtering to reject orientation-inconsistent matches.
int FeatureMatcher::search_by_projection(Frame &CurrentFrame, Keyframe pKF, const set<Pt> &sAlreadyFound, const float& radiusTh,
     const bool& useHighMatchingThreshold, const FeatureType& featType)
{
    Descriptor_Distance_Type descDistanceTh = descDistTh_low_reloc[featType];
    if(useHighMatchingThreshold)
        descDistanceTh = descDistTh_high_reloc[featType];

    const mat3f Rcw = CurrentFrame.Tcw.block<3,3>(0,0);
    const vec3f tcw = CurrentFrame.Tcw.block<3,1>(0,3);
    const vec3f Ow = -Rcw.transpose() * tcw;

    // Rotation Histogram (to check rotation consistency)
    int nMatches{0};
    float rotFactor{};
    vector<vector<int>>& rotHist = initRotationHistogram(rotFactor,HISTO_LENGTH);

    const vector<Pt> vpMPs = pKF->get_map_point_matches(featType);

    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        const Pt pMP = vpMPs[i];

        if(pMP)
        {
            if(!pMP->is_bad() && !sAlreadyFound.count(pMP))
            {
                //Project
                const vec3f x3Dw = pMP->get_world_pos();
                const vec3f x3Dc = Rcw * x3Dw + tcw;

                const float xc = x3Dc(0);
                const float yc = x3Dc(1);
                const float invzc = 1.0f / x3Dc(2);

                const float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
                const float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;

                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;

                // Compute predicted scale level
                const vec3f PO = x3Dw - Ow;
                const float dist3D = PO.norm();

                const float maxDistance = pMP->get_max_distance_invariance();
                const float minDistance = pMP->get_min_distance_invariance();

                // Depth must be inside the scale pyramid of the image
                if(dist3D<minDistance || dist3D>maxDistance)
                    continue;

                // Search in a window
                const float predictedSize = pMP->PredictSize(dist3D);
                const float radius = radiusScale * radiusTh * predictedSize;

                const vector<size_t> vIndices2 = CurrentFrame.get_features_in_area(u, v, radius, featType);

                if(vIndices2.empty())
                    continue;

                const cv::Mat refDescriptor = pMP->get_descriptor();
                Descriptor_Distance_Type bestDist{highest_possible_distance};
                int bestIdx2{-1};

                for(const size_t i2 : vIndices2)
                {
                    if(CurrentFrame.pts.at(featType)[i2])
                        continue;

                    const cv::Mat &descriptor = CurrentFrame.descriptors.at(featType).row(i2);
                    const Descriptor_Distance_Type descDist = descriptor_distance(refDescriptor,descriptor,pMP->featureType);

                    if(descDist < bestDist)
                    {
                        bestDist = descDist;
                        bestIdx2 = i2;
                    }
                }

                if(bestDist <= descDistanceTh)
                {
                    CurrentFrame.pts.at(featType)[bestIdx2]=pMP;
                    nMatches++;

                    if(mbCheckOrientation)
                        updateRotationHistogram(rotHist,bestIdx2, pKF->keypoints.at(featType)[i],CurrentFrame.keypoints.at(featType)[bestIdx2],rotFactor,HISTO_LENGTH);
                }
            }
        }
    }

    if(mbCheckOrientation)
        filterMatchesWithOrientation(rotHist,CurrentFrame.pts.at(featType),nMatches);

    return nMatches;
}

Descriptor_Distance_Type FeatureMatcher::descriptor_distance(const cv::Mat &a, const cv::Mat &b, const FeatureType& featureType_)
{
    const AF_VSLAM::Feature& ft = get_feature(featureType_);
    return ft.descriptor_distance(a,b);
}

void FeatureMatcher::setDescriptorDistanceThresholds(const string &feature_settings_yaml_file, const FeatureType& featureType) {

    AF_INFO("Loading Feature Matcher Settings from : " + feature_settings_yaml_file);
    cv::FileStorage fSettings(feature_settings_yaml_file, cv::FileStorage::READ);

    FeatureMatcher::TH_LOW[featureType] = fSettings["FeatureMatcher.TH_LOW"];
    FeatureMatcher::descDistTh_low_reloc[featureType] = fSettings["FeatureMatcher.descDistTh_high_reloc"];
    FeatureMatcher::descDistTh_high_reloc[featureType] = fSettings["FeatureMatcher.descDistTh_low_reloc"];

    AF_CONFIG_BEGIN("feature_settings_yaml_file");
        AF_CONFIG_FIELD("FeatureMatcher.TH_LOW", FeatureMatcher::TH_LOW[featureType]);
        AF_CONFIG_FIELD("FeatureMatcher.descDistTh_low_reloc", FeatureMatcher::descDistTh_low_reloc[featureType]);
        AF_CONFIG_FIELD("FeatureMatcher.descDistTh_high_reloc", FeatureMatcher::descDistTh_high_reloc[featureType]);
    AF_CONFIG_END();
}

vector<vector<int>>& FeatureMatcher::initRotationHistogram(float& rotFactor, const int& histLength){
    // Reused thread-local buffer: the old per-call construction allocated histLength
    // vectors (each reserve(500)) on every search_by_projection invocation.
    static thread_local vector<vector<int>> rotHist;
    rotHist.resize(histLength);
    for(auto& bin : rotHist)
        bin.clear();   // keeps each bin's capacity across calls
    rotFactor = 1.0f / static_cast<float>(histLength);
    return rotHist;
}

    void FeatureMatcher::updateRotationHistogram(vector<vector<int>>& rotHist,
                                                     const KeypointIndex& idx,
                                                     const cv::KeyPoint& keyPt, const cv::KeyPoint& refKeyPt,
                                                     const float& rotFactor, const int& histLength){
        float rot = keyPt.angle - refKeyPt.angle;
        if(rot < 0.0)
            rot += 360.0f;
        int bin = static_cast<int>(round(rot * rotFactor));
        if(bin == histLength)
            bin = 0;
        assert(bin >= 0 && bin < histLength);
        rotHist[bin].push_back(idx);
    }

    void FeatureMatcher::filterMatchesWithOrientation(vector<vector<int>>& rotHist, vector<Pt>& points, int& nMatches){
        int ind1{-1}, ind2{-1}, ind3{-1};
        computeThreeMaxima(rotHist,ind1,ind2,ind3);

        for(int i = 0; i < static_cast<int>(rotHist.size()); i++){
            if(i == ind1 || i == ind2 || i == ind3)
                continue;
            for(int j : rotHist[i]){
                points[j] = static_cast<Pt>(nullptr);
                nMatches--;
            }
        }
    }

    void FeatureMatcher::filterMatchesWithOrientation(vector<vector<int>>& rotHist, vector<int>& matches, int& nMatches){
        int ind1{-1}, ind2{-1}, ind3{-1};
        computeThreeMaxima(rotHist,ind1,ind2,ind3);

        for(int i = 0; i < static_cast<int>(rotHist.size()); i++){
            if(i == ind1 || i == ind2 || i == ind3)
                continue;
            for(int idx1 : rotHist[i]){
                if(matches[idx1] >= 0){
                    nMatches--;
                    matches[idx1] =-1;
                }
            }
        }
    }

    void FeatureMatcher::computeThreeMaxima(vector<vector<int>>& rotHist, int &ind1, int &ind2, int &ind3){
        int max1{0}, max2{0}, max3{0};
        for(int i = 0; i < static_cast<int>(rotHist.size()); i++)
        {
            const int s = static_cast<int>(rotHist[i].size());
            if(s > max1)
            {
                max3=max2;
                max2=max1;
                max1=s;
                ind3=ind2;
                ind2=ind1;
                ind1=i;
            }
            else if(s > max2)
            {
                max3=max2;
                max2=s;
                ind3=ind2;
                ind2=i;
            }
            else if(s > max3)
            {
                max3=s;
                ind3=i;
            }
        }

        if(max2 < 0.1f * static_cast<float>(max1))
        {
            ind2=-1;
            ind3=-1;
        }
        else if(max3 < 0.1f * static_cast<float>(max1))
        {
            ind3=-1;
        }
    }

    std::vector<cv::DMatch> FeatureMatcher::match_descriptors_only(
        const cv::Mat& desc1, const cv::Mat& desc2, const FeatureType& featType_){

        const AF_VSLAM::Feature& ft = get_feature(featType_);
        const MatcherType matcherType = ft.getMatcherType();

        std::vector<cv::DMatch> matches;
        switch(matcherType) {
            case LIGHTGLUE_SUPERPOINT:
            case LIGHTGLUE_ALIKED:
            case BF_L2:{
                matches = bruteforce_match_l2(desc1, desc2);
                break;
            }
            case LIGHTGLUE_SIFT:
            {
                sift_match_gpu_.SetDescriptors(0, desc1.rows, desc1.ptr<float>());
                sift_match_gpu_.SetDescriptors(1, desc2.rows, desc2.ptr<float>());
                constexpr int max_out = 4000;
                std::vector<std::array<uint32_t, 2>> match_buffer(max_out);
                int num_matches = sift_match_gpu_.GetSiftMatch(max_out, reinterpret_cast<uint32_t(*)[2]>(match_buffer.data()), 0.7f, 0.8f, 1);
                matches.clear();
                matches.reserve(num_matches);
                for (int i = 0; i < num_matches; ++i) {
                    matches.emplace_back(
                        static_cast<int>(match_buffer[i][0]),
                        static_cast<int>(match_buffer[i][1]),
                        0.0f);
                }
                break;
            }
            case BF_HAMMING:
            {
                matches = bruteforce_match_hamming(desc1, desc2);
                break;
            }
        }

        return matches;
    }

    std::vector<cv::DMatch> FeatureMatcher::match_descriptors(
        const cv::Mat& desc1, const cv::Mat& desc2,
        const std::vector<cv::KeyPoint>& kps1, const std::vector<cv::KeyPoint>& kps2,
        const FeatureType& featType_){

        const AF_VSLAM::Feature& ft = get_feature(featType_);
        const MatcherType matcherType = ft.getMatcherType();

        std::vector<cv::DMatch> matches;
        switch(matcherType) {
            case LIGHTGLUE_SIFT:
            case LIGHTGLUE_ALIKED:
            {
                matches = lightglueMatching(kps1, desc1, kps2, desc2, 0.0f);
                break;
            }
            case LIGHTGLUE_SUPERPOINT:
            {
                matches = matcherLightglueSuperpoint(kps1, desc1, kps2, desc2, 0.0f);
                break;
            }
            case BF_L2:
            {
                matches = bruteforce_match_l2(desc1, desc2);
                break;
            }
            case BF_HAMMING:
            {
                matches = bruteforce_match_hamming(desc1, desc2);
                break;
            }
        }

        return matches;
    }

    std::vector<cv::DMatch> FeatureMatcher::filter_matches_by_fundamental(std::vector<cv::DMatch>& matches,
        const std::vector<cv::KeyPoint>& kps1, const std::vector<cv::KeyPoint>& kps2,
        const int maxForRansac){

        std::sort(matches.begin(), matches.end(),
            [](const cv::DMatch& a, const cv::DMatch& b) { return a.distance < b.distance; });

        if (matches.size() > static_cast<size_t>(maxForRansac)) matches.resize(maxForRansac);

        // Build point correspondences ---
        std::vector<poselib::Point2D> pts1; pts1.reserve(matches.size());
        std::vector<poselib::Point2D> pts2; pts2.reserve(matches.size());
        for (const auto& m : matches) {
            // Safety: ensure indices are valid
            if (m.queryIdx < 0 || m.queryIdx >= static_cast<int>(kps1.size())) continue;
            if (m.trainIdx < 0 || m.trainIdx >= static_cast<int>(kps2.size())) continue;

            const auto& p1 = kps1[m.queryIdx].pt;
            const auto& p2 = kps2[m.trainIdx].pt;
            pts1.emplace_back(p1.x, p1.y);
            pts2.emplace_back(p2.x, p2.y);
        }

        if (pts1.size() < 8) return matches; // not enough after filtering

        constexpr double inlierThreshold = 3.0;  // pixels (Sampson / transfer error)
        constexpr size_t minInliersForModel = 8;

        // Robust filtering via PoseLib LO-RANSAC. Iterations are hard-bounded: OpenCV's
        // USAC_MAGSAC previously used here both threw an internal assertion on
        // H-degenerate+outlier inputs (crash) and ground for seconds per call on real
        // zero-baseline frames (the stop-section hiccups) — PoseLib does neither.
        //   1. Fundamental matrix (works at zero baseline too: F=[t]x fits identical points).
        //   2. If F finds no usable consensus, homography (the right model when the
        //      scene is planar/rotation-only).
        //   3. If both fail, pass matches through unfiltered; pose optimization's chi2
        //      outlier test downstream keeps that soft.
        std::vector<char> inlierMask;
        poselib::RelativePoseOptions fOpt;
        fOpt.max_error = inlierThreshold;
        fOpt.ransac.max_iterations = 1000;
        fOpt.ransac.min_iterations = 100;

        Eigen::Matrix3d F;
        poselib::RansacStats stats = poselib::estimate_fundamental(pts1, pts2, fOpt, &F, &inlierMask);

        if (stats.num_inliers < minInliersForModel) {
            poselib::HomographyOptions hOpt;
            hOpt.max_error = inlierThreshold;
            hOpt.ransac.max_iterations = 1000;
            hOpt.ransac.min_iterations = 100;

            Eigen::Matrix3d H;
            inlierMask.clear();
            stats = poselib::estimate_homography(pts1, pts2, hOpt, &H, &inlierMask);
            if (stats.num_inliers < minInliersForModel) {
                AF_WARN("filter_matches_by_fundamental: no F/H consensus ("
                    + std::to_string(stats.num_inliers) + " inliers), passing "
                    + std::to_string(matches.size()) + " matches unfiltered");
                return matches; // unfiltered fallback
            }
            AF_WARN("filter_matches_by_fundamental: F found no consensus, homography kept "
                + std::to_string(stats.num_inliers) + "/" + std::to_string(pts1.size()) + " matches");
        }

        if (inlierMask.size() != pts1.size())
            return matches;

        std::vector<cv::DMatch> inlierMatches;
        inlierMatches.reserve(matches.size());

        // Rebuild matchesUsed aligned with pts1/pts2:
        std::vector<cv::DMatch> matchesUsed;
        matchesUsed.reserve(matches.size());

        for (const auto& m : matches) {
            if (m.queryIdx < 0 || m.queryIdx >= static_cast<int>(kps1.size())) continue;
            if (m.trainIdx < 0 || m.trainIdx >= static_cast<int>(kps2.size())) continue;
            matchesUsed.push_back(m);
        }

        if (matchesUsed.size() != inlierMask.size())
            return matches; // alignment mismatch fallback

        for (size_t i = 0; i < inlierMask.size(); ++i) {
            if (inlierMask[i]) inlierMatches.push_back(matchesUsed[i]);
        }

        return inlierMatches;
    }

    std::map<FeatureType, std::vector<cv::DMatch>> FeatureMatcher::match_descriptors_parallel(
        const std::vector<FeatureType>& featureTypes,
        const std::map<FeatureType, cv::Mat> &desc1_, const std::map<FeatureType, cv::Mat> &desc2_,
        const std::map<FeatureType, std::vector<cv::KeyPoint>> &kps1_, const std::map<FeatureType, std::vector<cv::KeyPoint>> &kps2_
    ){

        struct MatchOut {
            FeatureType ft{};
            bool valid = false;
            std::vector<cv::DMatch> matches;
            size_t frameN = 0;
        };

        std::vector<FeatureType> jobs;
        jobs.reserve(featureTypes.size());
        for (const auto ft : featureTypes) {
            auto it1 = desc1_.find(ft);
            auto it2 = desc2_.find(ft);
            if (it1 == desc1_.end() || it2 == desc2_.end())
                continue;
            jobs.push_back(ft);
        }

            std::vector<MatchOut> outs(jobs.size());

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(jobs.size()); ++i) {
            const auto ft = jobs[i];
            auto& o = outs[i];
            o.ft = ft;
            o.valid = true;

            const auto& desc1 = desc1_.at(ft);
            const auto& desc2 = desc2_.at(ft);
            const auto& kps1  = kps1_.at(ft);
            const auto& kps2  = kps2_.at(ft);

            //o.frameN = (size_t)frame.N.at(ft);
            o.matches = match_descriptors(desc1, desc2, kps1, kps2, ft);
        }

        std::map<FeatureType, std::vector<cv::DMatch>> matches_by_type;
        for (auto& o : outs) {
            if (!o.valid) continue;
            matches_by_type[o.ft] = std::move(o.matches);
        }
        return matches_by_type;
    }

    std::vector<cv::DMatch> FeatureMatcher::serialFeatureMatching(
        const cv::Mat& desc1_, const cv::Mat& desc2_,
        const std::vector<cv::KeyPoint>& kps1_, const std::vector<cv::KeyPoint>& kps2_,
        const FeatureType& ft
    ){

        std::vector<cv::DMatch> matches = match_descriptors(desc1_, desc2_, kps1_, kps2_, ft);
        return matches;
    }

    } //namespace ORB_SLAM
