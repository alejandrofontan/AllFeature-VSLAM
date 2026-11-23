
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "afvslam_log.hpp"
#include "BruteForceMatcher.h"
#include "FeatureFactory.h"
#include "FeatureMatcher.h"
#include "onnx/FeatureModels.h"
#include "Report.h"

#include <opencv2/core/utility.hpp>
#include <opencv2/video/tracking.hpp>

#include <PoseLib/misc/essential.h>
#include <PoseLib/robust.h>

using namespace std;

namespace AF_VSLAM
{

namespace
{
// The size of the image a view's keypoints were detected in: its calibration's (as resized and cropped)
cv::Size image_size(const Calibration& calibration)
{
    return {calibration.width, calibration.height};
}

// Wall time since `start`, in ms: the report's `match.time_ms` and `filter.time_ms` entries
[[maybe_unused]] double elapsed_ms(const std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

// The LightGlue variant of a learned matcher type (nullptr for the brute-force ones)
const char* lightglue_variant(const MatcherType matcher_type)
{
    switch (matcher_type)
    {
        case LIGHTGLUE_SUPERPOINT: return "superpoint";
        case LIGHTGLUE_ALIKED: return "aliked";
        case LIGHTGLUE_RACO: return "raco";
        case LIGHTGLUE_SIFT: return "sift";
        case BF_L2:
        case BF_HAMMING: return nullptr;
    }
    return nullptr;   // not reached: a new MatcherType without a case is a compiler warning
}
}

// Loads (once per process, onnx::shared_lightglue) and warms up the LightGlue variant of every learned feature.
FeatureMatcher::FeatureMatcher(const std::vector<FeatureType>& feature_types, const std::string& name)
{
    AF_INFO("Initializing FeatureMatcher: " + name);

    // The diagnostics below are the same for every matcher (the sessions are process-wide): logged once
    static std::mutex logged_mutex;
    static bool opencv_logged = false;
    static std::set<std::string> variants_logged;
    std::lock_guard<std::mutex> lock(logged_mutex);

    if (!opencv_logged) {
        // Which parallel backend this OpenCV build runs: descriptor-matching cost depends on it
        opencv_logged = true;
        const std::string info = cv::getBuildInformation();
        const size_t p = info.find("Parallel framework");
        AF_INFO("OpenCV threading: getNumThreads=" << cv::getNumThreads() << " | "
                << (p == std::string::npos ? std::string("Parallel framework: (not reported)")
                                           : info.substr(p, info.find('\n', p) - p)));
    }
    for(const auto& feat_type : feature_types) {
        if (const char* variant = lightglue_variant(get_feature(feat_type).get_matcher_type())) {
            // Every session now (dynamic + each padded export): a missing file fails here, and no session is created
            // on the tracking thread later (seconds on Core ML even from its cache)
            const std::vector<int> lengths = onnx::preload_lightglue(variant);
            if (!variants_logged.insert(variant).second)
                continue;
            const auto& lg = onnx::shared_lightglue(variant);
            std::ostringstream padded;
            for (const int length : lengths)
                padded << " L" << length;
            AF_INFO("LightGlue (" << variant << ") ready: " << lg.model_path() << " (" << lg.precision()
                    << " weights) on onnxruntime/" << lg.provider()
                    << (lengths.empty() ? std::string() : " + padded exports" + padded.str()));
        }
    }
}

// ==================== Image to Image ====================

std::vector<cv::DMatch> FeatureMatcher::match_descriptors(
    const cv::Mat& desc1, const cv::Mat& desc2,
    const std::vector<cv::KeyPoint>& kps1, const std::vector<cv::KeyPoint>& kps2,
    const FeatureType& feat_type, const cv::Size image_size1, const cv::Size image_size2)
{
    // A side with no keypoints: nothing to match
    if(desc1.empty() || desc2.empty() || kps1.empty() || kps2.empty())
        return {};

    const MatcherType matcher_type = get_feature(feat_type).get_matcher_type();
    switch(matcher_type) {
        case LIGHTGLUE_SIFT:
        case LIGHTGLUE_ALIKED:
        case LIGHTGLUE_RACO:
        case LIGHTGLUE_SUPERPOINT:
        {
            try
            {
                return onnx::shared_lightglue(lightglue_variant(matcher_type),
                                              static_cast<int>(std::max(kps1.size(), kps2.size())))   // picks the export that fits
                           .match(kps1, desc1, kps2, desc2, image_size1, image_size2);
            }
            // A failed run is fatal: it means the GPU, the model or the inputs are wrong, and a run that
            // carried on without these matches would give results that look valid and are not. Caught
            // here because this also runs on worker threads and inside OpenMP regions, where an
            // escaping exception ends the process without saying why.
            catch(const std::exception& e)
            {
                AF_FATAL("LightGlue match failed (" << get_feature(feat_type).get_feature_name() << ", "
                         << kps1.size() << " x " << kps2.size() << " keypoints): " << e.what());
            }
        }
        case BF_L2:
            return bruteforce_match_l2(desc1, desc2);
        case BF_HAMMING:
            return bruteforce_match_hamming(desc1, desc2);
    }
    return {};   // not reached
}

std::vector<cv::DMatch> FeatureMatcher::filter_matches_geometric(const std::vector<cv::DMatch>& matches,
    const std::vector<cv::KeyPoint>& kps1, const std::vector<cv::KeyPoint>& kps2,
    const std::vector<mat2f>& sigma2_1, const std::vector<mat2f>& sigma2_2,
    const Calibration& calibration1, const Calibration& calibration2)
{
    if (matches.size() < filter_min_inliers) {
        AF_REPORT("filter.model", "", 0);   // too few matches to fit a model
        return matches;
    }

    // 1. Matched positions (undistorted pixels) and one pinhole camera per view
    std::vector<poselib::Point2D> points1, points2;   // in match order, so PoseLib's mask indexes `matches`
    points1.reserve(matches.size());
    points2.reserve(matches.size());
    for (const cv::DMatch& match : matches) {
        const cv::Point2f& p1 = kps1[match.queryIdx].pt;
        const cv::Point2f& p2 = kps2[match.trainIdx].pt;
        points1.emplace_back(p1.x, p1.y);
        points2.emplace_back(p2.x, p2.y);
    }
    const auto pinhole = [](const Calibration& calibration) {
        return poselib::Camera("PINHOLE", {calibration.fx, calibration.fy, calibration.cx, calibration.cy},
                               calibration.width, calibration.height);
    };
    const poselib::Camera camera1 = pinhole(calibration1);
    const poselib::Camera camera2 = pinhole(calibration2);

    // 2. Essential matrix: LO-RANSAC (5-point), then refined on its inliers
    poselib::RelativePoseOptions options;
    options.max_error = filter_inlier_threshold_px;                            // Sampson error, px
    options.ransac.min_iterations = filter_min_iterations;
    options.ransac.max_iterations = filter_max_iterations;
    options.bundle.loss_type = poselib::BundleOptions::LossType::TRUNCATED;    // final refinement
    options.bundle.loss_scale = filter_inlier_threshold_px;

    std::vector<char> ransac_mask;
    poselib::CameraPose pose;
    const poselib::RansacStats stats =
        poselib::estimate_relative_pose(points1, points2, camera1, camera2, options, &pose, &ransac_mask);
    AF_REPORT("filter.inliers_ransac", "", stats.num_inliers);
    if (stats.num_inliers < filter_min_inliers) {
        AF_WARN("filter_matches_geometric: no essential-matrix consensus ("
            + std::to_string(stats.num_inliers) + " inliers), passing "
            + std::to_string(matches.size()) + " matches unfiltered");
        AF_REPORT("filter.model", "", 0);
        return matches;
    }

    // 3. Every match re-tested against F (pixels) by its two keypoints' covariances
    Eigen::Matrix3d essential;
    poselib::essential_from_motion(pose, &essential);
    const auto inverse_K = [](const Calibration& calibration) {
        Eigen::Matrix3d K;
        K << calibration.fx, 0.0, calibration.cx,
             0.0, calibration.fy, calibration.cy,
             0.0, 0.0, 1.0;
        return Eigen::Matrix3d(K.inverse());
    };
    const Eigen::Matrix3d fundamental = inverse_K(calibration2).transpose() * essential * inverse_K(calibration1);   // K2^-T E K1^-1

    std::vector<cv::DMatch> inlier_matches;
    inlier_matches.reserve(stats.num_inliers);
    for (size_t i = 0; i < matches.size(); ++i) {
        const Eigen::Vector3d x1(points1[i](0), points1[i](1), 1.0);
        const Eigen::Vector3d x2(points2[i](0), points2[i](1), 1.0);
        const Eigen::Vector3d line2 = fundamental * x1;               // d(residual)/d(x2, y2)
        const Eigen::Vector3d line1 = fundamental.transpose() * x2;   // d(residual)/d(x1, y1)
        const double residual = x2.dot(line2);                        // x2^T F x1
        const mat2f& s1 = sigma2_1[matches[i].queryIdx];
        const mat2f& s2 = sigma2_2[matches[i].trainIdx];
        const Eigen::Vector2d j1 = line1.head<2>();
        const Eigen::Vector2d j2 = line2.head<2>();
        const double variance = j1.dot(s1.cast<double>() * j1) + j2.dot(s2.cast<double>() * j2);   // J1 S1 J1^T + J2 S2 J2^T
        if (variance > 0.0 && residual * residual < ChiSquared::dof1_95 * variance)     // 1 DoF: one epipolar equation
            inlier_matches.push_back(matches[i]);
    }
    // 4. Too few survivors: unfiltered
    AF_REPORT("filter.inliers", "", inlier_matches.size());
    if (inlier_matches.size() < filter_min_inliers) {
        AF_WARN("filter_matches_geometric: " + std::to_string(inlier_matches.size())
            + " matches within their keypoints' uncertainty, passing "
            + std::to_string(matches.size()) + " matches unfiltered");
        AF_REPORT("filter.model", "", 0);
        return matches;
    }
    AF_REPORT("filter.model", "", 1);
    return inlier_matches;
}

std::map<FeatureType, std::vector<cv::DMatch>> FeatureMatcher::filter_matches_geometric_jointly(
    const std::map<FeatureType, std::vector<cv::DMatch>>& matches_by_type,
    const FeatureData& features1, const FeatureData& features2,
    const Calibration& calibration1, const Calibration& calibration2)
{
    struct MatchOrigin
    {
        FeatureType feat_type;
        int idx1;   // per-type keypoint index in features1
        int idx2;   // per-type keypoint index in features2
    };
    // 1. Pool every type's matches; origins[slot] maps each one back
    std::vector<cv::DMatch> all_matches;
    std::vector<MatchOrigin> origins;
    std::vector<cv::KeyPoint> kps1, kps2;
    std::vector<mat2f> sigma2_1, sigma2_2;

    for (const auto& [ft, matches] : matches_by_type) {
        const KeypointData& data1 = features1.at(ft);
        const KeypointData& data2 = features2.at(ft);
        for (const cv::DMatch& match : matches) {
            const int slot = static_cast<int>(origins.size());   // the match's index in the pool
            origins.push_back(MatchOrigin{ft, match.queryIdx, match.trainIdx});
            kps1.push_back(data1.keypoints[match.queryIdx]);
            kps2.push_back(data2.keypoints[match.trainIdx]);
            sigma2_1.push_back(data1.sigma2[match.queryIdx]);
            sigma2_2.push_back(data2.sigma2[match.trainIdx]);
            all_matches.emplace_back(slot, slot, match.distance);
        }
    }

    // 2. One filter call, then each survivor back to its type and per-type indices
    [[maybe_unused]] const auto filter_start = std::chrono::steady_clock::now();
    const std::vector<cv::DMatch> kept =
        filter_matches_geometric(all_matches, kps1, kps2, sigma2_1, sigma2_2, calibration1, calibration2);
    AF_REPORT("filter.time_ms", "", elapsed_ms(filter_start));

    std::map<FeatureType, std::vector<cv::DMatch>> filtered_by_type;
    for (const cv::DMatch& match : kept) {
        const MatchOrigin& origin = origins[match.queryIdx];
        filtered_by_type[origin.feat_type].emplace_back(origin.idx1, origin.idx2, match.distance);
    }
    return filtered_by_type;
}

// Used in: Tracking::track_reference_keyframe, Tracking::relocalize
std::map<FeatureType, int> FeatureMatcher::match_keyframe_to_frame(const Keyframe& keyframe, Frame& frame,
    std::map<FeatureType, std::vector<Pt>>& map_pts_matches, const std::vector<FeatureType>& feat_types)
{
    std::map<FeatureType, std::vector<cv::DMatch>> matches_by_type;
    std::map<FeatureType, std::vector<Pt>> map_pts_kf;   // keyframe's map points per type (snapshot)

    map_pts_matches.clear();

    AF_REPORT_SCOPE();
    AF_REPORT("keyframe.frame_id", "", keyframe->frame_id);

    // 1. Descriptor matching, per feature type
    for (const auto& ft : feat_types) {
        if (!keyframe->features.has(ft) || !frame.features.has(ft))
            continue;

        map_pts_kf[ft] = keyframe->get_map_point_matches(ft);

        const KeypointData& keyframe_data = keyframe->features.at(ft);
        const KeypointData& frame_data = frame.features.at(ft);
        [[maybe_unused]] const auto match_start = std::chrono::steady_clock::now();
        std::vector<cv::DMatch> matches = match_descriptors(
            keyframe_data.descriptors, frame_data.descriptors,
            keyframe_data.keypoints, frame_data.keypoints, ft,
            image_size(*keyframe->calibration), image_size(*frame.calibration));
        AF_REPORT("match.time_ms", get_feature(ft).get_feature_name(), elapsed_ms(match_start));
        AF_REPORT("matches.raw", get_feature(ft).get_feature_name(), matches.size());

        if (!matches.empty())
            matches_by_type[ft] = std::move(matches);

        map_pts_matches[ft] = std::vector<Pt>(frame_data.count(), nullptr);
    }

    // 2. Geometric filter: one model for all feature types
    const std::map<FeatureType, std::vector<cv::DMatch>> robust_by_type =
        filter_matches_geometric_jointly(matches_by_type, keyframe->features, frame.features,
                                         *keyframe->calibration, *frame.calibration);

    // 3. Lift: a filtered pair gives the frame keypoint the keyframe's map point
    std::map<FeatureType, std::vector<cv::DMatch>> frame_side;   // filtered pairs, frame index first (the cache)

    std::map<FeatureType, int> matches_count;
    for (const auto& ft : feat_types)
        matches_count[ft] = 0;
    for (const auto& [ft, robust_matches] : robust_by_type) {
        std::vector<cv::DMatch>& frame_side_ft = frame_side[ft];
        frame_side_ft.reserve(robust_matches.size());
        const std::vector<Pt>& keyframe_pts = map_pts_kf.at(ft);
        std::vector<Pt>& frame_pts = map_pts_matches.at(ft);
        for (const cv::DMatch& match : robust_matches) {
            const int keyframe_idx = match.queryIdx;
            const int frame_idx = match.trainIdx;

            frame_side_ft.emplace_back(frame_idx, keyframe_idx, match.distance);

            const Pt& pt = keyframe_pts[keyframe_idx];
            if (!pt || pt->is_bad())
                continue;
            frame_pts[frame_idx] = pt;
            matches_count[ft]++;
        }
    }
    for ([[maybe_unused]] const auto& ft : feat_types) {
        AF_REPORT("matches.filtered", get_feature(ft).get_feature_name(),
                  frame_side.count(ft) ? frame_side.at(ft).size() : 0);
        AF_REPORT("matches.with_point", get_feature(ft).get_feature_name(), matches_count.at(ft));
    }

    // 4. Cache every filtered pair on the frame, keyed by the keyframe
    frame.cache_matched_pairs_feat_type.insert_or_assign(keyframe->frame_id, std::move(frame_side));

    return matches_count;
}

void FeatureMatcher::match_keyframe_to_keyframes(const Keyframe& keyframe, const std::vector<Keyframe>& others,
    const std::vector<FeatureType>& feat_types)
{
    const int num_others = static_cast<int>(others.size());
    const int num_feature_types = static_cast<int>(feat_types.size());

    // 1. Pending pairs: no cache entry yet, or an empty one
    std::vector<char> pending(num_others, 0);   // per other keyframe
    bool any_pending = false;
    {
        std::lock_guard<std::mutex> lock(keyframe->cache_mutex_);   // the cache is shared between threads
        for (int k = 0; k < num_others; k++) {
            if (others[k] == keyframe)
                continue;
            const auto it = keyframe->cache_matched_pairs_feat_type.find(others[k]->frame_id);
            pending[k] = it == keyframe->cache_matched_pairs_feat_type.end() || it->second.empty();
            any_pending = any_pending || pending[k];
        }
    }
    if (!any_pending)
        return;

    AF_REPORT_SCOPE();

    // 2. Descriptor matching: every pending pair x feature type in parallel
    std::vector<std::vector<std::vector<cv::DMatch>>> matches(   // [other][feature type]
        num_others, std::vector<std::vector<cv::DMatch>>(num_feature_types));
    [[maybe_unused]] std::vector<std::vector<double>> match_ms(   // [other][feature type], for the report
        num_others, std::vector<double>(num_feature_types, 0.0));

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int k = 0; k < num_others; k++) {
        for (int f = 0; f < num_feature_types; f++) {
            if (!pending[k])
                continue;
            const FeatureType ft = feat_types[f];
            if (!keyframe->features.has(ft) || !others[k]->features.has(ft))
                continue;
            const KeypointData& data1 = keyframe->features.at(ft);
            const KeypointData& data2 = others[k]->features.at(ft);
            const auto match_start = std::chrono::steady_clock::now();
            matches[k][f] = match_descriptors(data1.descriptors, data2.descriptors,
                                              data1.keypoints, data2.keypoints, ft,
                                              image_size(*keyframe->calibration), image_size(*others[k]->calibration));
            match_ms[k][f] = elapsed_ms(match_start);   // includes the wait for a shared LightGlue session
        }
    }

    // 3. Geometric filter and store, pair by pair (serial: keeps the report on this thread)
    [[maybe_unused]] std::vector<size_t> num_raw(num_feature_types, 0), num_filtered(num_feature_types, 0);
    for (int k = 0; k < num_others; k++) {
        if (!pending[k])
            continue;
        const Keyframe& other = others[k];

        std::map<FeatureType, std::vector<cv::DMatch>> raw_by_type;
        for (int f = 0; f < num_feature_types; f++) {
            num_raw[f] += matches[k][f].size();
            if (!matches[k][f].empty())
                raw_by_type[feat_types[f]] = std::move(matches[k][f]);
        }
        std::map<FeatureType, std::vector<cv::DMatch>> forward_by_type =
            filter_matches_geometric_jointly(raw_by_type, keyframe->features, other->features,
                                             *keyframe->calibration, *other->calibration);

        for (int f = 0; f < num_feature_types; f++) {
            const auto it = forward_by_type.find(feat_types[f]);
            if (it != forward_by_type.end())
                num_filtered[f] += it->second.size();
        }

        std::lock_guard<std::mutex> lock(keyframe->cache_mutex_);
        keyframe->cache_matched_pairs_feat_type.insert_or_assign(other->frame_id, std::move(forward_by_type));   // this keyframe only
    }

    // 4. Report: totals over the pairs matched in this call
    for (int f = 0; f < num_feature_types; f++) {
        [[maybe_unused]] double total_match_ms = 0.0;
        for (int k = 0; k < num_others; k++)
            total_match_ms += match_ms[k][f];
        AF_REPORT("match.time_ms", get_feature(feat_types[f]).get_feature_name(), total_match_ms);
        AF_REPORT("matches.raw", get_feature(feat_types[f]).get_feature_name(), num_raw[f]);
        AF_REPORT("matches.filtered", get_feature(feat_types[f]).get_feature_name(), num_filtered[f]);
    }
}

std::map<FeatureType, std::vector<std::pair<size_t, size_t>>> FeatureMatcher::get_keyframe_matches(const Keyframe& keyframe1, const Keyframe& keyframe2)
{
    std::map<FeatureType, std::vector<cv::DMatch>> cached_by_type;
    {
        std::lock_guard<std::mutex> lock(keyframe1->cache_mutex_);
        const auto cache_it = keyframe1->cache_matched_pairs_feat_type.find(keyframe2->frame_id);
        if (cache_it != keyframe1->cache_matched_pairs_feat_type.end())
            cached_by_type = cache_it->second;
    }

    std::map<FeatureType, std::vector<std::pair<size_t, size_t>>> matched_pairs;
    for (const auto& [ft, matches] : cached_by_type)
        for (const cv::DMatch& match : matches)
            matched_pairs[ft].emplace_back(size_t(match.queryIdx), size_t(match.trainIdx));
    return matched_pairs;
}

void FeatureMatcher::clear_keyframe_matches(const Keyframe& keyframe)
{
    std::map<FrameId, std::map<FeatureType, std::vector<cv::DMatch>>> released;
    {
        std::lock_guard<std::mutex> lock(keyframe->cache_mutex_);
        released.swap(keyframe->cache_matched_pairs_feat_type);
    }
}

std::map<FeatureType, std::vector<std::pair<size_t, size_t>>> FeatureMatcher::match_frames_for_initialization(const Frame& frame1, const Frame& frame2,
    const std::vector<FeatureType>& feat_types)
{
    AF_REPORT_SCOPE();

    // 1. Descriptor matching, per feature type (serial: keeps the tracking thread's run priority)
    std::map<FeatureType, std::vector<cv::DMatch>> matches_by_type;
    for (const auto& ft : feat_types) {
        if (!frame1.features.has(ft) || !frame2.features.has(ft))
            continue;
        const KeypointData& data1 = frame1.features.at(ft);
        const KeypointData& data2 = frame2.features.at(ft);
        [[maybe_unused]] const auto match_start = std::chrono::steady_clock::now();
        matches_by_type[ft] = match_descriptors(data1.descriptors, data2.descriptors,
                                                data1.keypoints, data2.keypoints, ft,
                                                image_size(*frame1.calibration), image_size(*frame2.calibration));
        AF_REPORT("match.time_ms", get_feature(ft).get_feature_name(), elapsed_ms(match_start));
        AF_REPORT("matches.raw", get_feature(ft).get_feature_name(), matches_by_type.at(ft).size());
    }

    // 2. Geometric filter: one model for all feature types
    const std::map<FeatureType, std::vector<cv::DMatch>> filtered_by_type =
        filter_matches_geometric_jointly(matches_by_type, frame1.features, frame2.features,
                                         *frame1.calibration, *frame2.calibration);

    // 3. Survivors as (index in frame1, index in frame2) pairs
    std::map<FeatureType, std::vector<std::pair<size_t, size_t>>> matched_pairs;
    for (const auto& ft : feat_types)
        matched_pairs[ft];   // an entry for every requested type: the caller reads with .at(ft)
    for (const auto& [ft, filtered] : filtered_by_type)
        for (const cv::DMatch& match : filtered)
            matched_pairs[ft].emplace_back(size_t(match.queryIdx), size_t(match.trainIdx));

    for ([[maybe_unused]] const auto& ft : feat_types)
        AF_REPORT("matches.filtered", get_feature(ft).get_feature_name(), matched_pairs.at(ft).size());

    return matched_pairs;
}

// ==================== Map to Image ====================

Descriptor_Distance_Type FeatureMatcher::descriptor_distance(const cv::Mat &a, const cv::Mat &b, const FeatureType& featureType_)
{
    const AF_VSLAM::Feature& ft = get_feature(featureType_);
    return ft.descriptor_distance(a,b);
}

namespace
{
// What the pose-guided search found for one feature type; the callers report it.
struct GateMatchCounts
{
    size_t candidates{0};          // candidate map points of this type
    size_t free_keypoints{0};      // keypoints with no map point yet
    size_t with_keypoint{0};       // candidates with at least one free keypoint in their gate
    size_t accepted{0};            // pairs where each side is the other's best descriptor match
};

template <typename KeypointsInArea, typename IsFree, typename Assign>
std::map<FeatureType, GateMatchCounts> match_points_in_gate(
    const AF_VSLAM::FeatureData& features,
    const std::vector<AF_VSLAM::Pt>& map_pts, const std::vector<cv::Point2f>& projections,   // projections[i] is map_pts[i]'s
    const float gate,                                   // max offset^T * information * offset (squared sigmas)
    const Descriptor_Distance_Type max_distance,        // the type's max(): no limit
    const KeypointsInArea& keypoints_in_area,           // (x, y, radius, type) -> keypoint indices near (x, y)
    const IsFree& is_free,                              // (type, keypoint) -> may it take a point
    const Assign& assign)                               // (type, keypoint, point): apply an accepted pair
{
    using AF_VSLAM::FeatureMatcher;
    using AF_VSLAM::Pt;
    constexpr Descriptor_Distance_Type no_distance = std::numeric_limits<Descriptor_Distance_Type>::max();

    // 1. Candidates by feature type: one descriptor read per point (it takes the point's lock)
    struct Candidates
    {
        std::vector<size_t> indices;          // into map_pts
        std::vector<cv::Mat> descriptors;
    };
    std::map<FeatureType, Candidates> candidates_by_type;
    for (size_t idx = 0; idx < map_pts.size(); idx++) {
        const Pt& pt = map_pts[idx];
        if (!pt || pt->is_bad())
            continue;

        cv::Mat descriptor = pt->get_descriptor();
        if (descriptor.empty())
            continue;

        Candidates& candidates = candidates_by_type[pt->featureType];
        candidates.indices.push_back(idx);
        candidates.descriptors.push_back(std::move(descriptor));
    }

    std::map<FeatureType, GateMatchCounts> counts_by_type;
    for (const auto& [ft, data] : features.per_type) {
        GateMatchCounts& counts = counts_by_type[ft];
        const size_t num_keypoints = data.count();

        // 2. Free keypoints, and the largest variance among them (it sets how far to search)
        std::vector<char> free_keypoint(num_keypoints, 0);
        float max_variance = 0.0f;   // bound on any free keypoint's variance along any direction
        for (size_t i = 0; i < num_keypoints; i++) {
            if (!is_free(ft, i))
                continue;
            free_keypoint[i] = 1;
            counts.free_keypoints++;
            const mat2f& sigma2 = data.sigma2[i];
            max_variance = std::max(max_variance,
                                    std::max(sigma2(0, 0), sigma2(1, 1)) + std::abs(sigma2(0, 1)));
        }

        const auto candidates_it = candidates_by_type.find(ft);
        if (candidates_it == candidates_by_type.end())
            continue;
        const Candidates& candidates = candidates_it->second;
        const size_t num_candidates = candidates.indices.size();
        counts.candidates = num_candidates;
        if (counts.free_keypoints == 0)
            continue;

        const std::vector<cv::KeyPoint>& keypoints = data.keypoints;
        const cv::Mat& descriptors = data.descriptors;

        // 3. Gate, then each side's best descriptor match among the pairs it allows
        const float search_radius = std::sqrt(gate * max_variance);   // covers the widest keypoint's gate
        std::vector<int> best_keypoint(num_candidates, -1);
        std::vector<Descriptor_Distance_Type> best_keypoint_distance(num_candidates, no_distance);
        std::vector<int> best_candidate(num_keypoints, -1);
        std::vector<Descriptor_Distance_Type> best_candidate_distance(num_keypoints, no_distance);

        for (size_t c = 0; c < num_candidates; c++) {
            const cv::Point2f& projection = projections[candidates.indices[c]];
            for (const size_t k : keypoints_in_area(projection.x, projection.y, search_radius, ft)) {
                if (!free_keypoint[k])
                    continue;

                const vec2f offset(keypoints[k].pt.x - projection.x, keypoints[k].pt.y - projection.y);
                if (offset.dot(data.information[k] * offset) >= gate)   // outside this keypoint's own gate
                    continue;

                const Descriptor_Distance_Type distance = FeatureMatcher::descriptor_distance(
                    candidates.descriptors[c], descriptors.row(static_cast<int>(k)), ft);
                if (distance < best_keypoint_distance[c]) {   // strict '<': the first of equal distances wins
                    best_keypoint_distance[c] = distance;
                    best_keypoint[c] = static_cast<int>(k);
                }
                if (distance < best_candidate_distance[k]) {
                    best_candidate_distance[k] = distance;
                    best_candidate[k] = static_cast<int>(c);
                }
            }
        }

        // 4. Accept the pairs where each side chose the other
        for (size_t c = 0; c < num_candidates; c++) {
            const int k = best_keypoint[c];
            if (k < 0)
                continue;
            counts.with_keypoint++;
            if (best_candidate[k] != static_cast<int>(c))   // not mutual
                continue;
            if (best_keypoint_distance[c] > max_distance)
                continue;
            assign(ft, static_cast<size_t>(k), map_pts[candidates.indices[c]]);
            counts.accepted++;
        }
    }
    return counts_by_type;
}
}

void FeatureMatcher::match_map_points_to_frame(Frame& frame, const std::vector<Pt>& map_pts,
                                               const std::vector<cv::Point2f>& projections)
{
    AF_REPORT_SCOPE();

    [[maybe_unused]] const std::map<FeatureType, GateMatchCounts> counts_by_type = match_points_in_gate(
        frame.features, map_pts, projections, projection_match_scale_sigma2,
        std::numeric_limits<Descriptor_Distance_Type>::max(),
        [&frame](const float x, const float y, const float radius, const FeatureType ft) {
            return frame.get_features_in_area(x, y, radius, ft);
        },
        [&frame](const FeatureType ft, const size_t keypoint_idx) {
            const Pt& pt = frame.pts.at(ft)[keypoint_idx];
            return !(pt && pt->number_of_observations() > 0);
        },
        [&frame](const FeatureType ft, const size_t keypoint_idx, const Pt& pt) {
            frame.pts.at(ft)[keypoint_idx] = pt;
        });

    for ([[maybe_unused]] const auto& [ft, counts] : counts_by_type) {
        AF_REPORT("points.candidates", get_feature(ft).get_feature_name(), counts.candidates);
        AF_REPORT("keypoints.unmatched", get_feature(ft).get_feature_name(), counts.free_keypoints);
        AF_REPORT("matches.raw", get_feature(ft).get_feature_name(), counts.with_keypoint);
        AF_REPORT("matches.accepted", get_feature(ft).get_feature_name(), counts.accepted);
    }
}

void FeatureMatcher::match_map_points_to_keyframe(const Keyframe& keyframe, const std::vector<Pt>& map_pts,
                                                  const std::vector<cv::Point2f>& projections,
                                                  const FeatureType feat_type, std::vector<Pt>& matched_points,
                                                  const float gate_scale_sigma2,
                                                  const Descriptor_Distance_Type max_descriptor_distance)
{
    if (keyframe->is_bad())   // a culled keyframe takes no new points
        return;

    AF_REPORT_SCOPE();

    [[maybe_unused]] const std::map<FeatureType, GateMatchCounts> counts_by_type = match_points_in_gate(
        keyframe->features, map_pts, projections, gate_scale_sigma2,
        max_descriptor_distance,
        [&keyframe](const float x, const float y, const float radius, const FeatureType ft) {
            return keyframe->get_features_in_area(x, y, radius, ft);
        },
        [&matched_points, feat_type](const FeatureType ft, const size_t keypoint_idx) {
            return ft == feat_type && keypoint_idx < matched_points.size() && !matched_points[keypoint_idx];
        },
        [&matched_points](const FeatureType, const size_t keypoint_idx, const Pt& pt) {
            matched_points[keypoint_idx] = pt;
        });

    [[maybe_unused]] const auto counts_it = counts_by_type.find(feat_type);
    AF_REPORT("points.candidates", get_feature(feat_type).get_feature_name(),
              counts_it == counts_by_type.end() ? 0 : counts_it->second.candidates);
    AF_REPORT("keypoints.unmatched", get_feature(feat_type).get_feature_name(),
              counts_it == counts_by_type.end() ? 0 : counts_it->second.free_keypoints);
    AF_REPORT("matches.raw", get_feature(feat_type).get_feature_name(),
              counts_it == counts_by_type.end() ? 0 : counts_it->second.with_keypoint);
    AF_REPORT("matches.accepted", get_feature(feat_type).get_feature_name(),
              counts_it == counts_by_type.end() ? 0 : counts_it->second.accepted);
}

// ==================== Optical Flow ====================

FeatureMatcher::FlowMatchCounts FeatureMatcher::match_frame_to_frame_by_flow(
    const Frame& last_frame, const cv::Mat& last_grey, Frame& frame, const cv::Mat& grey,
    const int window, const int pyramid_levels, const float max_backward_error, const float association_radius)
{
    AF_REPORT_SCOPE();

    FlowMatchCounts counts{};
    std::map<FeatureType, FlowMatchCounts> counts_by_feature;   // the report's per-feature split of counts
    for (const auto& [ft, pts] : last_frame.pts)
        counts_by_feature[ft] = {};
    const auto report = [&]
    {
        for ([[maybe_unused]] const auto& [ft, c] : counts_by_feature)
        {
            [[maybe_unused]] const std::string& feature = get_feature(ft).get_feature_name();
            AF_REPORT("flow.tracked", feature, c.tracked);
            AF_REPORT("flow.survived", feature, c.survived);
            AF_REPORT("flow.matched", feature, c.matched);
        }
        return counts;
    };

    // 1. The last frame's tracked points: raw (distorted) positions, the flow runs on the image as it is
    struct Track { FeatureType ft; int index; Pt map_point; };
    std::vector<Track> tracks;
    std::vector<cv::Point2f> prev_pts;
    for (const auto& [ft, pts] : last_frame.pts)
    {
        const auto& raw = last_frame.raw_keypoints.at(ft);
        const auto& outliers = last_frame.outliers.at(ft);
        for (size_t i = 0; i < pts.size(); i++)
        {
            if(!pts[i] || pts[i]->is_bad() || outliers[i])
                continue;
            tracks.push_back({ft, static_cast<int>(i), pts[i]});
            prev_pts.push_back(raw[i].pt);
            counts_by_feature[ft].tracked++;
        }
    }
    counts.tracked = static_cast<int>(prev_pts.size());
    if(prev_pts.empty())
        return report();

    // 2. KLT forward and backward; a track survives when both succeed and it returns to where it started
    std::vector<cv::Point2f> next_pts, back_pts;
    std::vector<uchar> status_fwd, status_bwd;
    std::vector<float> err;
    const cv::Size window_size(window, window);
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, flow_max_iterations, flow_epsilon);
    cv::calcOpticalFlowPyrLK(last_grey, grey, prev_pts, next_pts, status_fwd, err, window_size, pyramid_levels, criteria);
    cv::calcOpticalFlowPyrLK(grey, last_grey, next_pts, back_pts, status_bwd, err, window_size, pyramid_levels, criteria);
    std::vector<int> kept;
    std::vector<cv::Point2f> kept_pts;
    const float max_err2 = max_backward_error * max_backward_error;
    for (size_t t = 0; t < tracks.size(); t++)
    {
        if(!status_fwd[t] || !status_bwd[t])
            continue;
        const cv::Point2f d = back_pts[t] - prev_pts[t];
        if(d.dot(d) > max_err2)
            continue;
        const cv::Point2f& p = next_pts[t];
        if(p.x < 0.0f || p.y < 0.0f || p.x >= static_cast<float>(grey.cols) || p.y >= static_cast<float>(grey.rows))
            continue;
        kept.push_back(static_cast<int>(t));
        kept_pts.push_back(p);
        counts_by_feature[tracks[t].ft].survived++;
    }
    counts.survived = static_cast<int>(kept.size());
    if(kept.empty())
        return report();

    // 3. The frame's keypoints live in undistorted coordinates: undistort the flowed positions to look them up
    const std::vector<cv::Point2f> lookup_pts = frame.calibration->undistort(kept_pts);

    // 4. Snap each track to the nearest detected keypoint of its feature type; a keypoint claimed twice keeps
    //    the closer track
    std::map<FeatureType, std::vector<float>> best_dist2;
    for (const auto& [ft, data] : frame.features.per_type)
        best_dist2[ft].assign(data.count(), std::numeric_limits<float>::max());
    const float radius = association_radius;
    for (size_t k = 0; k < kept.size(); k++)
    {
        const Track& track = tracks[kept[k]];
        const cv::Point2f& u = lookup_pts[k];
        const std::vector<size_t> candidates = frame.get_features_in_area(u.x, u.y, radius, track.ft);
        int best = -1;
        float best_d2 = radius * radius;
        for (const size_t idx : candidates)
        {
            const cv::Point2f d = frame.features.at(track.ft).keypoints[idx].pt - u;
            const float d2 = d.dot(d);
            if(d2 < best_d2)
            {
                best_d2 = d2;
                best = static_cast<int>(idx);
            }
        }
        if(best < 0 || best_d2 >= best_dist2[track.ft][best])
            continue;
        auto& pts = frame.pts.at(track.ft);
        if(!pts[best])
        {
            ++counts.matched;                               // a re-claim replaces, the count stays
            ++counts_by_feature[track.ft].matched;
        }
        pts[best] = track.map_point;
        best_dist2[track.ft][best] = best_d2;
    }

    return report();
}

}
