
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

#include "afvslam_log.hpp"
#include "BruteForceMatcher.h"
#include "FeatureFactory.h"
#include "FeatureMatcher.h"
#include "onnx/FeatureModels.h"
#include "Report.h"

#include <opencv2/core/utility.hpp>

#include <PoseLib/robust.h>

using namespace std;

namespace AF_VSLAM
{

std::map<FeatureType, Descriptor_Distance_Type> FeatureMatcher::TH_LOW = {};

VerbosityLevel FeatureMatcher::verbosity{MEDIUM};

float FeatureMatcher::radiusScale{1.15f};

namespace
{
/// The LightGlue variant of a learned matcher type (nullptr for the brute-force ones).
const char* lightglue_variant(const MatcherType matcher_type)
{
    switch (matcher_type)
    {
        case LIGHTGLUE_SUPERPOINT: return "superpoint";
        case LIGHTGLUE_ALIKED: return "aliked";
        case LIGHTGLUE_SIFT: return "sift";
        default: return nullptr;
    }
}
} // namespace

// Loads (once per process, onnx::shared_lightglue) and warms up the LightGlue variant of every learned feature.
FeatureMatcher::FeatureMatcher(const std::vector<FeatureType>& feature_types,
    std::string name):
    name(name),
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
        if (const char* variant = lightglue_variant(get_feature(feat_type).get_matcher_type())) {
            // Every session now (dynamic + each padded export): a missing file fails here, and no session is created
            // on the tracking thread later (seconds on Core ML even from its cache)
            const std::vector<int> lengths = onnx::preload_lightglue(variant);
            const auto& lg = onnx::shared_lightglue(variant);
            std::ostringstream padded;
            for (const int length : lengths)
                padded << " L" << length;
            AF_INFO("LightGlue (" << variant << ") ready for " << name << ": " << lg.model_path() << " (" << lg.precision()
                    << " weights) on onnxruntime/" << lg.provider()
                    << (lengths.empty() ? std::string() : " + padded exports" + padded.str()));
        }
    }
}

// Used in: Tracking::track_reference_keyframe, Tracking::relocalize
std::map<FeatureType, int> FeatureMatcher::match_keyframe_to_frame(const Keyframe& keyframe, Frame& frame,
    std::map<FeatureType, std::vector<Pt>>& map_pts_matches, const std::vector<FeatureType>& feat_types)
{
    std::map<FeatureType, std::vector<cv::DMatch>> matches_by_type;
    std::map<FeatureType, std::vector<Pt>> map_pts_kf;

    map_pts_matches.clear();

    AF_REPORT_SCOPE();
    AF_REPORT("keyframe.frame_id", "", keyframe->frame_id);

    for (const auto& ft : feat_types) {
        if (!keyframe->features.has(ft) || !frame.features.has(ft))
            continue;

        map_pts_kf[ft] = keyframe->get_map_point_matches(ft);

        const KeypointData& keyframe_data = keyframe->features.at(ft);
        const KeypointData& frame_data = frame.features.at(ft);
        std::vector<cv::DMatch> matches = match_descriptors(
            keyframe_data.descriptors, frame_data.descriptors,
            keyframe_data.keypoints, frame_data.keypoints, ft,
            frame.calibration->width, frame.calibration->height);
        AF_REPORT("matches.raw", get_feature_name(ft), matches.size());

        if (!matches.empty())
            matches_by_type[ft] = std::move(matches);

        map_pts_matches[ft] = std::vector<Pt>(frame_data.count(), nullptr);
    }

    // One model for all feature types: they all see the same two views
    const std::map<FeatureType, std::vector<cv::DMatch>> robust_by_type =
        filter_matches_geometric_jointly(matches_by_type, keyframe->features, frame.features);

    std::map<FeatureType, std::vector<cv::DMatch>> frame_side;

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
        AF_REPORT("matches.filtered", get_feature_name(ft),
                  frame_side.count(ft) ? frame_side.at(ft).size() : 0);
        AF_REPORT("matches.with_point", get_feature_name(ft), matches_count.at(ft));
    }

    frame.cache_matched_pairs_feat_type.insert_or_assign(keyframe->frame_id, std::move(frame_side));

    return matches_count;
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

// The pose-guided search shared by match_map_points_to_frame and
// match_map_points_to_keyframe: gives a map point to the free keypoints of `features`, from
// candidate map points and where each projects (`projections[i]` belongs to `map_pts[i]`).
//
// Per feature type, a candidate and a free keypoint can be paired only if the keypoint lies
// inside the gate around the candidate's projection: the squared offset, in units of the
// keypoint's own position variance, below projection_match_scale_sigma2. Among the pairs the
// gate allows, a pair is accepted when each side is the other's best descriptor match: the
// candidate's nearest keypoint, and that keypoint's nearest candidate. So the search is local
// -- a correct keypoint next to the projection is never lost to a better-looking one elsewhere
// in the image -- and it needs no descriptor-distance threshold: mutual agreement is what
// rejects a candidate whose window holds nothing right.
//
// The three callables adapt it to where the keypoints live and where the result goes:
//   keypoints_in_area(x, y, radius, feature_type) -> indices of the keypoints near (x, y)
//   is_free(feature_type, keypoint_index)         -> whether that keypoint may take a point
//   assign(feature_type, keypoint_index, point)   -> give it one
template <typename KeypointsInArea, typename IsFree, typename Assign>
std::map<FeatureType, GateMatchCounts> match_points_in_gate(
    const AF_VSLAM::FeatureData& features,
    const std::vector<AF_VSLAM::Pt>& map_pts, const std::vector<cv::Point2f>& projections,
    const KeypointsInArea& keypoints_in_area, const IsFree& is_free, const Assign& assign)
{
    using AF_VSLAM::FeatureMatcher;
    using AF_VSLAM::Pt;
    constexpr Descriptor_Distance_Type no_distance = std::numeric_limits<Descriptor_Distance_Type>::max();
    constexpr float gate = FeatureMatcher::projection_match_scale_sigma2;

    // The candidates grouped by feature type: their index in map_pts and their descriptor. One
    // get_descriptor() per point -- it takes the point's lock -- and a point whose descriptor is
    // still empty is skipped.
    struct Candidates
    {
        std::vector<size_t> indices;
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

        // The free keypoints, and a bound on their position variance: the largest gate any of
        // them can have decides how far to look.
        std::vector<char> free_keypoint(num_keypoints, 0);
        float max_variance = 0.0f;
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

        // Each side's best partner among the pairs the gate allows. Strict '<' keeps the first
        // of equal distances, so the result does not depend on anything but the input order.
        const float search_radius = std::sqrt(gate * max_variance);
        std::vector<int> best_keypoint(num_candidates, -1);
        std::vector<Descriptor_Distance_Type> best_keypoint_distance(num_candidates, no_distance);
        std::vector<int> best_candidate(num_keypoints, -1);
        std::vector<Descriptor_Distance_Type> best_candidate_distance(num_keypoints, no_distance);

        for (size_t c = 0; c < num_candidates; c++) {
            const cv::Point2f& projection = projections[candidates.indices[c]];
            for (const size_t k : keypoints_in_area(projection.x, projection.y, search_radius, ft)) {
                if (!free_keypoint[k])
                    continue;

                // The gate proper: the offset in units of this keypoint's own uncertainty
                const vec2f offset(keypoints[k].pt.x - projection.x, keypoints[k].pt.y - projection.y);
                if (offset.dot(data.information[k] * offset) >= gate)
                    continue;

                const Descriptor_Distance_Type distance = FeatureMatcher::descriptor_distance(
                    candidates.descriptors[c], descriptors.row(static_cast<int>(k)), ft);
                if (distance < best_keypoint_distance[c]) {
                    best_keypoint_distance[c] = distance;
                    best_keypoint[c] = static_cast<int>(k);
                }
                if (distance < best_candidate_distance[k]) {
                    best_candidate_distance[k] = distance;
                    best_candidate[k] = static_cast<int>(c);
                }
            }
        }

        // Accept the pairs where each side chose the other
        for (size_t c = 0; c < num_candidates; c++) {
            const int k = best_keypoint[c];
            if (k < 0)
                continue;
            counts.with_keypoint++;
            if (best_candidate[k] != static_cast<int>(c))
                continue;
            assign(ft, static_cast<size_t>(k), map_pts[candidates.indices[c]]);
            counts.accepted++;
        }
    }
    return counts_by_type;
}
} // namespace

// Gives a map point to the frame keypoints that have none yet, from candidate map points and
// where each projects in the frame (`projections[i]` belongs to `map_pts[i]`): the
// pose-guided search of match_points_in_gate above, on the frame's keypoints. A keypoint is
// free when it has no map point, or one that nothing observes. Accepted matches are written to
// frame.pts.
// Reports, per feature type: points.candidates, keypoints.unmatched, matches.raw (candidates
// with at least one free keypoint in their gate) and matches.accepted.
// Used in: Tracking::search_local_points, Tracking::accept_relocalization_hypothesis
void FeatureMatcher::match_map_points_to_frame(Frame& frame, const std::vector<Pt>& map_pts,
                                               const std::vector<cv::Point2f>& projections)
{
    AF_REPORT_SCOPE();

    [[maybe_unused]] const std::map<FeatureType, GateMatchCounts> counts_by_type = match_points_in_gate(
        frame.features, map_pts, projections,
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
        AF_REPORT("points.candidates", get_feature_name(ft), counts.candidates);
        AF_REPORT("keypoints.unmatched", get_feature_name(ft), counts.free_keypoints);
        AF_REPORT("matches.raw", get_feature_name(ft), counts.with_keypoint);
        AF_REPORT("matches.accepted", get_feature_name(ft), counts.accepted);
    }
}

// The same search on a keyframe's keypoints of one feature type, writing into a vector the
// caller owns: `matched_points` has one slot per keypoint of `feat_type`, a keypoint is free
// when its slot is empty, and an accepted match fills it. The keyframe itself is not modified.
// Reports points.candidates, keypoints.unmatched, matches.raw and matches.accepted.
// Used in: LoopClosing::search_loop_map_points
void FeatureMatcher::match_map_points_to_keyframe(const Keyframe& keyframe, const std::vector<Pt>& map_pts,
                                                  const std::vector<cv::Point2f>& projections,
                                                  const FeatureType feat_type, std::vector<Pt>& matched_points)
{
    AF_REPORT_SCOPE();

    [[maybe_unused]] const std::map<FeatureType, GateMatchCounts> counts_by_type = match_points_in_gate(
        keyframe->features, map_pts, projections,
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
    AF_REPORT("points.candidates", get_feature_name(feat_type),
              counts_it == counts_by_type.end() ? 0 : counts_it->second.candidates);
    AF_REPORT("keypoints.unmatched", get_feature_name(feat_type),
              counts_it == counts_by_type.end() ? 0 : counts_it->second.free_keypoints);
    AF_REPORT("matches.raw", get_feature_name(feat_type),
              counts_it == counts_by_type.end() ? 0 : counts_it->second.with_keypoint);
    AF_REPORT("matches.accepted", get_feature_name(feat_type),
              counts_it == counts_by_type.end() ? 0 : counts_it->second.accepted);
}

// Makes sure the matches between `keyframe` and each of `others` are in `keyframe`'s match
// cache. Pairs already cached are skipped. For the rest: the descriptors of every feature type
// are matched, all (other, feature type) combinations in parallel; then, per pair of keyframes,
// the geometric filter runs on all its feature types jointly; the survivors are stored in
// `keyframe`'s cache, keyed by the other keyframe, per feature type and in per-type keypoint
// indices. Nothing is stored in the other keyframe: every reader looks the pair up from
// `keyframe`'s side.
//
// The cache is a hand-off, not a store: the caller reads the pairs back and then releases them
// with clear_keyframe_matches.
//
// This is the only writer of the cache for keyframe pairs. Its other writer is tracking
// (match_keyframe_to_frame, for a frame against its reference keyframe), which filters the same
// way, so a cached entry is always a geometrically filtered one.
// Used in: LocalMapping::create_new_map_points, LoopClosing::compute_sim3 and
// LoopClosing::search_loop_map_points. The pairs are read back with get_keyframe_matches.
void FeatureMatcher::match_keyframe_to_keyframes(const Keyframe& keyframe, const std::vector<Keyframe>& others,
    const std::vector<FeatureType>& feat_types)
{
    const int num_others = static_cast<int>(others.size());
    const int num_feature_types = static_cast<int>(feat_types.size());

    // The pairs still to match: no cache entry yet, or an empty one. The caches are shared
    // between the three threads (KeyFrame::cache_mutex_).
    std::vector<char> pending(num_others, 0);
    bool any_pending = false;
    {
        std::lock_guard<std::mutex> lock(keyframe->cache_mutex_);
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

    std::vector<std::vector<std::vector<cv::DMatch>>> matches(
        num_others, std::vector<std::vector<cv::DMatch>>(num_feature_types));

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
            matches[k][f] = match_descriptors(data1.descriptors, data2.descriptors,
                                              data1.keypoints, data2.keypoints, ft,
                                              keyframe->calibration->width, keyframe->calibration->height);
        }
    }

    // Filter and store, serially: the filter is a few milliseconds per pair, and this keeps the
    // report entries on the calling thread.
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
            filter_matches_geometric_jointly(raw_by_type, keyframe->features, other->features);

        for (int f = 0; f < num_feature_types; f++) {
            const auto it = forward_by_type.find(feat_types[f]);
            if (it != forward_by_type.end())
                num_filtered[f] += it->second.size();
        }

        std::lock_guard<std::mutex> lock(keyframe->cache_mutex_);
        keyframe->cache_matched_pairs_feat_type.insert_or_assign(other->frame_id, std::move(forward_by_type));
    }

    // Totals over the pairs matched in this call, per feature type.
    for (int f = 0; f < num_feature_types; f++) {
        AF_REPORT("matches.raw", get_feature_name(feat_types[f]), num_raw[f]);
        AF_REPORT("matches.filtered", get_feature_name(feat_types[f]), num_filtered[f]);
    }
}

// Releases every match list `keyframe` holds. Called by the thread that filled them, once it has
// read them back: Local Mapping after triangulating, Loop Closing when it is done with the
// keyframe. The cache is otherwise never pruned, and a keyframe's matches against its
// neighbors are a few hundred kilobytes. This includes the entry tracking left for the
// reference keyframe, which Local Mapping has used by then.
// Reports what was held: cache.entries (keyframes matched against) and cache.matches.
void FeatureMatcher::clear_keyframe_matches(const Keyframe& keyframe)
{
    std::map<FrameId, std::map<FeatureType, std::vector<cv::DMatch>>> released;
    {
        std::lock_guard<std::mutex> lock(keyframe->cache_mutex_);
        released.swap(keyframe->cache_matched_pairs_feat_type);
    }

#ifdef ALLFEATURE_REPORT
    size_t num_matches = 0;
    for (const auto& [frame_id, matches_by_type] : released)
        for (const auto& [ft, matches] : matches_by_type)
            num_matches += matches.size();
    AF_REPORT("cache.entries", "", released.size());
    AF_REPORT("cache.matches", "", num_matches);
#endif
}

// The matched keypoint pairs (index in keyframe1, index in keyframe2) that keyframe1's match
// cache holds for keyframe2, per feature type. A read only: it never matches. A pair of
// keyframes that was not cached first (match_keyframe_to_keyframes) gives no pairs.
// Used in: LocalMapping::create_new_map_points, LoopClosing::compute_sim3
std::map<FeatureType, std::vector<std::pair<size_t, size_t>>> FeatureMatcher::get_keyframe_matches(const Keyframe& keyframe1, const Keyframe& keyframe2)
{
    // A copy: the cache is shared between threads (KeyFrame::cache_mutex_)
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


// The matched keypoint pairs (index in frame1, index in frame2) of two frames, per feature
// type, for monocular initialization: the descriptors of each feature type are matched, and the
// geometric filter runs on all types jointly. The result has an entry for every requested
// feature type, empty when none of its matches survive -- the caller reads them with .at(ft).
// Reports matches.raw and matches.filtered per feature type.
// Used in: Tracking::attempt_monocular_initialization
std::map<FeatureType, std::vector<std::pair<size_t, size_t>>> FeatureMatcher::match_frames_for_initialization(const Frame& frame1, const Frame& frame2,
    const std::vector<FeatureType>& feat_types)
{
    AF_REPORT_SCOPE();

    // One feature type after the other, on the calling thread: with one learned feature the
    // LightGlue match dominates and there is nothing to gain from a parallel loop, and on this
    // thread the tracking thread's run priority applies to it (OrtSession, Threading).
    std::map<FeatureType, std::vector<cv::DMatch>> matches_by_type;
    for (const auto& ft : feat_types) {
        if (!frame1.features.has(ft) || !frame2.features.has(ft))
            continue;
        const KeypointData& data1 = frame1.features.at(ft);
        const KeypointData& data2 = frame2.features.at(ft);
        matches_by_type[ft] = match_descriptors(data1.descriptors, data2.descriptors,
                                                data1.keypoints, data2.keypoints, ft,
                                                frame1.calibration->width, frame1.calibration->height);
        AF_REPORT("matches.raw", get_feature_name(ft), matches_by_type.at(ft).size());
    }

    // One model for all feature types: they all see the same two views
    const std::map<FeatureType, std::vector<cv::DMatch>> filtered_by_type =
        filter_matches_geometric_jointly(matches_by_type, frame1.features, frame2.features);

    std::map<FeatureType, std::vector<std::pair<size_t, size_t>>> matched_pairs;
    for (const auto& ft : feat_types)
        matched_pairs[ft];
    for (const auto& [ft, filtered] : filtered_by_type)
        for (const cv::DMatch& match : filtered)
            matched_pairs[ft].emplace_back(size_t(match.queryIdx), size_t(match.trainIdx));

    for ([[maybe_unused]] const auto& ft : feat_types)
        AF_REPORT("matches.filtered", get_feature_name(ft), matched_pairs.at(ft).size());

    return matched_pairs;
}

// Projects each map point into the keyframe and attempts to fuse it with an existing keypoint.
// For each map point, applies visibility checks (depth, image bounds, scale, viewing angle),
// then finds the closest keypoint by descriptor distance within radius_th.
// If a match is found: replaces the weaker map point if one already exists, or adds a new observation.
// Returns the number of fused map points.
// Used in: LocalMapping::SearchInNeighbors
int FeatureMatcher::fuse_map_points_to_keyframe(const Keyframe& keyframe, const vector<Pt>& map_pts, const float& radius_th, const FeatureType& feat_type)
{
    // One lock: successive get_Rcw()/get_tcw() each take and release pose_mutex_,
    // so a set_pose landing between them hands back R and t from DIFFERENT poses.
    const Pose pose = keyframe->get_pose();

    const Descriptor_Distance_Type th_low = TH_LOW[feat_type];

    int n_fused = 0;
    const auto& kps_ft  = keyframe->features.at(feat_type).keypoints;
    const auto& desc_ft = keyframe->features.at(feat_type).descriptors;

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

        // Depth, image bounds, scale-invariance range and viewing angle
        const std::optional<Projection> projection =
            project_if_visible(pose, *keyframe->calibration, pt, cos_viewing_angle_th);
        if (!projection)
            continue;

        const float u = projection->uv.x;
        const float v = projection->uv.y;

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

            // e2 > chi2 * sigma2, the same shape as the other two chi-squared gates
            // (LocalMapping.cc:310/318, Sim3Solver.cc:90). The old form multiplied by a stored
            // 1/sigma2, rounded once at extraction and again here.
            if (e2 > chi2_perc * keyframe->features.get_sigma2(KeypointIndex(idx), feat_type))
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
            // Publish first, observe only if we won the slot: get_map_point above released the
            // lock, so another thread can fill best_idx in between. pt is an existing point
            // with a descriptor, so publishing before observing is safe here (unlike a freshly
            // created one -- see KeyFrame::make_map_point).
            if (keyframe->add_map_point(pt, best_idx))
                pt->add_observation(keyframe, best_idx);
        }
        n_fused++;
    }

    return n_fused;
}

// Sim3 overload: projects map points into a keyframe using a Sim3 transform and fuses them.
// Used in: LoopClosing
int FeatureMatcher::fuse_map_points_to_keyframe(Keyframe& keyframe, const mat4f& Scw,
    const vector<Pt> &map_pts, const float& radius_th, vector<Pt> &replace_pts, const FeatureType& feat_type)
{

    // Scw is a Sim3; the visibility tests want the SE3 inside it, scale divided out.
    const mat3f sRcw = Scw.block<3,3>(0,0);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    mat4f Tcw = mat4f::Identity();
    Tcw.block<3,3>(0,0) = sRcw / scw;
    Tcw.block<3,1>(0,3) = Scw.block<3,1>(0,3);

    // A non-finite Sim3 used to reach the loop and fail every per-point test (NaN compares
    // false), fusing nothing; this says so once instead.
    Pose pose;
    if(!pose.set(Tcw))
        return 0;

    // Set of MapPoints already found in the KeyFrame
    const std::vector<Pt> spAlreadyFound = keyframe->get_good_map_points(feat_type);

    int nFused=0;

    const size_t nPoints = map_pts.size();

    // For each candidate MapPoint project and match
    for(size_t iMP=0; iMP<nPoints; iMP++)
    {
        const Pt pMP = map_pts[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->is_bad() || contains_map_point(spAlreadyFound, pMP))
            continue;

        // Depth, image bounds, scale-invariance range and viewing angle
        const std::optional<Projection> projection =
            project_if_visible(pose, *keyframe->calibration, pMP, cos_viewing_angle_th);
        if(!projection)
            continue;

        const float u = projection->uv.x;
        const float v = projection->uv.y;
        const float dist3D = projection->dist;

        // Search in a radius
        const float predictedSize = pMP->predict_size(dist3D);
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
            const cv::Mat &descriptor = keyframe->features.at(feat_type).descriptors.row(idx);
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
                // Publish first, observe only on success -- see the note in the fuse above
                if(keyframe->add_map_point(pMP, bestIdx))
                    pMP->add_observation(keyframe, bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
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

    AF_CONFIG_BEGIN("feature_settings_yaml_file");
        AF_CONFIG_FIELD("FeatureMatcher.TH_LOW", FeatureMatcher::TH_LOW[featureType]);
    AF_CONFIG_END();
}

// The matches between two descriptor sets of one feature type, WITHOUT keypoints: for a side
// that has descriptors but no image positions, such as map points. LightGlue needs keypoints,
// so here every float descriptor, learned or not, is matched by L2 brute force, and every
// binary one by Hamming brute force. Brute force is mutual nearest neighbour, with no distance
// threshold. queryIdx indexes desc1, trainIdx desc2; distance is the L2 or Hamming distance.
// Used in: match_map_points_to_frame
std::vector<cv::DMatch> FeatureMatcher::match_descriptors_only(
    const cv::Mat& desc1, const cv::Mat& desc2, const FeatureType& feat_type)
{
    switch(get_feature(feat_type).get_matcher_type()) {
        case LIGHTGLUE_SUPERPOINT:
        case LIGHTGLUE_ALIKED:
        case LIGHTGLUE_SIFT:
        case BF_L2:
            return bruteforce_match_l2(desc1, desc2);
        case BF_HAMMING:
            return bruteforce_match_hamming(desc1, desc2);
    }
    return {};   // not reached: every MatcherType is handled above
}

// The matches between the keypoints of two images, for one feature type, with the matcher that
// feature declares (Feature::get_matcher_type): LightGlue for the learned features, mutual
// nearest neighbour brute force (L2 or Hamming, no distance threshold) for the others.
// queryIdx indexes the first set, trainIdx the second. What `distance` holds depends on the
// matcher: 1 - score for LightGlue (0..1), the L2 or Hamming distance for brute force. The two
// are not comparable, so nothing may rank matches of different feature types by it.
// `image_width` x `image_height` is the size of the image both keypoint sets were detected in.
// Used in: match_keyframe_to_frame, match_keyframe_to_keyframes, match_frames_for_initialization
std::vector<cv::DMatch> FeatureMatcher::match_descriptors(
    const cv::Mat& desc1, const cv::Mat& desc2,
    const std::vector<cv::KeyPoint>& kps1, const std::vector<cv::KeyPoint>& kps2,
    const FeatureType& feat_type, const int image_width, const int image_height)
{
    // A side without keypoints (a type a frame did not detect: its descriptors are an empty Mat)
    // has nothing to match, and its shape would fail the matchers' checks
    if(desc1.empty() || desc2.empty() || kps1.empty() || kps2.empty())
        return {};

    const MatcherType matcher_type = get_feature(feat_type).get_matcher_type();
    switch(matcher_type) {
        // LightGlue on ONNX Runtime: it normalises the keypoints with the size of the image they
        // were detected in, which the caller passes from the frame's calibration. That is the
        // image as resized AND cropped, not the size Tracking requested, which is larger when
        // the crop to multiples of 32 trims it (703x436 requested, 672x416 actual on ETH).
        case LIGHTGLUE_SIFT:
        case LIGHTGLUE_ALIKED:
        case LIGHTGLUE_SUPERPOINT:
        {
            // A failed Run (a transient Core ML / ONNX Runtime error) costs this match, not the
            // process: this runs on the tracking, mapping (inside OpenMP regions) and
            // loop-closing threads, where an escaping exception is std::terminate
            try
            {
                return onnx::shared_lightglue(lightglue_variant(matcher_type),
                                              static_cast<int>(std::max(kps1.size(), kps2.size())))
                           .match(kps1, desc1, kps2, desc2, image_width, image_height);
            }
            catch(const std::exception& e)
            {
                AF_WARN("LightGlue match failed, no matches for this pair: " << e.what());
                return {};
            }
        }
        case BF_L2:
            return bruteforce_match_l2(desc1, desc2);
        case BF_HAMMING:
            return bruteforce_match_hamming(desc1, desc2);
    }
    return {};   // not reached: every MatcherType is handled above
}

// Keeps the matches consistent with one two-view geometry, estimated with PoseLib LO-RANSAC:
//   1. a fundamental matrix (it works at zero baseline too: F = [t]x fits identical points);
//   2. if that finds no consensus, a homography, the right model for a planar or
//      rotation-only scene;
//   3. if both fail, or there are too few matches to fit either, the matches are returned
//      unfiltered -- pose optimization's chi2 outlier test downstream keeps that soft.
// The iterations are hard-bounded (filter_min_iterations .. filter_max_iterations).
//
// Every match takes part, and the result keeps the input order. There is no cap on the count:
// the cost is bounded by the iteration limit, and a cap by descriptor distance cannot be fair
// across feature types -- LightGlue stores 1 - score (0..1) and the brute-force matcher a
// Hamming or L2 distance (tens), so sorting them together ranks every learned match first, and
// the old cut to the best 2000 removed brute-force matches only.
//
// Which of the three happened is not returned; it is recorded as the report entry
// `filter.model`: 1 fundamental matrix, 2 homography, 0 unfiltered.
std::vector<cv::DMatch> FeatureMatcher::filter_matches_geometric(const std::vector<cv::DMatch>& matches,
    const std::vector<cv::KeyPoint>& kps1, const std::vector<cv::KeyPoint>& kps2)
{
    if (matches.size() < filter_min_inliers) {
        AF_REPORT("filter.model", "", 0);   // too few matches to fit a model
        return matches;
    }

    // Point correspondences, one per match and in the same order, so the inlier mask PoseLib
    // returns below indexes `matches` directly.
    std::vector<poselib::Point2D> points1, points2;
    points1.reserve(matches.size());
    points2.reserve(matches.size());
    for (const cv::DMatch& match : matches) {
        const cv::Point2f& p1 = kps1[match.queryIdx].pt;
        const cv::Point2f& p2 = kps2[match.trainIdx].pt;
        points1.emplace_back(p1.x, p1.y);
        points2.emplace_back(p2.x, p2.y);
    }

    // The same threshold and iteration bounds for both models.
    const auto configure = [](auto& options) {
        options.max_error = filter_inlier_threshold_px;
        options.ransac.min_iterations = filter_min_iterations;
        options.ransac.max_iterations = filter_max_iterations;
    };

    std::vector<char> inlier_mask;
    poselib::RelativePoseOptions fundamental_options;
    configure(fundamental_options);

    Eigen::Matrix3d fundamental;
    poselib::RansacStats stats =
        poselib::estimate_fundamental(points1, points2, fundamental_options, &fundamental, &inlier_mask);

    if (stats.num_inliers < filter_min_inliers) {
        poselib::HomographyOptions homography_options;
        configure(homography_options);

        Eigen::Matrix3d homography;
        inlier_mask.clear();
        stats = poselib::estimate_homography(points1, points2, homography_options, &homography, &inlier_mask);
        if (stats.num_inliers < filter_min_inliers) {
            AF_WARN("filter_matches_geometric: no F/H consensus ("
                + std::to_string(stats.num_inliers) + " inliers), passing "
                + std::to_string(matches.size()) + " matches unfiltered");
            AF_REPORT("filter.model", "", 0);
            return matches;
        }
        // Not a failure -- the homography is the right model for a planar or rotation-only
        // view -- so it is recorded, not warned about: on such a stretch it is every frame.
        AF_REPORT("filter.model", "", 2);
    }
    else
        AF_REPORT("filter.model", "", 1);

    // A model with at least filter_min_inliers inliers was found, and for such a model PoseLib
    // fills the mask with one entry per input point.
    std::vector<cv::DMatch> inlier_matches;
    inlier_matches.reserve(stats.num_inliers);
    for (size_t i = 0; i < matches.size(); ++i) {
        if (inlier_mask[i])
            inlier_matches.push_back(matches[i]);
    }

    return inlier_matches;
}

// filter_matches_geometric on the matches of several feature types at once. The filter addresses
// keypoints by index, so pooled match k points at slot k of two compact keypoint arrays holding
// only the matched keypoints, and origins[k] keeps the type and per-type indices it came from.
std::map<FeatureType, std::vector<cv::DMatch>> FeatureMatcher::filter_matches_geometric_jointly(
    const std::map<FeatureType, std::vector<cv::DMatch>>& matches_by_type,
    const FeatureData& features1, const FeatureData& features2)
{
    struct MatchOrigin
    {
        FeatureType feat_type;
        int idx1;   // per-type keypoint index in features1
        int idx2;   // per-type keypoint index in features2
    };
    std::vector<cv::DMatch> all_matches;
    std::vector<MatchOrigin> origins;
    std::vector<cv::KeyPoint> kps1, kps2;

    for (const auto& [ft, matches] : matches_by_type) {
        const std::vector<cv::KeyPoint>& keypoints1 = features1.at(ft).keypoints;
        const std::vector<cv::KeyPoint>& keypoints2 = features2.at(ft).keypoints;
        for (const cv::DMatch& match : matches) {
            const int slot = static_cast<int>(origins.size());
            origins.push_back(MatchOrigin{ft, match.queryIdx, match.trainIdx});
            kps1.push_back(keypoints1[match.queryIdx]);
            kps2.push_back(keypoints2[match.trainIdx]);
            all_matches.emplace_back(slot, slot, match.distance);
        }
    }

    std::map<FeatureType, std::vector<cv::DMatch>> filtered_by_type;
    for (const cv::DMatch& match : filter_matches_geometric(all_matches, kps1, kps2)) {
        const MatchOrigin& origin = origins[match.queryIdx];
        filtered_by_type[origin.feat_type].emplace_back(origin.idx1, origin.idx2, match.distance);
    }
    return filtered_by_type;
}

} // namespace AF_VSLAM
