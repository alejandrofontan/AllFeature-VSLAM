
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <unordered_set>
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
std::map<FeatureType, Descriptor_Distance_Type> FeatureMatcher::descDistTh_high_reloc = {};
std::map<FeatureType, Descriptor_Distance_Type> FeatureMatcher::descDistTh_low_reloc = {};

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
    struct MatchOrigin
    {
        FeatureType feat_type;
        int keyframe_idx;
        int frame_idx;
    };
    vector<cv::DMatch> all_matches;
    vector<MatchOrigin> origins;
    vector<cv::KeyPoint> keyframe_kps, frame_kps;
    map<FeatureType, vector<Pt>> map_pts_kf;

    map_pts_matches.clear();

    AF_REPORT_SCOPE();
    AF_REPORT("keyframe.frame_id", "", keyframe->frame_id);

    for (const auto& ft : feat_types) {
        auto it1 = keyframe->features.per_type.find(ft);
        auto it2 = frame.features.per_type.find(ft);

        if (it1 == keyframe->features.per_type.end() || it2 == frame.features.per_type.end())
            continue;

        map_pts_kf[ft] = keyframe->get_map_point_matches(ft);

        const auto& v1 = keyframe->features.at(ft).keypoints;
        const auto& v2 = frame.features.at(ft).keypoints;
        vector<cv::DMatch> matches = match_descriptors(
            it1->second.descriptors, it2->second.descriptors,
            v1, v2, ft);
        AF_REPORT("matches.raw", get_feature_name(ft), matches.size());

        for (cv::DMatch& m : matches) {
            const int slot = static_cast<int>(origins.size());
            origins.push_back(MatchOrigin{ft, m.queryIdx, m.trainIdx});
            keyframe_kps.push_back(v1[m.queryIdx]);
            frame_kps.push_back(v2[m.trainIdx]);

            m.queryIdx = slot;
            m.trainIdx = slot;
            all_matches.push_back(m);
        }

        map_pts_matches[ft] = vector<Pt>(frame.features.at(ft).count(), nullptr);
    }

    auto robust_matches = filter_matches_geometric(all_matches, keyframe_kps, frame_kps);

    std::map<FeatureType, std::vector<cv::DMatch>> frame_side;

    map<FeatureType, int> matches_count;
    for (const auto& ft : feat_types)
        matches_count[ft] = 0;
    for (const auto& m : robust_matches) {
        const MatchOrigin& origin = origins[m.queryIdx];
        const int query_idx = origin.keyframe_idx;
        const int train_idx = origin.frame_idx;
        const FeatureType feat_type = origin.feat_type;

        frame_side[feat_type].push_back(cv::DMatch(train_idx, query_idx, m.distance));

        const Pt pt = map_pts_kf[feat_type][query_idx];
        if (!pt || pt->is_bad())
            continue;
        map_pts_matches[feat_type][train_idx] = pt;
        matches_count[feat_type]++;
    }
    for ([[maybe_unused]] const auto& ft : feat_types) {
        AF_REPORT("matches.filtered", get_feature_name(ft),
                  frame_side.count(ft) ? frame_side.at(ft).size() : 0);
        AF_REPORT("matches.with_point", get_feature_name(ft), matches_count.at(ft));
    }

    frame.cache_matched_pairs_feat_type.insert_or_assign(keyframe->frame_id, std::move(frame_side));

    return matches_count;
}

// Gives a map point to the frame keypoints that have none yet, from the candidates the caller
// found visible. Per feature type: the unmatched keypoints are matched against the candidates by
// descriptor alone (cross-checked brute force, over the whole image), and a match is accepted if
// the squared offset between the keypoint and the candidate's projection is below
// projection_match_scale_sigma2 times the keypoint's own position variance.
//
// `projections[i]` is where `map_pts[i]` projects in this frame. Accepted matches are written to
// frame.pts. Reports, per feature type: points.candidates, keypoints.unmatched, matches.raw and
// matches.accepted.
// Used in: Tracking::search_local_points
void FeatureMatcher::match_map_points_to_frame(Frame& frame, const std::vector<Pt>& map_pts,
                                               const std::vector<cv::Point2f>& projections)
{
    AF_REPORT_SCOPE();

    // The candidates grouped by feature type: their index in map_pts and their descriptor. One
    // get_descriptor() per point -- it takes the point's lock -- and the cv::Mat header it
    // returns keeps that buffer alive until the rows are copied below. A point whose descriptor
    // is still empty is skipped: it would otherwise leave an uninitialised row in the matrix
    // (create does not zero, and copyTo from an empty Mat writes nothing), and that row would be
    // matched.
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

    for (const auto& [ft, data] : frame.features.per_type) {
        std::vector<Pt>& frame_pts = frame.pts.at(ft);

        // The keypoints with no observed map point yet.
        std::vector<size_t> keypoint_indices;
        keypoint_indices.reserve(data.count());
        for (size_t i = 0; i < data.count(); i++) {
            if (frame_pts[i] && frame_pts[i]->number_of_observations() > 0)
                continue;
            keypoint_indices.push_back(i);
        }

        const auto candidates_it = candidates_by_type.find(ft);
        AF_REPORT("points.candidates", get_feature_name(ft),
                  candidates_it == candidates_by_type.end() ? 0 : candidates_it->second.indices.size());
        AF_REPORT("keypoints.unmatched", get_feature_name(ft), keypoint_indices.size());

        if (keypoint_indices.empty() || candidates_it == candidates_by_type.end()) {
            AF_REPORT("matches.raw", get_feature_name(ft), 0);
            AF_REPORT("matches.accepted", get_feature_name(ft), 0);
            continue;
        }
        const Candidates& candidates = candidates_it->second;

        // Both sides as one descriptor matrix each, rows copied into a preallocated matrix:
        // cv::Mat::push_back reallocates and re-copies the whole matrix as it grows.
        const cv::Mat& first = candidates.descriptors.front();
        cv::Mat point_descriptors(static_cast<int>(candidates.descriptors.size()), first.cols, first.type());
        for (size_t k = 0; k < candidates.descriptors.size(); k++)
            candidates.descriptors[k].copyTo(point_descriptors.row(static_cast<int>(k)));

        const cv::Mat& frame_descriptors = data.descriptors;
        cv::Mat keypoint_descriptors(static_cast<int>(keypoint_indices.size()),
                                     frame_descriptors.cols, frame_descriptors.type());
        for (size_t k = 0; k < keypoint_indices.size(); k++)
            frame_descriptors.row(static_cast<int>(keypoint_indices[k]))
                .copyTo(keypoint_descriptors.row(static_cast<int>(k)));

        const std::vector<cv::DMatch> matches = match_descriptors_only(keypoint_descriptors, point_descriptors, ft);
        AF_REPORT("matches.raw", get_feature_name(ft), matches.size());

        // Accept a match only if the map point projects near the matched keypoint: the squared
        // Mahalanobis distance d^T * information * d must be below projection_match_scale_sigma2.
        // For an isotropic keypoint of variance s^2 that is |d|^2 < scale * s^2. A direct
        // distance test, not a get_features_in_area query: the keypoint is already known, so
        // there is nothing to search for. (That query also made grid membership an implicit
        // acceptance criterion -- a keypoint pos_in_grid rejected lost its match even when the
        // distance was fine.)
        const std::vector<cv::KeyPoint>& keypoints = data.keypoints;
        [[maybe_unused]] int num_accepted = 0;
        for (const cv::DMatch& match : matches) {
            const size_t keypoint_idx = keypoint_indices[match.queryIdx];
            const size_t point_idx = candidates.indices[match.trainIdx];

            const cv::Point2f& keypoint = keypoints[keypoint_idx].pt;
            const cv::Point2f& projection = projections[point_idx];
            const float dist_x = keypoint.x - projection.x;
            const float dist_y = keypoint.y - projection.y;
            const vec2f offset(dist_x, dist_y);
            if (offset.dot(data.information[keypoint_idx] * offset) < projection_match_scale_sigma2) {
                frame_pts[keypoint_idx] = map_pts[point_idx];
                num_accepted++;
            }
        }
        AF_REPORT("matches.accepted", get_feature_name(ft), num_accepted);
    }
}

// Matches two keyframes across all feature types to find candidate pairs for triangulation.
// Reuses cached matches from a prior tracking step if available; otherwise runs parallel
// descriptor matching followed by the geometric filter (filter_matches_geometric).
// Outputs only pairs where neither keyframe has an existing 3D map point for those keypoints.
// Used in: LocalMapping::CreateNewMapPoints
map<FeatureType, vector<pair<size_t,size_t>>> FeatureMatcher::match_keyframes(const Keyframe& keyframe1, const Keyframe& keyframe2,
                                const std::vector<FeatureType>& feat_types){

    if (keyframe1 == keyframe2)   // nothing to match (and the two-keyframe cache lock below needs two mutexes)
        return {};

    // Reuse cached matches if keyframe1 already has matches stored for keyframe2 (a copy: the cache is shared
    // between threads, KeyFrame::cache_mutex_)
    map<FeatureType, vector<cv::DMatch>> cached_by_type;
    {
        std::lock_guard<std::mutex> lock(keyframe1->cache_mutex_);
        const auto cache_it = keyframe1->cache_matched_pairs_feat_type.find(keyframe2->frame_id);
        if (cache_it != keyframe1->cache_matched_pairs_feat_type.end())
            cached_by_type = cache_it->second;
    }
    const bool cached = !cached_by_type.empty();

    // No cache — run parallel descriptor matching across all feature types
    map<FeatureType, vector<cv::DMatch>> matches_by_type;
    if (!cached) {
        matches_by_type = match_descriptors_parallel(feat_types, keyframe1->features, keyframe2->features);
    }

    map<FeatureType, vector<pair<size_t,size_t>>> matched_pairs;

    // The cache already stores per-type keypoint indices, so a hit needs no stacking and no decoding
    if (cached) {
        for (const auto& [ft, matches] : cached_by_type)
            for (const auto& m : matches)
                matched_pairs[ft].emplace_back(size_t(m.queryIdx), size_t(m.trainIdx));
        return matched_pairs;
    }

    // Stack all feature types into one index space (each type's indices offset by the keypoint counts of
    // the earlier ones) so the fundamental matrix is estimated from every feature jointly, and keep the
    // tables that decode a stacked index back to its feature type and per-type index.
    vector<cv::KeyPoint> kps1, kps2;
    vector<size_t> kps1_indexes, kps2_indexes;
    vector<cv::DMatch> all_matches;
    vector<FeatureType> used_feature_types;
    for (const auto& ft : feat_types) {

        // Ensure both frames contain the requested feature type
        auto it1 = keyframe1->features.per_type.find(ft);
        auto it2 = keyframe2->features.per_type.find(ft);
        if (it1 == keyframe1->features.per_type.end() || it2 == keyframe2->features.per_type.end())
            continue;

        // Accumulate keypoints and track per-feature-type original indices for later lookup
        auto const& v1 = keyframe1->features.at(ft).keypoints;
        auto const& v2 = keyframe2->features.at(ft).keypoints;

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

    // Filter outliers jointly across all feature types
    if (all_matches.size() < min_match_keyframes)
        return matched_pairs; // Not enough matches to reliably estimate fundamental matrix

    const vector<cv::DMatch> filtered_matches = filter_matches_geometric(all_matches, kps1, kps2);

    // Decode the stacked indices back to per-type ones, both for the caller and for the cache
    std::map<FeatureType, vector<cv::DMatch>> forward_by_type, backward_by_type;
    for (const auto& m : filtered_matches) {
        const size_t query_idx = kps1_indexes[m.queryIdx];
        const size_t train_idx = kps2_indexes[m.trainIdx];
        const FeatureType feat_type = used_feature_types[m.queryIdx];

        matched_pairs[feat_type].emplace_back(query_idx, train_idx);
        forward_by_type[feat_type].push_back(cv::DMatch(int(query_idx), int(train_idx), m.distance));
        backward_by_type[feat_type].push_back(cv::DMatch(int(train_idx), int(query_idx), m.distance));
    }

    {
        std::scoped_lock lock(keyframe1->cache_mutex_, keyframe2->cache_mutex_);
        keyframe1->cache_matched_pairs_feat_type.insert_or_assign(keyframe2->frame_id, std::move(forward_by_type));
        keyframe2->cache_matched_pairs_feat_type.insert_or_assign(keyframe1->frame_id, std::move(backward_by_type));
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

// Matches keypoints between two frames across all feature types for monocular initialization.
// Descriptors are matched in parallel per feature type, then all matches are pooled and filtered
// jointly with the geometric filter (filter_matches_geometric). Matched index pairs are written into
// matched_pairs per feature type.
// Used in: Tracking::monocular_initialization
map<FeatureType, vector<pair<size_t, size_t>>> FeatureMatcher::match_frames_for_initialization(const Frame& F1, const Frame& F2,
    const vector<FeatureType>& feat_types)
{
    map<FeatureType, vector<cv::DMatch>> matches_by_type;
    matches_by_type = match_descriptors_parallel(feat_types, F1.features, F2.features);

    // Offset match indices to account for previously appended feature types
    vector<cv::KeyPoint> kps1, kps2;
    vector<size_t> kps1_indexes, kps2_indexes;
    vector<cv::DMatch> all_matches;
    vector<FeatureType> used_feature_types;
    for (const auto& ft : feat_types) {
        // Ensure both frames contain the requested feature type
        auto it1 = F1.features.per_type.find(ft);
        auto it2 = F2.features.per_type.find(ft);
        if (it1 == F1.features.per_type.end() || it2 == F2.features.per_type.end())
            continue;

        // Accumulate keypoints and track per-feature-type original indices for later lookup
        auto const& v1 = F1.features.at(ft).keypoints;
        auto const& v2 = F2.features.at(ft).keypoints;

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

    // Filter outliers jointly across all feature types (PoseLib LO-RANSAC, fundamental matrix)
    auto filtered_matches = filter_matches_geometric(all_matches, kps1, kps2);

    // One entry per requested feature type, empty when none of its matches survive (callers read them with .at(ft))
    std::map<FeatureType, std::vector<pair<size_t, size_t>>> matched_pairs;
    for (const auto& ft : feat_types)
        matched_pairs[ft];
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
/// @note Used in: LoopClosing::search_loop_map_points
int FeatureMatcher::search_by_projection_for_compute_sim3(const Keyframe& keyframe, const mat4f& Scw,
    const vector<Pt>& pts, vector<Pt>& pts_matched,
    const map<PtId, Keyframe>& pt_to_keyframe_id)
{
    const float fx = keyframe->calibration->fx;
    const float fy = keyframe->calibration->fy;
    const float cx = keyframe->calibration->cx;
    const float cy = keyframe->calibration->cy;

    // Decompose Sim3 into rotation and translation
    const mat3f sRcw = Scw.block<3,3>(0,0);
    const float  scw  = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    const mat3f Rcw  = sRcw / scw;
    const vec3f tcw  = Scw.block<3,1>(0,3);

    // Build the set of already-matched map points to skip
    unordered_set<Pt> pts_already_found(pts_matched.begin(), pts_matched.end());
    pts_already_found.erase(nullptr);

    int num_matches = 0;
    const float radius2 = radiusTh_factor * radiusTh_factor;

    for (const Pt& pt : pts) {
        if (pt->is_bad() || pts_already_found.count(pt))
            continue;

        // Look up cached match index for this map point (per-type indices; read-only lookups under the keyframe's
        // cache lock, which is a leaf: get_index_in_keyframe takes the point's own lock, so it runs first)
        const Keyframe& ref_kf = pt_to_keyframe_id.at(pt->ptId);
        const int ref_idx = pt->get_index_in_keyframe(ref_kf);
        int bestIdx{-1};
        {
            std::lock_guard<std::mutex> lock(keyframe->cache_mutex_);
            const auto cache_it = keyframe->cache_matched_pairs_feat_type.find(ref_kf->frame_id);
            if (cache_it == keyframe->cache_matched_pairs_feat_type.end())
                continue;
            const auto type_it = cache_it->second.find(pt->featureType);
            if (type_it == cache_it->second.end())
                continue;
            for (const auto& m : type_it->second) {
                if (m.trainIdx == ref_idx) {
                    bestIdx = m.queryIdx;
                    break;
                }
            }
        }
        if (bestIdx < 0 || static_cast<size_t>(bestIdx) >= pts_matched.size())
            continue;

        // Project map point into the keyframe
        const vec3f p3Dc = Rcw * pt->get_world_pos() + tcw;
        if (p3Dc(2) < 0.0f)
            continue;

        const float invz = 1.0f / p3Dc(2);
        const float u = fx * p3Dc(0) * invz + cx;
        const float v = fy * p3Dc(1) * invz + cy;

        if (!keyframe->calibration->is_in_image(u, v))
            continue;

        // Accept match only if bestIdx falls within the projected search radius. A direct distance
        // test, not a get_features_in_area query: bestIdx is already known, so there is nothing to
        // search for. (That query also made grid membership an implicit acceptance criterion.)
        const cv::Point2f& kp = keyframe->features.at(pt->featureType).keypoints[bestIdx].pt;
        const float dist_x = kp.x - u;
        const float dist_y = kp.y - v;
        if (dist_x * dist_x + dist_y * dist_y < radius2) {
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

// Projects map points from a keyframe into a current frame for relocalization.
// Applies rotation histogram filtering to reject orientation-inconsistent matches.
int FeatureMatcher::search_by_projection(Frame &CurrentFrame, Keyframe pKF, const set<Pt> &sAlreadyFound, const float& radiusTh,
     const bool& useHighMatchingThreshold, const FeatureType& feature_type)
{
    Descriptor_Distance_Type descDistanceTh = descDistTh_low_reloc[feature_type];
    if(useHighMatchingThreshold)
        descDistanceTh = descDistTh_high_reloc[feature_type];

    // Rotation Histogram (to check rotation consistency)
    int nMatches{0};
    float rotFactor{};
    vector<vector<int>>& rotHist = initRotationHistogram(rotFactor,HISTO_LENGTH);

    const vector<Pt> vpMPs = pKF->get_map_point_matches(feature_type);
    const Calibration& cal = *CurrentFrame.calibration;   // hoisted out of the loop below

    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        const Pt pMP = vpMPs[i];

        if(pMP)
        {
            if(!pMP->is_bad() && !sAlreadyFound.count(pMP))
            {
                // Depth, image bounds and the depth window, as everywhere else.
                // -1 for the viewing-angle limit disables that test (no cosine is below -1):
                // relocalization is the one path that must accept a point seen from any
                // direction, since the whole point is that the current viewpoint is unknown.
                // This is also the first time the path rejects a point BEHIND the camera --
                // it used to divide by x3Dc(2) without checking its sign.
                const std::optional<Projection> projection =
                    project_if_visible(CurrentFrame.get_pose(), cal, pMP, -1.0f);
                if(!projection)
                    continue;

                const float u = projection->uv.x;
                const float v = projection->uv.y;
                const float dist3D = projection->dist;

                // Search in a window
                const float predictedSize = pMP->predict_size(dist3D);
                const float radius = radiusScale * radiusTh * predictedSize;

                const vector<size_t> vIndices2 = CurrentFrame.get_features_in_area(u, v, radius, feature_type);

                if(vIndices2.empty())
                    continue;

                const cv::Mat refDescriptor = pMP->get_descriptor();
                Descriptor_Distance_Type bestDist{highest_possible_distance};
                int bestIdx2{-1};

                for(const size_t i2 : vIndices2)
                {
                    if(CurrentFrame.pts.at(feature_type)[i2])
                        continue;

                    const cv::Mat &descriptor = CurrentFrame.features.at(feature_type).descriptors.row(i2);
                    const Descriptor_Distance_Type descDist = descriptor_distance(refDescriptor,descriptor,pMP->featureType);

                    if(descDist < bestDist)
                    {
                        bestDist = descDist;
                        bestIdx2 = i2;
                    }
                }

                if(bestDist <= descDistanceTh)
                {
                    CurrentFrame.pts.at(feature_type)[bestIdx2]=pMP;
                    nMatches++;

                    if(mbCheckOrientation)
                        updateRotationHistogram(rotHist,bestIdx2, pKF->features.at(feature_type).keypoints[i],CurrentFrame.features.at(feature_type).keypoints[bestIdx2],rotFactor,HISTO_LENGTH);
                }
            }
        }
    }

    if(mbCheckOrientation)
        filterMatchesWithOrientation(rotHist,CurrentFrame.pts.at(feature_type),nMatches);

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

    // Bin a match by the rotation between its two keypoints, so filterMatchesWithOrientation can
    // keep only the three most populated bins -- matches whose orientation change disagrees with
    // the frame's consensus are outliers.
    //
    // Inert for aliked128 and superpoint256: AlikedOnnx and SuperPointOnnx build every keypoint
    // with angle 0 (both detectors are orientation-free), so every match lands in bin 0 and the
    // filter keeps all of them. Harmless -- one full bin is still "the top bins" -- but it means
    // those two features get no orientation check at all, not a lenient one. orb32, brisk48,
    // kaze64, akaze61 and sift128 all carry a real angle from their detector.
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
        const MatcherType matcherType = ft.get_matcher_type();

        std::vector<cv::DMatch> matches;
        switch(matcherType) {
            // descriptor-only matching of every float descriptor, learned or not: the cross-checked brute force
            // (SIFT came through SiftMatchGPU's ratio test until 2026-10-16)
            case LIGHTGLUE_SUPERPOINT:
            case LIGHTGLUE_ALIKED:
            case LIGHTGLUE_SIFT:
            case BF_L2:{
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

    std::vector<cv::DMatch> FeatureMatcher::match_descriptors(
        const cv::Mat& desc1, const cv::Mat& desc2,
        const std::vector<cv::KeyPoint>& kps1, const std::vector<cv::KeyPoint>& kps2,
        const FeatureType& featType_){

        const AF_VSLAM::Feature& ft = get_feature(featType_);
        const MatcherType matcherType = ft.get_matcher_type();

        std::vector<cv::DMatch> matches;
        // A side without keypoints (a type a frame did not detect: its descriptors are an empty Mat) has nothing to
        // match, and its shape would fail the matchers' checks
        if(desc1.empty() || desc2.empty() || kps1.empty() || kps2.empty())
            return matches;
        switch(matcherType) {
            // LightGlue on ONNX Runtime: the keypoints are normalised with the real image size inside
            // (the TensorRT path normalised with a 320x240 config until 2026-10-16)
            case LIGHTGLUE_SIFT:
            case LIGHTGLUE_ALIKED:
            case LIGHTGLUE_SUPERPOINT:
            {
                // A failed Run (a transient Core ML / ONNX Runtime error) costs this match, not the process: this
                // runs on the tracking, mapping (inside OpenMP regions) and loop-closing threads, where an escaping
                // exception is std::terminate
                try
                {
                    matches = onnx::shared_lightglue(lightglue_variant(matcherType),
                                                     static_cast<int>(std::max(kps1.size(), kps2.size())))
                                  .match(kps1, desc1, kps2, desc2, image_width, image_height);
                }
                catch(const std::exception& e)
                {
                    AF_WARN("LightGlue match failed, no matches for this pair: " << e.what());
                    matches.clear();
                }
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

    std::map<FeatureType, std::vector<cv::DMatch>> FeatureMatcher::match_descriptors_parallel(
        const std::vector<FeatureType>& feature_types,
        const FeatureData& features1, const FeatureData& features2
    ){

        struct MatchOut {
            FeatureType ft{};
            bool valid = false;
            std::vector<cv::DMatch> matches;
            size_t frameN = 0;
        };

        std::vector<FeatureType> jobs;
        jobs.reserve(feature_types.size());
        for (const auto ft : feature_types) {
            if (!features1.has(ft) || !features2.has(ft))
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

            const KeypointData& data1 = features1.at(ft);
            const KeypointData& data2 = features2.at(ft);
            o.matches = match_descriptors(data1.descriptors, data2.descriptors,
                                          data1.keypoints, data2.keypoints, ft);
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
