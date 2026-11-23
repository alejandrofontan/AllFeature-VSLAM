#include "LocalMapping.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <placecell/placecell.h>

#include "afvslam_log.hpp"
#include "ErrorModel.h"
#include "FeatureFactory.h"
#include "FeatureMatcher.h"
#include "KeyFrame.h"
#include "KeyframeInformation.h"
#include "LoopClosing.h"
#include "Map.h"
#include "MapPoint.h"
#include "Optimizer.h"
#include "Report.h"
#include "Utils.h"
#include "Viewer.h"

namespace AF_VSLAM
{

LocalMapping::LocalMapping(std::shared_ptr<Map> map, const std::vector<FeatureType>& feature_types):
    map_(std::move(map)),
    matcher_(std::make_unique<FeatureMatcher>(feature_types, THREAD_NAME))
{
}

LocalMapping::~LocalMapping() = default;   // here, where FeatureMatcher is complete

void LocalMapping::run()
{
    // Loop Closing receives every processed keyframe: the thread cannot work without it
    if(!loop_closer_)
    {
        AF_ERROR("[LocalMapping] started without Loop Closing (set_loop_closer before the thread)");
        exit(-1);
    }

    {
        std::lock_guard<std::mutex> lock(finish_mutex_);
        finished_ = false;
    }

    while(true)
    {
        // Tracking sees Local Mapping as busy until this iteration's work is done
        set_accept_keyframes(false);

        if(has_new_keyframes())
            process_keyframe();
        else if(stop_if_requested())
        {
            // Paused by a loop closure: idle until release() (or shutdown)
            while(is_stopped() && !is_finish_requested())
                std::this_thread::sleep_for(IDLE_POLL_PERIOD);
            if(is_finish_requested())
                break;
        }

        reset_if_requested();

        // Tracking sees Local Mapping as idle again
        set_accept_keyframes(true);

        if(is_finish_requested())
            break;

        // Yield only when idle: a queued keyframe is processed immediately
        if(!has_new_keyframes())
            std::this_thread::sleep_for(IDLE_POLL_PERIOD);
    }

    set_finished();
}

// One mapping iteration for the keyframe at the head of the queue
void LocalMapping::process_keyframe()
{
    const auto t_start = std::chrono::steady_clock::now();

    // Take the keyframe: every stage below works on current_keyframe_
    {
        std::lock_guard<std::mutex> lock(new_keyframes_mutex_);
        current_keyframe_ = new_keyframes_.front();
        new_keyframes_.pop_front();
    }

    // Report: label this thread's entries with the keyframe's frame; the scope writes time_ms on exit
    AF_REPORT_CONTEXT(THREAD_NAME, current_keyframe_->frame_id);
    AF_REPORT_SCOPE();

    // Register the keyframe: observations, covisibility graph, map
    register_keyframe();

    // Cull recent map points that failed their probation
    cull_map_points();

    // Create map points: triangulation against neighbors, and sensor depth
    create_new_map_points();

    // The matches against the neighbors are not read again
    FeatureMatcher::clear_keyframe_matches(current_keyframe_);

    // Fuse duplicated points with the neighbors
    fuse_with_neighbors();

    // Local bundle adjustment around the keyframe
    local_bundle_adjustment();

    // Cull keyframes the others already explain
    cull_keyframes();

    // Hand the keyframe to Loop Closing
    loop_closer_->insert_keyframe(current_keyframe_);

    // Map size after this iteration
    AF_REPORT("map.keyframes", "", map_->keyframes_in_map());
    AF_REPORT("map.points", "", map_->map_points_in_map());

    // Iteration time in whole ms
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    local_mapping_times_[static_cast<int>(elapsed_ms)]++;
    if(viewer_)
        viewer_->set_local_mapping_time_median(map_median(local_mapping_times_));
}

// Stage 1: make the keyframe part of the map
void LocalMapping::register_keyframe()
{
    AF_REPORT_SCOPE();

    // Place recognition descriptor of the keyframe's image
    current_keyframe_->compute_global_descriptor();

    // The keyframe becomes an observer of the points Tracking matched into it
    for(const FeatureType feature_type : current_keyframe_->feature_types)
    {
        const std::vector<Pt> map_points = current_keyframe_->get_map_point_matches(feature_type);
        const KeypointIndex num_slots = static_cast<KeypointIndex>(map_points.size());
        for(KeypointIndex index = 0; index < num_slots; index++)
        {
            Pt map_point = map_points[index];
            if(!map_point)
                continue;

            // Culled or fused since Tracking matched it
            if(map_point->is_bad())
            {
                // Fused into another point: follow the chain to the one that survives
                Pt survivor = map_point->get_replaced();
                while(survivor && survivor->is_bad())
                    survivor = survivor->get_replaced();

                // The survivor takes the slot, unless the keyframe already holds it in another one
                const bool adopt = survivor && survivor->featureType == feature_type
                                   && std::find(map_points.begin(), map_points.end(), survivor) == map_points.end();
                if(!current_keyframe_->replace_map_point_match(map_point, index, adopt ? survivor : nullptr) || !adopt)
                    continue;
                map_point = survivor;
            }

            const KeypointIndex observed_index = map_point->get_index_in_keyframe(current_keyframe_);
            if(observed_index < 0)
                map_point->add_observation(current_keyframe_, index);
            // Already observing through this slot: the point was created with the keyframe, so it is on probation
            else if(observed_index == index)
                recent_map_points_.push_back(map_point);
            // Already observing through another slot: one point cannot hold two keypoints
            else
                current_keyframe_->erase_map_point_match(map_point, index);
        }
    }

    // Covisibility links from the shared points
    current_keyframe_->update_connections();

    // Into the map, and into the information kernel
    map_->add_keyframe(current_keyframe_);
    if(keyframe_information_)
        keyframe_information_->on_keyframe_processed(current_keyframe_);
}

// Stage 2: probation of the points created in the last few keyframes
void LocalMapping::cull_map_points()
{
    AF_REPORT_SCOPE();
    const int current_id = static_cast<int>(current_keyframe_->keyId);

    // Report: what became of the points on probation, per feature
    struct ProbationCounts { int points{0}, dead{0}, culled_found_ratio{0}, culled_observations{0}, graduated{0}; };
    std::map<FeatureType, ProbationCounts> counts_by_feature;

    recent_map_points_.remove_if([current_id, &counts_by_feature](const Pt& map_point)
    {
        ProbationCounts& counts = counts_by_feature[map_point->featureType];
        counts.points++;

        // Culled or fused elsewhere meanwhile: it just leaves the list
        if(map_point->is_bad())
        {
            counts.dead++;
            return true;
        }

        // In view often, matched rarely (a new point starts at ratio 1)
        if(map_point->get_found_ratio() < params.map_point_culling_min_found_ratio)
        {
            map_point->set_bad_flag();
            counts.culled_found_ratio++;
            return true;
        }

        // Old enough to have been observed again, and was not. Age = keyframes created since the point was;
        // an observation with sensor depth counts more than one
        const int age = current_id - static_cast<int>(map_point->first_keyframe_id);
        if(age >= params.map_point_culling_observation_test_age
           && map_point->number_of_observations() < params.map_point_culling_min_observations)
        {
            map_point->set_bad_flag();
            counts.culled_observations++;
            return true;
        }

        // Survived: leaves probation once old enough, and is not tested again
        if(age >= params.map_point_culling_probation_age)
        {
            counts.graduated++;
            return true;
        }
        return false;
    });

    for([[maybe_unused]] const auto& [feature_type, counts] : counts_by_feature)
    {
        const std::string& feature = get_feature(feature_type).get_feature_name();
        AF_REPORT("probation.points", feature, counts.points);
        AF_REPORT("probation.dead", feature, counts.dead);
        AF_REPORT("culled.found_ratio", feature, counts.culled_found_ratio);
        AF_REPORT("culled.observations", feature, counts.culled_observations);
        AF_REPORT("probation.graduated", feature, counts.graduated);
        AF_REPORT("probation.waiting", feature, counts.points - counts.dead - counts.culled_found_ratio
                                                - counts.culled_observations - counts.graduated);
    }
}

// The keyframes to triangulate against: the most covisible ones with enough baseline
std::vector<Keyframe> LocalMapping::select_triangulation_neighbors() const
{
    AF_REPORT_SCOPE();
    const std::vector<Keyframe> covisible = current_keyframe_->get_covisible_keyframes();   // most covisible first
    const vec3f centre = current_keyframe_->get_pose().twc;

    const size_t num_wanted = static_cast<size_t>(std::max(params.create_new_map_points_keyframes, 0));
    std::vector<Keyframe> neighbors;
    neighbors.reserve(num_wanted);
    [[maybe_unused]] size_t num_examined = 0;
    [[maybe_unused]] float min_baseline_ratio = std::numeric_limits<float>::max();

    // A neighbor that is too close is replaced by the next one down the list
    for(const Keyframe& candidate : covisible)
    {
        if(neighbors.size() >= num_wanted)
            break;
        num_examined++;
        if(candidate->is_bad())
            continue;

        // Baseline against the neighbor's median scene depth: too little gives no parallax
        const float baseline = (candidate->get_pose().twc - centre).norm();
        const std::optional<float> scene_depth = candidate->compute_scene_median_depth();
        if(!scene_depth || *scene_depth <= 0.0f
           || baseline / *scene_depth < params.create_new_map_points_min_baseline_depth_ratio)
            continue;
        neighbors.push_back(candidate);
        min_baseline_ratio = std::min(min_baseline_ratio, baseline / *scene_depth);
    }

    AF_REPORT("neighbors.covisible", "", covisible.size());
    AF_REPORT("neighbors.examined", "", num_examined);
    AF_REPORT("neighbors.selected", "", neighbors.size());
    if(!neighbors.empty())
        AF_REPORT("neighbors.baseline_ratio_min", "", min_baseline_ratio);
    return neighbors;
}

namespace
{

// Degrees in one radian
constexpr float DEGREES_PER_RADIAN{57.29577951308232f};

// One keyframe as the point-creation code needs it: its pose (read once, under one lock), its intrinsics and
// its keypoints.
struct KeyframeView
{
    explicit KeyframeView(Keyframe keyframe_)
        : keyframe(std::move(keyframe_)), pose(keyframe->get_pose()), calibration(*keyframe->calibration) {}

    // The keypoint's viewing ray in the camera frame, at depth 1
    [[nodiscard]] vec3f normalized(const cv::KeyPoint& keypoint) const
    {
        return vec3f{(keypoint.pt.x - calibration.cx) * calibration.invfx,
                     (keypoint.pt.y - calibration.cy) * calibration.invfy, 1.0f};
    }

    // The world point on that ray at the given inverse depth
    [[nodiscard]] vec3f back_project(const vec3f& normalized_keypoint, float inv_depth) const
    {
        return pose.Rwc * (normalized_keypoint / inv_depth) + pose.twc;
    }

    // True when the world point lies in front of the camera and agrees with the keypoint, by the optimizer's own
    // residual and threshold: error^T * information * error of the pixel error against Error.Chi2Mono, or, when
    // the keypoint has sensor depth, with the inverse-depth error added, against Error.Chi2RGBD. A point at
    // infinity (a degenerate triangulation) fails here.
    [[nodiscard]] bool observes(const vec3f& x3D, const cv::KeyPoint& keypoint, KeypointIndex index,
                                FeatureType feature_type) const
    {
        const float z = pose.Rcw.row(2).dot(x3D) + pose.tcw(2);
        if(!(z > 0))
            return false;
        const float u = calibration.fx * (pose.Rcw.row(0).dot(x3D) + pose.tcw(0)) / z + calibration.cx;
        const float v = calibration.fy * (pose.Rcw.row(1).dot(x3D) + pose.tcw(1)) / z + calibration.cy;
        const vec2f error{u - keypoint.pt.x, v - keypoint.pt.y};
        const float chi2_pixel = error.dot(keyframe->features.get_information_2d(index, feature_type) * error);

        const KeypointData& data = keyframe->features.at(feature_type);
        const float inv_depth = data.inv_depth[index];
        if(!(inv_depth > 0.0f))
            return chi2_pixel <= ErrorModel::params.chi2_mono;

        const float error_inv_depth = inv_depth - 1.0f / z;
        return chi2_pixel + error_inv_depth * error_inv_depth / data.sigma2_inv_depth[index]
               <= ErrorModel::params.chi2_rgbd;
    }

    // How much a back-projection from this keypoint's sensor depth is worth: the inverse variance of that depth.
    // depth = 1 / inv_depth, so its variance is sigma2_inv_depth / inv_depth^4.
    [[nodiscard]] float depth_weight(KeypointIndex index, FeatureType feature_type) const
    {
        const KeypointData& data = keyframe->features.at(feature_type);
        const float inv_depth = data.inv_depth[index];
        return inv_depth * inv_depth * inv_depth * inv_depth / data.sigma2_inv_depth[index];
    }

    const Keyframe keyframe;
    const Pose pose;
    const Calibration& calibration;   // owned by keyframe
};

// The point takes the keyframe's free keypoint. Published first, observed only if the slot was won: another
// thread can fill it meanwhile.
bool observe(const Keyframe& keyframe, const Pt& pt, KeypointIndex index)
{
    if(!keyframe->add_map_point(pt, index))
        return false;
    pt->add_observation(keyframe, index);
    return true;
}

// Two points that are one: the one with more observations survives
void merge(const Pt& a, const Pt& b)
{
    if(a->number_of_observations() > b->number_of_observations())
        b->replace(a);
    else
        a->replace(b);
}

// One matched pair of keypoints: keypoint index1 of view1's keyframe and index2 of view2's
struct MatchedPair
{
    const KeyframeView& view1;
    const KeyframeView& view2;
    KeypointIndex index1, index2;
    const cv::KeyPoint& keypoint1;
    const cv::KeyPoint& keypoint2;
    FeatureType feature_type;
};

// Report: what became of the matched pairs of one feature
struct PairCounts
{
    int matched{0}, free{0}, rejected_parallax{0}, rejected_gate{0}, triangulated{0}, depth{0};
    int same_point{0}, one_point{0}, two_points{0}, observed{0}, merged{0};   // the pairs that are not free
    std::vector<float> parallax_cos;   // of the triangulated points
};

// A pair with a point on either side is a fusion statement, not a new point
void fuse_matched_pair(const MatchedPair& pair, const Pt& pt1, const Pt& pt2, PairCounts& counts)
{
    if(pt1 == pt2)
    {
        counts.same_point++;
        return;
    }

    // Two points matched to each other: one point, if each reprojects onto the other's keypoint
    if(pt1 && pt2)
    {
        counts.two_points++;
        if(!pt1->is_bad() && !pt2->is_bad()
           && pair.view2.observes(pt1->get_world_pos(), pair.keypoint2, pair.index2, pair.feature_type)
           && pair.view1.observes(pt2->get_world_pos(), pair.keypoint1, pair.index1, pair.feature_type))
        {
            merge(pt1, pt2);
            counts.merged++;
        }
        return;
    }

    // A point matched to a free keypoint: it is seen there too, if it reprojects onto it
    counts.one_point++;
    const Pt& pt = pt1 ? pt1 : pt2;
    const KeyframeView& view = pt1 ? pair.view2 : pair.view1;
    const cv::KeyPoint& keypoint = pt1 ? pair.keypoint2 : pair.keypoint1;
    const KeypointIndex index = pt1 ? pair.index2 : pair.index1;
    if(!pt->is_bad() && !pt->is_in_keyframe(view.keyframe)
       && view.observes(pt->get_world_pos(), keypoint, index, pair.feature_type)
       && observe(view.keyframe, pt, index))
        counts.observed++;
}

// Two free keypoints: a new point, placed from sensor depth when there is any. Null when the pair gives none.
Pt create_point_from_pair(const MatchedPair& pair, Map& map, PairCounts& counts)
{
    const KeyframeView& view1 = pair.view1;
    const KeyframeView& view2 = pair.view2;
    const vec3f xn1 = view1.normalized(pair.keypoint1);
    const vec3f xn2 = view2.normalized(pair.keypoint2);
    const float inv_depth1 = view1.keyframe->features.at(pair.feature_type).inv_depth[pair.index1];
    const float inv_depth2 = view2.keyframe->features.at(pair.feature_type).inv_depth[pair.index2];
    const bool have_depth1 = inv_depth1 > 0.0f;
    const bool have_depth2 = inv_depth2 > 0.0f;

    vec3f x3D;
    float cos_parallax = 1.0f;   // of the two rays; set when the point is triangulated
    if(have_depth1 && have_depth2)
    {
        // Two depth readings: their mean, each weighted by how certain it is. The gates below reject
        // the pair when the result disagrees with either reading.
        const float w1 = view1.depth_weight(pair.index1, pair.feature_type);
        const float w2 = view2.depth_weight(pair.index2, pair.feature_type);
        x3D = (w1 * view1.back_project(xn1, inv_depth1) + w2 * view2.back_project(xn2, inv_depth2)) / (w1 + w2);
    }
    // One depth reading: its back-projection
    else if(have_depth1)
        x3D = view1.back_project(xn1, inv_depth1);
    else if(have_depth2)
        x3D = view2.back_project(xn2, inv_depth2);
    // No depth: two-view triangulation, if the rays have enough parallax
    else
    {
        const vec3f ray1 = view1.pose.Rwc * xn1;
        const vec3f ray2 = view2.pose.Rwc * xn2;
        cos_parallax = ray1.dot(ray2) / (ray1.norm() * ray2.norm());
        if(!(cos_parallax > LocalMapping::params.create_new_map_points_min_parallax_cos
             && cos_parallax < LocalMapping::params.create_new_map_points_max_parallax_cos))
        {
            counts.rejected_parallax++;
            return nullptr;
        }

        const mat4f& Tcw1 = view1.pose.Tcw;
        const mat4f& Tcw2 = view2.pose.Tcw;
        Eigen::Matrix<float, 4, 4> A;
        A.row(0) = xn1(0) * Tcw1.row(2) - Tcw1.row(0);
        A.row(1) = xn1(1) * Tcw1.row(2) - Tcw1.row(1);
        A.row(2) = xn2(0) * Tcw2.row(2) - Tcw2.row(0);
        A.row(3) = xn2(1) * Tcw2.row(2) - Tcw2.row(1);

        const Eigen::JacobiSVD<Eigen::Matrix<float, 4, 4>> svd(A, Eigen::ComputeFullV);
        const Eigen::Vector4f x_h = svd.matrixV().col(3);
        x3D = x_h.head<3>() / x_h(3);
    }

    // In front of both cameras and inside both keypoints' gates
    if(!view1.observes(x3D, pair.keypoint1, pair.index1, pair.feature_type)
       || !view2.observes(x3D, pair.keypoint2, pair.index2, pair.feature_type))
    {
        counts.rejected_gate++;
        return nullptr;
    }

    const Pt map_point = map.create_monocular_map_point(x3D, view1.keyframe, pair.index1,
                                                        view2.keyframe, pair.index2, pair.feature_type);
    if(!map_point)
        return nullptr;   // another thread took a slot meanwhile
    if(have_depth1 || have_depth2)
        counts.depth++;
    else
    {
        counts.triangulated++;
        counts.parallax_cos.push_back(cos_parallax);
    }
    return map_point;
}

}

// Stage 3: new map points, and fusion, from the matches against the neighbors
void LocalMapping::create_new_map_points()
{
    AF_REPORT_SCOPE();

    // Neighbors with enough baseline, matched against the keyframe all at once
    const std::vector<Keyframe> neighbors = select_triangulation_neighbors();
    matcher_->match_keyframe_to_keyframes(current_keyframe_, neighbors, current_keyframe_->feature_types);

    const KeyframeView view1(current_keyframe_);
    std::map<FeatureType, PairCounts> counts_by_feature;

    for(const Keyframe& neighbor : neighbors)
    {
        const KeyframeView view2(neighbor);
        const std::map<FeatureType, std::vector<std::pair<size_t, size_t>>> matched_indices =
            FeatureMatcher::get_keyframe_matches(current_keyframe_, neighbor);

        for(const auto& [feature_type, pairs] : matched_indices)
        {
            const auto& keypoints1 = current_keyframe_->features.at(feature_type).keypoints;
            const auto& keypoints2 = neighbor->features.at(feature_type).keypoints;
            PairCounts& counts = counts_by_feature[feature_type];
            counts.matched += static_cast<int>(pairs.size());

            for(const auto& [match1, match2] : pairs)
            {
                const KeypointIndex index1 = static_cast<KeypointIndex>(match1);
                const KeypointIndex index2 = static_cast<KeypointIndex>(match2);
                const MatchedPair pair{view1, view2, index1, index2, keypoints1[match1], keypoints2[match2],
                                       feature_type};

                // A point on either side: fusion. Two free keypoints: a new point, on probation.
                const Pt pt1 = current_keyframe_->get_map_point(index1, feature_type);
                const Pt pt2 = neighbor->get_map_point(index2, feature_type);
                if(pt1 || pt2)
                {
                    fuse_matched_pair(pair, pt1, pt2, counts);
                    continue;
                }
                counts.free++;
                if(const Pt map_point = create_point_from_pair(pair, *map_, counts))
                    recent_map_points_.push_back(map_point);
            }
        }
    }

    for([[maybe_unused]] auto& [feature_type, counts] : counts_by_feature)
    {
        const std::string& feature = get_feature(feature_type).get_feature_name();
        AF_REPORT("pairs.matched", feature, counts.matched);
        AF_REPORT("pairs.free", feature, counts.free);
        AF_REPORT("pairs.same_point", feature, counts.same_point);
        AF_REPORT("pairs.one_point", feature, counts.one_point);
        AF_REPORT("pairs.two_points", feature, counts.two_points);
        AF_REPORT("fused.observed", feature, counts.observed);
        AF_REPORT("fused.merged", feature, counts.merged);
        AF_REPORT("rejected.parallax", feature, counts.rejected_parallax);
        AF_REPORT("rejected.gate", feature, counts.rejected_gate);
        AF_REPORT("created.triangulated", feature, counts.triangulated);
        AF_REPORT("created.depth", feature, counts.depth);
        if(!counts.parallax_cos.empty())
        {
            // Median parallax of the triangulated points, in degrees
            auto& parallax_cos = counts.parallax_cos;
            const auto middle = parallax_cos.begin() + static_cast<std::ptrdiff_t>(parallax_cos.size() / 2);
            std::nth_element(parallax_cos.begin(), middle, parallax_cos.end());
            AF_REPORT("created.parallax_deg", feature,
                      std::acos(std::clamp(*middle, -1.0f, 1.0f)) * DEGREES_PER_RADIAN);
        }
    }

    // Keypoints with sensor depth and still no point
    create_depth_seeded_points();
}

// Points from sensor depth for the keypoints no neighbor matched (RGB-D only)
void LocalMapping::create_depth_seeded_points()
{
    AF_REPORT_SCOPE();
    const KeyframeView view(current_keyframe_);

    for(const FeatureType feature_type : current_keyframe_->feature_types)
    {
        const auto& inv_depth = current_keyframe_->features.at(feature_type).inv_depth;
        const auto& keypoints = current_keyframe_->features.at(feature_type).keypoints;
        const KeypointIndex num_keypoints = static_cast<KeypointIndex>(inv_depth.size());
        [[maybe_unused]] size_t num_with_depth = 0, num_created = 0;   // for the report
        for(KeypointIndex index = 0; index < num_keypoints; index++)
        {
            // No sensor depth here, or the keypoint already has a point
            if(inv_depth[index] <= 0.0f)
                continue;
            num_with_depth++;
            if(current_keyframe_->get_map_point(index, feature_type))
                continue;

            // One observation so far: probation decides whether another keyframe confirms it
            const vec3f position = view.back_project(view.normalized(keypoints[index]), inv_depth[index]);
            if(const Pt map_point = map_->create_map_point(position, current_keyframe_, index, feature_type))
            {
                recent_map_points_.push_back(map_point);
                num_created++;
            }
        }

        // Nothing is written for a keyframe without sensor depth
        if(num_with_depth > 0)
        {
            const std::string& feature = get_feature(feature_type).get_feature_name();
            AF_REPORT("keypoints.with_depth", feature, num_with_depth);
            AF_REPORT("created.depth_seeded", feature, num_created);
        }
    }
}

// Stage 4: fuse duplicated points with the neighbors, by projection
void LocalMapping::fuse_with_neighbors()
{
    AF_REPORT_SCOPE();

    // Targets: the best covisible keyframes and a few of theirs, each once
    std::vector<Keyframe> targets;
    std::unordered_set<KeyframeId> target_ids;
    [[maybe_unused]] size_t num_first = 0;   // for the report
    const std::vector<Keyframe> neighbors =
        current_keyframe_->get_best_covisibility_keyframes(params.fuse_with_neighbors_keyframes);
    for(const Keyframe& neighbor : neighbors)
    {
        if(neighbor->is_bad())
            continue;
        num_first++;
        if(!target_ids.insert(neighbor->keyId).second)
            continue;   // already a target, through an earlier neighbor: its own neighbors are not added
        targets.push_back(neighbor);

        const std::vector<Keyframe> seconds =
            neighbor->get_best_covisibility_keyframes(params.fuse_with_neighbors_second_keyframes);
        for(const Keyframe& second : seconds)
        {
            if(second->is_bad() || second->keyId == current_keyframe_->keyId
               || !target_ids.insert(second->keyId).second)
                continue;
            targets.push_back(second);
        }
    }
    AF_REPORT("targets.first", "", num_first);
    AF_REPORT("targets.second", "", targets.size() - num_first);

    for(const FeatureType feature_type : current_keyframe_->feature_types)
    {
        estimate_fuse_max_descriptor_distance(feature_type);

        // This keyframe's points, projected into every target. Read again for each one: a merge in one target
        // replaces a point of this keyframe, and the next target must get the survivor.
        [[maybe_unused]] FuseCounts into_targets;
        for(const Keyframe& target : targets)
        {
            const FuseCounts counts =
                fuse_into_keyframe(target, current_keyframe_->get_map_point_matches(feature_type), feature_type);
            into_targets.not_visible += counts.not_visible;
            into_targets.observed += counts.observed;
            into_targets.merged += counts.merged;
        }

        // Every target's points, each once, projected into this keyframe
        std::vector<Pt> candidates;
        std::unordered_set<PtId> candidate_ids;
        for(const Keyframe& target : targets)
        {
            for(const Pt& candidate : target->get_map_point_matches(feature_type))
            {
                if(!candidate || candidate->is_bad() || !candidate_ids.insert(candidate->ptId).second)
                    continue;
                candidates.push_back(candidate);
            }
        }
        [[maybe_unused]] const FuseCounts into_current =
            fuse_into_keyframe(current_keyframe_, candidates, feature_type);

        const std::string& feature = get_feature(feature_type).get_feature_name();
        AF_REPORT("into_targets.not_visible", feature, into_targets.not_visible);
        AF_REPORT("into_targets.observed", feature, into_targets.observed);
        AF_REPORT("into_targets.merged", feature, into_targets.merged);
        AF_REPORT("into_current.candidates", feature, candidates.size());
        AF_REPORT("into_current.not_visible", feature, into_current.not_visible);
        AF_REPORT("into_current.observed", feature, into_current.observed);
        AF_REPORT("into_current.merged", feature, into_current.merged);
    }

    // Covisibility links from the points gained
    current_keyframe_->update_connections();
}

// The descriptor limit of the fusion, for one feature: how far a point's descriptor lies from its observations
void LocalMapping::estimate_fuse_max_descriptor_distance(const FeatureType feature_type)
{
    std::vector<Descriptor_Distance_Type> distances;
    for(const Pt& map_point : current_keyframe_->get_map_point_matches(feature_type))
    {
        if(!map_point || map_point->is_bad())
            continue;
        const cv::Mat descriptor = map_point->get_descriptor();
        if(descriptor.empty())
            continue;
        bool self_skipped = false;
        for(const auto& [id, observation] : map_point->get_observations())
        {
            if(!observation.keyframe || observation.keyframe->is_bad())
                continue;
            const Descriptor_Distance_Type distance = FeatureMatcher::descriptor_distance(descriptor,
                observation.keyframe->features.at(feature_type).descriptors.row(observation.index), feature_type);

            // The point's descriptor is a copy of one observation's: that one, at distance 0, is not a sample
            if(!self_skipped && distance == 0)
            {
                self_skipped = true;
                continue;
            }
            distances.push_back(distance);
        }
    }

    // Too few samples to trust: the previous estimate stays
    const std::string& feature = get_feature(feature_type).get_feature_name();
    AF_REPORT("descriptor.samples", feature, distances.size());
    if(distances.size() < static_cast<size_t>(params.fuse_estimate_min_samples))
        return;

    // The quantile becomes the map's limit, read by this thread's fusion and by Loop Closing's
    const size_t index = std::min(distances.size() - 1,
                                  static_cast<size_t>(params.fuse_descriptor_quantile * distances.size()));
    std::nth_element(distances.begin(), distances.begin() + static_cast<std::ptrdiff_t>(index), distances.end());
    map_->set_fuse_max_descriptor_distance(feature_type, distances[index]);
    AF_REPORT("descriptor.fuse_max", feature, distances[index]);
}

// Fuses map_points into one keyframe by projection: an observation at a free keypoint, a merge at a taken one
LocalMapping::FuseCounts LocalMapping::fuse_into_keyframe(const Keyframe& keyframe, const std::vector<Pt>& map_points,
                                                          const FeatureType feature_type)
{
    // No descriptor limit yet for this feature: nothing is fused
    const std::optional<Descriptor_Distance_Type> max_descriptor_distance =
        map_->get_fuse_max_descriptor_distance(feature_type);
    if(!max_descriptor_distance)
        return {};

    // The points the keyframe can see: in front, in the image, at a plausible distance and viewing angle
    FuseCounts counts;
    const Pose pose = keyframe->get_pose();
    std::vector<Pt> candidates;
    std::vector<cv::Point2f> projections;
    for(const Pt& pt : map_points)
    {
        if(!pt || pt->is_bad() || pt->featureType != feature_type || pt->is_in_keyframe(keyframe))
            continue;
        const std::optional<Projection> projection =
            project_if_visible(pose, *keyframe->calibration, pt, params.fuse_viewing_cos_limit);
        if(!projection)
        {
            counts.not_visible++;
            continue;
        }
        candidates.push_back(pt);
        projections.push_back(projection->uv);
    }

    // Matched by position (gate around the projection) and descriptor (below the limit), mutually
    std::vector<Pt> matches(keyframe->features.at(feature_type).count());
    matcher_->match_map_points_to_keyframe(keyframe, candidates, projections, feature_type, matches,
                                           ErrorModel::params.chi2_mono, *max_descriptor_distance);

    const KeypointIndex num_keypoints = static_cast<KeypointIndex>(matches.size());
    for(KeypointIndex index = 0; index < num_keypoints; index++)
    {
        const Pt& pt = matches[index];
        if(!pt || pt->is_bad() || pt->is_in_keyframe(keyframe))
            continue;

        const Pt existing = keyframe->get_map_point(index, feature_type);
        // The keypoint holds another point: the two are one, the better observed survives
        if(existing)
        {
            if(existing->is_bad())
                continue;
            merge(existing, pt);
            counts.merged++;
        }
        // A free keypoint: the point is seen there too
        else if(observe(keyframe, pt, index))
            counts.observed++;
    }
    return counts;
}

// Stage 5: local bundle adjustment around the keyframe
void LocalMapping::local_bundle_adjustment()
{
    AF_REPORT_SCOPE();
    [[maybe_unused]] const Optimizer::LocalBaCounts counts =
        Optimizer::local_bundle_adjustment(current_keyframe_, map_);
    AF_REPORT("keyframes.local", "", counts.local_keyframes);
    AF_REPORT("keyframes.fixed", "", counts.fixed_keyframes);
    AF_REPORT("keyframes.oldest_fixed", "", counts.oldest_fixed);
    AF_REPORT("points", "", counts.points);
    AF_REPORT("edges", "", counts.edges);
    AF_REPORT("outliers.erased", "", counts.outliers_erased);
}

// Stage 6: cull keyframes the others explain (placecell's joint-information culler)
void LocalMapping::cull_keyframes()
{
    AF_REPORT_SCOPE();

    // No kernel: nothing is culled (warned once)
    if(!keyframe_information_ || keyframe_information_->place_cell().size() == 0)
    {
        if(!warned_no_kernel_)
        {
            AF_WARN("[LocalMapping] no keyframe information kernel (InformationKernel: megaloc needs a build with "
                    "MegaLoc): keyframes are never culled; set InformationKernel: covisibility");
            warned_no_kernel_ = true;
        }
        return;
    }
    placecell::PlaceCell& place_cell = keyframe_information_->place_cell();

    // Keyframes that died elsewhere become culling history in placecell too
    std::unordered_map<FrameId, Keyframe> keyframe_by_frame_id;
    std::vector<Keyframe> alive_keyframes;
    for(const Keyframe& keyframe : map_->get_all_keyframes())
    {
        keyframe_by_frame_id.emplace(keyframe->frame_id, keyframe);
        if(!keyframe->is_bad())
            alive_keyframes.push_back(keyframe);
    }
    for(const placecell::PlaceCell::ExternalId id : place_cell.external_ids())
    {
        if(place_cell.is_culled(id))
            continue;
        const auto it = keyframe_by_frame_id.find(static_cast<FrameId>(id));
        if(it == keyframe_by_frame_id.end() || it->second->is_bad())
            place_cell.set_culled(id);
    }

    // The kernel follows the map: every alive keyframe's current points
    keyframe_information_->refresh(alive_keyframes);

    // Tau is read live: the viewer's slider can move it. Keyframe 0 is never culled (placecell), nor the
    // current keyframe (protect_last is at least 1).
    placecell::PlaceCell::CullParameters cull_parameters;
    cull_parameters.method = KEYFRAME_CULLING_METHOD;
    cull_parameters.objective = params.keyframe_culling_objective;
    cull_parameters.max_unexplained = params.keyframe_culling_max_unexplained.load();
    cull_parameters.centred = params.keyframe_culling_centred;
    cull_parameters.min_keyframes = params.keyframe_culling_min_keyframes;
    cull_parameters.protect_last = std::max(params.keyframe_culling_min_age, 1);
    cull_parameters.max_per_call = params.keyframe_culling_max_per_call;

    // Local scope: the current keyframe and its covisibles only
    const bool local_scope = (params.keyframe_culling_scope == "local");
    std::vector<placecell::PlaceCell::ExternalId> window;
    if(local_scope)
    {
        window.push_back(current_keyframe_->frame_id);
        for(const Keyframe& keyframe : current_keyframe_->get_covisible_keyframes())
            window.push_back(keyframe->frame_id);
    }

    // A cull pinned by Loop Closing is refused, and placecell moves on to the next candidate
    const auto try_cull = [&keyframe_by_frame_id](const placecell::PlaceCell::ExternalId id) -> bool
    {
        const auto it = keyframe_by_frame_id.find(static_cast<FrameId>(id));
        if(it == keyframe_by_frame_id.end())
            return false;
        it->second->set_bad_flag();
        return it->second->is_bad();
    };

    const placecell::PlaceCell::CullReport report =
        place_cell.cull_keyframes(cull_parameters, try_cull, local_scope ? &window : nullptr);

    AF_REPORT("culled", "", report.culled.size());
    AF_REPORT("candidates", "", report.candidates);
    AF_REPORT("alive_after", "", report.alive_after);
    AF_REPORT("history_worst_unexplained", "", report.worst_history);
    AF_REPORT("history_over_budget", "", report.history_over_budget);
    AF_REPORT("max_per_call_reached", "", report.reached_max_per_call ? 1 : 0);

    // Keyframes ever inserted that the alive ones no longer explain within tau: warned whenever the count changes
    if(report.history_over_budget != last_over_budget_)
    {
        if(report.history_over_budget > 0)
            AF_WARN("[LocalMapping] cull_keyframes: " << report.history_over_budget << " keyframes ever inserted are "
                    << "more than tau unexplained (worst " << std::fixed << std::setprecision(3) << report.worst_history
                    << " > tau " << cull_parameters.max_unexplained << ")");
        last_over_budget_ = report.history_over_budget;
    }
}

}
