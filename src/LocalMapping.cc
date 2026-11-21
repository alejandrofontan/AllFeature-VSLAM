#include "LocalMapping.h"

#include <placecell/placecell.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <Eigen/Dense>

#include "ErrorModel.h"
#include "FeatureMatcher.h"
#include "KeyframeInformation.h"
#include "LoopClosing.h"
#include "Map.h"
#include "Optimizer.h"
#include "Report.h"
#include "Viewer.h"
#include "Utils.h"
#include "afvslam_log.hpp"

namespace AF_VSLAM
{

LocalMapping::LocalMapping(std::shared_ptr<Map> map, const std::vector<FeatureType>& feature_types):
    map_(std::move(map)),
    matcher_(std::make_shared<FeatureMatcher>(feature_types, "LocalMapping"))
{
}

void LocalMapping::run()
{
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
    AF_REPORT_CONTEXT("local_mapping", current_keyframe_->frame_id);
    AF_REPORT_SCOPE();

    // Register the keyframe: observations, covisibility graph, map
    register_keyframe();

    // Cull recent map points that failed their probation
    cull_map_points();

    // Create map points: triangulation against neighbours, and sensor depth
    create_new_map_points();

    // The matches against the neighbours are not read again
    FeatureMatcher::clear_keyframe_matches(current_keyframe_);

    // Fuse duplicated points with the neighbours
    search_in_neighbors();

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
    local_mapping_times_[int(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count())]++;
    if(viewer_)
        viewer_->set_local_mapping_time_median(map_median(local_mapping_times_));
}

// Stage 1: make the keyframe part of the map
void LocalMapping::register_keyframe()
{
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
    struct Counts { int points{0}, dead{0}, culled_found_ratio{0}, culled_observations{0}, graduated{0}; };
    std::map<FeatureType, Counts> counts_by_feature;

    recent_map_points_.remove_if([current_id, &counts_by_feature](const Pt& map_point)
    {
        Counts& counts = counts_by_feature[map_point->featureType];
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

// One keyframe as the point-creation code needs it: its pose (read once, under one lock), its intrinsics and
// its keypoints.
struct KeyframeView
{
    explicit KeyframeView(const Keyframe& keyframe_)
        : keyframe(keyframe_), pose(keyframe_->get_pose()), calibration(*keyframe_->calibration) {}

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

    // True when the world point lies in front of the camera and reprojects onto the keypoint within the
    // chi-square gate of the keypoint's own 2x2 covariance: error^T * information * error, the same quantity the
    // optimizer and the matcher's gate test. For a keypoint with an isotropic sigma this is error^2 / sigma^2.
    [[nodiscard]] bool observes(const vec3f& x3D, const cv::KeyPoint& keypoint, size_t index,
                                FeatureType feature_type) const
    {
        const float z = pose.Rcw.row(2).dot(x3D) + pose.tcw(2);
        if(z <= 0)
            return false;
        const float u = calibration.fx * (pose.Rcw.row(0).dot(x3D) + pose.tcw(0)) / z + calibration.cx;
        const float v = calibration.fy * (pose.Rcw.row(1).dot(x3D) + pose.tcw(1)) / z + calibration.cy;
        const vec2f error{u - keypoint.pt.x, v - keypoint.pt.y};
        return error.dot(keyframe->features.get_information_2d(index, feature_type) * error)
               <= ErrorModel::params.chi2_mono;
    }

    const Keyframe& keyframe;
    const Pose pose;
    const Calibration& calibration;
};

}

void LocalMapping::create_new_map_points()
{
    const std::vector<Keyframe> neighbors = select_triangulation_neighbors();

    // Match the new keyframe against all its neighbors at once; the loop below reads the pairs back
    matcher_->match_keyframe_to_keyframes(current_keyframe_, neighbors, current_keyframe_->feature_types);

    const KeyframeView view1(current_keyframe_);

    // For each neighbor, place each matched keypoint pair in 3D: from sensor depth when either
    // view has it (cross-validated by the reprojection gates below), by two-view triangulation
    // otherwise.
    for(const Keyframe& neighbor : neighbors)
    {
        const KeyframeView view2(neighbor);
        const mat4f& Tcw1 = view1.pose.Tcw;
        const mat4f& Tcw2 = view2.pose.Tcw;

        const std::map<FeatureType, std::vector<std::pair<size_t, size_t>>> matched_indices =
            FeatureMatcher::get_keyframe_matches(current_keyframe_, neighbor);

        for(const auto& [feature_type, pairs] : matched_indices)
        {
            const auto& inv_depth1_ft = current_keyframe_->features.at(feature_type).inv_depth;
            const auto& inv_depth2_ft = neighbor->features.at(feature_type).inv_depth;
            const auto& keypoints1_ft = current_keyframe_->features.at(feature_type).keypoints;
            const auto& keypoints2_ft = neighbor->features.at(feature_type).keypoints;

            for(const auto& [idx1, idx2] : pairs)
            {
                // Only a pair where neither keypoint has a map point yet is a candidate to triangulate
                if(current_keyframe_->get_map_point(idx1, feature_type) || neighbor->get_map_point(idx2, feature_type))
                    continue;

                const cv::KeyPoint& kp1 = keypoints1_ft[idx1];
                const cv::KeyPoint& kp2 = keypoints2_ft[idx2];

                // Viewing rays and their parallax
                const vec3f xn1 = view1.normalized(kp1);
                const vec3f xn2 = view2.normalized(kp2);
                const vec3f ray1 = view1.pose.Rwc * xn1;
                const vec3f ray2 = view2.pose.Rwc * xn2;
                const float cos_parallax = ray1.dot(ray2) / (ray1.norm() * ray2.norm());

                const float inv_depth1 = inv_depth1_ft[idx1];
                const float inv_depth2 = inv_depth2_ft[idx2];
                const bool have_depth1 = inv_depth1 > 0.0f;
                const bool have_depth2 = inv_depth2 > 0.0f;

                vec3f x3D;
                if(have_depth1 && have_depth2)
                {
                    // Two independent sensor depth readings of the same point: back-project
                    // each from its own keyframe and average. The reprojection gates below
                    // then validate agreement between them in both views.
                    x3D = 0.5f * (view1.back_project(xn1, inv_depth1) + view2.back_project(xn2, inv_depth2));
                }
                else if(have_depth1)
                {
                    // Back-project from this keyframe's own measured depth: no second-view
                    // triangulation, no parallax requirement. The reprojection gate against
                    // the neighbor below still cross-validates it.
                    x3D = view1.back_project(xn1, inv_depth1);
                }
                else if(have_depth2)
                {
                    // Symmetric case: back-project from the neighbor's depth,
                    // cross-validated against this keyframe
                    x3D = view2.back_project(xn2, inv_depth2);
                }
                else if(cos_parallax > 0 && cos_parallax < params.create_new_map_points_max_parallax_cos)
                {
                    // No sensor depth on either side: linear two-view triangulation (SVD)
                    Eigen::Matrix<float, 4, 4> A;
                    A.row(0) = xn1(0) * Tcw1.row(2) - Tcw1.row(0);
                    A.row(1) = xn1(1) * Tcw1.row(2) - Tcw1.row(1);
                    A.row(2) = xn2(0) * Tcw2.row(2) - Tcw2.row(0);
                    A.row(3) = xn2(1) * Tcw2.row(2) - Tcw2.row(1);

                    const Eigen::JacobiSVD<Eigen::Matrix<float, 4, 4>> svd(A, Eigen::ComputeFullV);
                    const Eigen::Vector4f x_h = svd.matrixV().col(3);
                    if(std::abs(x_h(3)) < HOMOGENEOUS_W_EPSILON)
                        continue;   // degenerate solve
                    x3D = x_h.head<3>() / x_h(3);
                }
                else
                    continue; // no depth, and too little parallax to triangulate

                // In front of both cameras, and within the reprojection gate of both keypoints
                if(!view1.observes(x3D, kp1, idx1, feature_type) || !view2.observes(x3D, kp2, idx2, feature_type))
                    continue;

                const Pt map_point = map_->create_monocular_map_point(
                    x3D, current_keyframe_, KeypointIndex(idx1), neighbor, KeypointIndex(idx2), feature_type);
                if(!map_point)
                    continue;   // a slot was taken between the emptiness check and the write
                recent_map_points_.push_back(map_point);
            }
        }
    }

    create_depth_seeded_points();
}

void LocalMapping::create_depth_seeded_points()
{
    // Back-project still-unmatched keypoints straight from the keyframe's own sensor depth.
    // Triangulation only creates points for keypoints with a 2D feature match in a covisible
    // neighbor -- a keypoint with a valid depth reading but no such match (textureless
    // region, repeated pattern, fast motion) is exactly where sensor depth helps most.
    // Same trust policy as the triangulation depth branches: any inv_depth > 0, no range gate.
    const KeyframeView view(current_keyframe_);

    for(const FeatureType feature_type : current_keyframe_->feature_types)
    {
        const auto& inv_depth = current_keyframe_->features.at(feature_type).inv_depth;
        const auto& keypoints = current_keyframe_->features.at(feature_type).keypoints;
        for(size_t idx = 0; idx < inv_depth.size(); idx++)
        {
            if(inv_depth[idx] <= 0.0f)
                continue; // no valid sensor depth at this keypoint
            if(current_keyframe_->get_map_point(idx, feature_type))
                continue; // already has a map point (from tracking, or the triangulation loop)

            const Pt map_point = map_->create_map_point(view.back_project(view.normalized(keypoints[idx]), inv_depth[idx]),
                                                        current_keyframe_, KeypointIndex(idx), feature_type);
            if(!map_point)
                continue;   // a slot was taken between the emptiness check and the write
            recent_map_points_.push_back(map_point);
        }
    }
}

void LocalMapping::search_in_neighbors()
{

    // Fuse target set: the best covisible keyframes plus a few of THEIR best covisible keyframes, each once (a
    // keyframe reachable through two neighbors is fused once, not twice). Collected once for all feature types.
    std::vector<Keyframe> targets;
    std::unordered_set<KeyframeId> target_ids;
    for(const Keyframe& neighbor : current_keyframe_->get_best_covisibility_keyframes(params.search_in_neighbors_keyframes))
    {
        if(neighbor->is_bad() || !target_ids.insert(neighbor->keyId).second)
            continue;
        targets.push_back(neighbor);

        for(const Keyframe& second : neighbor->get_best_covisibility_keyframes(params.search_in_neighbors_second_keyframes))
        {
            if(second->is_bad() || second->keyId == current_keyframe_->keyId || !target_ids.insert(second->keyId).second)
                continue;
            targets.push_back(second);
        }
    }

    for(const FeatureType feature_type : current_keyframe_->feature_types)
    {
        estimate_fuse_max_descriptor_distance(feature_type);

        // Project this keyframe's map points into each target and fuse duplicates.
        // By-value snapshot: get_map_point_matches copies under the keyframe's mutex.
        const std::vector<Pt> map_points = current_keyframe_->get_map_point_matches(feature_type);
        for(const Keyframe& target : targets)
            fuse_into_keyframe(target, map_points, feature_type);

        // Project the targets' map points into this keyframe and fuse duplicates (each point once)
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
        fuse_into_keyframe(current_keyframe_, candidates, feature_type);

        // Refresh the surviving matches (fusion may have replaced points)
        for(const Pt& map_point : current_keyframe_->get_map_point_matches(feature_type))
        {
            if(map_point && !map_point->is_bad())
            {
                map_point->compute_distinctive_descriptors();
                map_point->update_normal_and_depth();
            }
        }

    }

    // Update links in the covisibility graph
    current_keyframe_->update_connections();
}

// How far apart in descriptor space a correct match lies, for one feature type: for each map
// point of the current keyframe, the distance from its distinctive descriptor to the descriptor
// of every other observation (the observation it was chosen from, at distance 0, is skipped once).
// The fuse_descriptor_quantile quantile of these becomes the map's fuse descriptor distance; a
// keyframe whose points have no second observation leaves the previous estimate in place.
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
            if(!self_skipped && distance == Descriptor_Distance_Type(0))
            {
                self_skipped = true;
                continue;
            }
            distances.push_back(distance);
        }
    }

    if(distances.empty())
        return;

    const float quantile = std::clamp(params.fuse_descriptor_quantile, 0.0f, 1.0f);
    const size_t index = std::min(distances.size() - 1, static_cast<size_t>(quantile * distances.size()));
    std::nth_element(distances.begin(), distances.begin() + static_cast<std::ptrdiff_t>(index), distances.end());
    map_->set_fuse_max_descriptor_distance(feature_type, distances[index]);
}

// Fuses `map_points` into `keyframe`'s keypoints of `feature_type`. Each point the keyframe does not already
// observe is projected with the keyframe's pose (project_if_visible: depth, image, scale range, viewing angle)
// and matched by FeatureMatcher's gated mutual search, every keypoint a target whether it holds a point or not,
// at most the map's fuse descriptor distance apart. A match on an empty keypoint adds the observation; a match
// on a keypoint holding another point merges the two, the one with more observations surviving. A point or
// keyframe slot changed by an earlier merge of this call is re-checked.
void LocalMapping::fuse_into_keyframe(const Keyframe& keyframe, const std::vector<Pt>& map_points,
                                      const FeatureType feature_type)
{
    // No estimate yet for this type (no point with two observations): nothing is fused
    const std::optional<Descriptor_Distance_Type> max_descriptor_distance =
        map_->get_fuse_max_descriptor_distance(feature_type);
    if(!max_descriptor_distance)
        return;

    // One pose for all points: get_pose() takes the keyframe's pose lock once
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
            continue;
        candidates.push_back(pt);
        projections.push_back(projection->uv);
    }

    std::vector<Pt> matches(keyframe->features.at(feature_type).count());
    matcher_->match_map_points_to_keyframe(keyframe, candidates, projections, feature_type, matches,
                                           ErrorModel::params.chi2_mono, *max_descriptor_distance);

    for(size_t idx = 0; idx < matches.size(); idx++)
    {
        const Pt& pt = matches[idx];
        if(!pt || pt->is_bad() || pt->is_in_keyframe(keyframe))
            continue;

        const KeypointIndex index = static_cast<KeypointIndex>(idx);
        const Pt existing = keyframe->get_map_point(index, feature_type);
        if(existing)
        {
            if(existing->is_bad())
                continue;
            if(existing->number_of_observations() > pt->number_of_observations())
                pt->replace(existing);
            else
                existing->replace(pt);
        }
        // Publish first, observe only if the slot was won: another thread can fill it after
        // get_map_point. pt already has a descriptor, so publishing before observing is safe.
        else if(keyframe->add_map_point(pt, index))
            pt->add_observation(keyframe, index);
    }
}

void LocalMapping::local_bundle_adjustment()
{
    // Local BA needs more keyframes than this in the map
    if(map_->keyframes_in_map() <= static_cast<size_t>(Optimizer::params.local_ba_min_keyframes))
        return;
    Optimizer::local_bundle_adjustment(current_keyframe_, map_);
}

void LocalMapping::cull_keyframes()
{
    // The joint-information culling maths ("gram-greedy") lives in placecell
    // (PlaceCell::cull_keyframes); this adapter supplies the host side: the kernel's
    // refresh, parameters, the covisibility window for the local scope, reconciliation
    // of externally removed keyframes, execution of each cull (set_bad_flag), and logging.
    // No kernel (InformationKernel: megaloc in a build without MegaLoc): nothing is culled
    if(!keyframe_information_ || keyframe_information_->place_cell().size() == 0){
        if(!warned_no_kernel_){
            AF_WARN("[LocalMapping] no keyframe information kernel (InformationKernel: megaloc needs a build with "
                    "MegaLoc): keyframes are never culled; set InformationKernel: covisibility");
            warned_no_kernel_ = true;
        }
        return;
    }
    placecell::PlaceCell& place_cell = keyframe_information_->place_cell();

    // Resolve every stored row back to a keyframe. A row that no longer resolves (or
    // resolves to a bad keyframe) was removed outside this culler -- tell placecell
    // so it becomes culling history there too.
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
        const auto it = keyframe_by_frame_id.find(FrameId(id));
        if(it == keyframe_by_frame_id.end() || it->second->is_bad())
            place_cell.set_culled(id);
    }

    // Bring the kernel up to date with the map (covisibility: every alive keyframe's
    // current map-point set, after this cycle's point culling / creation / fusion;
    // megaloc: nothing to do)
    keyframe_information_->refresh(alive_keyframes);

    placecell::PlaceCell::CullParameters cull_parameters;
    cull_parameters.method = KEYFRAME_CULLING_METHOD;
    cull_parameters.objective = params.keyframe_culling_objective;
    cull_parameters.max_unexplained = params.keyframe_culling_max_unexplained.load();   // live: Viewer slider
    cull_parameters.centred = params.keyframe_culling_centred;
    cull_parameters.min_keyframes = params.keyframe_culling_min_keyframes;
    // min_age keyframes back from the current one, measured in stored rows; >= 1 so
    // the current keyframe itself is never culled (KF0 is covered by protect_first)
    cull_parameters.protect_last = std::max(params.keyframe_culling_min_age, 1);
    cull_parameters.max_per_call = params.keyframe_culling_max_per_call;

    const bool local_scope = (params.keyframe_culling_scope == "local");
    std::vector<placecell::PlaceCell::ExternalId> window;
    if(local_scope){
        window.push_back(current_keyframe_->frame_id);
        for(const Keyframe& keyframe : current_keyframe_->get_covisible_keyframes())
            window.push_back(keyframe->frame_id);
    }

    const auto try_cull = [&](const placecell::PlaceCell::ExternalId id) -> bool {
        const auto it = keyframe_by_frame_id.find(FrameId(id));
        if(it == keyframe_by_frame_id.end())
            return false;
        it->second->set_bad_flag();
        // Deferred by pin() (loop closing holds it): leave it alive; placecell
        // never retries a refused candidate within the same call
        return it->second->is_bad();
    };

    const placecell::PlaceCell::CullReport report =
        place_cell.cull_keyframes(cull_parameters, try_cull, local_scope ? &window : nullptr);

    auto fmt = [](double x) { std::ostringstream s; s << std::fixed << std::setprecision(3) << x; return s.str(); };
    for(const auto& culled : report.culled){
        const auto it = keyframe_by_frame_id.find(FrameId(culled.id));
        AF_INFO("[VPR] culled keyframe " << (it != keyframe_by_frame_id.end() ? std::to_string(it->second->keyId) : "?")
                << " (frame " << culled.id
                << "): unique information " << fmt(culled.unique_information)
                << ", worst unexplained keyframe after the cull " << fmt(culled.worst_unexplained_after)
                << ", alive keyframes " << culled.alive_after);
    }
    const double tau = double(cull_parameters.max_unexplained);
    if(!report.culled.empty()){
        AF_INFO("[VPR] cull_keyframes[" << params.keyframe_culling_scope << (params.keyframe_culling_centred ? ", centred" : ", raw")
                << "]: culled " << report.culled.size() << " of " << report.candidates
                << " candidates (" << report.alive_after << " alive in scope, " << report.views_total
                << " keyframes ever); worst unexplained keyframe "
                << fmt(report.worst_history) << " (tau " << fmt(tau) << ")"
                << (report.reached_max_per_call ? " [per-call limit reached]" : ""));
    } else {
        // Threshold lowered below earlier culls: report the over-budget history once per change
        if(report.history_over_budget != last_over_budget_){
            if(report.history_over_budget > 0)
                AF_INFO("[VPR] cull_keyframes: " << report.history_over_budget << " culled keyframes are above tau (max unexplained "
                        << fmt(report.worst_history) << " > tau " << fmt(tau) << ") — only keyframes that do not explain them can be culled");
            last_over_budget_ = report.history_over_budget;
        }
    }
}

}
