/**
 * Local Mapping thread: the map back-end of AllFeature-VSLAM. Consumes the
 * keyframes Tracking inserts (run/process_keyframe), culls recent map points,
 * creates new ones (sensor depth and/or two-view triangulation), fuses
 * duplicates with covisible neighbors, runs the local bundle adjustment, and
 * culls redundant keyframes. Auxiliary members (parameter loading, thread
 * synchronization, the online VPR matrix, profiling) live in
 * LocalMapping_aux.{h,cc}.
 */
#include "LocalMapping.h"
#include "LocalMapping_aux.h"

#include <placecell/placecell.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>

#include <Eigen/Dense>

#include "FeatureMatcher.h"
#include "LoopClosing.h"
#include "Optimizer.h"
#include "Report.h"
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
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
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
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }

    set_finished();
}

// One full mapping iteration for the keyframe at the head of the queue. Every stage
// runs for every keyframe: the refinement stages (fuse, local BA, keyframe culling)
// are no longer skipped when Tracking has meanwhile queued another keyframe. Each
// stage function records its own PROFILING_EXHAUSTIVE histogram.
void LocalMapping::process_keyframe()
{
    LocalMappingProfiler profiler{};

    // Global descriptor, map-point associations, covisibility graph, map insertion
    process_new_keyframe();

    // Enforce the quality checks on recently added map points
    cull_map_points();

    create_new_map_points();

    // The matches against the neighbors have been triangulated from; nothing reads them again
    FeatureMatcher::clear_keyframe_matches(current_keyframe_);

    // Find more matches in neighbor keyframes and fuse point duplications
    search_in_neighbors();

    local_bundle_adjustment();

#ifdef ALLFEATURE_REPORT
    // Report the keyframe's place in the covisibility graph once it has been fused with its
    // neighbors and adjusted, and how many map points it observes. Inside the #ifdef because it
    // copies the connection list and the map-point vectors only to count them.
    {
        const std::vector<Connection> connections = current_keyframe_->get_covisible_connections();
        long weight_total = 0;
        int weight_max = 0;
        for(const Connection& connection : connections)
        {
            weight_total += connection.weight;
            weight_max = std::max(weight_max, connection.weight);
        }
        AF_REPORT("connections.count", "", connections.size());
        AF_REPORT("connections.weight_total", "", weight_total);
        AF_REPORT("connections.weight_max", "", weight_max);
        for(const FeatureType feature_type : current_keyframe_->feature_types)
        {
            size_t num_points = 0;
            for(const Pt& point : current_keyframe_->get_map_point_matches(feature_type))
                if(point && !point->is_bad())
                    num_points++;
            AF_REPORT("keyframe.map_points", get_feature(feature_type).get_feature_name(), num_points);
        }
    }
#endif

    cull_keyframes();

    loop_closer_->insert_keyframe(current_keyframe_);

    // Median excludes the current iteration (recorded below), matching the original order
    if(viewer_)
        viewer_->set_runLocalMapping_time_median(map_median(local_mapping_times_));
    profiler.iteration_done(local_mapping_times_);
    log_profile();
}

void LocalMapping::process_new_keyframe()
{
    {
        std::lock_guard<std::mutex> lock(new_keyframes_mutex_);
        current_keyframe_ = new_keyframes_.front();
        new_keyframes_.pop_front();
    }
    AF_REPORT_CONTEXT("local_mapping", current_keyframe_->frame_id);

    // Global descriptor (VPR image embedding, backend-dependent)
    current_keyframe_->compute_global_descriptor();

    // Register the keyframe as an observer of the map points Tracking matched into it
    // (add_observation also refreshes the point's descriptor and normal/depth). Points
    // that already observe this keyframe -- created WITH it by the initializer -- enter
    // the probation list instead (cull_map_points).
    for(const FeatureType feature_type : current_keyframe_->feature_types)
    {
        // By-value snapshot: get_map_point_matches copies under the keyframe's mutex
        const std::vector<Pt> map_points = current_keyframe_->get_map_point_matches(feature_type);
        for(size_t i = 0; i < map_points.size(); i++)
        {
            const Pt& map_point = map_points[i];
            if(!map_point || map_point->is_bad())
                continue;
            if(!map_point->is_in_keyframe(current_keyframe_))
                map_point->add_observation(current_keyframe_, i);
            else
                recent_map_points_.push_back(map_point);
        }
    }

    // Update links in the covisibility graph
    current_keyframe_->update_connections();

    map_->add_keyframe(current_keyframe_);

    // Register the keyframe in the information kernel so it explains the next frames
    // right away (megaloc: no-op, compute_global_descriptor() already stored the
    // descriptor; covisibility: its current map-point set)
    if(keyframe_information_)
        keyframe_information_->on_keyframe_processed(current_keyframe_);
}

void LocalMapping::cull_map_points()
{
    // Probation for recently added map points: cull the ones that fail their quality
    // checks, and graduate survivors out of the probation list after
    // map_point_culling_probation_age keyframes (they stay in the map for good).
    const int current_id = int(current_keyframe_->keyId);

    recent_map_points_.remove_if([&](const Pt& map_point)
    {
        if(map_point->is_bad())
            return true; // already culled elsewhere: just drop it from the list

        // Visible (in frustum) often but actually matched rarely: unreliable point.
        // Applies to every feature type uniformly (issue #11: an earlier version
        // exempted all but feature_types[0]).
        if(map_point->get_found_ratio() < params.map_point_culling_min_found_ratio)
        {
            map_point->set_bad_flag();
            return true;
        }

        // Old enough to have been re-observed, but wasn't
        // (number_of_observations counts depth-verified observations twice)
        const int age = current_id - int(map_point->first_keyframe_id);
        if(age >= params.map_point_culling_observation_test_age
           && map_point->number_of_observations() <= params.map_point_culling_min_observations)
        {
            map_point->set_bad_flag();
            return true;
        }

        return age >= params.map_point_culling_probation_age;
    });
}

// The covisible keyframes to triangulate against: the create_new_map_points_keyframes most
// covisible ones THAT HAVE ENOUGH BASELINE to the current keyframe, relative to their scene
// depth -- a near-zero-parallax pair only yields ill-conditioned triangulations. The covisible
// keyframes are walked from most to least covisible until that many pass, so a neighbor that is
// too close is replaced by the next one down the list rather than just dropped. Selected before
// any matching, so no neighbor is matched and filtered only to be skipped.
std::vector<Keyframe> LocalMapping::select_triangulation_neighbors() const
{
    const std::vector<Keyframe> covisible = current_keyframe_->get_covisible_keyframes();   // most covisible first
    const auto [Tcw1, Twc1, Rcw1, Rwc1, tcw1, twc1] = current_keyframe_->get_pose();

    const size_t num_wanted = static_cast<size_t>(std::max(params.create_new_map_points_keyframes, 0));
    std::vector<Keyframe> neighbors;
    neighbors.reserve(num_wanted);
    [[maybe_unused]] size_t num_examined = 0;
    for(const Keyframe& candidate : covisible)
    {
        if(neighbors.size() >= num_wanted)
            break;
        num_examined++;

        const auto [Tcw2, Twc2, Rcw2, Rwc2, tcw2, twc2] = candidate->get_pose();
        const float baseline = (twc2 - twc1).norm();
        // No map points, or a scene sitting on the camera plane: nothing to measure the
        // baseline against. The old form divided by the -1 sentinel and skipped because the
        // result came out negative -- and divided by a legitimate 0 to get inf, which did NOT
        // skip.
        const std::optional<float> scene_depth = candidate->compute_scene_median_depth();
        if(!scene_depth || *scene_depth <= 0.0f
           || baseline / *scene_depth < params.create_new_map_points_min_baseline_depth_ratio)
            continue;
        neighbors.push_back(candidate);
    }

    // How far down the covisibility list the selection had to go, and how many it found
    AF_REPORT("neighbors.covisible", "", covisible.size());
    AF_REPORT("neighbors.examined", "", num_examined);
    AF_REPORT("neighbors.enough_baseline", "", neighbors.size());
    return neighbors;
}

void LocalMapping::create_new_map_points()
{
    StageTimer timer{};
    AF_REPORT_SCOPE();

    const std::vector<Keyframe> neighbors = select_triangulation_neighbors();

    const auto [Tcw1, Twc1, Rcw1, Rwc1, tcw1, twc1] = current_keyframe_->get_pose();
    const float fx1 = current_keyframe_->calibration->fx, fy1 = current_keyframe_->calibration->fy;
    const float cx1 = current_keyframe_->calibration->cx, cy1 = current_keyframe_->calibration->cy;
    const float invfx1 = current_keyframe_->calibration->invfx, invfy1 = current_keyframe_->calibration->invfy;

    // Match the new keyframe against all its neighbors at once; the loop below reads the pairs back
    matcher_->match_keyframe_to_keyframes(current_keyframe_, neighbors, current_keyframe_->feature_types);

    // What becomes of the matched pairs, per feature type; reported at the end
    struct TriangulationCounts
    {
        size_t pairs{0};          // matched pairs read back
        size_t candidates{0};     // neither keypoint has a map point yet
        size_t no_parallax{0};    // no depth and too little parallax, or a degenerate solve
        size_t behind_camera{0};
        size_t reprojection{0};   // rejected by the reprojection gate in either view
        size_t slot_taken{0};     // a keypoint got a map point between the check and the write
        size_t created{0};
    };
    std::map<FeatureType, TriangulationCounts> triangulation_counts;

    // For each neighbor, place each matched keypoint pair in 3D: from sensor depth when either
    // view has it (cross-validated by the reprojection gates below), by two-view triangulation
    // otherwise.
    for(const Keyframe& neighbor : neighbors)
    {
        const auto [Tcw2, Twc2, Rcw2, Rwc2, tcw2, twc2] = neighbor->get_pose();
        const float fx2 = neighbor->calibration->fx, fy2 = neighbor->calibration->fy;
        const float cx2 = neighbor->calibration->cx, cy2 = neighbor->calibration->cy;
        const float invfx2 = neighbor->calibration->invfx, invfy2 = neighbor->calibration->invfy;

        const std::map<FeatureType, std::vector<std::pair<size_t, size_t>>> matched_indices =
            FeatureMatcher::get_keyframe_matches(current_keyframe_, neighbor);

        for(const auto& [feature_type, pairs] : matched_indices)
        {
            const auto& inv_depth1_ft = current_keyframe_->features.at(feature_type).inv_depth;
            const auto& inv_depth2_ft = neighbor->features.at(feature_type).inv_depth;
            const auto& keypoints1_ft = current_keyframe_->features.at(feature_type).keypoints;
            const auto& keypoints2_ft = neighbor->features.at(feature_type).keypoints;
            TriangulationCounts& counts = triangulation_counts[feature_type];
            counts.pairs += pairs.size();

            for(const auto& [idx1, idx2] : pairs)
            {
                // Only a pair where neither keypoint has a map point yet is a candidate to triangulate
                if(current_keyframe_->get_map_point(idx1, feature_type) || neighbor->get_map_point(idx2, feature_type))
                    continue;
                counts.candidates++;

                const cv::KeyPoint& kp1 = keypoints1_ft[idx1];
                const cv::KeyPoint& kp2 = keypoints2_ft[idx2];

                // Viewing rays and their parallax
                const vec3f xn1{(kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0f};
                const vec3f xn2{(kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0f};
                const vec3f ray1 = Rwc1 * xn1;
                const vec3f ray2 = Rwc2 * xn2;
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
                    const vec3f x3D_1 = Rwc1 * (xn1 / inv_depth1) + twc1;
                    const vec3f x3D_2 = Rwc2 * (xn2 / inv_depth2) + twc2;
                    x3D = 0.5f * (x3D_1 + x3D_2);
                }
                else if(have_depth1)
                {
                    // Back-project from this keyframe's own measured depth: no second-view
                    // triangulation, no parallax requirement. The reprojection gate against
                    // the neighbor below still cross-validates it.
                    x3D = Rwc1 * (xn1 / inv_depth1) + twc1;
                }
                else if(have_depth2)
                {
                    // Symmetric case: back-project from the neighbor's depth,
                    // cross-validated against this keyframe
                    x3D = Rwc2 * (xn2 / inv_depth2) + twc2;
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
                    {
                        counts.no_parallax++;
                        continue;
                    }
                    x3D = x_h.head<3>() / x_h(3);
                }
                else
                {
                    counts.no_parallax++;
                    continue; // no depth, and too little parallax to triangulate
                }

                // The point must lie in front of both cameras
                const float z1 = Rcw1.row(2).dot(x3D) + tcw1(2);
                const float z2 = Rcw2.row(2).dot(x3D) + tcw2(2);
                if(z1 <= 0 || z2 <= 0)
                {
                    counts.behind_camera++;
                    continue;
                }

                // Reprojection gate in this keyframe
                const float u1 = fx1 * (Rcw1.row(0).dot(x3D) + tcw1(0)) / z1 + cx1;
                const float v1 = fy1 * (Rcw1.row(1).dot(x3D) + tcw1(1)) / z1 + cy1;
                const float err_x1 = u1 - kp1.pt.x;
                const float err_y1 = v1 - kp1.pt.y;
                if(err_x1 * err_x1 + err_y1 * err_y1 > ErrorModel::params.chi2_mono * current_keyframe_->features.get_sigma2(idx1, feature_type))
                {
                    counts.reprojection++;
                    continue;
                }

                // Reprojection gate in the neighbor
                const float u2 = fx2 * (Rcw2.row(0).dot(x3D) + tcw2(0)) / z2 + cx2;
                const float v2 = fy2 * (Rcw2.row(1).dot(x3D) + tcw2(1)) / z2 + cy2;
                const float err_x2 = u2 - kp2.pt.x;
                const float err_y2 = v2 - kp2.pt.y;
                if(err_x2 * err_x2 + err_y2 * err_y2 > ErrorModel::params.chi2_mono * neighbor->features.get_sigma2(idx2, feature_type))
                {
                    counts.reprojection++;
                    continue;
                }

                const Pt map_point = map_->create_monocular_map_point(
                    x3D, current_keyframe_, KeypointIndex(idx1), neighbor, KeypointIndex(idx2), feature_type);
                if(!map_point)
                {
                    counts.slot_taken++;
                    continue;   // a slot was taken between the emptiness check and the write
                }
                recent_map_points_.push_back(map_point);
                counts.created++;
            }
        }
    }

    for([[maybe_unused]] const auto& [feature_type, counts] : triangulation_counts)
    {
        AF_REPORT("triangulation.pairs", get_feature(feature_type).get_feature_name(), counts.pairs);
        AF_REPORT("triangulation.candidates", get_feature(feature_type).get_feature_name(), counts.candidates);
        AF_REPORT("triangulation.no_parallax", get_feature(feature_type).get_feature_name(), counts.no_parallax);
        AF_REPORT("triangulation.behind_camera", get_feature(feature_type).get_feature_name(), counts.behind_camera);
        AF_REPORT("triangulation.reprojection", get_feature(feature_type).get_feature_name(), counts.reprojection);
        AF_REPORT("triangulation.slot_taken", get_feature(feature_type).get_feature_name(), counts.slot_taken);
        AF_REPORT("triangulation.created", get_feature(feature_type).get_feature_name(), counts.created);
    }

    create_depth_seeded_points();

    timer.record(create_new_map_points_times_);
}

void LocalMapping::create_depth_seeded_points()
{
    // Back-project still-unmatched keypoints straight from the keyframe's own sensor depth.
    // Triangulation only creates points for keypoints with a 2D feature match in a covisible
    // neighbor -- a keypoint with a valid depth reading but no such match (textureless
    // region, repeated pattern, fast motion) is exactly where sensor depth helps most.
    // Same trust policy as the triangulation depth branches: any inv_depth > 0, no range gate.
    const auto [Tcw1, Twc1, Rcw1, Rwc1, tcw1, twc1] = current_keyframe_->get_pose();
    const float cx1 = current_keyframe_->calibration->cx, cy1 = current_keyframe_->calibration->cy;
    const float invfx1 = current_keyframe_->calibration->invfx, invfy1 = current_keyframe_->calibration->invfy;

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

            const cv::KeyPoint& kp = keypoints[idx];
            const vec3f xn{(kp.pt.x - cx1) * invfx1, (kp.pt.y - cy1) * invfy1, 1.0f};
            const Pt map_point = map_->create_map_point(Rwc1 * (xn / inv_depth[idx]) + twc1,
                                                        current_keyframe_, KeypointIndex(idx), feature_type);
            if(!map_point)
                continue;   // a slot was taken between the emptiness check and the write
            recent_map_points_.push_back(map_point);
        }
    }
}

void LocalMapping::search_in_neighbors()
{
    AF_REPORT_SCOPE();
    StageTimer timer{};

    // Fuse target set: the best covisible keyframes plus a few of THEIR best covisible
    // keyframes, deduplicated via the fuse-target mark. Collected ONCE for all feature
    // types: an earlier per-feature-type version re-ran this collection each call, and
    // the mark made every call after the first come up empty -- silently fusing only
    // the first feature type. (Second neighbors are marked too, so a keyframe reachable
    // through two different neighbors is fused once, not twice.)
    std::vector<Keyframe> targets;
    for(const Keyframe& neighbor : current_keyframe_->get_best_covisibility_keyframes(params.search_in_neighbors_keyframes))
    {
        if(neighbor->is_bad() || neighbor->fuse_target_for_keyframe == current_keyframe_->keyId)
            continue;
        neighbor->fuse_target_for_keyframe = current_keyframe_->keyId;
        targets.push_back(neighbor);

        for(const Keyframe& second : neighbor->get_best_covisibility_keyframes(params.search_in_neighbors_second_keyframes))
        {
            if(second->is_bad() || second->fuse_target_for_keyframe == current_keyframe_->keyId
               || second->keyId == current_keyframe_->keyId)
                continue;
            second->fuse_target_for_keyframe = current_keyframe_->keyId;
            targets.push_back(second);
        }
    }

    AF_REPORT("fuse.targets", "", targets.size());

    for(const FeatureType feature_type : current_keyframe_->feature_types)
    {
        estimate_fuse_max_descriptor_distance(feature_type);

        // Project this keyframe's map points into each target and fuse duplicates.
        // By-value snapshot: get_map_point_matches copies under the keyframe's mutex.
        const std::vector<Pt> map_points = current_keyframe_->get_map_point_matches(feature_type);
        [[maybe_unused]] FuseCounts into_targets{};
        for(const Keyframe& target : targets)
        {
            const FuseCounts counts = fuse_into_keyframe(target, map_points, feature_type);
            into_targets.merged += counts.merged;
            into_targets.observed += counts.observed;
        }

        // Project the targets' map points into this keyframe and fuse duplicates
        std::vector<Pt> candidates;
        candidates.reserve(targets.size() * map_points.size());
        for(const Keyframe& target : targets)
        {
            for(const Pt& candidate : target->get_map_point_matches(feature_type))
            {
                if(!candidate || candidate->is_bad()
                   || candidate->fuse_candidate_for_keyframe == current_keyframe_->keyId)
                    continue;
                candidate->fuse_candidate_for_keyframe = current_keyframe_->keyId;
                candidates.push_back(candidate);
            }
        }
        [[maybe_unused]] const FuseCounts into_current = fuse_into_keyframe(current_keyframe_, candidates, feature_type);

        AF_REPORT("fuse.points", get_feature(feature_type).get_feature_name(),
                  std::count_if(map_points.begin(), map_points.end(), [](const Pt& pt) { return pt != nullptr; }));
        AF_REPORT("fuse.into_targets", get_feature(feature_type).get_feature_name(), into_targets.merged + into_targets.observed);
        AF_REPORT("fuse.into_targets_merged", get_feature(feature_type).get_feature_name(), into_targets.merged);
        AF_REPORT("fuse.candidates", get_feature(feature_type).get_feature_name(), candidates.size());
        AF_REPORT("fuse.into_current", get_feature(feature_type).get_feature_name(), into_current.merged + into_current.observed);
        AF_REPORT("fuse.into_current_merged", get_feature(feature_type).get_feature_name(), into_current.merged);

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

    timer.record(search_in_neighbors_times_);
}

// Fuses `map_points` into `keyframe`'s keypoints of `feature_type`. Each point the keyframe does
// not already observe is projected with the keyframe's pose (project_if_visible: depth, image,
// scale range, viewing angle) and matched by FeatureMatcher's gated mutual search, every keypoint
// a target whether it holds a point or not, at most the map's fuse descriptor distance apart. A match on an empty keypoint adds the observation; a
// match on a keypoint holding another point merges the two, the one with more observations
// surviving. A point or keyframe slot changed by an earlier merge of this call is re-checked.
// How far apart in descriptor space a correct match lies, for one feature type: for each map
// point of the current keyframe, the distance from its distinctive descriptor to the descriptor
// of every other observation (the observation it was chosen from, at distance 0, is skipped once).
// The fuse_descriptor_quantile quantile of these becomes the map's fuse descriptor distance; a
// keyframe whose points have no second observation leaves the previous estimate in place.
// Reports descdist.count and descdist.fuse_max.
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

    AF_REPORT("descdist.count", get_feature(feature_type).get_feature_name(), distances.size());
    if(distances.empty())
        return;

    const float quantile = std::clamp(params.fuse_descriptor_quantile, 0.0f, 1.0f);
    const size_t index = std::min(distances.size() - 1, static_cast<size_t>(quantile * distances.size()));
    std::nth_element(distances.begin(), distances.begin() + static_cast<std::ptrdiff_t>(index), distances.end());
    map_->set_fuse_max_descriptor_distance(feature_type, distances[index]);
    AF_REPORT("descdist.fuse_max", get_feature(feature_type).get_feature_name(), distances[index]);
}

LocalMapping::FuseCounts LocalMapping::fuse_into_keyframe(const Keyframe& keyframe, const std::vector<Pt>& map_points,
                                                          const FeatureType feature_type)
{
    // No estimate yet for this type (no point with two observations): nothing is fused
    const std::optional<Descriptor_Distance_Type> max_descriptor_distance =
        map_->get_fuse_max_descriptor_distance(feature_type);
    if(!max_descriptor_distance)
        return {};

    // One pose for all points: get_pose() takes the keyframe's pose lock once
    const Pose pose = keyframe->get_pose();

    std::vector<Pt> candidates;
    std::vector<cv::Point2f> projections;
    for(const Pt& pt : map_points)
    {
        if(!pt || pt->is_bad() || pt->featureType != feature_type || pt->is_in_keyframe(keyframe))
            continue;
        const std::optional<Projection> projection =
            project_if_visible(pose, *keyframe->calibration, pt, FUSE_VIEWING_COS_LIMIT);
        if(!projection)
            continue;
        candidates.push_back(pt);
        projections.push_back(projection->uv);
    }

    std::vector<Pt> matches(keyframe->features.at(feature_type).count());
    matcher_->match_map_points_to_keyframe(keyframe, candidates, projections, feature_type, matches,
                                           ErrorModel::params.chi2_mono, *max_descriptor_distance);

    FuseCounts counts{};
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
            counts.merged++;
        }
        // Publish first, observe only if the slot was won: another thread can fill it after
        // get_map_point. pt already has a descriptor, so publishing before observing is safe.
        else if(keyframe->add_map_point(pt, index))
        {
            pt->add_observation(keyframe, index);
            counts.observed++;
        }
    }
    return counts;
}

void LocalMapping::local_bundle_adjustment()
{
    // Local BA needs more keyframes than this in the map
    if(map_->keyframes_in_map() <= static_cast<size_t>(Optimizer::params.local_ba_min_keyframes))
        return;

    StageTimer timer{};
    Optimizer::local_bundle_adjustment(current_keyframe_, map_);
    timer.record(local_ba_times_);
}

void LocalMapping::cull_keyframes()
{
    // The joint-information culling maths ("gram-greedy") lives in placecell
    // (PlaceCell::cull_keyframes); this adapter supplies the host side: the kernel's
    // refresh, parameters, the covisibility window for the local scope, reconciliation
    // of externally removed keyframes, execution of each cull (set_bad_flag), and logging.
    // No kernel (InformationKernel: megaloc in a build without MegaLoc): nothing is culled
    if(!keyframe_information_ || keyframe_information_->place_cell().size() == 0){
        static bool warned{false};
        if(!warned){
            AF_WARN("[LocalMapping] no keyframe information kernel (InformationKernel: megaloc needs a build with "
                    "MegaLoc): keyframes are never culled; set InformationKernel: covisibility");
            warned = true;
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
    cull_parameters.method = "gram-greedy";
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
        static int last_over_budget{-1};
        if(report.history_over_budget != last_over_budget){
            if(report.history_over_budget > 0)
                AF_INFO("[VPR] cull_keyframes: " << report.history_over_budget << " culled keyframes are above tau (max unexplained "
                        << fmt(report.worst_history) << " > tau " << fmt(tau) << ") — only keyframes that do not explain them can be culled");
            last_over_budget = report.history_over_budget;
        }
    }
}

} // namespace AF_VSLAM
