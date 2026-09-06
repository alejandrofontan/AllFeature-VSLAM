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
#include "Utils.h"
#include "afvslam_log.hpp"

namespace AF_VSLAM
{

LocalMapping::LocalMapping(std::shared_ptr<Map> map, const std::vector<FeatureType>& feature_types,
                           const int image_width, const int image_height):
    map_(std::move(map)),
    matcher_(std::make_shared<FeatureMatcher>(image_width, image_height, feature_types, "LocalMapping"))
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

// One full mapping iteration for the keyframe at the head of the queue. The first
// stages always run; the refinement stages (fuse, local BA, keyframe culling) are
// skipped when Tracking has meanwhile queued another keyframe -- draining the queue
// first bounds keyframe latency and returns the busy flag to Tracking sooner.
void LocalMapping::process_keyframe()
{
    LocalMappingProfiler profiler{};

    // Global descriptor, map-point associations, covisibility graph, map insertion
    process_new_keyframe();

    // Enforce the quality checks on recently added map points
    cull_map_points();

    {
        StageTimer timer{};
        create_new_map_points();
        timer.record(create_new_map_points_times_);
    }

    if(!has_new_keyframes())
    {
        // Find more matches in neighbor keyframes and fuse point duplications
        StageTimer timer{};
        search_in_neighbors();
        timer.record(search_in_neighbors_times_);
    }

    if(!has_new_keyframes())
    {
        if(map_->keyframes_in_map() > LOCAL_BA_MIN_KEYFRAMES)
        {
            StageTimer timer{};
            Optimizer::LocalBundleAdjustment(current_keyframe_, map_);
            timer.record(local_ba_times_);
        }
        cull_keyframes();
    }

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

    // Global descriptor (VPR image embedding, backend-dependent)
    current_keyframe_->compute_global_descriptor();

    // Register the keyframe as an observer of the map points Tracking matched into it
    // (add_observation also refreshes the point's descriptor and normal/depth). Points
    // that already observe this keyframe -- created WITH it by the initializer -- enter
    // the probation list instead (cull_map_points).
    for(const FeatureType feature_type : current_keyframe_->featureTypes)
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

    // The online keyframe VPR kernel needs no explicit growth here anymore: placecell
    // grew it when compute_global_descriptor() stored this keyframe's descriptor.
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
        // exempted all but featureTypes[0]).
        if(map_point->get_found_ratio() < params.map_point_culling_min_found_ratio)
        {
            map_point->set_bad_flag();
            return true;
        }

        // Old enough to have been re-observed, but wasn't
        // (number_of_observations counts depth-verified observations twice)
        const int age = current_id - int(map_point->mnFirstKFid);
        if(age >= params.map_point_culling_observation_test_age
           && map_point->number_of_observations() <= params.map_point_culling_min_observations)
        {
            map_point->set_bad_flag();
            return true;
        }

        return age >= params.map_point_culling_probation_age;
    });
}

void LocalMapping::create_new_map_points()
{
    const std::vector<Keyframe> neighbors =
        current_keyframe_->get_best_covisibility_keyframes(params.create_new_map_points_keyframes);

    mat4f Twc1, Tcw1;
    mat3f Rwc1, Rcw1;
    vec3f twc1, tcw1;
    current_keyframe_->getFullPose(Twc1, Rwc1, twc1, Tcw1, Rcw1, tcw1);

    float fx1, fy1, cx1, cy1, invfx1, invfy1;
    current_keyframe_->getFullIntrinsics(fx1, fy1, cx1, cy1, invfx1, invfy1);

    cache_neighbor_matches(neighbors);

    // For each neighbor with enough baseline, place each matched keypoint pair in 3D:
    // from sensor depth when either view has it (cross-validated by the reprojection
    // gates below), by two-view triangulation otherwise.
    for(const Keyframe& neighbor : neighbors)
    {
        mat4f Twc2, Tcw2;
        mat3f Rwc2, Rcw2;
        vec3f twc2, tcw2;
        neighbor->getFullPose(Twc2, Rwc2, twc2, Tcw2, Rcw2, tcw2);

        float fx2, fy2, cx2, cy2, invfx2, invfy2;
        neighbor->getFullIntrinsics(fx2, fy2, cx2, cy2, invfx2, invfy2);

        // Discard neighbors whose baseline is too short relative to their scene depth:
        // a near-zero-parallax pair only yields ill-conditioned triangulations
        const float baseline = (twc2 - twc1).norm();
        if(baseline / neighbor->compute_scene_median_depth(2) < params.create_new_map_points_min_baseline_depth_ratio)
            continue;

        std::map<FeatureType, std::vector<std::pair<size_t, size_t>>> matched_indices;
        matcher_->match_keyframes_for_triangulation(current_keyframe_, neighbor, matched_indices,
                                                    current_keyframe_->featureTypes);

        for(const auto& [feature_type, pairs] : matched_indices)
        {
            const auto& inv_depth1_ft = current_keyframe_->inv_depth.at(feature_type);
            const auto& inv_depth2_ft = neighbor->inv_depth.at(feature_type);
            const auto& keypoints1_ft = current_keyframe_->keypoints.at(feature_type);
            const auto& keypoints2_ft = neighbor->keypoints.at(feature_type);

            for(const auto& [idx1, idx2] : pairs)
            {
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
                        continue;
                    x3D = x_h.head<3>() / x_h(3);
                }
                else
                    continue; // no depth, and too little parallax to triangulate

                // The point must lie in front of both cameras
                const float z1 = Rcw1.row(2).dot(x3D) + tcw1(2);
                if(z1 <= 0)
                    continue;
                const float z2 = Rcw2.row(2).dot(x3D) + tcw2(2);
                if(z2 <= 0)
                    continue;

                // Reprojection gate in this keyframe
                const float u1 = fx1 * (Rcw1.row(0).dot(x3D) + tcw1(0)) / z1 + cx1;
                const float v1 = fy1 * (Rcw1.row(1).dot(x3D) + tcw1(1)) / z1 + cy1;
                const float err_x1 = u1 - kp1.pt.x;
                const float err_y1 = v1 - kp1.pt.y;
                if(err_x1 * err_x1 + err_y1 * err_y1 > CHI2_2DOF * current_keyframe_->GetKeyPt1DSigma2(idx1, feature_type))
                    continue;

                // Reprojection gate in the neighbor
                const float u2 = fx2 * (Rcw2.row(0).dot(x3D) + tcw2(0)) / z2 + cx2;
                const float v2 = fy2 * (Rcw2.row(1).dot(x3D) + tcw2(1)) / z2 + cy2;
                const float err_x2 = u2 - kp2.pt.x;
                const float err_y2 = v2 - kp2.pt.y;
                if(err_x2 * err_x2 + err_y2 * err_y2 > CHI2_2DOF * neighbor->GetKeyPt1DSigma2(idx2, feature_type))
                    continue;

                const Pt map_point = current_keyframe_->create_monocular_map_point(
                    x3D, KeypointIndex(idx1), neighbor, KeypointIndex(idx2), feature_type);
                recent_map_points_.push_back(map_point);
            }
        }
    }

    create_depth_seeded_points();
}

void LocalMapping::cache_neighbor_matches(const std::vector<Keyframe>& neighbors)
{
    // Brute-force descriptor matching of the new keyframe against every neighbor it has no
    // cached matches with yet, all (neighbor, feature type) pairs in parallel. Results are
    // stored symmetrically in both keyframes' caches, per feature type and stacked (with
    // keypoint-index offsets) over all feature types.
    const auto& feature_types = current_keyframe_->featureTypes;
    const int num_neighbors = int(neighbors.size());
    const int num_feature_types = int(feature_types.size());

    std::vector<char> cached(num_neighbors, 0);
    for(int k = 0; k < num_neighbors; k++)
    {
        const auto it = current_keyframe_->cache_matched_pairs.find(neighbors[k]->frame_id);
        cached[k] = it != current_keyframe_->cache_matched_pairs.end() && !it->second.empty();
    }

    std::vector<std::vector<std::vector<cv::DMatch>>> matches(
        num_neighbors, std::vector<std::vector<cv::DMatch>>(num_feature_types));

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for(int k = 0; k < num_neighbors; k++)
    {
        for(int f = 0; f < num_feature_types; f++)
        {
            if(cached[k])
                continue;
            const Keyframe& neighbor = neighbors[k];
            const FeatureType feature_type = feature_types[f];
            matches[k][f] = matcher_->serialFeatureMatching(
                current_keyframe_->descriptors.at(feature_type), neighbor->descriptors.at(feature_type),
                current_keyframe_->keypoints.at(feature_type), neighbor->keypoints.at(feature_type),
                feature_type);
        }
    }

    // Store (serial): per-feature-type caches in both directions, plus the stacked list
    for(int k = 0; k < num_neighbors; k++)
    {
        if(cached[k])
            continue;
        const Keyframe& neighbor = neighbors[k];

        auto& forward_by_type = current_keyframe_->cache_matched_pairs_feat_type[neighbor->frame_id];
        auto& backward_by_type = neighbor->cache_matched_pairs_feat_type[current_keyframe_->frame_id];

        std::vector<cv::DMatch> stacked;
        int query_offset = 0;
        int train_offset = 0;
        for(int f = 0; f < num_feature_types; f++)
        {
            const FeatureType feature_type = feature_types[f];
            stacked.reserve(stacked.size() + matches[k][f].size());
            for(const cv::DMatch& match : matches[k][f])
            {
                forward_by_type[feature_type].push_back(match);
                backward_by_type[feature_type].push_back(cv::DMatch(match.trainIdx, match.queryIdx, match.distance));

                cv::DMatch stacked_match = match;
                stacked_match.queryIdx += query_offset;
                stacked_match.trainIdx += train_offset;
                stacked.push_back(stacked_match);
            }
            query_offset += int(current_keyframe_->keypoints.at(feature_type).size());
            train_offset += int(neighbor->keypoints.at(feature_type).size());
        }

        neighbor->cache_matched_pairs.insert_or_assign(current_keyframe_->frame_id,
                                                       FeatureMatcher::swap_match_direction(stacked));
        current_keyframe_->cache_matched_pairs.insert_or_assign(neighbor->frame_id, std::move(stacked));
    }
}

void LocalMapping::create_depth_seeded_points()
{
    // Back-project still-unmatched keypoints straight from the keyframe's own sensor depth.
    // Triangulation only creates points for keypoints with a 2D feature match in a covisible
    // neighbor -- a keypoint with a valid depth reading but no such match (textureless
    // region, repeated pattern, fast motion) is exactly where sensor depth helps most.
    // Same trust policy as the triangulation depth branches: any inv_depth > 0, no range gate.
    mat4f Twc1, Tcw1;
    mat3f Rwc1, Rcw1;
    vec3f twc1, tcw1;
    current_keyframe_->getFullPose(Twc1, Rwc1, twc1, Tcw1, Rcw1, tcw1);

    float fx1, fy1, cx1, cy1, invfx1, invfy1;
    current_keyframe_->getFullIntrinsics(fx1, fy1, cx1, cy1, invfx1, invfy1);

    for(const FeatureType feature_type : current_keyframe_->featureTypes)
    {
        const auto& inv_depth = current_keyframe_->inv_depth.at(feature_type);
        const auto& keypoints = current_keyframe_->keypoints.at(feature_type);
        for(size_t idx = 0; idx < inv_depth.size(); idx++)
        {
            if(inv_depth[idx] <= 0.0f)
                continue; // no valid sensor depth at this keypoint
            if(current_keyframe_->get_map_point(idx, feature_type))
                continue; // already has a map point (from tracking, or the triangulation loop)

            const cv::KeyPoint& kp = keypoints[idx];
            const vec3f xn{(kp.pt.x - cx1) * invfx1, (kp.pt.y - cy1) * invfy1, 1.0f};
            const Pt map_point = current_keyframe_->create_map_point(Rwc1 * (xn / inv_depth[idx]) + twc1,
                                                                     KeypointIndex(idx), feature_type);
            recent_map_points_.push_back(map_point);
        }
    }
}

void LocalMapping::search_in_neighbors()
{
    // Fuse target set: the best covisible keyframes plus a few of THEIR best covisible
    // keyframes, deduplicated via the fuse-target mark. Collected ONCE for all feature
    // types: an earlier per-feature-type version re-ran this collection each call, and
    // the mark made every call after the first come up empty -- silently fusing only
    // the first feature type. (Second neighbors are marked too, so a keyframe reachable
    // through two different neighbors is fused once, not twice.)
    std::vector<Keyframe> targets;
    for(const Keyframe& neighbor : current_keyframe_->get_best_covisibility_keyframes(params.search_in_neighbors_keyframes))
    {
        if(neighbor->is_bad() || neighbor->mnFuseTargetForKF == current_keyframe_->keyId)
            continue;
        neighbor->mnFuseTargetForKF = current_keyframe_->keyId;
        targets.push_back(neighbor);

        for(const Keyframe& second : neighbor->get_best_covisibility_keyframes(params.search_in_neighbors_second_keyframes))
        {
            if(second->is_bad() || second->mnFuseTargetForKF == current_keyframe_->keyId
               || second->keyId == current_keyframe_->keyId)
                continue;
            second->mnFuseTargetForKF = current_keyframe_->keyId;
            targets.push_back(second);
        }
    }

    for(const FeatureType feature_type : current_keyframe_->featureTypes)
    {
        // Project this keyframe's map points into each target and fuse duplicates.
        // By-value snapshot: get_map_point_matches copies under the keyframe's mutex.
        const std::vector<Pt> map_points = current_keyframe_->get_map_point_matches(feature_type);
        for(const Keyframe& target : targets)
            matcher_->fuse_map_points_to_keyframe(target, map_points, params.search_in_neighbors_radius, feature_type);

        // Project the targets' map points into this keyframe and fuse duplicates
        std::vector<Pt> candidates;
        candidates.reserve(targets.size() * map_points.size());
        for(const Keyframe& target : targets)
        {
            for(const Pt& candidate : target->get_map_point_matches(feature_type))
            {
                if(!candidate || candidate->is_bad()
                   || candidate->mnFuseCandidateForKF == current_keyframe_->keyId)
                    continue;
                candidate->mnFuseCandidateForKF = current_keyframe_->keyId;
                candidates.push_back(candidate);
            }
        }
        matcher_->fuse_map_points_to_keyframe(current_keyframe_, candidates, params.search_in_neighbors_radius, feature_type);

        // Refresh the surviving matches (fusion may have replaced points)
        for(const Pt& map_point : current_keyframe_->get_map_point_matches(feature_type))
        {
            if(map_point && !map_point->is_bad())
            {
                map_point->ComputeDistinctiveDescriptors();
                map_point->UpdateNormalAndDepth();
            }
        }
    }

    // Update links in the covisibility graph
    current_keyframe_->update_connections();
}

void LocalMapping::cull_keyframes()
{
    if(params.keyframe_culling_method == "information"){
        // The information method needs the online kernel, i.e. an image-embedding VPR backend
        // (vpr: megaloc). Without stored descriptors, degrade to the heuristic once, loudly.
        if(!place_cell_ || place_cell_->size() == 0){
            static bool warned{false};
            if(!warned){
                AF_WARN("[LocalMapping] KeyframeCullingMethod: information requested but keyframes carry no global "
                        "descriptor (vpr is not megaloc) — falling back to the heuristic culling; set vpr: megaloc to use it");
                warned = true;
            }
            cull_keyframes_heuristic();
            return;
        }
        cull_keyframes_information();
        return;
    }
    cull_keyframes_heuristic();
}

void LocalMapping::cull_keyframes_heuristic()
{
    // A covisible keyframe is redundant when more than
    // params.keyframe_culling_redundancy_ratio of its (non-bad) map points are each
    // observed by at least params.keyframe_culling_min_observations OTHER non-bad
    // keyframes. The test runs per feature type, and a keyframe redundant for ANY
    // of its feature types is culled. (Stock ORB-SLAM2's extra requirement that the
    // other observations be at the same or a finer scale level is not applied.)

    // By-value snapshot: get_covisible_keyframes copies under the keyframe's mutex
    const std::vector<Keyframe> local_keyframes = current_keyframe_->get_covisible_keyframes();
    for(const Keyframe& keyframe : local_keyframes){
        if(keyframe->keyId == 0)
            continue; // the first keyframe anchors the map and is never culled

        for(const FeatureType ft : keyframe->featureTypes){
            // By-value snapshot: get_map_point_matches copies under the keyframe's mutex
            const std::vector<Pt> map_points = keyframe->get_map_point_matches(ft);

            int num_points = 0;
            int num_redundant = 0;
            for(const Pt& pt : map_points){
                if(!pt || pt->is_bad())
                    continue;
                num_points++;

                // number_of_observations counts depth-verified observations twice
                if(pt->number_of_observations() <= params.keyframe_culling_min_observations)
                    continue;

                // By-value snapshot: get_observations copies under the point's mutex
                const std::map<KeyframeId, Obs> observations = pt->get_observations();
                int num_other_observers = 0;
                for(const auto& [key_id, obs] : observations){
                    if(key_id == keyframe->keyId || obs->projKeyframe->is_bad())
                        continue;
                    if(++num_other_observers >= params.keyframe_culling_min_observations)
                        break;
                }
                if(num_other_observers >= params.keyframe_culling_min_observations)
                    num_redundant++;
            }

            if(num_redundant <= params.keyframe_culling_redundancy_ratio * static_cast<float>(num_points))
                continue;

            bool protected_keyframe{false};

            if(!protected_keyframe)
                keyframe->set_bad_flag();
        }
    }
}

void LocalMapping::cull_keyframes_information()
{
    // The joint-information culling maths ("gram-greedy") lives in placecell
    // (PlaceCell::cull_keyframes); this adapter supplies the host side: parameters,
    // the covisibility window for the local scope, reconciliation of externally
    // removed keyframes, execution of each cull (set_bad_flag), and logging.
    if(!place_cell_)
        return;

    // Resolve every stored row back to a keyframe. A row that no longer resolves (or
    // resolves to a bad keyframe) was removed outside this culler (e.g. by the
    // heuristic method) -- tell placecell so it becomes culling history there too.
    std::unordered_map<FrameId, Keyframe> keyframe_by_frame_id;
    for(const Keyframe& keyframe : map_->GetAllKeyFrames())
        keyframe_by_frame_id.emplace(keyframe->frame_id, keyframe);
    for(const placecell::PlaceCell::ExternalId id : place_cell_->external_ids())
    {
        if(place_cell_->is_culled(id))
            continue;
        const auto it = keyframe_by_frame_id.find(FrameId(id));
        if(it == keyframe_by_frame_id.end() || it->second->is_bad())
            place_cell_->set_culled(id);
    }

    placecell::PlaceCell::CullParameters cull_parameters;
    cull_parameters.method = "gram-greedy";
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
        // Deferred by SetNotErase (loop closing holds it): leave it alive; placecell
        // never retries a refused candidate within the same call
        return it->second->is_bad();
    };

    const placecell::PlaceCell::CullReport report =
        place_cell_->cull_keyframes(cull_parameters, try_cull, local_scope ? &window : nullptr);

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
