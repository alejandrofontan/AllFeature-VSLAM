/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * Loop Closing thread: the loop pipeline of AllFeature-VSLAM (run: detect_loop,
 * compute_sim3, search_loop_map_points, correct_loop, and the global BA thread).
 * Auxiliary members (parameter loading, thread synchronization) live in
 * LoopClosing_aux.cc.
 */
#include "LoopClosing.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include "Converter.h"
#include "FeatureMatcher.h"
#include "LocalMapping.h"
#include "MapDrawer.h"
#include "Optimizer.h"
#include "Sim3Solver.h"
#include "afvslam_log.hpp"

namespace AF_VSLAM
{

LoopClosing::LoopClosing(std::shared_ptr<Map> map, std::shared_ptr<PlaceRecognition> place_recognition,
                         std::shared_ptr<LocalMapping> local_mapper, std::shared_ptr<MapDrawer> map_drawer,
                         const bool fix_scale, const std::vector<FeatureType>& feature_types,
                         const int image_width, const int image_height):
    verification_feature_(place_recognition->verification_feature()),
    feature_types_(feature_types),
    fix_scale_(fix_scale),
    map_(std::move(map)),
    place_recognition_(std::move(place_recognition)),
    local_mapper_(std::move(local_mapper)),
    map_drawer_(std::move(map_drawer)),
    matcher_(std::make_shared<FeatureMatcher>(image_width, image_height, feature_types, "LoopClosing", 0.8, true))
{
}

LoopClosing::~LoopClosing()
{
    // A completed global BA leaves its thread joinable (a superseded one was detached by
    // correct_loop); System::Shutdown waits for is_gba_running() to clear before this runs
    if(gba_thread_.joinable())
        gba_thread_.join();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// # Main loop
void LoopClosing::run()
{
    {
        std::lock_guard<std::mutex> lock(finish_mutex_);
        finished_ = false;
    }

    while(true)
    {
        // One queued keyframe per iteration: candidates, geometric verification, correction
        if(has_new_keyframes() && detect_loop() && compute_sim3() && search_loop_map_points())
        {
            const auto t_start = std::chrono::steady_clock::now();
            correct_loop();
            map_drawer_->AddLoopClosureKeyframe(current_keyframe_->get_pose_inverse());
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
            AF_INFO("[LoopClosing] loop closed: keyframes " << current_keyframe_->keyId << " <-> " << matched_keyframe_->keyId
                    << " in " << seconds << " s (correction + essential graph; the global BA runs in its own thread)");
            std::cout.flush();
        }

        reset_if_requested();

        if(is_finish_requested())
            break;

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    set_finished();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// # Loop detection
bool LoopClosing::detect_loop()
{
    {
        std::lock_guard<std::mutex> lock(new_keyframes_mutex_);
        current_keyframe_ = new_keyframes_.front();
        new_keyframes_.pop_front();
        // Keyframe culling must not delete the keyframe while this thread works on it.
        // SetErase lifts the guard (a culling deferred meanwhile is applied then); on
        // success compute_sim3/correct_loop keep it until the loop is closed or rejected.
        current_keyframe_->SetNotErase();
    }

    // Candidates retrieved by the VPR backend (covisible keyframes excluded), kept only
    // when their covisibility group was also retrieved by the previous keyframes. No
    // detection right after a loop closure, nor for the first keyframes of a map (both
    // gates share the counter: keyIds start at 0); the consistency groups are left as
    // they are meanwhile.
    loop_candidates_.clear();
    const KeyframeId min_keyframes_between_loops = static_cast<KeyframeId>(params.min_keyframes_between_loops);
    if(current_keyframe_->keyId >= last_loop_keyframe_id_ + min_keyframes_between_loops)
        loop_candidates_ = consistent_loop_candidates(place_recognition_->detect_loop_candidates(current_keyframe_));

    // Into the database only after its own query
    place_recognition_->add(current_keyframe_);

    if(loop_candidates_.empty())
    {
        current_keyframe_->SetErase();
        return false;
    }
    return true;
}

// True when the two sorted id lists share a keyframe
static bool share_keyframe(const std::vector<KeyframeId>& a, const std::vector<KeyframeId>& b)
{
    auto ia = a.begin();
    auto ib = b.begin();
    while(ia != a.end() && ib != b.end())
    {
        if(*ia == *ib)
            return true;
        if(*ia < *ib)
            ++ia;
        else
            ++ib;
    }
    return false;
}

// Covisibility-consistency vote (ORB-SLAM2). Each candidate expands to its covisibility
// group: the candidate plus the keyframes connected to it. A group is consistent with a
// group kept from the previous keyframe when the two share a keyframe; it then inherits
// that group's consistency count plus one, and a count reaching
// params.covisibility_consistency_threshold makes the candidate a loop candidate. The
// groups built here replace consistent_groups_ for the next keyframe (no candidates ->
// no groups). Rules kept from the original: a previous group passes its count to the
// first candidate consistent with it only, so the same group is never carried twice;
// a candidate consistent with several previous groups is stored once per group, and
// enters the result once; a candidate consistent with no previous group starts a new
// group with count 0.
std::vector<Keyframe> LoopClosing::consistent_loop_candidates(const std::vector<Keyframe>& candidates)
{
    std::vector<Keyframe> loop_candidates;
    std::vector<ConsistentGroup> current_groups;
    std::vector<bool> carried(consistent_groups_.size(), false);   // previous group already passed on

    for(const Keyframe& candidate : candidates)
    {
        // Covisibility group as sorted ids (GetConnectedKeyFrames is ordered by id)
        ConsistentGroup group;
        for(const auto& [id, keyframe] : candidate->GetConnectedKeyFrames())
            group.keyframes.push_back(id);
        group.keyframes.insert(std::lower_bound(group.keyframes.begin(), group.keyframes.end(), candidate->keyId),
                               candidate->keyId);

        bool consistent_with_some_group = false;
        bool enough_consistent = false;
        for(size_t i = 0; i < consistent_groups_.size(); i++)
        {
            const ConsistentGroup& previous = consistent_groups_[i];
            if(!share_keyframe(group.keyframes, previous.keyframes))
                continue;
            consistent_with_some_group = true;

            const int consistency = previous.consistency + 1;
            if(!carried[i])
            {
                current_groups.push_back(ConsistentGroup{group.keyframes, consistency});
                carried[i] = true;
            }
            if(consistency >= params.covisibility_consistency_threshold && !enough_consistent)
            {
                loop_candidates.push_back(candidate);
                enough_consistent = true;
            }
        }

        if(!consistent_with_some_group)
            current_groups.push_back(std::move(group));
    }

    consistent_groups_ = std::move(current_groups);
    return loop_candidates;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// # Sim3 verification

// RANSAC schedule of the per-candidate Sim3 solvers and the Sim3 optimization threshold
static constexpr double sim3_ransac_probability = 0.99;
static constexpr int sim3_ransac_min_inliers = 20;
static constexpr int sim3_ransac_max_iterations = 300;
static constexpr int sim3_ransac_iterations_per_round = 5;   // rounds alternate over the candidates
static constexpr float sim3_optimization_chi2 = 10.0f;       // Optimizer::OptimizeSim3 outlier threshold

bool LoopClosing::compute_sim3()
{
    // One Sim3 solver per candidate with enough feature matches to the current keyframe.
    // Only verification_feature_'s matches drive the Sim3: the other feature types are
    // matched (match cache) but unused downstream.
    struct Candidate
    {
        Keyframe keyframe;
        std::vector<Pt> matches;               // per keypoint of the current keyframe: the candidate's map point matched to it, or null
        std::unique_ptr<Sim3Solver> solver;    // null once the candidate is discarded
    };
    std::vector<Candidate> candidates;
    candidates.reserve(loop_candidates_.size());
    int num_alive = 0;

    const size_t num_current_keypoints = current_keyframe_->get_map_point_matches(verification_feature_).size();
    for(const Keyframe& keyframe : loop_candidates_)
    {
        // Keyframe culling must not delete a candidate while it is being verified
        keyframe->SetNotErase();

        Candidate candidate{keyframe, {}, nullptr};
        if(!keyframe->is_bad())
        {
            std::map<FeatureType, std::vector<std::pair<size_t,size_t>>> matched_pairs;
            matcher_->match_keyframes_for_compute_sim3(current_keyframe_, keyframe, matched_pairs, feature_types_);
            const std::vector<std::pair<size_t,size_t>>& pairs = matched_pairs[verification_feature_];

            // By-value snapshot: get_map_point_matches copies under the keyframe's mutex
            const std::vector<Pt> candidate_points = keyframe->get_map_point_matches(verification_feature_);
            candidate.matches.assign(num_current_keypoints, nullptr);
            for(const auto& [current_index, candidate_index] : pairs)
                candidate.matches[current_index] = candidate_points[candidate_index];

            if(static_cast<int>(pairs.size()) >= params.sim3_min_matches)
            {
                candidate.solver = std::make_unique<Sim3Solver>(current_keyframe_, keyframe, candidate.matches,
                                                                verification_feature_, fix_scale_);
                candidate.solver->set_ransac_parameters(sim3_ransac_probability, sim3_ransac_min_inliers,
                                                        sim3_ransac_max_iterations);
                num_alive++;
            }
        }
        candidates.push_back(std::move(candidate));
    }

    // Alternate RANSAC rounds over the surviving candidates until one Sim3 passes the
    // optimization or every candidate has exhausted its iteration budget
    bool matched = false;
    while(num_alive > 0 && !matched)
    {
        for(Candidate& candidate : candidates)
        {
            if(!candidate.solver)
                continue;

            std::vector<bool> inliers;
            int num_inliers = 0;
            bool no_more = false;
            const cv::Mat Scm = candidate.solver->iterate(sim3_ransac_iterations_per_round, no_more, inliers, num_inliers);

            if(!Scm.empty())
            {
                // RANSAC consensus: optimize the Sim3 over the candidate's inlier matches
                // (OptimizeSim3 nulls the matches it rejects)
                std::vector<Pt> inlier_matches(candidate.matches.size(), nullptr);
                for(size_t j = 0; j < inliers.size(); j++)
                    if(inliers[j])
                        inlier_matches[j] = candidate.matches[j];

                g2o::Sim3 g2o_Scm(candidate.solver->GetEstimatedRotation().cast<double>(),
                                  candidate.solver->GetEstimatedTranslation().cast<double>(),
                                  candidate.solver->GetEstimatedScale());
                const int num_optimized_inliers = Optimizer::OptimizeSim3(current_keyframe_, candidate.keyframe, inlier_matches,
                                                                          g2o_Scm, sim3_optimization_chi2, fix_scale_,
                                                                          verification_feature_);
                if(num_optimized_inliers >= params.sim3_min_inliers)
                {
                    matched = true;
                    matched_keyframe_ = candidate.keyframe;
                    // World -> current keyframe through the loop side: Scm * Smw
                    const g2o::Sim3 g2o_Smw(candidate.keyframe->get_rotation().cast<double>(),
                                            candidate.keyframe->get_translation().cast<double>(), 1.0);
                    g2o_Scw_ = g2o_Scm * g2o_Smw;
                    Scw_ = Converter::to_matrix4f(g2o_Scw_);
                    loop_matched_points_ = std::move(inlier_matches);
                    break;
                }
            }

            // Iteration budget exhausted without an accepted Sim3: discard
            if(no_more)
            {
                candidate.solver.reset();
                num_alive--;
            }
        }
    }

    if(!matched)
        release_loop_candidates(false);
    return matched;
}

// Map points seen by the matched keyframe and its covisibles, projected into the current
// keyframe with the loop Sim3 to find matches beyond the RANSAC inliers. The loop is
// accepted when enough points match; the erase guards of the candidates are lifted
// either way (matched keyframe excepted on acceptance).
bool LoopClosing::search_loop_map_points()
{
    std::vector<Keyframe> loop_keyframes = matched_keyframe_->get_covisible_keyframes();
    loop_keyframes.push_back(matched_keyframe_);

    loop_map_points_.clear();
    std::map<PtId, Keyframe> point_keyframe;   // the loop keyframe each point was collected from
    for(const Keyframe& keyframe : loop_keyframes)
    {
        // Side effect only: fills the current keyframe's match cache for this keyframe,
        // which search_by_projection_for_compute_sim3 reads (the pairs themselves are not needed)
        matcher_->match_keyframes(current_keyframe_, keyframe, feature_types_);

        for(const Pt& point : keyframe->get_map_point_matches(verification_feature_))
        {
            if(!point || point->is_bad() || point->mnLoopPointForKF == current_keyframe_->keyId)
                continue;
            loop_map_points_.push_back(point);
            point->mnLoopPointForKF = current_keyframe_->keyId;   // collected once per loop
            point_keyframe[point->ptId] = keyframe;
        }
    }

    matcher_->search_by_projection_for_compute_sim3(current_keyframe_, Scw_, loop_map_points_, loop_matched_points_, point_keyframe);

    const long num_matches = std::count_if(loop_matched_points_.begin(), loop_matched_points_.end(),
                                           [](const Pt& point) { return point != nullptr; });
    const bool accepted = num_matches >= params.loop_min_matches;
    release_loop_candidates(accepted);
    return accepted;
}

// Lifts the erase guards set for the verification: every candidate's, except the matched
// keyframe's when the loop is accepted, and the current keyframe's when it is rejected
// (an accepted loop keeps both guarded until correct_loop pins them with the loop edge)
void LoopClosing::release_loop_candidates(const bool loop_accepted)
{
    for(const Keyframe& candidate : loop_candidates_)
        if(!loop_accepted || candidate != matched_keyframe_)
            candidate->SetErase();
    if(!loop_accepted)
        current_keyframe_->SetErase();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// # Loop correction
void LoopClosing::correct_loop()
{
    AF_INFO("[LoopClosing] loop detected: keyframe " << current_keyframe_->keyId << " -> " << matched_keyframe_->keyId);
    std::cout.flush();

    // Pause Local Mapping: no keyframe may be inserted while the map is corrected
    local_mapper_->request_stop();

    // A running global BA belongs to a superseded loop: make it abort and drop its result
    {
        std::lock_guard<std::mutex> lock(gba_mutex_);
        if(gba_running_)
        {
            gba_stop_ = true;
            ++gba_generation_;
            if(gba_thread_.joinable())
                gba_thread_.detach();   // exits on its own once g2o sees gba_stop_ (or its generation check fails)
        }
    }

    // Wait until Local Mapping has effectively stopped
    while(!local_mapper_->is_stopped())
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

    // Refresh the current keyframe's covisibility before propagating the correction
    current_keyframe_->update_connections();

    // The current keyframe and its covisibles get the corrected pose Scw, propagated
    // through their relative poses; both versions feed the essential-graph optimization
    std::vector<Keyframe> connected_keyframes = current_keyframe_->get_covisible_keyframes();
    connected_keyframes.push_back(current_keyframe_);

    KeyframePoses corrected_poses, uncorrected_poses;
    corrected_poses[current_keyframe_] = g2o_Scw_;
    const mat4f Twc = current_keyframe_->get_pose_inverse();

    {
        std::unique_lock<std::mutex> lock(map_->map_update_mutex_);

        for(const Keyframe& keyframe : connected_keyframes)
        {
            const mat4f Tiw = keyframe->get_pose();
            if(keyframe != current_keyframe_)
            {
                // Corrected: relative pose to the current keyframe composed with Scw
                const mat4f Tic = Tiw * Twc;
                const mat3f Ric = Tic.block<3,3>(0,0);
                const vec3f tic = Tic.block<3,1>(0,3);
                const g2o::Sim3 g2o_Sic(Ric.cast<double>(), tic.cast<double>(), 1.0);
                corrected_poses[keyframe] = g2o_Sic * g2o_Scw_;
            }
            const mat3f Riw = Tiw.block<3,3>(0,0);
            const vec3f tiw = Tiw.block<3,1>(0,3);
            uncorrected_poses[keyframe] = g2o::Sim3(Riw.cast<double>(), tiw.cast<double>(), 1.0);
        }

        // Map points seen by these keyframes (every feature type: the essential-graph
        // optimization later assumes all of a corrected keyframe's points were moved):
        // un-project with the old pose, re-project with the corrected one, once each
        for(const auto& [keyframe, g2o_corrected_Siw] : corrected_poses)
        {
            const g2o::Sim3 g2o_corrected_Swi = g2o_corrected_Siw.inverse();
            const g2o::Sim3& g2o_Siw = uncorrected_poses[keyframe];

            for(const FeatureType feature_type : keyframe->featureTypes)
            {
                for(const Pt& point : keyframe->get_map_point_matches(feature_type))
                {
                    if(!point || point->is_bad() || point->mnCorrectedByKF == current_keyframe_->keyId)
                        continue;
                    const Eigen::Vector3d corrected = g2o_corrected_Swi.map(g2o_Siw.map(point->get_world_pos().cast<double>()));
                    point->set_world_pos(corrected.cast<float>());
                    point->mnCorrectedByKF = current_keyframe_->keyId;
                    point->mnCorrectedReference = keyframe->keyId;
                    point->UpdateNormalAndDepth();
                }
            }

            // Keyframe pose from the corrected Sim3: [R t/s; 0 1]
            const Eigen::Matrix3d R = g2o_corrected_Siw.rotation().toRotationMatrix();
            const Eigen::Vector3d t = g2o_corrected_Siw.translation() / g2o_corrected_Siw.scale();
            keyframe->set_pose(Converter::to_matrix4f(R, t));
            keyframe->update_connections();
        }

        // Loop fusion: the loop-side points matched into the current keyframe replace its
        // own (or fill the keypoints that had none)
        for(size_t i = 0; i < loop_matched_points_.size(); i++)
        {
            const Pt& loop_point = loop_matched_points_[i];
            if(!loop_point)
                continue;
            const Pt current_point = current_keyframe_->get_map_point(i, verification_feature_);
            if(current_point)
                current_point->replace(loop_point);
            else
            {
                current_keyframe_->add_map_point(loop_point, KeypointIndex(i));
                loop_point->add_observation(current_keyframe_, KeypointIndex(i));
            }
        }
    }

    // Project the loop-side points into the corrected keyframes and fuse the duplicates
    search_and_fuse(corrected_poses);

    // The fusion linked both sides of the loop in the covisibility graph: those links are
    // the loop edges of the essential graph
    std::map<KeyframeId, LoopConnections> loop_connections = loop_connections_after_fusion(connected_keyframes);
    Optimizer::OptimizeEssentialGraph(map_, matched_keyframe_, current_keyframe_, uncorrected_poses, corrected_poses,
                                      loop_connections, fix_scale_);
    map_->InformNewBigChange();

    // Loop edge (also pins both keyframes: never culled)
    matched_keyframe_->AddLoopEdge(current_keyframe_);
    current_keyframe_->AddLoopEdge(matched_keyframe_);

    // Global BA in its own thread; Local Mapping resumes meanwhile. A previous BA has
    // either finished (joinable, join returns at once) or was detached above.
    if(gba_thread_.joinable())
        gba_thread_.join();
    {
        std::lock_guard<std::mutex> lock(gba_mutex_);
        gba_running_ = true;
        gba_stop_ = false;
        gba_thread_ = std::thread(&LoopClosing::run_global_bundle_adjustment, this, current_keyframe_->keyId);
    }

    local_mapper_->release();

    last_loop_keyframe_id_ = current_keyframe_->keyId;
}

// Covisibility links each corrected keyframe gained through the fusion: its connections
// now, minus the ones it had before and minus the corrected keyframes themselves
std::map<KeyframeId, LoopConnections> LoopClosing::loop_connections_after_fusion(const std::vector<Keyframe>& connected_keyframes)
{
    std::map<KeyframeId, LoopConnections> loop_connections;
    for(const Keyframe& keyframe : connected_keyframes)
    {
        const std::vector<Keyframe> previous_neighbors = keyframe->get_covisible_keyframes();
        keyframe->update_connections();

        LoopConnections& links = loop_connections[keyframe->keyId];
        links.keyframe = keyframe;
        links.connections = keyframe->GetConnectedKeyFrames();
        for(const Keyframe& neighbor : previous_neighbors)
            links.connections.erase(neighbor->keyId);
        for(const Keyframe& corrected : connected_keyframes)
            links.connections.erase(corrected->keyId);
    }
    return loop_connections;
}

// Projects the loop-side map points into every corrected keyframe (with its corrected
// pose) and merges the duplicates found there into the loop-side points
void LoopClosing::search_and_fuse(const KeyframePoses& corrected_poses)
{
    for(const auto& [corrected_keyframe, g2o_Scw] : corrected_poses)
    {
        Keyframe keyframe = corrected_keyframe;   // fuse_map_points_to_keyframe takes a non-const reference
        const mat4f Scw = Converter::to_matrix4f(g2o_Scw);

        std::vector<Pt> replaced_points(loop_map_points_.size(), nullptr);
        matcher_->fuse_map_points_to_keyframe(keyframe, Scw, loop_map_points_, params.fuse_radius, replaced_points,
                                              verification_feature_);

        std::unique_lock<std::mutex> lock(map_->map_update_mutex_);
        for(size_t i = 0; i < loop_map_points_.size(); i++)
            if(replaced_points[i])
                replaced_points[i]->replace(loop_map_points_[i]);
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// # Global bundle adjustment

// Runs in gba_thread_ (launched by correct_loop). Local Mapping keeps working during the
// BA, so what it creates meanwhile is not part of it: apply_gba_correction propagates the
// result to those keyframes and points.
void LoopClosing::run_global_bundle_adjustment(const KeyframeId loop_keyframe_id)
{
    AF_INFO("[LoopClosing] global bundle adjustment started (loop keyframe " << loop_keyframe_id << ")");
    std::cout.flush();

    int generation = 0;
    {
        std::lock_guard<std::mutex> lock(gba_mutex_);
        generation = gba_generation_;
    }

    Optimizer::global_bundle_adjustment(map_, params.gba_iterations, &gba_stop_, loop_keyframe_id, false);

    {
        std::lock_guard<std::mutex> lock(gba_mutex_);
        // Superseded by a newer loop closure (correct_loop bumped the generation): the
        // result belongs to an outdated map, and the new BA owns the running flag (issue #35)
        if(generation != gba_generation_)
            return;

        if(!gba_stop_)
            apply_gba_correction(loop_keyframe_id);

        gba_running_ = false;
    }
}

// Under gba_mutex_. Pauses Local Mapping, writes the optimized poses and positions, and
// propagates the correction to what was created during the BA: a keyframe not in the BA
// inherits its spanning-tree parent's correction through their relative pose, a point
// not in the BA its reference keyframe's.
void LoopClosing::apply_gba_correction(const KeyframeId loop_keyframe_id)
{
    AF_INFO("[LoopClosing] global bundle adjustment finished, updating the map ...");
    std::cout.flush();

    local_mapper_->request_stop();
    while(!local_mapper_->is_stopped() && !local_mapper_->is_finished())
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

    std::unique_lock<std::mutex> lock(map_->map_update_mutex_);

    // Keyframes, breadth-first from the map origins along the spanning tree
    std::list<Keyframe> pending(map_->keyframe_origins_.begin(), map_->keyframe_origins_.end());
    while(!pending.empty())
    {
        const Keyframe keyframe = pending.front();
        pending.pop_front();

        const mat4f Twc = keyframe->get_pose_inverse();
        for(const Keyframe& child : keyframe->get_children())
        {
            if(child->mnBAGlobalForKF != loop_keyframe_id)
            {
                const mat4f Tchild_parent = child->get_pose() * Twc;
                child->TcwGBA = Tchild_parent * keyframe->TcwGBA;
                child->mnBAGlobalForKF = loop_keyframe_id;
            }
            pending.push_back(child);
        }

        keyframe->TcwBefGBA = keyframe->get_pose();
        keyframe->set_pose(keyframe->TcwGBA);
    }

    // Map points: optimized ones take their BA position, the others follow their reference keyframe
    for(const Pt& point : map_->get_all_map_points())
    {
        if(point->is_bad())
            continue;

        if(point->mnBAGlobalForKF == loop_keyframe_id)
        {
            point->set_world_pos(point->PosGBA);
            continue;
        }

        const Keyframe reference = point->GetReferenceKeyFrame();
        if(reference->mnBAGlobalForKF != loop_keyframe_id)
            continue;

        // Un-project with the reference's pose before the BA, re-project with the corrected one
        const mat4f& Tcw = reference->TcwBefGBA;
        const vec3f Xc = Tcw.block<3,3>(0,0) * point->get_world_pos() + Tcw.block<3,1>(0,3);
        const mat4f Twc = reference->get_pose_inverse();
        point->set_world_pos(Twc.block<3,3>(0,0) * Xc + Twc.block<3,1>(0,3));
    }

    map_->InformNewBigChange();
    local_mapper_->release();

    AF_INFO("[LoopClosing] map updated");
    std::cout.flush();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace AF_VSLAM
