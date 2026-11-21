/**
 * Module: AllFeature-VSLAM - Optimizer.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.2)
 * - Version: 1.1
 * - Created: 2026-01-22
 * - Updated: 2026-11-21 (cleanup pass: dead code, comments, names and style)
 * - License: GPLv3 License
 */

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

#pragma once

#include <map>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include "g2o/types/types_seven_dof_expmap.h"

#include "ErrorModel.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "LoopClosing.h"
#include "Map.h"
#include "MapPoint.h"
#include "Types.h"

namespace AF_VSLAM
{

struct OptimizerParameters
{
    // The outlier thresholds and Huber deltas are ErrorModel::params (ErrorModel.h), shared with the other stages

    // pose_optimization()
    int pose_iterations{10};           // per round
    int pose_rounds{4};                // optimize, classify inliers/outliers, repeat; each round restarts from the entry pose
    int pose_rounds_with_kernel{3};    // the Huber kernel is dropped after this many rounds
    int pose_min_correspondences{3};   // fewer matches: nothing is optimized
    int pose_min_inliers{10};          // fewer inliers after a round: no further rounds
    int pose_warn_min_correspondences{20};   // the rejection warning needs at least this many matches ...
    float pose_warn_outlier_ratio{0.5f};     // ... and more than this share of them rejected

    // local_bundle_adjustment()
    int local_ba_min_keyframes{2};     // the map needs more keyframes than this
    int local_ba_neighbors{10};        // best covisible keyframes optimized with the current one
    int local_ba_iterations{5};        // first round, every observation
    int local_ba_refine_iterations{10};   // second round, without the outliers of the first

    // global_bundle_adjustment()
    int final_gba_iterations{100};     // the bundle adjustment at shutdown (System::GBA)
    bool loop_gba_robust{false};       // Huber kernel in the bundle adjustment after a loop closure

    // optimize_essential_graph()
    double essential_graph_lambda_init{1e-16};
    int essential_graph_min_weight{100};
    int essential_graph_iterations{20};
    double essential_graph_edge_information{1.0};   // every edge (spanning tree, loop, covisibility) weighs the same

    // optimize_sim3()
    int sim3_iterations{5};
    int sim3_more_iterations_after_outliers{10};
    int sim3_more_iterations{5};
    int sim3_min_correspondences{10};
    float sim3_chi2{10.0f};            // outlier threshold of both reprojections (not the 95% value)
};

class Optimizer
{
public:
    // Bundle adjustment over every keyframe and point of the map. With loop_keyframe_id == 0 the result is written
    // to the map; otherwise to the *_gba fields, tagged with that id, for the caller to apply.
    static void global_bundle_adjustment(const std::shared_ptr<Map>& map, int iterations = 5,
                                         bool* stop_flag = nullptr, KeyframeId loop_keyframe_id = 0,
                                         bool robust = true);

    static void local_bundle_adjustment(const Keyframe& current_keyframe, const std::shared_ptr<Map>& map);

    // Returns the number of inliers. use_depth_channel = false books RGB-D observations as plain 2D reprojection
    // edges: used by track_reference_keyframe's divergence rescue (see Tracking.cc).
    static int pose_optimization(Frame* frame, bool use_depth_channel = true);

    // if fix_scale is true, 6DoF optimization (stereo,rgbd), 7DoF otherwise (mono)
    static void optimize_essential_graph(const std::shared_ptr<Map>& map, const Keyframe& matched_keyframe,
                                         const Keyframe& current_keyframe,
                                         const LoopClosing::KeyframePoses& uncorrected_poses,
                                         const LoopClosing::KeyframePoses& corrected_poses,
                                         std::map<KeyframeId, LoopConnections>& loop_connections,
                                         bool fix_scale);

    // Returns the number of inliers; the matches it rejects are nulled in matches1.
    // if fix_scale is true, optimize SE3 (stereo,rgbd), Sim3 otherwise (mono)
    static int optimize_sim3(const Keyframe& keyframe1, const Keyframe& keyframe2, std::vector<Pt>& matches1,
                             g2o::Sim3& S12, bool fix_scale, FeatureType feature_type);

    // Tunable parameters, loaded from the settings YAML at System startup
    static OptimizerParameters params;
    static void LoadParameters(const cv::FileStorage& fSettings);

private:
    static void bundle_adjustment(const std::vector<Keyframe>& keyframes, const std::vector<Pt>& map_points,
                                  int iterations, bool* stop_flag, KeyframeId loop_keyframe_id, bool robust);
};

} // namespace AF_VSLAM
