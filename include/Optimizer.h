#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include "g2o/types/types_seven_dof_expmap.h"

#include "ErrorModel.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "LoopTypes.h"
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
    int local_ba_min_fixed_mono{2};    // fewer fixed keyframes: the oldest local ones are fixed too (pose + scale)
    int local_ba_min_fixed_rgbd{1};    // the same with sensor-depth edges (pose only: depth fixes the scale)

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
    // to the map; otherwise to the *_gba fields, tagged with that id, for the caller to apply. stop_flag, set from
    // another thread, ends the optimization after the iteration in progress.
    static void global_bundle_adjustment(const std::shared_ptr<Map>& map, int iterations = 5,
                                         const std::atomic<bool>* stop_flag = nullptr, KeyframeId loop_keyframe_id = 0,
                                         bool robust = true);

    // What one local bundle adjustment did; all zero when the map is too small to run it
    struct LocalBaCounts { int local_keyframes{0}, fixed_keyframes{0}, oldest_fixed{0}, points{0}, edges{0},
                           outliers_erased{0}; };
    static LocalBaCounts local_bundle_adjustment(const Keyframe& current_keyframe, const std::shared_ptr<Map>& map);

    // Returns the number of inliers. use_depth_channel = false books RGB-D observations as plain 2D reprojection
    // edges: used by track_reference_keyframe's divergence rescue (see Tracking.cc).
    static int pose_optimization(Frame* frame, bool use_depth_channel = true);

    // if fix_scale is true, 6DoF optimization (stereo,rgbd), 7DoF otherwise (mono)
    static void optimize_essential_graph(const std::shared_ptr<Map>& map, const Keyframe& matched_keyframe,
                                         const Keyframe& current_keyframe,
                                         const KeyframePoses& uncorrected_poses,
                                         const KeyframePoses& corrected_poses,
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
                                  int iterations, const std::atomic<bool>* stop_flag, KeyframeId loop_keyframe_id, bool robust);
};

}
