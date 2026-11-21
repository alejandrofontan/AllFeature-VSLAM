#include "Optimizer.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/StdVector>
#include "g2o/core/block_solver.h"
#include "g2o/core/hyper_graph_action.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/solvers/linear_solver_dense.h"
#include "g2o/solvers/linear_solver_eigen.h"
#include "g2o/types/types_seven_dof_expmap.h"
#include "g2o/types/types_six_dof_expmap.h"

#include "afvslam_log.hpp"
#include "Converter.h"
#include "SettingsReader.h"

namespace AF_VSLAM
{

OptimizerParameters Optimizer::params{};

void Optimizer::LoadParameters(const cv::FileStorage &fSettings)
{
    auto read_if_present = [&fSettings](const char* key, auto& field)
    {
        read_setting_if_present(fSettings, key, field);
    };

    read_if_present("Optimizer.PoseOptIterations", params.pose_iterations);
    read_if_present("Optimizer.PoseOptRounds", params.pose_rounds);
    read_if_present("Optimizer.PoseOptRoundsWithKernel", params.pose_rounds_with_kernel);
    read_if_present("Optimizer.PoseOptMinCorrespondences", params.pose_min_correspondences);
    read_if_present("Optimizer.PoseOptMinInliers", params.pose_min_inliers);
    read_if_present("Optimizer.PoseOptWarnMinCorrespondences", params.pose_warn_min_correspondences);
    read_if_present("Optimizer.PoseOptWarnOutlierRatio", params.pose_warn_outlier_ratio);

    read_if_present("Optimizer.LocalBaMinKeyframes", params.local_ba_min_keyframes);
    read_if_present("Optimizer.LocalBaNeighbors", params.local_ba_neighbors);
    read_if_present("Optimizer.LocalBaIterations", params.local_ba_iterations);
    read_if_present("Optimizer.LocalBaRefineIterations", params.local_ba_refine_iterations);

    read_if_present("Optimizer.FinalGbaIterations", params.final_gba_iterations);
    int loop_gba_robust = params.loop_gba_robust ? 1 : 0;   // a 0/1 flag in the file
    read_if_present("Optimizer.LoopGbaRobust", loop_gba_robust);
    params.loop_gba_robust = loop_gba_robust != 0;

    read_if_present("Optimizer.EssGraphLambdaInit", params.essential_graph_lambda_init);
    read_if_present("Optimizer.EssGraphMinFeat", params.essential_graph_min_weight);
    read_if_present("Optimizer.EssGraphIterations", params.essential_graph_iterations);
    read_if_present("Optimizer.EssGraphEdgeInformation", params.essential_graph_edge_information);

    read_if_present("Optimizer.Sim3Iterations", params.sim3_iterations);
    read_if_present("Optimizer.Sim3MoreIterationsHigh", params.sim3_more_iterations_after_outliers);
    read_if_present("Optimizer.Sim3MoreIterationsLow", params.sim3_more_iterations);
    read_if_present("Optimizer.Sim3MinCorrespondences", params.sim3_min_correspondences);
    read_if_present("Optimizer.Sim3Chi2", params.sim3_chi2);
}

// Per-observation information (1/variance) for the RGB-D inverse-depth residual, from
// Frame/KeyFrame::sigma2_inv_depth (Frame::get_depth's quadratic depth-noise model).
//
// sigma2_inv_depth is strictly positive wherever this is called: Frame::get_depth writes it in the
// same branch as inv_depth, and all three call sites below build the edge only for inv_depth > 0.
// No fallback: a zero would be a broken invariant, and it gives an infinite weight here, not a
// substituted one. Local BA then refuses the non-finite result (it keeps the previous poses and
// positions); the other optimizations write what comes out.
static double rgbd_inv_depth_information(float sigma2_inv_depth)
{
    return 1.0 / sigma2_inv_depth;
}

// What one keypoint contributes to an edge: (u, v) with the keypoint's 2x2 information, plus, when it carries a
// sensor depth, the inverse depth with its own information. A mono edge takes the first two rows only.
struct KeypointMeasurement
{
    bool has_depth{false};
    Eigen::Vector3d value{Eigen::Vector3d::Zero()};          // u, v, inverse depth (0 without depth)
    Eigen::Matrix3d information{Eigen::Matrix3d::Zero()};    // 2x2 block, and (2,2) with depth

    [[nodiscard]] Eigen::Vector2d pixel() const { return value.head<2>(); }
    [[nodiscard]] Eigen::Matrix2d pixel_information() const { return information.topLeftCorner<2,2>(); }
};

// use_depth = false reads an RGB-D keypoint as a mono one
static KeypointMeasurement keypoint_measurement(const FeatureData& features, FeatureType feature_type,
                                                KeypointIndex index, bool use_depth = true)
{
    const KeypointData& data = features.at(feature_type);
    const cv::KeyPoint& keypoint = data.keypoints[index];
    const float inv_depth = data.inv_depth[index];

    KeypointMeasurement measurement;
    measurement.has_depth = use_depth && inv_depth > 0.0f;

    // Assembled in single precision, as the keypoint stores it, then widened
    mat3f information = mat3f::Zero();
    information.block<2,2>(0,0) = features.get_information_2d(index, feature_type);
    if(measurement.has_depth)
        information(2,2) = static_cast<float>(rgbd_inv_depth_information(data.sigma2_inv_depth[index]));
    measurement.information = information.cast<double>();

    measurement.value << keypoint.pt.x, keypoint.pt.y, (measurement.has_depth ? inv_depth : 0.0f);
    return measurement;
}

// Copies the caller's atomic stop flag into the plain bool g2o polls; run by g2o after every iteration, on the
// optimizing thread.
struct StopFlagMirror : g2o::HyperGraphAction
{
    StopFlagMirror(const std::atomic<bool>* source_, bool& mirror_) : source(source_), mirror(mirror_) {}

    g2o::HyperGraphAction* operator()(const g2o::HyperGraph*, Parameters* = nullptr) override
    {
        if(source)
            mirror = source->load();
        return this;
    }

    const std::atomic<bool>* source;
    bool& mirror;
};

// A Levenberg-Marquardt algorithm on the given block and linear solvers; the optimizer it is set on owns it
template<typename BlockSolverT, template<typename> class LinearSolverT>
static g2o::OptimizationAlgorithmLevenberg* make_levenberg()
{
    auto* linear_solver = new LinearSolverT<typename BlockSolverT::PoseMatrixType>();
    return new g2o::OptimizationAlgorithmLevenberg(new BlockSolverT(linear_solver));
}

// Builds and registers a binary (point + pose) g2o edge shared by the mono/RGBD
// bundle_adjustment/local_bundle_adjustment branches: sets both vertices, measurement,
// information, robust kernel and camera intrinsics, then adds it to the optimizer.
// Returns nullptr and adds nothing when either vertex is missing (a keyframe that started observing the point
// after the vertices were laid out): g2o's addEdge dereferences a null vertex.
template<typename EdgeT, typename MeasT, typename InfoT>
static EdgeT* create_ba_edge(g2o::SparseOptimizer& optimizer, int point_id, int keyframe_id,
                           const MeasT& measurement, const InfoT& information,
                           double huber_delta, const Keyframe& keyframe, bool use_robust_kernel = true)
{
    auto* point_vertex = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(point_id));
    auto* pose_vertex = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(keyframe_id));
    if(!point_vertex || !pose_vertex)
        return nullptr;

    EdgeT* e = new EdgeT();
    e->setVertex(0, point_vertex);
    e->setVertex(1, pose_vertex);
    e->setMeasurement(measurement);
    e->setInformation(information);

    if(use_robust_kernel)
    {
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(huber_delta);
    }

    e->fx = keyframe->calibration->fx;
    e->fy = keyframe->calibration->fy;
    e->cx = keyframe->calibration->cx;
    e->cy = keyframe->calibration->cy;

    optimizer.addEdge(e);
    return e;
}

// One BA edge plus the (keyframe, map point) pair it connects -- lets local_bundle_adjustment
// track edges/keyframes/points as a single vector per edge type instead of three parallel ones.
// Named BAObservation (not Observation) to avoid colliding with AF_VSLAM::Observation
// (include/MapPoint.h), which is a map point's record of one keyframe that saw it.
template<typename EdgeT>
struct BAObservation
{
    EdgeT* edge;
    Keyframe keyframe;
    Pt pt;
};

// Marks each edge of one type bad (level 1) when its chi-square exceeds threshold or its
// point is behind the camera, and always disables its robust kernel -- the pass BA runs
// once, between its initial and refining optimization rounds.
template<typename EdgeT>
static void mark_ba_outliers(const std::vector<BAObservation<EdgeT>>& observations, float chi2_threshold)
{
    for(const auto& obs : observations)
    {
        if(obs.pt->is_bad())
            continue;

        // Recompute at the current estimates: after a rejected LM trial g2o restores the
        // vertices but leaves cached edge errors at the rejected pose (see
        // classify_pose_only_edges) — classifying from stale chi2 here would mark good
        // observations as outliers.
        obs.edge->computeError();
        if(obs.edge->chi2() > chi2_threshold || !obs.edge->isDepthPositive())
            obs.edge->setLevel(1);

        obs.edge->setRobustKernel(0);
    }
}

// Collects (keyframe, map point) pairs whose edge chi-square exceeds threshold, or whose
// point is behind the camera, for erasure after BA's final optimization round.
template<typename EdgeT>
static void collect_ba_outliers(const std::vector<BAObservation<EdgeT>>& observations, float chi2_threshold,
                              std::vector<std::pair<Keyframe,Pt>>& to_erase)
{
    for(const auto& obs : observations)
    {
        if(obs.pt->is_bad())
            continue;

        // Same stale-chi2 hazard as mark_ba_outliers — erasing observations from the map
        // based on a rejected trial pose would silently damage the map.
        obs.edge->computeError();
        if(obs.edge->chi2() > chi2_threshold || !obs.edge->isDepthPositive())
            to_erase.push_back(std::make_pair(obs.keyframe, obs.pt));
    }
}

void Optimizer::global_bundle_adjustment(const std::shared_ptr<Map>& map, int iterations,
                                         const std::atomic<bool>* stop_flag,
                                         KeyframeId loop_keyframe_id, bool robust)
{
    // Both arrive ordered by id (the map's sets are id-ordered): g2o's result depends, in round-off, on the order
    // vertices and edges are added, so every run optimizes the same graph (issue #16)
    const std::vector<Keyframe> keyframes = map->get_all_keyframes();
    const std::vector<Pt> map_points = map->get_all_map_points();

    bundle_adjustment(keyframes,map_points,iterations,stop_flag, loop_keyframe_id, robust);
}


void Optimizer::bundle_adjustment(const std::vector<Keyframe>& keyframes, const std::vector<Pt>& map_points,
                                  int iterations, const std::atomic<bool>* stop_flag, KeyframeId loop_keyframe_id, bool robust)
{
    std::vector<bool> point_excluded;
    point_excluded.resize(map_points.size());

    // g2o polls a plain bool: `stop` belongs to this thread and is refreshed from the caller's atomic flag after
    // every iteration, so the two threads never touch the same plain variable. Declared before the optimizer,
    // which holds pointers to both.
    bool stop = stop_flag && stop_flag->load();
    StopFlagMirror stop_mirror{stop_flag, stop};

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(make_levenberg<g2o::BlockSolver_6_3, g2o::LinearSolverEigen>());

    if(stop_flag)
    {
        optimizer.setForceStopFlag(&stop);
        optimizer.addPostIterationAction(&stop_mirror);
    }

    long unsigned int max_keyframe_id = 0;

    // Set KeyFrame vertices
    for(const Keyframe& keyframe : keyframes)
    {
        if(keyframe->is_bad())
            continue;
        g2o::VertexSE3Expmap * pose_vertex = new g2o::VertexSE3Expmap();
        pose_vertex->setEstimate(Converter::toSE3Quat(keyframe->get_Tcw()));
        pose_vertex->setId(keyframe->keyId);
        pose_vertex->setFixed(keyframe->is_origin());
        optimizer.addVertex(pose_vertex);
        if(keyframe->keyId > max_keyframe_id)
            max_keyframe_id=keyframe->keyId;
    }

    // Set MapPoint vertices
    for(size_t i = 0; i < map_points.size(); i++)
    {
        const Pt& pt = map_points[i];
        const FeatureType feature_type = pt->featureType;

        if(pt->is_bad())
            continue;
        g2o::VertexSBAPointXYZ* point_vertex = new g2o::VertexSBAPointXYZ();

        point_vertex->setEstimate(pt->get_world_pos().cast<double>());
        const int point_id = pt->ptId + max_keyframe_id+1;
        point_vertex->setId(point_id);
        point_vertex->setMarginalized(true);
        optimizer.addVertex(point_vertex);

       const std::map<KeyframeId, Observation> observations = pt->get_observations();

        int num_edges = 0;
        //SET EDGES
        for(auto& obs: observations)
        {

            Keyframe keyframe = obs.second.keyframe;
            if(keyframe->is_bad() || keyframe->keyId > max_keyframe_id)
                continue;

            const KeypointMeasurement measurement = keypoint_measurement(keyframe->features, feature_type, obs.second.index);
            bool added = false;
            if(measurement.has_depth)   // RGB-D sensor-depth observation
                added = create_ba_edge<g2o::EdgeRGBDSE3ProjectXYZ>(
                    optimizer, point_id, keyframe->keyId, measurement.value, measurement.information,
                    ErrorModel::params.huber_rgbd(), keyframe, robust) != nullptr;
            else                        // mono observation
                added = create_ba_edge<g2o::EdgeSE3ProjectXYZ>(
                    optimizer, point_id, keyframe->keyId, measurement.pixel(), measurement.pixel_information(),
                    ErrorModel::params.huber_mono(), keyframe, robust) != nullptr;
            if(added)
                num_edges++;
        }

        if(num_edges==0)
        {
            optimizer.removeVertex(point_vertex);
            point_excluded[i]=true;
        }
        else
        {
            point_excluded[i]=false;
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(iterations);

    // Recover optimized data. A degenerate edge can drive the solution to NaN or infinity: such a pose or position
    // is not written, the keyframe or point keeps what it had
    int num_rejected = 0;

    //Keyframes
    for(const Keyframe& keyframe : keyframes)
    {
        if(keyframe->is_bad())
            continue;
        g2o::VertexSE3Expmap* pose_vertex = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(keyframe->keyId));
        const mat4f Tcw = Converter::to_matrix4f(pose_vertex->estimate());
        if(!Tcw.allFinite())   // left untagged after a loop closure: it then inherits its parent's correction
        {
            num_rejected++;
            continue;
        }

        if(loop_keyframe_id==0)
        {
            keyframe->set_pose(Tcw);
        }
        else
        {
            keyframe->Tcw_gba = Tcw;
            keyframe->ba_global_for_keyframe = loop_keyframe_id;
        }
    }

    //Points
    for(size_t i=0; i<map_points.size(); i++)
    {
        if(point_excluded[i])
            continue;

        Pt pt = map_points[i];

        if(pt->is_bad())
            continue;

        g2o::VertexSBAPointXYZ* point_vertex = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pt->ptId + max_keyframe_id+1));
        const vec3f position = point_vertex->estimate().cast<float>();
        if(!position.allFinite())
        {
            num_rejected++;
            continue;
        }

        if(loop_keyframe_id==0)
        {
            pt->set_world_pos(position);
            pt->update_normal_and_depth();
        }
        else
        {
            pt->position_gba = position;
            pt->ba_global_for_keyframe = loop_keyframe_id;
        }
    }

    if(num_rejected > 0)
        AF_WARN("bundle_adjustment: " << num_rejected << " non-finite poses or positions were not written");
}

// Builds and registers a unary pose-only g2o edge shared by the mono/RGBD
// pose_optimization() branches: sets the frame vertex, measurement, information,
// robust kernel, camera intrinsics and the (fixed) map point position, then adds it
// to the optimizer.
template<typename EdgeT, typename MeasT, typename InfoT>
static EdgeT* create_pose_only_edge(g2o::SparseOptimizer& optimizer, const MeasT& measurement,
                                 const InfoT& information, double huber_delta,
                                 const Frame* frame, const vec3f& Xw)
{
    EdgeT* e = new EdgeT();
    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
    e->setMeasurement(measurement);
    e->setInformation(information);

    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    rk->setDelta(huber_delta);

    e->fx = frame->calibration->fx;
    e->fy = frame->calibration->fy;
    e->cx = frame->calibration->cx;
    e->cy = frame->calibration->cy;
    e->Xw[0] = Xw(0);
    e->Xw[1] = Xw(1);
    e->Xw[2] = Xw(2);

    optimizer.addEdge(e);
    return e;
}

// Classifies every edge of one type as inlier/outlier against its chi-square threshold
// for the current optimization round.
template<typename EdgeT>
static void classify_pose_only_edges(const std::map<FeatureType, std::vector<EdgeT*>>& edges_by_feature,
                                  const std::map<FeatureType, std::vector<size_t>>& index_by_feature,
                                  Frame* frame, float chi2_threshold, bool disable_robust_kernel, int& num_outliers)
{
    for (auto& [ft, edges] : edges_by_feature)
    {
        for (size_t i = 0, iend = edges.size(); i < iend; i++)
        {
            EdgeT* e = edges[i];
            const size_t idx = index_by_feature.at(ft)[i];

            // Always recompute at the CURRENT vertex estimate. g2o's LM pops (restores) the
            // estimate after a rejected trial step but leaves every edge's cached _error at
            // the rejected pose — classifying from stale chi2() then rejects all
            // correspondences of a perfectly converged pose (observed: frame 3951, 620
            // matches at 1.5px median residual classified 617/620 outliers -> tracking lost).
            e->computeError();

            const bool is_outlier = e->chi2() > chi2_threshold;
            frame->outliers.at(ft)[idx] = is_outlier;
            e->setLevel(is_outlier ? 1 : 0);
            if(is_outlier)
                num_outliers++;

            if(disable_robust_kernel)
                e->setRobustKernel(0);
        }
    }
}

int Optimizer::pose_optimization(Frame* frame, bool use_depth_channel)
{
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(make_levenberg<g2o::BlockSolver_6_3, g2o::LinearSolverDense>());

    int num_correspondences=0;

    // Set Frame vertex
    g2o::VertexSE3Expmap * pose_vertex = new g2o::VertexSE3Expmap();
    pose_vertex->setEstimate(Converter::toSE3Quat(frame->get_Tcw()));
    pose_vertex->setId(0);
    pose_vertex->setFixed(false);
    optimizer.addVertex(pose_vertex);

    // Set MapPoint vertices
    std::map<FeatureType, std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*>> mono_edges;
    std::map<FeatureType, std::vector<g2o::EdgeRGBDSE3ProjectXYZOnlyPose*>> rgbd_edges;
    std::map<FeatureType, std::vector<size_t>> mono_indices, rgbd_indices;
    for (const auto& [ft, data] : frame->features.per_type){
        mono_edges[ft].reserve(data.count());
        mono_indices[ft].reserve(data.count());
        rgbd_edges[ft].reserve(data.count());
        rgbd_indices[ft].reserve(data.count());
    }

    // No lock here: each position is read atomically under the point's own mutex. They do not all come from one
    // instant, which no constraint here needs, and the optimize() calls below run unlocked anyway.
    for (auto& [ft, pts] : frame->pts) {
        const int num_keypoints = static_cast<int>(frame->features.at(ft).count());
        auto& mono_edges_ft = mono_edges.at(ft);
        auto& mono_indices_ft = mono_indices.at(ft);
        auto& rgbd_edges_ft = rgbd_edges.at(ft);
        auto& rgbd_indices_ft = rgbd_indices.at(ft);
        for(int i = 0; i < num_keypoints; i++)
        {
            Pt pt = pts[i];
            if(!pt)
                continue;

            num_correspondences++;
            frame->outliers[ft][i] = false;

            const vec3f Xw = pt->get_world_pos();
            const KeypointMeasurement measurement = keypoint_measurement(frame->features, ft, i, use_depth_channel);
            if(measurement.has_depth)   // RGB-D sensor-depth observation
            {
                auto* e = create_pose_only_edge<g2o::EdgeRGBDSE3ProjectXYZOnlyPose>(
                    optimizer, measurement.value, measurement.information, ErrorModel::params.huber_rgbd(), frame, Xw);
                rgbd_edges_ft.push_back(e);
                rgbd_indices_ft.push_back(i);
            }
            else                        // mono observation
            {
                auto* e = create_pose_only_edge<g2o::EdgeSE3ProjectXYZOnlyPose>(
                    optimizer, measurement.pixel(), measurement.pixel_information(), ErrorModel::params.huber_mono(), frame, Xw);
                mono_edges_ft.push_back(e);
                mono_indices_ft.push_back(i);
            }
        }
    }

    if(num_correspondences < params.pose_min_correspondences)
        return 0;

    // pose_rounds optimizations; after each one every observation is classified as inlier or outlier.
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    int num_outliers=0;
    std::vector<int> pass_inliers(params.pose_rounds, -1);
    for(int it=0; it<params.pose_rounds; it++)
    {
        pose_vertex->setEstimate(Converter::toSE3Quat(frame->get_Tcw()));
        optimizer.initializeOptimization(0);
        optimizer.optimize(params.pose_iterations);

        num_outliers=0;
        const bool disable_robust_kernel = (it == params.pose_rounds_with_kernel - 1);
        classify_pose_only_edges(mono_edges, mono_indices, frame, ErrorModel::params.chi2_mono, disable_robust_kernel, num_outliers);
        classify_pose_only_edges(rgbd_edges, rgbd_indices, frame, ErrorModel::params.chi2_rgbd, disable_robust_kernel, num_outliers);
        pass_inliers[it] = num_correspondences - num_outliers;

        // Stop refining once too few inliers remain to constrain the pose. This is NOT the
        // same as optimizer.edges().size(): outliers are excluded via setLevel(1), never
        // erased from the graph, so that count never changes across iterations.
        if(num_correspondences - num_outliers < params.pose_min_inliers)
            break;
    }

    // Diagnostic: when pose optimization rejects most of its correspondences, report the
    // per-channel breakdown and the error split of the rejected RGB-D edges (pixel vs
    // inverse-depth residual), so a tracking loss right after can be attributed to the
    // 2D geometry or to the depth channel.
    if(num_correspondences >= params.pose_warn_min_correspondences
       && num_outliers > params.pose_warn_outlier_ratio * num_correspondences)
    {
        const auto median = [](std::vector<float>& v) -> float {
            if(v.empty()) return -1.0f;
            auto mid = v.begin() + v.size() / 2;
            std::nth_element(v.begin(), mid, v.end());
            return *mid;
        };

        int num_mono = 0, num_mono_outliers = 0, num_rgbd = 0, num_rgbd_outliers = 0;
        std::vector<float> rgbd_outlier_px, rgbd_outlier_inv_depth, rgbd_outlier_depth, mono_outlier_px;
        for(const auto& [ft, edges] : mono_edges)
        {
            for(size_t i = 0; i < edges.size(); i++)
            {
                num_mono++;
                if(frame->outliers.at(ft)[mono_indices.at(ft)[i]])
                {
                    num_mono_outliers++;
                    mono_outlier_px.push_back(static_cast<float>(edges[i]->error().norm()));
                }
            }
        }
        for(const auto& [ft, edges] : rgbd_edges)
        {
            for(size_t i = 0; i < edges.size(); i++)
            {
                num_rgbd++;
                const size_t idx = rgbd_indices.at(ft)[i];
                if(frame->outliers.at(ft)[idx])
                {
                    num_rgbd_outliers++;
                    const Eigen::Vector3d err = edges[i]->error();
                    rgbd_outlier_px.push_back(static_cast<float>(err.head<2>().norm()));
                    rgbd_outlier_inv_depth.push_back(static_cast<float>(std::abs(err(2))));
                    const float inv_depth = frame->features.at(ft).inv_depth[idx];
                    if(inv_depth > 0.0f)
                        rgbd_outlier_depth.push_back(1.0f / inv_depth);
                }
            }
        }
        // Initial (frame's pose is still the entry pose here) -> optimized pose delta:
        // large delta = the optimizer moved far (divergence / dragged); tiny delta with
        // uniform pixel errors = the entry pose itself was already inconsistent.
        const g2o::SE3Quat optimized_pose = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0))->estimate();
        const g2o::SE3Quat delta = optimized_pose * Converter::toSE3Quat(frame->get_Tcw()).inverse();
        const double delta_translation = delta.translation().norm();
        const double delta_rotation_deg = 2.0 * std::asin(std::min(1.0, delta.rotation().vec().norm())) * 180.0 / M_PI;

        std::string pass_inliers_text;
        for(const int inliers : pass_inliers)
            pass_inliers_text += (pass_inliers_text.empty() ? "" : ",") + std::to_string(inliers);

        AF_WARN("pose_optimization: rejected " << num_outliers << "/" << num_correspondences
                << " (mono " << num_mono_outliers << "/" << num_mono
                << ", rgbd " << num_rgbd_outliers << "/" << num_rgbd << ")"
                << " | pass_inliers=[" << pass_inliers_text << "]"
                << " | poseDelta: trans=" << delta_translation << "m rot=" << delta_rotation_deg << "deg"
                << " | rejected-rgbd medians: pxErr=" << median(rgbd_outlier_px)
                << " invDepthErr=" << median(rgbd_outlier_inv_depth)
                << " depth=" << median(rgbd_outlier_depth) << "m"
                << " | rejected-mono median pxErr=" << median(mono_outlier_px)
                << " | frame=" << frame->frame_id);
    }

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* optimized_vertex = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    const mat4f final_pose = Converter::to_matrix4f(optimized_vertex->estimate());
    if(!final_pose.allFinite())   // the frame keeps its entry pose and reports no inliers
    {
        AF_WARN("pose_optimization: non-finite pose, not written | frame=" << frame->frame_id);
        return 0;
    }
    frame->set_pose(final_pose);

    return num_correspondences-num_outliers;
}

void Optimizer::local_bundle_adjustment(const Keyframe& current_keyframe, const std::shared_ptr<Map>& map)
{
    // Which keyframes and points are already in the problem (a bad neighbour is marked local too, so that it is
    // not taken as a fixed keyframe either)
    std::unordered_set<KeyframeId> local_keyframe_ids, fixed_keyframe_ids;
    std::unordered_set<PtId> local_point_ids;

    // Local KeyFrames: First Breath Search from Current Keyframe
    std::vector<Keyframe> local_keyframes;

    local_keyframes.push_back(current_keyframe);
    local_keyframe_ids.insert(current_keyframe->keyId);

    std::vector<Keyframe> neighbors = current_keyframe->get_best_covisibility_keyframes(params.local_ba_neighbors);

    for(const Keyframe& keyframe : neighbors)
    {
        local_keyframe_ids.insert(keyframe->keyId);
        if(!keyframe->is_bad())
            local_keyframes.push_back(keyframe);
    }

    // Local MapPoints seen in Local KeyFrames
    std::vector<Pt> local_points;
    for(const Keyframe& keyframe : local_keyframes)
    {
        for(const auto& ft : keyframe->feature_types)
        {
            for(const Pt& pt : keyframe->get_map_point_matches(ft))
            {
                if(pt && !pt->is_bad() && local_point_ids.insert(pt->ptId).second)
                    local_points.push_back(pt);
            }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    std::vector<Keyframe> fixed_keyframes;
    for(const Pt& pt : local_points)
    {
        for(const auto& [key_id, keyframe] : pt->get_observers())
        {
            if(local_keyframe_ids.count(keyframe->keyId) == 0 && fixed_keyframe_ids.insert(keyframe->keyId).second
               && !keyframe->is_bad())
                fixed_keyframes.push_back(keyframe);
        }
    }

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(make_levenberg<g2o::BlockSolver_6_3, g2o::LinearSolverEigen>());

    unsigned long max_keyframe_id = 0;

    // Set Local KeyFrame vertices
    for(const Keyframe& keyframe : local_keyframes)
    {
        g2o::VertexSE3Expmap * pose_vertex = new g2o::VertexSE3Expmap();
        pose_vertex->setEstimate(Converter::toSE3Quat(keyframe->get_Tcw()));
        pose_vertex->setId(keyframe->keyId);
        pose_vertex->setFixed(keyframe->is_origin());

        optimizer.addVertex(pose_vertex);
        if(keyframe->keyId>max_keyframe_id)
            max_keyframe_id=keyframe->keyId;
    }

    // Set Fixed KeyFrame vertices
    for(const Keyframe& keyframe : fixed_keyframes)
    {
        g2o::VertexSE3Expmap * pose_vertex = new g2o::VertexSE3Expmap();
        pose_vertex->setEstimate(Converter::toSE3Quat(keyframe->get_Tcw()));
        pose_vertex->setId(keyframe->keyId);
        pose_vertex->setFixed(true);
        optimizer.addVertex(pose_vertex);
        if(keyframe->keyId > max_keyframe_id)
            max_keyframe_id=keyframe->keyId;
    }

    // Set MapPoint vertices
    int expected_edges = 0;
    for(const Pt& pt : local_points)
        expected_edges += pt->num_observing_keyframes();

    std::vector<BAObservation<g2o::EdgeSE3ProjectXYZ>> mono_observations;
    mono_observations.reserve(expected_edges);

    std::vector<BAObservation<g2o::EdgeRGBDSE3ProjectXYZ>> rgbd_observations;
    rgbd_observations.reserve(expected_edges);

    for(const Pt& pt : local_points)
    {
        const FeatureType feature_type = pt->featureType;

        g2o::VertexSBAPointXYZ* point_vertex = new g2o::VertexSBAPointXYZ();
        point_vertex->setEstimate(pt->get_world_pos().cast<double>());
        int point_id = pt->ptId + max_keyframe_id+1;
        point_vertex->setId(point_id);
        point_vertex->setMarginalized(true);
        optimizer.addVertex(point_vertex);

        std::map<KeyframeId, Observation> observations = pt->get_observations();

        //Set edges
        for(auto& obs: observations)
        {
            Keyframe keyframe = obs.second.keyframe;

            if(!keyframe->is_bad())
            {
                const KeypointMeasurement measurement = keypoint_measurement(keyframe->features, feature_type, obs.second.index);
                if(measurement.has_depth)   // RGB-D sensor-depth observation
                {
                    auto* e = create_ba_edge<g2o::EdgeRGBDSE3ProjectXYZ>(
                        optimizer, point_id, keyframe->keyId, measurement.value, measurement.information,
                        ErrorModel::params.huber_rgbd(), keyframe);
                    if(e)
                        rgbd_observations.push_back({e, keyframe, pt});
                }
                else                        // mono observation
                {
                    auto* e = create_ba_edge<g2o::EdgeSE3ProjectXYZ>(
                        optimizer, point_id, keyframe->keyId, measurement.pixel(), measurement.pixel_information(),
                        ErrorModel::params.huber_mono(), keyframe);
                    if(e)
                        mono_observations.push_back({e, keyframe, pt});
                }
            }
        }
    }

    optimizer.initializeOptimization();
    optimizer.optimize(params.local_ba_iterations);

    // Check inlier observations
    mark_ba_outliers(mono_observations, ErrorModel::params.chi2_mono);
    mark_ba_outliers(rgbd_observations, ErrorModel::params.chi2_rgbd);

    // Optimize again without the outliers
    optimizer.initializeOptimization(0);
    optimizer.optimize(params.local_ba_refine_iterations);

    std::vector<std::pair<Keyframe,Pt> > to_erase;
    to_erase.reserve(mono_observations.size()+rgbd_observations.size());

    // Check inlier observations
    collect_ba_outliers(mono_observations, ErrorModel::params.chi2_mono, to_erase);
    collect_ba_outliers(rgbd_observations, ErrorModel::params.chi2_rgbd, to_erase);

    // Get Map Mutex

    std::unique_lock<std::mutex> lock(map->map_update_mutex_);
    for(const auto& [keyframe, pt] : to_erase)
    {
        keyframe->erase_map_point_match(pt);
        pt->erase_observation(keyframe);
    }

    // Recover optimized data
    int num_rejected = 0;

    //Keyframes
    for(const Keyframe& keyframe : local_keyframes)
    {
        g2o::VertexSE3Expmap* pose_vertex = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(keyframe->keyId));
        g2o::SE3Quat optimized_pose = pose_vertex->estimate();
        // A degenerate edge (a point at ~zero depth) can drive the solution to NaN: keep the previous pose rather
        // than write NaN into the map (from where it reaches tracking, grids and indices)
        const mat4f Tcw = Converter::to_matrix4f(optimized_pose);
        if(!Tcw.allFinite())
        {
            num_rejected++;
            continue;
        }
        keyframe->set_pose(Tcw);
    }

    //Points
    for(const Pt& pt : local_points)
    {
        g2o::VertexSBAPointXYZ* point_vertex = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pt->ptId + max_keyframe_id+1));
        const vec3f position = point_vertex->estimate().cast<float>();
        if(!position.allFinite())
        {
            num_rejected++;
            continue;
        }
        pt->set_world_pos(position);
        pt->update_normal_and_depth();
    }

    if(num_rejected > 0)
        AF_WARN("local_bundle_adjustment: " << num_rejected << " non-finite poses or positions were not written");
}


void Optimizer::optimize_essential_graph(const std::shared_ptr<Map>& map, const Keyframe& matched_keyframe,
                                         const Keyframe& current_keyframe,
                                         const KeyframePoses& uncorrected_poses,
                                         const KeyframePoses& corrected_poses,
                                         std::map<KeyframeId, LoopConnections>& loop_connections,
                                         bool fix_scale)
{
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::OptimizationAlgorithmLevenberg* solver = make_levenberg<g2o::BlockSolver_7_3, g2o::LinearSolverEigen>();
    solver->setUserLambdaInit(params.essential_graph_lambda_init);
    optimizer.setAlgorithm(solver);

    const std::vector<Keyframe> keyframes = map->get_all_keyframes();
    const std::vector<Pt> map_points = map->get_all_map_points();

    const unsigned int max_keyframe_id = map->get_max_keyframe_id();

    std::vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > Scw(max_keyframe_id+1);
    std::vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > corrected_Swc(max_keyframe_id+1);

    // Set KeyFrame vertices
    for(const Keyframe& keyframe : keyframes)
    {
        if(keyframe->is_bad())
            continue;
        g2o::VertexSim3Expmap* sim3_vertex = new g2o::VertexSim3Expmap();

        const int id_i = keyframe->keyId;

        const auto it = corrected_poses.find(keyframe);

        if(it!=corrected_poses.end())
        {
            Scw[id_i] = it->second;
            sim3_vertex->setEstimate(it->second);
        }
        else
        {
            // One lock: R and t from the same pose
            const Pose pose = keyframe->get_pose();
            Eigen::Matrix<double,3,3> Rcw = pose.Rcw.cast<double>();
            Eigen::Matrix<double,3,1> tcw = pose.tcw.cast<double>();
            g2o::Sim3 Siw(Rcw,tcw,1.0);
            Scw[id_i] = Siw;
            sim3_vertex->setEstimate(Siw);
        }

        if(keyframe==matched_keyframe)
            sim3_vertex->setFixed(true);

        sim3_vertex->setId(id_i);
        sim3_vertex->setMarginalized(false);
        sim3_vertex->_fix_scale = fix_scale;

        optimizer.addVertex(sim3_vertex);
    }


    std::set<std::pair<long unsigned int,long unsigned int> > inserted_edges;

    const Eigen::Matrix<double,7,7> identity_information =
        params.essential_graph_edge_information * Eigen::Matrix<double,7,7>::Identity();

    // Every edge between keyframes that both have a vertex: bad keyframes get none (above), yet a bad keyframe can
    // still be a parent, a loop edge or a loop connection of a good one, and g2o's addEdge dereferences a null vertex
    const auto add_sim3_edge = [&](const long unsigned int id_j, const long unsigned int id_i, const g2o::Sim3& Sji) -> bool
    {
        auto* vj = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(int(id_j)));
        auto* vi = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(int(id_i)));
        if(!vj || !vi)
            return false;
        g2o::EdgeSim3* e = new g2o::EdgeSim3();
        e->setVertex(1, vj);
        e->setVertex(0, vi);
        e->setMeasurement(Sji);
        e->information() = identity_information;
        optimizer.addEdge(e);
        return true;
    };

    // Set Loop edges
    for(const auto& [key_id, loop_connection] : loop_connections)
    {
        const Keyframe& keyframe = loop_connection.keyframe;
        const KeyframeId id_i = keyframe->keyId;
        const g2o::Sim3 Siw = Scw[id_i];
        const g2o::Sim3 Swi = Siw.inverse();

        for(const auto& [id_j, connected] : loop_connection.connections)
        {
            if((id_i != current_keyframe->keyId || id_j != matched_keyframe->keyId)
               && keyframe->get_weight(connected) < params.essential_graph_min_weight)
                continue;

            const g2o::Sim3 Sjw = Scw[id_j];
            const g2o::Sim3 Sji = Sjw * Swi;

            if(add_sim3_edge(id_j, id_i, Sji))
                inserted_edges.insert(std::make_pair(std::min(id_i,id_j),std::max(id_i,id_j)));
        }
    }

    // A keyframe's pose before the loop correction: the corrected side keeps its uncorrected pose for these edges
    const auto uncorrected_Scw = [&](const Keyframe& keyframe) -> const g2o::Sim3&
    {
        const auto it = uncorrected_poses.find(keyframe);
        return it != uncorrected_poses.end() ? it->second : Scw[keyframe->keyId];
    };

    // Set normal edges
    for(const Keyframe& keyframe : keyframes)
    {
        const int id_i = keyframe->keyId;
        const g2o::Sim3 Swi = uncorrected_Scw(keyframe).inverse();

        Keyframe parent = keyframe->get_parent();

        // Spanning tree edge
        if(parent)
        {
            const g2o::Sim3 Sji = uncorrected_Scw(parent) * Swi;
            add_sim3_edge(parent->keyId, id_i, Sji);
        }

        // Loop edges
        const KeyframeIdSet loop_edges = keyframe->get_loop_edges();
        for(const Keyframe& loop_edge_keyframe : loop_edges)
        {
            if(loop_edge_keyframe->keyId < keyframe->keyId)
            {
                const g2o::Sim3 Sli = uncorrected_Scw(loop_edge_keyframe) * Swi;
                add_sim3_edge(loop_edge_keyframe->keyId, id_i, Sli);
            }
        }

        // Covisibility graph edges
        const std::vector<Keyframe> covisibles = keyframe->get_covisibles_by_weight(params.essential_graph_min_weight);
        for(const Keyframe& neighbor : covisibles)
        {
            if(neighbor && neighbor!=parent && !keyframe->has_child(neighbor) && !loop_edges.count(neighbor))
            {
                if(!neighbor->is_bad() && neighbor->keyId<keyframe->keyId)
                {
                    if(inserted_edges.count(std::make_pair(std::min(keyframe->keyId,neighbor->keyId),std::max(keyframe->keyId,neighbor->keyId))))
                        continue;

                    const g2o::Sim3 Sni = uncorrected_Scw(neighbor) * Swi;
                    add_sim3_edge(neighbor->keyId, id_i, Sni);
                }
            }
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(params.essential_graph_iterations);

    std::unique_lock<std::mutex> lock(map->map_update_mutex_);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    std::vector<bool> corrected(corrected_Swc.size(), false);   // keyframes whose optimized pose was written
    int num_rejected = 0;
    for(const Keyframe& keyframe : keyframes)
    {

        const int id_i = keyframe->keyId;

        g2o::VertexSim3Expmap* sim3_vertex = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(id_i));
        if(!sim3_vertex)   // a bad keyframe (no vertex): nothing was optimized for it
            continue;
        g2o::Sim3 corrected_Siw =  sim3_vertex->estimate();
        Eigen::Matrix3d R = corrected_Siw.rotation().toRotationMatrix();
        Eigen::Vector3d t = corrected_Siw.translation();
        double s = corrected_Siw.scale();

        t *=(1./s); //[R t/s;0 1]

        mat4f Tiw = Converter::to_matrix4f(R,t);
        if(!Tiw.allFinite())   // keeps its pose; its points are not corrected either (below)
        {
            num_rejected++;
            continue;
        }

        corrected_Swc[id_i]=corrected_Siw.inverse();
        corrected[id_i] = true;
        keyframe->set_pose(Tiw);
    }
    if(num_rejected > 0)
        AF_WARN("optimize_essential_graph: " << num_rejected << " non-finite keyframe poses were not written");

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for(const Pt& pt : map_points)
    {

        if(pt->is_bad())
            continue;

        int reference_id;
        if(pt->corrected_by_keyframe==current_keyframe->keyId)
        {
            reference_id = pt->corrected_reference;
        }
        else
        {
            Keyframe reference_keyframe = pt->get_reference_keyframe();
            if(!reference_keyframe)
                continue;
            reference_id = reference_keyframe->keyId;
        }
        if(reference_id < 0 || size_t(reference_id) >= Scw.size() || size_t(reference_id) >= corrected_Swc.size())
            continue;
        if(!corrected[reference_id])   // a reference with no optimized pose: the point stays where it is
            continue;

        g2o::Sim3 Srw = Scw[reference_id];
        g2o::Sim3 corrected_Swr = corrected_Swc[reference_id];

        vec3f position = pt->get_world_pos();
        Eigen::Matrix<double,3,1> position_d = position.cast<double>();
        Eigen::Matrix<double,3,1> corrected_position = corrected_Swr.map(Srw.map(position_d));

        pt->set_world_pos(corrected_position.cast<float>());

        pt->update_normal_and_depth();
    }
}

int Optimizer::optimize_sim3(const Keyframe& keyframe1, const Keyframe& keyframe2, std::vector<Pt>& matches1,
                             g2o::Sim3& S12, bool fix_scale, FeatureType feature_type)
{
    const float chi2_threshold = params.sim3_chi2;
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(make_levenberg<g2o::BlockSolverX, g2o::LinearSolverDense>());

    const Calibration& calibration1 = *keyframe1->calibration;
    const Calibration& calibration2 = *keyframe2->calibration;

    // Camera poses
    // One lock each: R and t must come from the same pose
    const Pose pose1 = keyframe1->get_pose();
    const Pose pose2 = keyframe2->get_pose();
    const mat3f& R1w = pose1.Rcw;
    const vec3f& t1w = pose1.tcw;
    const mat3f& R2w = pose2.Rcw;
    const vec3f& t2w = pose2.tcw;

    // Set Sim3 vertex
    g2o::VertexSim3Expmap * sim3_vertex = new g2o::VertexSim3Expmap();
    sim3_vertex->_fix_scale=fix_scale;
    sim3_vertex->setEstimate(S12);
    sim3_vertex->setId(0);
    sim3_vertex->setFixed(false);
    sim3_vertex->_principle_point1 << calibration1.cx, calibration1.cy;
    sim3_vertex->_focal_length1 << calibration1.fx, calibration1.fy;
    sim3_vertex->_principle_point2 << calibration2.cx, calibration2.cy;
    sim3_vertex->_focal_length2 << calibration2.fx, calibration2.fy;
    optimizer.addVertex(sim3_vertex);

    // Set MapPoint vertices
    const int num_matches = matches1.size();
    const std::vector<Pt> map_points1 = keyframe1->get_map_point_matches(feature_type);
    std::vector<g2o::EdgeSim3ProjectXYZ*> edges12;
    std::vector<g2o::EdgeInverseSim3ProjectXYZ*> edges21;
    std::vector<size_t> edge_indices;

    edge_indices.reserve(2*num_matches);
    edges12.reserve(2*num_matches);
    edges21.reserve(2*num_matches);

    const float huber_delta = std::sqrt(chi2_threshold);

    int num_correspondences = 0;

    for(int i=0; i<num_matches; i++)
    {
        if(!matches1[i] || i >= int(map_points1.size()))
            continue;

        // map_points1 is a fresh snapshot: LocalMapping keeps running during the Sim3 search and may have erased
        // the current keyframe's point in this slot since the matches were collected
        Pt pt1 = map_points1[i];
        Pt pt2 = matches1[i];
        if(!pt1 || !pt2)
            continue;
        if(pt2->featureType != feature_type)   // pt1 comes from keyframe1's slots of feature_type
            continue;

        const int point_id1 = 2*i+1;
        const int point_id2 = 2*(i+1);

        const int index2 = pt2->get_index_in_keyframe(keyframe2);

        if(pt1->is_bad() || pt2->is_bad() || index2<0)
            continue;

        g2o::VertexSBAPointXYZ* point_vertex1 = new g2o::VertexSBAPointXYZ();
        vec3f P3D1w = pt1->get_world_pos();
        vec3f P3D1c = R1w * P3D1w + t1w;
        point_vertex1->setEstimate(P3D1c.cast<double>());
        point_vertex1->setId(point_id1);
        point_vertex1->setFixed(true);
        optimizer.addVertex(point_vertex1);

        g2o::VertexSBAPointXYZ* point_vertex2 = new g2o::VertexSBAPointXYZ();
        vec3f P3D2w = pt2->get_world_pos();
        vec3f P3D2c = R2w * P3D2w + t2w;
        point_vertex2->setEstimate(P3D2c.cast<double>());
        point_vertex2->setId(point_id2);
        point_vertex2->setFixed(true);
        optimizer.addVertex(point_vertex2);

        num_correspondences++;

        // Set edge x1 = S12*X2
        Eigen::Matrix<double,2,1> measurement1;
        const cv::KeyPoint &keypoint1 = keyframe1->features.at(feature_type).keypoints[i];
        measurement1 << keypoint1.pt.x, keypoint1.pt.y;

        g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(point_id2)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));

        e12->setMeasurement(measurement1);
        e12->setInformation(keyframe1->features.get_information_2d(i, feature_type).cast<double>());

        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(huber_delta);
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        Eigen::Matrix<double,2,1> measurement2;
        const cv::KeyPoint &keypoint2 = keyframe2->features.at(feature_type).keypoints[index2];
        measurement2 << keypoint2.pt.x, keypoint2.pt.y;

        g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(point_id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));

        e21->setMeasurement(measurement2);
        e21->setInformation(keyframe2->features.get_information_2d(index2, feature_type).cast<double>());

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(huber_delta);
        optimizer.addEdge(e21);

        edges12.push_back(e12);
        edges21.push_back(e21);
        edge_indices.push_back(i);
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(params.sim3_iterations);

    // Check inliers
    int num_outliers=0;
    for(size_t i=0; i<edges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = edges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = edges21[i];
        if(!e12 || !e21)
            continue;

        // Recompute at the restored estimate (stale-chi2 hazard after a rejected LM trial,
        // see classify_pose_only_edges).
        e12->computeError();
        e21->computeError();
        if(e12->chi2()>chi2_threshold || e21->chi2()>chi2_threshold)
        {
            size_t idx = edge_indices[i];
            matches1[idx]=static_cast<Pt>(nullptr);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            edges12[i]=static_cast<g2o::EdgeSim3ProjectXYZ*>(nullptr);
            edges21[i]=static_cast<g2o::EdgeInverseSim3ProjectXYZ*>(nullptr);
            num_outliers++;
        }
    }

    int more_iterations;
    if(num_outliers > 0)
        more_iterations = params.sim3_more_iterations_after_outliers;
    else
        more_iterations = params.sim3_more_iterations;

    if(num_correspondences - num_outliers < params.sim3_min_correspondences)
        return 0;

    // Optimize again only with inliers

    optimizer.initializeOptimization();
    optimizer.optimize(more_iterations);

    int num_inliers = 0;
    for(size_t i=0; i<edges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = edges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = edges21[i];
        if(!e12 || !e21)
            continue;

        e12->computeError();
        e21->computeError();
        if(e12->chi2()>chi2_threshold || e21->chi2()>chi2_threshold)
        {
            size_t idx = edge_indices[i];
            matches1[idx]=static_cast<Pt>(nullptr);
        }
        else
            num_inliers++;
    }

    // Recover optimized Sim3
    g2o::VertexSim3Expmap* optimized_vertex = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    const g2o::Sim3 optimized_S12 = optimized_vertex->estimate();
    if(!optimized_S12.rotation().coeffs().allFinite() || !optimized_S12.translation().allFinite()
       || !std::isfinite(optimized_S12.scale()))   // S12 keeps the caller's estimate; the candidate is rejected
    {
        AF_WARN("optimize_sim3: non-finite Sim3, not written");
        return 0;
    }
    S12 = optimized_S12;

    return num_inliers;
}


}