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

#include "Optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/linear_solver_eigen.h"
#include "g2o/types/types_six_dof_expmap.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/solvers/linear_solver_dense.h"
#include "g2o/types/types_seven_dof_expmap.h"

#include<Eigen/StdVector>

#include "Converter.h"

#include<mutex>
#include <algorithm>
#include <vector>

#include "afvslam_log.hpp"

namespace AF_VSLAM
{

OptimizerParameters Optimizer::params{};

void Optimizer::LoadParameters(const cv::FileStorage &fSettings)
{
    auto readIfPresent = [&fSettings](const char* key, auto& field)
    {
        const cv::FileNode node = fSettings[key];
        if(!node.empty())
            node >> field;
    };

    readIfPresent("Optimizer.Chi2Mono", params.chi2_2dof);
    readIfPresent("Optimizer.Chi2Stereo", params.chi2_3dof);
    readIfPresent("Optimizer.Chi2RGBD", params.chi2_3dof_rgbd);
    readIfPresent("Optimizer.InvDepthInfo", params.invDepthInfo);
    readIfPresent("Optimizer.PoseOptIterations", params.numItPoseOpt);
    readIfPresent("Optimizer.EssGraphLambdaInit", params.userLambdaInit);
    readIfPresent("Optimizer.EssGraphMinFeat", params.minFeat);
    readIfPresent("Optimizer.EssGraphIterations", params.numItEssGraphOpt);
    readIfPresent("Optimizer.Sim3Iterations", params.numItSim3Opt);
    readIfPresent("Optimizer.Sim3MoreIterationsHigh", params.nMoreItHigh);
    readIfPresent("Optimizer.Sim3MoreIterationsLow", params.nMoreItLow);
    readIfPresent("Optimizer.Sim3MinCorrespondences", params.minNCorrespondences);

    // Derived, always recomputed -- never read from YAML directly, so they can't desync from chi2_*
    params.thHuber_2dof = sqrtf(params.chi2_2dof);
    params.thHuber_3dof = sqrtf(params.chi2_3dof);
    params.thHuber_3dof_rgbd = sqrtf(params.chi2_3dof_rgbd);
}

// Per-observation information (1/variance) for the RGB-D inverse-depth residual, from
// Frame/KeyFrame::sigma2invDepth (Frame::GetDepth's quadratic depth-noise model) instead of the
// single global params.invDepthInfo placeholder every observation used to share. Falls back to
// that placeholder if sigma2 is non-positive -- shouldn't happen in practice, since Frame::GetDepth
// always sets sigma2invDepth alongside invDepth, but keeps this safe if that invariant ever breaks.
static double RGBDInvDepthInformation(float sigma2invDepth)
{
    return sigma2invDepth > 0.0f ? 1.0 / sigma2invDepth : Optimizer::params.invDepthInfo;
}

// Builds and registers a binary (point + pose) g2o edge shared by the mono/stereo/RGBD
// BundleAdjustment/LocalBundleAdjustment branches: sets both vertices, measurement,
// information, robust kernel and camera intrinsics, then adds it to the optimizer.
// Edge-type-specific extras (e.g. EdgeStereoSE3ProjectXYZ::bf) are set by the caller
// on the returned pointer.
template<typename EdgeT, typename MeasT, typename InfoT>
static EdgeT* CreateBAEdge(g2o::SparseOptimizer& optimizer, int pointId, int keyframeId,
                           const MeasT& measurement, const InfoT& information,
                           double huberDelta, const Keyframe& pKFi, bool useRobustKernel = true)
{
    EdgeT* e = new EdgeT();
    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pointId)));
    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(keyframeId)));
    e->setMeasurement(measurement);
    e->setInformation(information);

    if(useRobustKernel)
    {
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(huberDelta);
    }

    e->fx = pKFi->fx;
    e->fy = pKFi->fy;
    e->cx = pKFi->cx;
    e->cy = pKFi->cy;

    optimizer.addEdge(e);
    return e;
}

// One BA edge plus the (keyframe, map point) pair it connects -- lets LocalBundleAdjustment
// track edges/keyframes/points as a single vector per edge type instead of three parallel ones.
// Named BAObservation (not Observation) to avoid colliding with AF_VSLAM::Observation
// (include/Observation.h), the unrelated projKeyframe/projIndex pairing used elsewhere.
template<typename EdgeT>
struct BAObservation
{
    EdgeT* edge;
    Keyframe kf;
    Pt mp;
};

// Marks each edge of one type bad (level 1) when its chi-square exceeds threshold or its
// point is behind the camera, and always disables its robust kernel -- the pass BA runs
// once, between its initial and refining optimization rounds.
template<typename EdgeT>
static void MarkBAOutliers(const vector<BAObservation<EdgeT>>& observations, float chi2Threshold)
{
    for(const auto& obs : observations)
    {
        if(obs.mp->is_bad())
            continue;

        // Recompute at the current estimates: after a rejected LM trial g2o restores the
        // vertices but leaves cached edge errors at the rejected pose (see
        // ClassifyPoseOnlyEdges) — classifying from stale chi2 here would mark good
        // observations as outliers.
        obs.edge->computeError();
        if(obs.edge->chi2() > chi2Threshold || !obs.edge->isDepthPositive())
            obs.edge->setLevel(1);

        obs.edge->setRobustKernel(0);
    }
}

// Collects (keyframe, map point) pairs whose edge chi-square exceeds threshold, or whose
// point is behind the camera, for erasure after BA's final optimization round.
template<typename EdgeT>
static void CollectBAOutliers(const vector<BAObservation<EdgeT>>& observations, float chi2Threshold,
                              vector<pair<Keyframe,Pt>>& vToErase)
{
    for(const auto& obs : observations)
    {
        if(obs.mp->is_bad())
            continue;

        // Same stale-chi2 hazard as MarkBAOutliers — erasing observations from the map
        // based on a rejected trial pose would silently damage the map.
        obs.edge->computeError();
        if(obs.edge->chi2() > chi2Threshold || !obs.edge->isDepthPositive())
            vToErase.push_back(make_pair(obs.kf, obs.mp));
    }
}

void Optimizer::global_bundle_adjustment(shared_ptr<Map> pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<Keyframe> vpKFs = pMap->GetAllKeyFrames();
    vector<Pt> vpMP = pMap->get_all_map_points();
    BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag, nLoopKF, bRobust);
}


void Optimizer::BundleAdjustment(const vector<Keyframe > &vpKFs, const vector<Pt> &vpMP,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);


    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    // Set KeyFrame vertices
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        Keyframe pKF = vpKFs[i];
        if(pKF->is_bad())
            continue;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKF->get_pose()));
        vSE3->setId(pKF->keyId);
        vSE3->setFixed(pKF->keyId == 0);
        optimizer.addVertex(vSE3);
        if(pKF->keyId > maxKFid)
            maxKFid=pKF->keyId;
    }

    // Set MapPoint vertices
    for(size_t i=0; i<vpMP.size(); i++)
    {
        Pt pMP = vpMP[i];
        FeatureType featType = pMP->featureType;

        if(pMP->is_bad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();

        vPoint->setEstimate(pMP->get_world_pos().cast<double>());
        const int id = pMP->ptId + maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

       const map<KeyframeId ,Obs> observations = pMP->GetObservations();

        int nEdges = 0;
        //SET EDGES
        for(auto& obs: observations)
        {

            Keyframe pKF = obs.second->projKeyframe;
            if(pKF->is_bad() || pKF->keyId > maxKFid)
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->keypoints.at(featType)[obs.second->projIndex];
            const float invDepth_i = pKF->invDepth.at(featType)[obs.second->projIndex];

            if(invDepth_i > 0.0f)   // RGB-D sensor-depth observation
            {
                Eigen::Matrix<double,3,1> obs3D;
                obs3D << kpUn.pt.x, kpUn.pt.y, invDepth_i;

                const float sigma2invDepth_i = pKF->sigma2invDepth.at(featType)[obs.second->projIndex];
                mat3f infMat = mat3f::Zero();
                infMat.block<2,2>(0,0) = pKF->GetKeyPt2DInf(obs.second->projIndex, featType);
                infMat(2,2) = static_cast<float>(RGBDInvDepthInformation(sigma2invDepth_i));

                CreateBAEdge<g2o::EdgeRGBDSE3ProjectXYZ>(
                    optimizer, id, pKF->keyId, obs3D, infMat.cast<double>(), params.thHuber_3dof_rgbd, pKF, bRobust);
            }
            else if(pKF->mvuRight.at(featType)[obs.second->projIndex]<0)   // mono observation
            {
                Eigen::Matrix<double,2,1> obs2D;
                obs2D << kpUn.pt.x, kpUn.pt.y;

                CreateBAEdge<g2o::EdgeSE3ProjectXYZ>(
                    optimizer, id, pKF->keyId, obs2D,
                    pKF->GetKeyPt2DInf(obs.second->projIndex, featType).cast<double>(), params.thHuber_2dof, pKF, bRobust);
            }
            else // Stereo observation
            {
                Eigen::Matrix<double,3,1> obs3D;
                obs3D << kpUn.pt.x, kpUn.pt.y, pKF->mvuRight.at(featType)[obs.second->projIndex];

                CreateBAEdge<g2o::EdgeStereoSE3ProjectXYZ>(
                    optimizer, id, pKF->keyId, obs3D,
                    pKF->GetKeyPt3DInf(obs.second->projIndex, featType).cast<double>(), params.thHuber_3dof, pKF, bRobust)
                    ->bf = pKF->mbf;
            }
        }

        if(nEdges==0)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;
        }
        else
        {
            vbNotIncludedMP[i]=false;
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Recover optimized data
    //Keyframes
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        Keyframe pKF = vpKFs[i];
        if(pKF->is_bad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->keyId));
        g2o::SE3Quat SE3quat = vSE3->estimate();

        if(nLoopKF==0)
        {
            pKF->set_pose(Converter::toMatrix4f(SE3quat));
        }
        else
        {
            pKF->TcwGBA = Converter::toMatrix4f(SE3quat);
            pKF->mnBAGlobalForKF = nLoopKF;
        }
    }

    //Points
    for(size_t i=0; i<vpMP.size(); i++)
    {
        if(vbNotIncludedMP[i])
            continue;

        Pt pMP = vpMP[i];

        if(pMP->is_bad())
            continue;

        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->ptId + maxKFid+1));

        if(nLoopKF==0)
        {
            pMP->set_world_pos(vPoint->estimate().cast<float>());
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            pMP->PosGBA = vPoint->estimate().cast<float>();
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }

}

// Builds and registers a unary pose-only g2o edge shared by the mono/stereo/RGBD
// PoseOptimization() branches: sets the frame vertex, measurement, information,
// robust kernel, camera intrinsics and the (fixed) map point position, then adds it
// to the optimizer. Edge-type-specific extras (e.g. EdgeStereoSE3ProjectXYZOnlyPose::bf)
// are set by the caller on the returned pointer.
template<typename EdgeT, typename MeasT, typename InfoT>
static EdgeT* CreatePoseOnlyEdge(g2o::SparseOptimizer& optimizer, const MeasT& measurement,
                                 const InfoT& information, double huberDelta,
                                 const Frame* pFrame, const vec3f& Xw)
{
    EdgeT* e = new EdgeT();
    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
    e->setMeasurement(measurement);
    e->setInformation(information);

    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    rk->setDelta(huberDelta);

    e->fx = pFrame->fx;
    e->fy = pFrame->fy;
    e->cx = pFrame->cx;
    e->cy = pFrame->cy;
    e->Xw[0] = Xw(0);
    e->Xw[1] = Xw(1);
    e->Xw[2] = Xw(2);

    optimizer.addEdge(e);
    return e;
}

// Classifies every edge of one type as inlier/outlier against its chi-square threshold
// for the current optimization round -- the logic PoseOptimization() previously
// repeated once per edge type (mono/stereo/RGBD).
template<typename EdgeT>
static void ClassifyPoseOnlyEdges(const std::map<FeatureType, vector<EdgeT*>>& edgesByFeature,
                                  const std::map<FeatureType, vector<size_t>>& indexByFeature,
                                  Frame* pFrame, float chi2Threshold, bool disableRobustKernel, int& nBad)
{
    for (auto& [ft, edges] : edgesByFeature)
    {
        for (size_t i = 0, iend = edges.size(); i < iend; i++)
        {
            EdgeT* e = edges[i];
            const size_t idx = indexByFeature.at(ft)[i];

            // Always recompute at the CURRENT vertex estimate. g2o's LM pops (restores) the
            // estimate after a rejected trial step but leaves every edge's cached _error at
            // the rejected pose — classifying from stale chi2() then rejects all
            // correspondences of a perfectly converged pose (observed: frame 3951, 620
            // matches at 1.5px median residual classified 617/620 outliers -> tracking lost).
            e->computeError();

            const bool isOutlier = e->chi2() > chi2Threshold;
            pFrame->mvbOutlier.at(ft)[idx] = isOutlier;
            e->setLevel(isOutlier ? 1 : 0);
            if(isOutlier)
                nBad++;

            if(disableRobustKernel)
                e->setRobustKernel(0);
        }
    }
}

int Optimizer::PoseOptimization(Frame *pFrame, const bool useDepthChannel)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;

    // Set Frame vertex
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->Tcw));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    std::map<FeatureType, std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*>> vpEdgesMono;
    std::map<FeatureType, vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*>> vpEdgesStereo;
    std::map<FeatureType, vector<g2o::EdgeRGBDSE3ProjectXYZOnlyPose*>> vpEdgesRGBD;
    std::map<FeatureType, vector<size_t>> vnIndexEdgeMono, vnIndexEdgeStereo, vnIndexEdgeRGBD;
    for (const auto& [ft, N] : pFrame->N){
        vpEdgesMono[ft].reserve(N);
        vnIndexEdgeMono[ft].reserve(N);
        vpEdgesStereo[ft].reserve(N);
        vnIndexEdgeStereo[ft].reserve(N);
        vpEdgesRGBD[ft].reserve(N);
        vnIndexEdgeRGBD[ft].reserve(N);
    }

    {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);
    for (auto& [ft, pts] : pFrame->pts) {
        const int N_ft = pFrame->N.at(ft);
        const auto& keypointsFt = pFrame->keypoints.at(ft);
        const auto& invDepthFt = pFrame->invDepth.at(ft);
        const auto& sigma2invDepthFt = pFrame->sigma2invDepth.at(ft);
        const auto& mvuRightFt = pFrame->mvuRight.at(ft);
        auto& edgesMonoFt = vpEdgesMono.at(ft);
        auto& idxMonoFt = vnIndexEdgeMono.at(ft);
        auto& edgesStereoFt = vpEdgesStereo.at(ft);
        auto& idxStereoFt = vnIndexEdgeStereo.at(ft);
        auto& edgesRGBDFt = vpEdgesRGBD.at(ft);
        auto& idxRGBDFt = vnIndexEdgeRGBD.at(ft);
        for(int i = 0; i < N_ft; i++)
        {
            Pt pMP = pts[i];
            if(!pMP)
                continue;

            nInitialCorrespondences++;
            pFrame->mvbOutlier[ft][i] = false;

            const cv::KeyPoint &kpUn = keypointsFt[i];
            const vec3f Xw = pMP->get_world_pos();
            const float invDepth_i = invDepthFt[i];

            if(useDepthChannel && invDepth_i > 0.0f)   // RGB-D sensor-depth observation
            {
                Eigen::Matrix<double,3,1> obs3d;
                obs3d << kpUn.pt.x, kpUn.pt.y, invDepth_i;

                mat3f infMat = mat3f::Zero();
                infMat.block<2,2>(0,0) = pFrame->GetKeyPt2DInf(i, ft);
                infMat(2,2) = static_cast<float>(RGBDInvDepthInformation(sigma2invDepthFt[i]));

                auto* e = CreatePoseOnlyEdge<g2o::EdgeRGBDSE3ProjectXYZOnlyPose>(
                    optimizer, obs3d, infMat.cast<double>(), params.thHuber_3dof_rgbd, pFrame, Xw);

                edgesRGBDFt.push_back(e);
                idxRGBDFt.push_back(i);
            }
            else if(mvuRightFt[i] < 0)   // mono observation
            {
                Eigen::Matrix<double,2,1> obs2D;
                obs2D << kpUn.pt.x, kpUn.pt.y;

                auto* e = CreatePoseOnlyEdge<g2o::EdgeSE3ProjectXYZOnlyPose>(
                    optimizer, obs2D, pFrame->GetKeyPt2DInf(i, ft).cast<double>(), params.thHuber_2dof, pFrame, Xw);

                edgesMonoFt.push_back(e);
                idxMonoFt.push_back(i);
            }
            else   // Stereo observation
            {
                Eigen::Matrix<double,3,1> obs3d;
                obs3d << kpUn.pt.x, kpUn.pt.y, mvuRightFt[i];

                auto* e = CreatePoseOnlyEdge<g2o::EdgeStereoSE3ProjectXYZOnlyPose>(
                    optimizer, obs3d, pFrame->GetKeyPt3DInf(i, ft).cast<double>(), params.thHuber_3dof, pFrame, Xw);
                e->bf = pFrame->mbf;

                edgesStereoFt.push_back(e);
                idxStereoFt.push_back(i);
            }
        }
    }
    }

    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4]={params.chi2_2dof,params.chi2_2dof,params.chi2_2dof,params.chi2_2dof};
    const float chi2Stereo[4]={params.chi2_3dof,params.chi2_3dof,params.chi2_3dof,params.chi2_3dof};
    const float chi2RGBD[4]={params.chi2_3dof_rgbd,params.chi2_3dof_rgbd,params.chi2_3dof_rgbd,params.chi2_3dof_rgbd};
    const int its[4]={params.numItPoseOpt,params.numItPoseOpt,params.numItPoseOpt,params.numItPoseOpt};

    int nBad=0;
    int passInliers[4] = {-1,-1,-1,-1};
    for(size_t it=0; it<4; it++)
    {
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->Tcw));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad=0;
        const bool disableRobustKernel = (it==2);
        ClassifyPoseOnlyEdges(vpEdgesMono, vnIndexEdgeMono, pFrame, chi2Mono[it], disableRobustKernel, nBad);
        ClassifyPoseOnlyEdges(vpEdgesStereo, vnIndexEdgeStereo, pFrame, chi2Stereo[it], disableRobustKernel, nBad);
        ClassifyPoseOnlyEdges(vpEdgesRGBD, vnIndexEdgeRGBD, pFrame, chi2RGBD[it], disableRobustKernel, nBad);
        passInliers[it] = nInitialCorrespondences - nBad;

        // Stop refining once too few inliers remain to constrain the pose. This is NOT the
        // same as optimizer.edges().size(): outliers are excluded via setLevel(1), never
        // erased from the graph, so that count never changes across iterations.
        if(nInitialCorrespondences - nBad < 10)
            break;
    }

    // Diagnostic: when pose optimization rejects most of its correspondences, report the
    // per-channel breakdown and the error split of the rejected RGB-D edges (pixel vs
    // inverse-depth residual), so a tracking loss right after can be attributed to the
    // 2D geometry or to the depth channel (see CLAUDE.md, Stop-Induced Keyframe Runaway).
    if(nInitialCorrespondences >= 20 && nBad * 2 > nInitialCorrespondences)
    {
        const auto median = [](std::vector<float>& v) -> float {
            if(v.empty()) return -1.0f;
            auto mid = v.begin() + v.size() / 2;
            std::nth_element(v.begin(), mid, v.end());
            return *mid;
        };

        int nMono = 0, nMonoBad = 0, nRGBD = 0, nRGBDBad = 0, nStereo = 0, nStereoBad = 0;
        std::vector<float> rgbdBadPx, rgbdBadInvD, rgbdBadDepth, monoBadPx;
        for(const auto& [ft, edges] : vpEdgesMono)
        {
            for(size_t i = 0; i < edges.size(); i++)
            {
                nMono++;
                if(pFrame->mvbOutlier.at(ft)[vnIndexEdgeMono.at(ft)[i]])
                {
                    nMonoBad++;
                    monoBadPx.push_back(static_cast<float>(edges[i]->error().norm()));
                }
            }
        }
        for(const auto& [ft, edges] : vpEdgesStereo)
        {
            for(size_t i = 0; i < edges.size(); i++)
            {
                nStereo++;
                if(pFrame->mvbOutlier.at(ft)[vnIndexEdgeStereo.at(ft)[i]])
                    nStereoBad++;
            }
        }
        for(const auto& [ft, edges] : vpEdgesRGBD)
        {
            for(size_t i = 0; i < edges.size(); i++)
            {
                nRGBD++;
                const size_t idx = vnIndexEdgeRGBD.at(ft)[i];
                if(pFrame->mvbOutlier.at(ft)[idx])
                {
                    nRGBDBad++;
                    const Eigen::Vector3d err = edges[i]->error();
                    rgbdBadPx.push_back(static_cast<float>(err.head<2>().norm()));
                    rgbdBadInvD.push_back(static_cast<float>(std::abs(err(2))));
                    const float invD = pFrame->invDepth.at(ft)[idx];
                    if(invD > 0.0f)
                        rgbdBadDepth.push_back(1.0f / invD);
                }
            }
        }
        // Initial (pFrame->Tcw is still the entry pose here) -> optimized pose delta:
        // large delta = the optimizer moved far (divergence / dragged); tiny delta with
        // uniform pixel errors = the entry pose itself was already inconsistent.
        const g2o::SE3Quat optPose = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0))->estimate();
        const g2o::SE3Quat delta = optPose * Converter::toSE3Quat(pFrame->Tcw).inverse();
        const double dTrans = delta.translation().norm();
        const double dRotDeg = 2.0 * std::asin(std::min(1.0, delta.rotation().vec().norm())) * 180.0 / M_PI;

        AF_WARN("PoseOptimization: rejected " << nBad << "/" << nInitialCorrespondences
                << " (mono " << nMonoBad << "/" << nMono
                << ", stereo " << nStereoBad << "/" << nStereo
                << ", rgbd " << nRGBDBad << "/" << nRGBD << ")"
                << " | passInliers=[" << passInliers[0] << "," << passInliers[1]
                << "," << passInliers[2] << "," << passInliers[3] << "]"
                << " | poseDelta: trans=" << dTrans << "m rot=" << dRotDeg << "deg"
                << " | rejected-rgbd medians: pxErr=" << median(rgbdBadPx)
                << " invDepthErr=" << median(rgbdBadInvD)
                << " depth=" << median(rgbdBadDepth) << "m"
                << " | rejected-mono median pxErr=" << median(monoBadPx)
                << " | frame=" << pFrame->mnId);
    }

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    pFrame->set_pose(Converter::toMatrix4f(SE3quat_recov));

    return nInitialCorrespondences-nBad;
}

void Optimizer::LocalBundleAdjustment(Keyframe pKF, [[maybe_unused]] bool* pbStopFlag, shared_ptr<Map> pMap)
{

    // Local KeyFrames: First Breath Search from Current Keyframe
    list<Keyframe> lLocalKeyFrames;

    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->keyId;

    #ifdef ALLFEATURE_REAL_TIME
    vector<Keyframe> vNeighKFs = pKF->GetBestCovisibilityKeyFrames(10);
    //vector<Keyframe> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    #else
    vector<Keyframe> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    #endif

    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        Keyframe pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->keyId;
        if(!pKFi->is_bad())
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames
    list<Pt> llocalPts;
    for(list<Keyframe>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        for (const auto& ft : (*lit)->featureTypes){
            vector<Pt> vpMPs = (*lit)->get_map_point_matches(ft);
            for(vector<Pt>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
            {
                Pt pMP = *vit;
                if(pMP)
                    if(!pMP->is_bad())
                        if(pMP->mnBALocalForKF!=pKF->keyId)
                        {
                            llocalPts.push_back(pMP);
                            pMP->mnBALocalForKF=pKF->keyId;
                        }
            }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    list<Keyframe> lFixedCameras;
    for(list<Pt>::iterator lit=llocalPts.begin(), lend=llocalPts.end(); lit!=lend; lit++)
    {
        map<KeyframeId , Obs> observations = (*lit)->GetObservations();
        for(auto& obs :observations)
        {
            Keyframe pKFi = obs.second->projKeyframe;

            if(pKFi->mnBALocalForKF!=pKF->keyId && pKFi->mnBAFixedForKF != pKF->keyId)
            {
                pKFi->mnBAFixedForKF=pKF->keyId;
                if(!pKFi->is_bad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // if(pbStopFlag)
    //     optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // Set Local KeyFrame vertices
    for(list<Keyframe>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        Keyframe pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->get_pose()));
        vSE3->setId(pKFi->keyId);
        vSE3->setFixed(pKFi->keyId==0);

        optimizer.addVertex(vSE3);
        if(pKFi->keyId>maxKFid)
            maxKFid=pKFi->keyId;
    }

    // Set Fixed KeyFrame vertices
    for(list<Keyframe>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        Keyframe pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->get_pose()));
        vSE3->setId(pKFi->keyId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->keyId > maxKFid)
            maxKFid=pKFi->keyId;
    }

    // Set MapPoint vertices
    int nExpectedSize = 0;
    for(const Pt& pMP : llocalPts)
        nExpectedSize += pMP->Getnumber_of_observations();

    vector<BAObservation<g2o::EdgeSE3ProjectXYZ>> obsMono;
    obsMono.reserve(nExpectedSize);

    vector<BAObservation<g2o::EdgeStereoSE3ProjectXYZ>> obsStereo;
    obsStereo.reserve(nExpectedSize);

    vector<BAObservation<g2o::EdgeRGBDSE3ProjectXYZ>> obsRGBD;
    obsRGBD.reserve(nExpectedSize);

    for(list<Pt>::iterator lit=llocalPts.begin(), lend=llocalPts.end(); lit!=lend; lit++)
    {
        Pt pMP = *lit;
        const FeatureType featType = pMP->featureType;

        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->get_world_pos().cast<double>());
        int id = pMP->ptId + maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        map<KeyframeId , Obs> observations = pMP->GetObservations();

        //Set edges
        for(auto& obs: observations)
        {
            Keyframe pKFi = obs.second->projKeyframe;

            if(!pKFi->is_bad())
            {
                const cv::KeyPoint &kpUn = pKFi->keypoints.at(featType)[obs.second->projIndex];
                const float invDepth_i = pKFi->invDepth.at(featType)[obs.second->projIndex];

                if(invDepth_i > 0.0f)   // RGB-D sensor-depth observation
                {
                    Eigen::Matrix<double,3,1> obs3D;
                    obs3D << kpUn.pt.x, kpUn.pt.y, invDepth_i;

                    const float sigma2invDepth_i = pKFi->sigma2invDepth.at(featType)[obs.second->projIndex];
                    mat3f infMat = mat3f::Zero();
                    infMat.block<2,2>(0,0) = pKFi->GetKeyPt2DInf(obs.second->projIndex, featType);
                    infMat(2,2) = static_cast<float>(RGBDInvDepthInformation(sigma2invDepth_i));

                    auto* e = CreateBAEdge<g2o::EdgeRGBDSE3ProjectXYZ>(
                        optimizer, id, pKFi->keyId, obs3D, infMat.cast<double>(), params.thHuber_3dof_rgbd, pKFi);

                    obsRGBD.push_back({e, pKFi, pMP});
                }
                else if(pKFi->mvuRight.at(featType)[obs.second->projIndex] < 0)   // mono observation
                {
                    Eigen::Matrix<double,2,1> obs2D;
                    obs2D << kpUn.pt.x, kpUn.pt.y;

                    auto* e = CreateBAEdge<g2o::EdgeSE3ProjectXYZ>(
                        optimizer, id, pKFi->keyId, obs2D,
                        pKFi->GetKeyPt2DInf(obs.second->projIndex, featType).cast<double>(), params.thHuber_2dof, pKFi);

                    obsMono.push_back({e, pKFi, pMP});
                }
                else // Stereo observation
                {
                    Eigen::Matrix<double,3,1> obs3D;
                    obs3D << kpUn.pt.x, kpUn.pt.y, pKFi->mvuRight.at(featType)[obs.second->projIndex];

                    auto* e = CreateBAEdge<g2o::EdgeStereoSE3ProjectXYZ>(
                        optimizer, id, pKFi->keyId, obs3D,
                        pKFi->GetKeyPt3DInf(obs.second->projIndex, featType).cast<double>(), params.thHuber_3dof, pKFi);
                    e->bf = pKFi->mbf;

                    obsStereo.push_back({e, pKFi, pMP});
                }
            }
        }
    }

    // if(pbStopFlag)
    //     if(*pbStopFlag)
    //         return;

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore= true;

    // if(pbStopFlag)
    //     if(*pbStopFlag)
    //         bDoMore = false;

    if(bDoMore)
    {

    // Check inlier observations
    MarkBAOutliers(obsMono, params.chi2_2dof);
    MarkBAOutliers(obsStereo, params.chi2_3dof);
    MarkBAOutliers(obsRGBD, params.chi2_3dof_rgbd);

    // Optimize again without the outliers

    optimizer.initializeOptimization(0);
    optimizer.optimize(10);

    }

    vector<pair<Keyframe,Pt> > vToErase;
    vToErase.reserve(obsMono.size()+obsStereo.size()+obsRGBD.size());

    // Check inlier observations
    CollectBAOutliers(obsMono, params.chi2_2dof, vToErase);
    CollectBAOutliers(obsStereo, params.chi2_3dof, vToErase);
    CollectBAOutliers(obsRGBD, params.chi2_3dof_rgbd, vToErase);

    // Get Map Mutex

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);
    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            Keyframe pKFi = vToErase[i].first;
            Pt pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data

    //Keyframes
    for(list<Keyframe>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        Keyframe pKF = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->keyId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->set_pose(Converter::toMatrix4f(SE3quat));
    }

    //Points
    for(list<Pt>::iterator lit=llocalPts.begin(), lend=llocalPts.end(); lit!=lend; lit++)
    {
        Pt pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->ptId + maxKFid+1));
        pMP->set_world_pos(vPoint->estimate().cast<float>());
        pMP->UpdateNormalAndDepth();
    }
}


void Optimizer::OptimizeEssentialGraph(shared_ptr<Map> pMap, Keyframe pLoopKF, Keyframe pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       std::map<KeyframeId, LoopConnections> &loopConnections,
                                       const bool &bFixScale)
{
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
            new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(params.userLambdaInit);
    optimizer.setAlgorithm(solver);

    const vector<Keyframe> vpKFs = pMap->GetAllKeyFrames();
    const vector<Pt> vpMPs = pMap->get_all_map_points();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);

    // Set KeyFrame vertices
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        Keyframe pKF = vpKFs[i];
        if(pKF->is_bad())
            continue;
        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKF->keyId;

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

        if(it!=CorrectedSim3.end())
        {
            vScw[nIDi] = it->second;
            VSim3->setEstimate(it->second);
        }
        else
        {
            Eigen::Matrix<double,3,3> Rcw = pKF->get_rotation().cast<double>();
            Eigen::Matrix<double,3,1> tcw = pKF->get_translation().cast<double>();
            g2o::Sim3 Siw(Rcw,tcw,1.0);
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        if(pKF==pLoopKF)
            VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = bFixScale;

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;
    }


    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;

    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

    // Set Loop edges
    for(auto& mit : loopConnections)
    {
        Keyframe pKF = mit.second.keyframe;
        const long unsigned int nIDi = pKF->keyId;
        const map<KeyframeId, Keyframe> &spConnections = mit.second.connections;
        const g2o::Sim3 Siw = vScw[nIDi];
        const g2o::Sim3 Swi = Siw.inverse();

        for(auto& sit : spConnections)
        {
            const long unsigned int nIDj = sit.first;
            if((nIDi!=pCurKF->keyId || nIDj!=pLoopKF->keyId) && pKF->GetWeight(sit.second) < params.minFeat)
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;

            optimizer.addEdge(e);

            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));
        }
    }

    // Set normal edges
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        Keyframe pKF = vpKFs[i];

        const int nIDi = pKF->keyId;

        g2o::Sim3 Swi;

        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

        if(iti!=NonCorrectedSim3.end())
            Swi = (iti->second).inverse();
        else
            Swi = vScw[nIDi].inverse();

        Keyframe pParentKF = pKF->GetParent();

        // Spanning tree edge
        if(pParentKF)
        {
            int nIDj = pParentKF->keyId;

            g2o::Sim3 Sjw;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            if(itj!=NonCorrectedSim3.end())
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];

            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // Loop edges
        const set<Keyframe> sLoopEdges = pKF->GetLoopEdges();
        for(set<Keyframe>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            Keyframe pLKF = *sit;
            if(pLKF->keyId < pKF->keyId)
            {
                g2o::Sim3 Slw;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                if(itl!=NonCorrectedSim3.end())
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->keyId];

                g2o::Sim3 Sli = Slw * Swi;
                g2o::EdgeSim3* el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->keyId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Covisibility graph edges
        const vector<Keyframe> vpConnectedKFs = pKF->GetCovisiblesByWeight(params.minFeat);
        for(vector<Keyframe>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            Keyframe pKFn = *vit;
            if(pKFn && pKFn!=pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
            {
                if(!pKFn->is_bad() && pKFn->keyId<pKF->keyId)
                {
                    if(sInsertedEdges.count(make_pair(min(pKF->keyId,pKFn->keyId),max(pKF->keyId,pKFn->keyId))))
                        continue;

                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    if(itn!=NonCorrectedSim3.end())
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->keyId];

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3* en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->keyId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(params.numItEssGraphOpt);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for(size_t i=0;i<vpKFs.size();i++)
    {
        Keyframe pKFi = vpKFs[i];

        const int nIDi = pKFi->keyId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        double s = CorrectedSiw.scale();

        eigt *=(1./s); //[R t/s;0 1]

        mat4f Tiw = Converter::toMatrix4f(eigR,eigt);

        pKFi->set_pose(Tiw);
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        Pt pMP = vpMPs[i];

        if(pMP->is_bad())
            continue;

        int nIDr;
        if(pMP->mnCorrectedByKF==pCurKF->keyId)
        {
            nIDr = pMP->mnCorrectedReference;
        }
        else
        {
            Keyframe pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->keyId;
        }


        g2o::Sim3 Srw = vScw[nIDr];
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        vec3f P3Dw = pMP->get_world_pos();
        Eigen::Matrix<double,3,1> eigP3Dw = P3Dw.cast<double>();
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

        pMP->set_world_pos(eigCorrectedP3Dw.cast<float>());

        pMP->UpdateNormalAndDepth();
    }
}

int Optimizer::OptimizeSim3(Keyframe pKF1, Keyframe pKF2, vector<Pt > &vpMatches1, g2o::Sim3 &g2oS12, const float th2, const bool bFixScale,const FeatureType& featureType)
{

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Calibration
    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    // Camera poses
    const mat3f R1w = pKF1->get_rotation();
    const vec3f t1w = pKF1->get_translation();
    const mat3f R2w = pKF2->get_rotation();
    const vec3f t2w = pKF2->get_translation();

    // Set Sim3 vertex
    g2o::VertexSim3Expmap * vSim3 = new g2o::VertexSim3Expmap();
    vSim3->_fix_scale=bFixScale;
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->_principle_point1[0] = K1.at<float>(0,2);
    vSim3->_principle_point1[1] = K1.at<float>(1,2);
    vSim3->_focal_length1[0] = K1.at<float>(0,0);
    vSim3->_focal_length1[1] = K1.at<float>(1,1);
    vSim3->_principle_point2[0] = K2.at<float>(0,2);
    vSim3->_principle_point2[1] = K2.at<float>(1,2);
    vSim3->_focal_length2[0] = K2.at<float>(0,0);
    vSim3->_focal_length2[1] = K2.at<float>(1,1);
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices
    const int N = vpMatches1.size();
    const vector<Pt> vpMapPoints1 = pKF1->get_map_point_matches(featureType);
    vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12;
    vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21;
    vector<size_t> vnIndexEdge;

    vnIndexEdge.reserve(2*N);
    vpEdges12.reserve(2*N);
    vpEdges21.reserve(2*N);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;

    for(int i=0; i<N; i++)
    {
        if(!vpMatches1[i])
            continue;

        Pt pMP1 = vpMapPoints1[i];
        Pt pMP2 = vpMatches1[i];
        const FeatureType featType1 = pMP1->featureType;
        const FeatureType featType2 = pMP2->featureType;
        if (featType1 != featType2)
            continue;

        const int id1 = 2*i+1;
        const int id2 = 2*(i+1);

        const int i2 = pMP2->GetIndexInKeyFrame(pKF2);

        if(pMP1 && pMP2)
        {
            if(!pMP1->is_bad() && !pMP2->is_bad() && i2>=0)
            {
                g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
                vec3f P3D1w = pMP1->get_world_pos();
                vec3f P3D1c = R1w * P3D1w + t1w;
                vPoint1->setEstimate(P3D1c.cast<double>());
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                vec3f P3D2w = pMP2->get_world_pos();
                vec3f P3D2c = R2w * P3D2w + t2w;
                vPoint2->setEstimate(P3D2c.cast<double>());
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
                continue;
        }
        else
            continue;

        nCorrespondences++;

        // Set edge x1 = S12*X2
        Eigen::Matrix<double,2,1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->keypoints.at(featType1)[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));

        e12->setMeasurement(obs1);
        e12->setInformation(pKF1->GetKeyPt2DInf(i, featType1).cast<double>());

        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        Eigen::Matrix<double,2,1> obs2;
        const cv::KeyPoint &kpUn2 = pKF2->keypoints.at(featType2)[i2];
        obs2 << kpUn2.pt.x, kpUn2.pt.y;

        g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));

        e21->setMeasurement(obs2);
        e21->setInformation(pKF2->GetKeyPt2DInf(i2, featType2).cast<double>());

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(params.numItSim3Opt);

    // Check inliers
    int nBad=0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        // Recompute at the restored estimate (stale-chi2 hazard after a rejected LM trial,
        // see ClassifyPoseOnlyEdges).
        e12->computeError();
        e21->computeError();
        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<Pt>(NULL);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i]=static_cast<g2o::EdgeSim3ProjectXYZ*>(NULL);
            vpEdges21[i]=static_cast<g2o::EdgeInverseSim3ProjectXYZ*>(NULL);
            nBad++;
        }
    }

    int nMoreIterations;
    if(nBad > 0)
        nMoreIterations = params.nMoreItHigh;
    else
        nMoreIterations = params.nMoreItLow;

    if(nCorrespondences - nBad < params.minNCorrespondences)
        return 0;

    // Optimize again only with inliers

    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        e12->computeError();
        e21->computeError();
        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<Pt>(NULL);
        }
        else
            nIn++;
    }

    // Recover optimized Sim3
    g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    g2oS12= vSim3_recov->estimate();

    return nIn;
}


} //namespace ORB_SLAM
