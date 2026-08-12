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

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "LoopClosing.h"
#include "Frame.h"

#include "g2o/types/types_seven_dof_expmap.h"
namespace AF_VSLAM
{

class LoopClosing;

struct OptimizerParameters
{
    // Chi-square outlier thresholds (95% for 2/3-DoF residuals)
    float chi2_2dof{5.991f};
    float chi2_3dof{7.815f};
    float chi2_3dof_rgbd{7.815f};        // split from chi2_3dof so RGB-D can be tuned independently

    // Huber deltas -- DERIVED from chi2_* by LoadParameters(), never set independently
    // (previously separate statics that could silently desync if chi2_* ever changed after startup)
    float thHuber_2dof{2.4477f};
    float thHuber_3dof{2.7955f};
    float thHuber_3dof_rgbd{2.7955f};

    // RGB-D depth-channel measurement information (1/variance) for the inverse-depth residual
    double invDepthInfo{20000.0};

    // PoseOptimization()
    int numItPoseOpt{10};

    // OptimizeEssentialGraph()
    double userLambdaInit{1e-16};
    int minFeat{100};
    int numItEssGraphOpt{20};

    // OptimizeSim3()
    int numItSim3Opt{5};
    int nMoreItHigh{10};
    int nMoreItLow{5};
    int minNCorrespondences{10};
};

class Optimizer
{
public:
    void static BundleAdjustment(const std::vector<Keyframe> &vpKF, const std::vector<Pt> &vpMP,
                                 int nIterations = 5, bool *pbStopFlag=NULL, const unsigned long nLoopKF=0,
                                 const bool bRobust = true);
    void static GlobalBundleAdjustemnt(shared_ptr<Map> pMap, int nIterations=5, bool *pbStopFlag=NULL,
                                       const unsigned long nLoopKF=0, const bool bRobust = true);
    void static LocalBundleAdjustment(Keyframe pKF, bool *pbStopFlag, shared_ptr<Map>pMap);
    // useDepthChannel=false books RGB-D observations as plain 2D reprojection edges —
    // used by TrackReferenceKeyFrame's divergence rescue (see Tracking.cc).
    int static PoseOptimization(Frame* pFrame, const bool useDepthChannel = true);

    // if bFixScale is true, 6DoF optimization (stereo,rgbd), 7DoF otherwise (mono)
    void static OptimizeEssentialGraph(shared_ptr<Map>pMap, Keyframe pLoopKF, Keyframe pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       std::map<KeyframeId, LoopConnections> &loopConnections,
                                       const bool &bFixScale);

    // if bFixScale is true, optimize SE3 (stereo,rgbd), Sim3 otherwise (mono)
    static int OptimizeSim3(Keyframe pKF1, Keyframe pKF2, std::vector<Pt> &vpMatches1,
                            g2o::Sim3 &g2oS12, const float th2, const bool bFixScale, const FeatureType& featureType);

    // Tunable parameters, loaded from the settings YAML at System startup
    static OptimizerParameters params;
    static void LoadParameters(const cv::FileStorage &fSettings);

};

} //namespace ORB_SLAM

#endif // OPTIMIZER_H
