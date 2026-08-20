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


#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/core/core.hpp>
#include <mutex>
#include <deque>

#include "Viewer.h"
#include "FrameDrawer.h"
#include "Map.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Frame.h"
#include "FeatureVocabulary.h"
#include "KeyFrameDatabase.h"
#include "FeatureExtractor.h"
#include "Initializer.h"
#include "MapDrawer.h"
#include "System.h"
#include"FeatureMatcher.h"
#include "Utils.h"
#include "TrackingLostException.h"

namespace AF_VSLAM
{

class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class System;

// Tunable tracking heuristics, loaded from the settings YAML at System startup.
// Same pattern as OptimizerParameters: compiled-in defaults, overridden only by
// keys present in the file, so settings YAMLs without a Tracking.* block keep
// working unchanged.
struct TrackingParameters
{
    // monocular_initialization()
    int init_min_keypoints{100};     // min keypoints to accept a frame for two-view init
    float init_sigma{1.0f};          // measurement sigma for the two-view initializer
    int init_min_matches{100};       // min correspondences to attempt initialization
    int init_ransac_iterations{200}; // initializer RANSAC iteration budget
};

class Tracking
{

public:
    Tracking(System* pSys, shared_ptr<Vocabulary> vocabulary, std::shared_ptr<FrameDrawer> pFrameDrawer, std::shared_ptr<MapDrawer> pMapDrawer, shared_ptr<Map> pMap,
             shared_ptr<KeyFrameDatabase> pKFDB,
             const string &strCalibrationPath, const string &strSettingPath,
             const std::map<FeatureType, string>& feature_settings_yaml_file,
             const int sensor,
             const vector<FeatureType>& featureTypes,
             const bool& fixImageSize = false);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    mat4f GrabImageMonocular(Image &im, const double &timestamp);

    void SetLocalMapper(std::shared_ptr<LocalMapping> localMapper_);
    void SetLoopClosing(std::shared_ptr<LoopClosing> loopClosing_);
    void SetViewer(shared_ptr<Viewer> viewer_);

    // Load new settings
    // The focal lenght should be similar or scale prediction will fail when projecting points
    // TODO: Modify MapPoint::PredictScale to take into account focal lenght
    void ChangeCalibration(const string &strSettingPath);

public:
    VerbosityLevel verbosity{LOW};

    // Tunable parameters, loaded from the settings YAML at System startup
    static TrackingParameters params;
    static void LoadParameters(const cv::FileStorage &fSettings);

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Input sensor
    int mSensor;

    // Current Frame
    Frame currentFrame;
    cv::Mat mImGray;
    cv::Mat mImMask; // segmentation mask of the current image (1 = static, 0 = dynamic); empty when segmentation is off
    std::string imName;

    // Initialization Variables (mono)
    std::vector<int> init_matches_;
    std::vector<vec3f> init_points3d_;
    Frame initial_frame_;

    std::map<FeatureType, std::vector<int>> matches_per_feature_;
    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<mat4f> mlRelativeFramePoses;
    list<Keyframe> mlpReferences;
    list<double> mlFrameTimes;
    list<bool> mlbLost;

    void Reset();

    // Reason (with statistics) reported by the most recent TrackingLostException, kept around so
    // the "reset if lost soon after init" branch in Track() can reference why tracking was lost.
    std::string mLastTrackingLostReason{};

    size_t numTrackedFrames{2};

    vector<FeatureType> featureTypes{};
    int featureInitialization{0};
    int featureRelocalization{0};

    int get_image_width() const {return w;};
    int get_image_height() const {return h;};

    ////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////
    // Profiling
    std::map<int, int> resize_times{};
    std::map<int, int> frame_times{};
    std::map<int, int> tracking_times{};
    std::map<int, int> track_ref_times{};
    std::map<int, int> pose_opt_times{};
    std::map<int, int> local_map_times{};
    std::map<int, int> grabImageMonocular_times{};

    ////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////

protected:

    // Main tracking function. It is independent of the input sensor.
    void Track();

    // Map initialization for monocular
    void monocular_initialization(const FeatureType& featureType);
    void CreateInitialMapMonocular(const FeatureType& featureType);

    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame(const bool& optimizePose = true);
    void UpdateLastFrame();

    bool Relocalization(const FeatureType& featureType);

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();

    bool TrackLocalMap();
    void SearchLocalPoints();

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    // Median pixel displacement of map points shared between currentFrame and lastFrame.
    // Scale-free stationarity signal for NeedNewKeyFrame; returns -1 if too few shared
    // points to be meaningful (gate then stays inactive).
    float MedianFlowFromLastFrame() const;

    void loadCameraParameters(const string &strCalibrationPath, const string &strSettingPath);
    shared_ptr<FeatureExtractor> getFeatureExtractor(const int& scaleNumFeaturesMonocular_,
                                                     const string &featureSettingsYamlFile,
                                                     const FeatureType& featureType);
    static void getGrayImage(cv::Mat& im, const bool& rgb);

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO;

    //Other Thread Pointers
    std::shared_ptr<LocalMapping> localMapper;
    std::shared_ptr<LoopClosing> loopClosing;

    // Features
    std::map<FeatureType, shared_ptr<FeatureExtractor>> featureExtractorLeft, featureExtractorRight;
    std::map<FeatureType, shared_ptr<FeatureExtractor>> initFeatureExtractor;

    //BoW
    shared_ptr<Vocabulary> vocabulary;
    shared_ptr<KeyFrameDatabase> keyFrameDB;

    // Initalization (only for monocular)
    shared_ptr<Initializer> initializer_;

    //Local Map
    Keyframe refKeyframe;
    std::vector<Keyframe> localKeyframes;
    std::vector<Pt> localPts;

    // System
    System* mpSystem;

    //Drawers
    std::shared_ptr<Viewer> viewer;
    std::shared_ptr<FrameDrawer> frameDrawer;
    std::shared_ptr<MapDrawer> mapDrawer;

    // Map
    shared_ptr<Map> map;

    //Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;
    float mbf;
    int w{};
    int h{};

    // New KeyFrame rules (according to fps)
    size_t minFrames;
    size_t maxFrames;
    float fps;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth;

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor;

    //Current matches in frame
    int mnMatchesInliers;

    // Rolling inlier history (last inliersHistorySize tracked frames) — reference for the
    // emergency-keyframe trigger in NeedNewKeyFrame(). Comparing against recent frames
    // instead of refKeyframe->TrackedMapPoints() avoids the self-inflating feedback loop
    // where every inserted keyframe grows the reference stat via post-hoc triangulation
    // (see CLAUDE.md, Stop-Induced Keyframe Runaway Investigation).
    std::deque<int> recentInliersHistory;
    FrameId lastEmergencyKFId{0};

    // Last Frame, KeyFrame and Relocalisation Info
    Keyframe lastKeyFrame;
    Frame lastFrame;
    KeyframeId lastKeyFrameId;
    FrameId lastRelocFrameId;

    //Motion Model
    mat4f mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB{true};

    list<Pt> mlpTemporalPoints;

    // Fix image size to nominal size 307200 pixels
    bool fixImageSize{false};

    //////////////////////////////////////////////// Heuristics
    // Tracking()
    const int scaleNumFeaturesMonocular{4};

    // UpdateLocalKeyFrames()
    const int _maxNumKey_{80};
    const int _bestCovKey_{10};

    // Matching
    const float nnratio_monoInit{0.9f};
    const float nnratio_trackRefKey{0.7f};
    const float nnratio_trackMotModel{0.9f};
    const float nnratio_slp{0.8f};
    const float nnratio_low_reloc{0.75f};
    const float nnratio_high_reloc{0.9f};

    const float radiusTh_high_trackMotModel{15.0f};
    const float radiusTh_low_trackMotModel{7.0f};
    const float radiusTh_scale_trackMotModel{2.0f};
    const float radiusTh_high_reloc{10.0f};
    const float radiusTh_low_reloc{3.0f};

    const float viewingCosLimit_slp{0.5f};

    //////////////////////////////////////////////// Constants

    // CreateInitialMapMonocular()
    const int keyframeTrackedMapPoints{100};
    const int minDepthSamples_createInitialMap{10};

    // TrackReferenceKeyFrame()
    const int TRACK_REFERENCE_KEYFRAME_MIN_MATCHES_HIGH{15};
    const int TRACK_REFERENCE_KEYFRAME_MIN_MATCHES_LOW{10};
    // Prior-consistency gate on the globally-matched map points: drop matches whose map
    // point projects behind/at the camera or farther than this from its keypoint under the
    // motion prior (genuine matches measure <= ~11 px there; garbage depth-seeded points
    // measure 1e2-1e6 px — see the 2026-08-13 tracking-loss investigation in CLAUDE.md).
    const float reprojPriorGate{80.0f};
    const float minDepthPriorGate{0.2f};

    // TrackLocalMap()
    const int minMatches_trackLocalMap_high{50};
    const int minMatches_trackLocalMap_low{30};

    // NeedNewKeyFrame()
    // Stationarity gate: median frame-to-frame flow (px) below which no keyframe is
    // inserted (camera considered static — new keyframes would only feed zero-baseline
    // triangulation). Gate needs at least minSharedPtsForFlow shared points to engage.
    const float minMedianFlow_needNewKey{1.0f};
    const int minSharedPtsForFlow{20};
    // Emergency keyframes: reference window and refire cooldown (frames).
    const size_t inliersHistorySize{30};
    const int emergencyKFCooldown{10};
    const float refRatio_high_needNewKey{0.9f};
    const int nMinObs_high{3};
    const int nMinObs_low{2};
    const int minNKFs{2};
    const int minTrackedClose{100};
    const int minNonTrackedClose{70};
    const float nRefMatchesDrop{0.25f};
    const int minMatchesInliers{15};
    const int minKeyframesInQueue{3};

    // CreateNewKeyFrame()
    const int minNumPoints_createNewKey{100};

    // SearchLocalPoints()
    const int idSum{2};

    // Relocalization()
    const int minNmatches{15};
    const int nGood_high{50};
    const int nGood_medium{30};
    const int nGood_low{10};

    // # Iterations
    const int numItGBA{20}; // CreateInitialMapMonocular()
    const int numItpSolver{5}; // Relocalization

    // Memory
    const int scaleReserveKey{3}; // UpdateLocalKeyFrames()

    // RANSAC
    const float ransac_probability{0.99};
    const int ransac_minInliers{10};
    const int ransac_maxIterations{3000};
    const int ransac_minSet{4};
    const float ransac_epsilon{0.5};
    const float ransac_th2{5.991};

    // loadCameraParameters()
    const float fps0{30.0f};
    const int numFeatures0{1000};

    // GrabImageRGBD()
    const float mDepthMapFactor_th{1e-5};

    // Track()
    const int minKeyframesInMap{5};

    std::shared_ptr<FeatureMatcher> matcher;

    bool emergencyKeyframe{false};

};

} //namespace ORB_SLAM

#endif // TRACKING_H
