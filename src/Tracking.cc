


#include "Tracking.h"

#include <opencv2/core/core.hpp>
//#include<opencv2/features2d/features2d.hpp>

#include "FrameDrawer.h"
#include "Converter.h"
#include "Map.h"
#include "Initializer.h"
#include "afvslam_log.hpp"

#include "Optimizer.h"
#include "PnPsolver.h"
#include "Utils.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include <mutex>

#include <yaml-cpp/yaml.h>

using namespace std;

namespace AF_VSLAM
{

TrackingParameters Tracking::params{};

void Tracking::LoadParameters(const cv::FileStorage &fSettings)
{
    auto readIfPresent = [&fSettings](const char* key, auto& field)
    {
        const cv::FileNode node = fSettings[key];
        if(!node.empty())
            node >> field;
    };

    readIfPresent("Tracking.InitMinKeypoints", params.init_min_keypoints);
    readIfPresent("Tracking.InitSigma", params.init_sigma);
    readIfPresent("Tracking.InitMinMatches", params.init_min_matches);
    readIfPresent("Tracking.InitRansacIterations", params.init_ransac_iterations);
    readIfPresent("Tracking.InitMinMedianDisparity", params.init_min_median_disparity);
}

Tracking::Tracking(System *pSys, shared_ptr<Vocabulary> vocabulary,
                   std::shared_ptr<FrameDrawer> frameDrawer, std::shared_ptr<MapDrawer> mapDrawer,
                   shared_ptr<Map> map, shared_ptr<KeyFrameDatabase> pKFDB,
                   const string &strCalibrationPath, const string &strSettingPath,
                   const std::map<FeatureType, string>& feature_settings_yaml_file,
                   const int sensor,
                   const vector<FeatureType>& featureTypes,
                   const bool& fixImageSize):
    mState(NO_IMAGES_YET), mSensor(sensor), feature_types_(featureTypes), mbVO(false), vocabulary(vocabulary),
    keyFrameDB(pKFDB), mpSystem(pSys), viewer(static_cast<shared_ptr<Viewer>>(nullptr)),
    frameDrawer(frameDrawer), mapDrawer(mapDrawer), map(map), lastRelocFrameId(0), fixImageSize(fixImageSize)
{
    // Load camera parameters from settings yaml file
    Tracking::loadCameraParameters(strCalibrationPath, strSettingPath);

    // Max/Min Frames to insert keyframes and to check relocalisation
    minFrames = 0;
    maxFrames = size_t(fps);

    // Load feature parameters from settings yaml file
    for (auto& ft: featureTypes){
        featureExtractorLeft[ft] = Tracking::getFeatureExtractor(1, feature_settings_yaml_file.at(ft), ft);
        initFeatureExtractor[ft] = Tracking::getFeatureExtractor(scaleNumFeaturesMonocular , feature_settings_yaml_file.at(ft), ft);
    }

    matcher_ = std::make_shared<FeatureMatcher>(w, h, featureTypes, "Tracking");
}

void Tracking::SetLocalMapper(std::shared_ptr<LocalMapping> localMapper_)
{
    localMapper = localMapper_;
}

void Tracking::SetLoopClosing(std::shared_ptr<LoopClosing> loopClosing_)
{
    loopClosing = loopClosing_;
}

void Tracking::SetViewer(shared_ptr<Viewer> viewer_)
{
    viewer = viewer_;
}

mat4f Tracking::GrabImageMonocular(Image &im, const double &timestamp)
{
    std::chrono::steady_clock::time_point t_start_0 = std::chrono::steady_clock::now();
    ///////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Convert image to grayscale and resize
    im.GetGrayImage(mbRGB);
    if(fixImageSize)
        im.FixImageSize(w,h);

    mImGray = im.grayImg;
    mImMask = im.mask;
    imName = im.imageName;

    std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
    double t_duration = std::chrono::duration_cast<std::chrono::duration<double> >(t_end - t_start_0).count();

#ifdef PROFILING_EXHAUSTIVE
    resize_times[int(1000 * t_duration)]++;
#endif

    ///////////////////////////////////////////////////////////////////////////////
    // Create Frame (extract features)
#ifdef PROFILING_EXHAUSTIVE
    std::chrono::steady_clock::time_point t_start_1 = std::chrono::steady_clock::now();
#endif

    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        current_frame_ = Frame(im,timestamp,initFeatureExtractor,vocabulary,mK,mDistCoef,mbf,mThDepth);
    else
        current_frame_ = Frame(im,timestamp,featureExtractorLeft,vocabulary,mK,mDistCoef,mbf,mThDepth);

#ifdef PROFILING_EXHAUSTIVE
    t_end = std::chrono::steady_clock::now();
    t_duration = std::chrono::duration_cast<std::chrono::duration<double> >(t_end - t_start_1).count();
    frame_times[int(1000 * t_duration)]++;
#endif
    ///////////////////////////////////////////////////////////////////////////////
    // Track Frame
#ifdef PROFILING_EXHAUSTIVE
    std::chrono::steady_clock::time_point t_start_2 = std::chrono::steady_clock::now();
#endif

    Track();

#ifdef PROFILING_EXHAUSTIVE
    t_end = std::chrono::steady_clock::now();
    t_duration = std::chrono::duration_cast<std::chrono::duration<double> >(t_end - t_start_2).count();
    if(mState == OK)
        tracking_times[int(1000 * t_duration)]++;
#endif

    if(viewer)
        viewer->set_grabImageMonocular_time_median(map_median(grabImageMonocular_times));

#ifdef PROFILING_EXHAUSTIVE
    AF_PROFILE_BEGIN("Tracking Profiling");
    AF_PROFILE_FIELD(resize_times,             "Resize Image");
    AF_PROFILE_FIELD(frame_times,              "Frame Creation");
    AF_PROFILE_FIELD(tracking_times,           "Tracking");
    AF_PROFILE_FIELD(track_ref_times,          "  Track Ref");
    AF_PROFILE_FIELD(pose_opt_times,           "  Pose Optimization");
    AF_PROFILE_FIELD(local_map_times,          "  Track Local Map");
    AF_PROFILE_FIELD(grabImageMonocular_times, "Grab Image Monocular");
    AF_PROFILE_END();
#endif

    t_end = std::chrono::steady_clock::now();
    t_duration = std::chrono::duration_cast<std::chrono::duration<double> >(t_end - t_start_0).count();

    if(mState == OK)
        grabImageMonocular_times[int(1000 * t_duration)]++;

    return current_frame_.Tcw;
}

void Tracking::Track()
{


    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    // Stage timing: report where the time went whenever a frame stalls visibly
    // (hiccup diagnosis; threshold ~3 frame periods at 20 fps).
    using StageClock = std::chrono::steady_clock;
    const auto stageMs = [](StageClock::time_point a, StageClock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const auto tTrackStart = StageClock::now();

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(map->mMutexMapUpdate);
    const auto tLockAcquired = StageClock::now();
    const double msLockWait = stageMs(tTrackStart, tLockAcquired);
    double msTrackRef = 0.0, msLocalMap = 0.0, msEmergencyWait = 0.0;

    if(mState==NOT_INITIALIZED)
    {

        monocular_initialization(feature_types_[featureInitialization]);
        frameDrawer->Update(this);

        if(mState!=OK)
            return;
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;
        if(mState==OK)
        {
            // Local Mapping might have changed some MapPoints tracked in last frame
            CheckReplacedInLastFrame();
            try
            {
                bOK = TrackReferenceKeyFrame();
            }
            catch(const TrackingLostException& e)
            {
                mLastTrackingLostReason = e.what();
                AF_WARN("Tracking lost — " << mLastTrackingLostReason);
                bOK = false;
            }
        }
        else
        {
            bOK = Relocalization(feature_types_[featureRelocalization]);
        }


        current_frame_.refKeyframe = refKeyframe;

        const auto tAfterTrackRef = StageClock::now();

        // If we have an initial estimation of the camera pose and matching. Track the local map.
#ifdef PROFILING_EXHAUSTIVE
        std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();
#endif

        if(bOK)
        {
            try
            {
                bOK = TrackLocalMap();
            }
            catch(const TrackingLostException& e)
            {
                mLastTrackingLostReason = e.what();
                AF_WARN("Tracking lost — " << mLastTrackingLostReason);
                bOK = false;
            }
        }

        msTrackRef = stageMs(tLockAcquired, tAfterTrackRef);
        msLocalMap = stageMs(tAfterTrackRef, StageClock::now());

#ifdef PROFILING_EXHAUSTIVE
        std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
        double t_duration = std::chrono::duration_cast<std::chrono::duration<double> >(t_end - t_start).count();
        local_map_times[int(1000 * t_duration)]++;
#endif

        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // Update drawer
        frameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            ++numTrackedFrames;

            // Low-rate heartbeat so post-mortems can see the inlier/map trend leading
            // into a tracking loss, not just the loss line itself.
            if(current_frame_.mnId % 100 == 0)
            {
                AF_INFO("Track heartbeat | frame=" << current_frame_.mnId
                        << " inliers=" << mnMatchesInliers
                        << " localPts=" << localPts.size()
                        << " KFs=" << map->KeyFramesInMap()
                        << " mapPts=" << map->MapPointsInMap());
                std::cout.flush(); // stdout is fully buffered under the runner's redirect
            }

            // Update motion model
            if(last_frame_.Tcw(3,3) == 1.0f)
            {
                mat4f LastTwc{mat4f::Identity()};
                LastTwc.block<3,3>(0,0) = last_frame_.GetRotationInverse();
                LastTwc.block<3,1>(0,3) = last_frame_.get_camera_center();
                mVelocity = current_frame_.Tcw * LastTwc;
            }
            else
                mVelocity = mat4f::Zero();

            mapDrawer->SetCurrentCameraPose(current_frame_.Tcw);

            // Clean VO matches
            for (auto& [ft, N] : current_frame_.N) {
                for(int i = 0; i < N; i++)
                {
                    Pt pMP = current_frame_.pts.at(ft)[i];
                    if(pMP)
                        if(pMP->number_of_observations() < 1)
                        {
                            current_frame_.mvbOutlier.at(ft)[i] = false;
                            current_frame_.pts.at(ft)[i]=static_cast<Pt>(nullptr);
                        }
                }
            }
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();


            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so, we discard them in the frame.
            for (auto& [ft, N] : current_frame_.N) {
                for(int i =0; i < N; i++)
                {
                    if(current_frame_.pts.at(ft)[i] && current_frame_.mvbOutlier.at(ft)[i])
                        current_frame_.pts.at(ft)[i]=static_cast<Pt>(nullptr);
                }
            }
        }

        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(map->KeyFramesInMap() <= static_cast<size_t>(minKeyframesInMap))
            {
                AF_WARN("Track lost soon after initialisation (" << map->KeyFramesInMap() << " <= "
                        << minKeyframesInMap << " keyframes in map), reason: " << mLastTrackingLostReason
                        << " — reseting...");
                mpSystem->Reset();
                return;
            }
        }

        if(!current_frame_.refKeyframe)
            current_frame_.refKeyframe = refKeyframe;

        last_frame_ = Frame(current_frame_);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterward.
    if(current_frame_.Tcw(3,3) == 1.0f)
    {
        mat4f Tcr = current_frame_.Tcw * current_frame_.refKeyframe->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(refKeyframe);
        mlFrameTimes.push_back(current_frame_.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

    if (emergencyKeyframe){
        std::cout << "Tracking::Track: emergency keyframe triggered, waiting for local mapping to be idle..." << std::endl;
        const auto tWaitStart = StageClock::now();
        lock.unlock();
        bool localMappingIdle = localMapper->AcceptKeyFrames();
        while(!localMappingIdle)
        {
            // Wait until Local Mapping is idle
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            localMappingIdle = localMapper->AcceptKeyFrames();
        }
        emergencyKeyframe = false;
        msEmergencyWait = stageMs(tWaitStart, StageClock::now());
        std::cout << "Tracking::Track: local mapping is idle, inserting emergency keyframe..." << std::endl;
    }

    // Hiccup diagnosis: whenever this frame stalled visibly, say where the time went.
    // "other" covers keyframe decision/creation, drawer updates, and pose bookkeeping.
    const double msTotal = stageMs(tTrackStart, StageClock::now());
    if (msTotal > 150.0)
    {
        AF_WARN("Track: slow frame, " << int(msTotal) << " ms"
                << " (mapMutexWait=" << int(msLockWait)
                << ", trackRef=" << int(msTrackRef)
                << ", localMap=" << int(msLocalMap)
                << ", emergencyWait=" << int(msEmergencyWait)
                << ", other=" << int(msTotal - msLockWait - msTrackRef - msLocalMap - msEmergencyWait)
                << ") | frame=" << current_frame_.mnId);
    }
}

void Tracking::monocular_initialization(FeatureType feature_type)
{
    if(!initializer_)
    {
        // Set Reference Frame
        if(current_frame_.keypoints.at(feature_type).size() > static_cast<size_t>(params.init_min_keypoints))
        {
            initial_frame_ = current_frame_;
            last_frame_ = current_frame_;
            initializer_ = std::make_shared<Initializer>(current_frame_, params.init_sigma, params.init_ransac_iterations, feature_type);
            return;
        }
    }
    else
    {
        // Try to initialize
        if(current_frame_.keypoints.at(feature_type).size() <= static_cast<size_t>(params.init_min_keypoints))
        {
            initializer_ = nullptr;
            return;
        }

        // Find correspondences
        const auto matched_pairs = matcher_->match_frames_for_initialization(initial_frame_, current_frame_, feature_types_);

        // Fill matches_per_feature_ (used later in create_initial_map_monocular) and
        // init_matches_ (flat over all feature types, the structure the initializer uses).
        init_matches_.clear();
        size_t offset2 = 0;
        for (const auto& ft : feature_types_) {
            const size_t offset1 = init_matches_.size();
            init_matches_.resize(offset1 + initial_frame_.keypoints.at(ft).size(), -1);
            auto& matches_ft = matches_per_feature_[ft];
            matches_ft.assign(initial_frame_.keypoints.at(ft).size(), -1);
            for (const auto& m : matched_pairs.at(ft)) {
                matches_ft[m.first] = m.second;
                init_matches_[offset1 + m.first] = m.second + offset2;
            }
            offset2 += current_frame_.keypoints.at(ft).size();
        }

        // Check if there are enough correspondences
        const size_t num_matches = matched_pairs.at(feature_type).size();
        if(num_matches < static_cast<size_t>(params.init_min_matches))
        {
            initializer_ = nullptr;
            return;
        }

        // Disparity gate: with (near-)zero baseline — e.g. a static camera — two-view
        // initialization can only produce ill-conditioned geometry, so don't attempt it.
        // Keep the reference frame: disparity only grows once the camera starts moving.
        {
            const auto& keypoints1 = initial_frame_.keypoints.at(feature_type);
            const auto& keypoints2 = current_frame_.keypoints.at(feature_type);
            std::vector<float> disparities;
            disparities.reserve(num_matches);
            for (const auto& m : matched_pairs.at(feature_type))
                disparities.push_back(static_cast<float>(cv::norm(keypoints2[m.second].pt - keypoints1[m.first].pt)));
            const auto mid = disparities.begin() + disparities.size() / 2;
            std::nth_element(disparities.begin(), mid, disparities.end());
            if (*mid < params.init_min_median_disparity)
                return;
        }

        mat3f Rcw{}; // Current Camera Rotation
        vec3f tcw{}; // Current Camera Translation
        std::vector<bool> triangulated;
        if(initializer_->initialize(current_frame_, init_matches_, Rcw, tcw, init_points3d_, triangulated))
        {
            // Discard matches the initializer could not triangulate
            for (size_t j = 0; j < init_matches_.size(); j++)
                if (init_matches_[j] >= 0 && !triangulated[j])
                    init_matches_[j] = -1;

            // Set Frame Poses
            initial_frame_.set_pose(mat4f::Identity());
            mat4f Tcw{mat4f::Identity()};
            Tcw.block<3,3>(0,0) = Rcw;
            Tcw.block<3,1>(0,3) = tcw;
            current_frame_.set_pose(Tcw);
            create_initial_map_monocular(feature_type);
        }
    }
}

void Tracking::create_initial_map_monocular(FeatureType feature_type)
{

    // Create KeyFrames
    Keyframe pKFini = make_shared<KeyFrame>(initial_frame_, map, keyFrameDB);
    Keyframe pKFcur = make_shared<KeyFrame>(current_frame_, map, keyFrameDB);


    pKFini->ComputeBoW(feature_type);
    pKFcur->ComputeBoW(feature_type);

    // Insert KFs in the map
    map->AddKeyFrame(pKFini);
    map->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    int j = -1;
    for (const auto& ft : feature_types_) {
        const auto matches = matches_per_feature_[ft];

        for (size_t i = 0; i < matches.size(); i++) {
            j++;
            if (init_matches_[j] < 0)
                continue;

            //Create MapPoint.
            Pt pMP = pKFcur->CreateMonocularMapPoint(init_points3d_[j], KeypointIndex(matches[i]),
                                                    pKFini, KeypointIndex(i), ft);
            //Fill Current Frame structure
            current_frame_.pts[ft][matches[i]] = pMP;
            current_frame_.mvbOutlier[ft][matches[i]] = false;

            //Add to Map
            map->add_map_point(pMP);
        }
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    AF_INFO("New Map created with " << map->MapPointsInMap() << " points");

    Optimizer::GlobalBundleAdjustemnt(map,numItGBA);

    // Set the initial map's scale: prefer a depth-verified scale over the arbitrary
    // monocular "median depth = 1" convention, when enough points have valid sensor depth.
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1) < keyframeTrackedMapPoints)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    vector<float> depthRatios;
    for (const auto& ft : feature_types_) {
        const auto& invDepthKF = pKFini->invDepth.at(ft);
        vector<Pt> vpAllMapPoints = pKFini->get_map_point_matches(ft);
        for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
        {
            if(!vpAllMapPoints[iMP])
                continue;
            float sensorInvDepth = invDepthKF[iMP];
            if(sensorInvDepth <= 0.0f)
                continue;
            float triangulatedDepth = vpAllMapPoints[iMP]->get_world_pos()(2);
            if(triangulatedDepth <= 0.0f)
                continue;
            depthRatios.push_back((1.0f / sensorInvDepth) / triangulatedDepth);
        }
    }

    float invMedianDepth;
    if((int)depthRatios.size() >= minDepthSamples_createInitialMap)
    {
        sort(depthRatios.begin(), depthRatios.end());
        invMedianDepth = depthRatios[depthRatios.size() / 2];
    }
    else
    {
        invMedianDepth = 1.0f / medianDepth;
    }

    // Scale initial baseline
    mat4f Tc2w = pKFcur->GetPose();
    Tc2w.block<3,1>(0,3) *= invMedianDepth;
    pKFcur->set_pose(Tc2w);

    // Scale points

    for (const auto& ft : feature_types_) {
        vector<Pt> vpAllMapPoints = pKFini->get_map_point_matches(ft);
        for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
        {
            if(vpAllMapPoints[iMP])
            {
                Pt pMP = vpAllMapPoints[iMP];
                pMP->SetWorldPos(pMP->get_world_pos()*invMedianDepth);
            }
        }
    }


    localMapper->InsertKeyFrame(pKFini);
    localMapper->InsertKeyFrame(pKFcur);

    current_frame_.set_pose(pKFcur->GetPose());
    lastKeyFrameId=current_frame_.mnId;
    lastKeyFrame = pKFcur;

    localKeyframes.push_back(pKFcur);
    localKeyframes.push_back(pKFini);
    localPts = map->GetAllMapPoints();
    refKeyframe = pKFcur;
    current_frame_.refKeyframe = pKFcur;

    last_frame_ = Frame(current_frame_);

    map->SetReferenceMapPoints(localPts);

    mapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    map->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;
}

void Tracking::CheckReplacedInLastFrame()
{
    for (auto& [ft, pts] : last_frame_.pts) {
        for(int i = 0; i<last_frame_.N.at(ft); i++)
        {
            Pt pMP = last_frame_.pts.at(ft)[i];

            if(pMP)
            {
                Pt pRep = pMP->GetReplaced();
                if(pRep)
                {
                    last_frame_.pts.at(ft)[i] = pRep;
                }
            }
        }
    }
}


bool Tracking::TrackReferenceKeyFrame(const bool& optimizePose)
{
    #ifdef PROFILING_EXHAUSTIVE
            std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();
    #endif

    // Feature Matching
    std::map<FeatureType, std::vector<Pt>> mapPointMatches;
    std::map<FeatureType, int> nmatches_ft = matcher_->match_keyframe_to_frame(refKeyframe, current_frame_, mapPointMatches, current_frame_.featureTypes);
    int nmatches = 0;
    for (auto& [ft, N] : current_frame_.N)
    {
        current_frame_.pts[ft] = mapPointMatches[ft];
        current_frame_.mvbOutlier[ft] = vector<bool>(mapPointMatches[ft].size(), false);
        nmatches += nmatches_ft[ft];
    }


    if (!optimizePose)
        return true;

    if(nmatches < TRACK_REFERENCE_KEYFRAME_MIN_MATCHES_HIGH)
    {
        std::ostringstream reason;
        reason << "TrackReferenceKeyFrame: insufficient matches to reference keyframe (nmatches="
               << nmatches << " < " << TRACK_REFERENCE_KEYFRAME_MIN_MATCHES_HIGH << ")"
               << " | frame=" << current_frame_.mnId << " refKeyframe=" << refKeyframe->keyId;
        throw TrackingLostException(reason.str());
    }

#ifdef PROFILING_EXHAUSTIVE
    std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
    double t_duration = std::chrono::duration_cast<std::chrono::duration<double> >(t_end - t_start).count();
    track_ref_times[int(1000 * t_duration)]++;
    t_start = std::chrono::steady_clock::now();
#endif

    // Optimize Pose — seed with the constant-velocity prediction when the motion model
    // is valid (mVelocity is Zero after a loss/reloc). Seeding with last_frame_.Tcw alone
    // (constant-position) leaves every pixel residual ~one frame of motion large at the
    // start: on fast sequences that Huber-saturates the 2D terms while the high-information
    // RGB-D inverse-depth terms stay quadratic — and depth only constrains the z-component,
    // so pass-1 LM could walk into a wrong rotation/lateral basin and reject everything
    // (observed: frame 3325, 445 matches -> 0 inliers, poseDelta 1.48m/3.5deg, instantly
    // recoverable by from-scratch PnP relocalization).
    const mat4f posePrior = (mVelocity(3,3) == 1.0f) ? mat4f(mVelocity * last_frame_.Tcw)
                                                     : last_frame_.Tcw;

    // Prior-consistency gate. match_keyframe_to_frame is GLOBAL descriptor matching — unlike
    // stock ORB-SLAM2's projection-windowed SearchByProjection, nothing bounds how far a
    // matched map point may project from its keypoint. A depth-seeded point born from a spiky
    // sensor-depth reading can sit meters too close, cross the camera plane within a frame or
    // two of vehicle motion, and project 1e3-1e6 px away while its 2D descriptor match (and
    // its F-filter check, which never sees the 3D position) stays perfectly valid. Huber's
    // linear tail times the fx/z Jacobian explosion then lets a handful of such edges drag
    // the whole pose (observed: frame 17166, 10 of 469 matches at 100-654000 px dragged the
    // pose 25 deg -> every genuine match rejected -> tracking lost). Genuine matches sit
    // within ~11 px of the motion prior (measured across collapse dumps), so a generous gate
    // loses nothing.
    {
        const mat3f Rcw_prior = posePrior.block<3,3>(0,0);
        const vec3f tcw_prior = posePrior.block<3,1>(0,3);
        const float fx = mK.at<float>(0,0), fy = mK.at<float>(1,1);
        const float cx = mK.at<float>(0,2), cy = mK.at<float>(1,2);
        int nGated = 0;
        for (auto& [ft, N_ft] : current_frame_.N)
        {
            const auto& kps = current_frame_.keypoints.at(ft);
            for(int i = 0; i < N_ft; i++)
            {
                const Pt& pt = current_frame_.pts.at(ft)[i];
                if(!pt)
                    continue;
                const vec3f Xc = Rcw_prior * pt->get_world_pos() + tcw_prior;
                bool bad = Xc(2) < minDepthPriorGate;
                if(!bad)
                {
                    const float du = fx * Xc(0) / Xc(2) + cx - kps[i].pt.x;
                    const float dv = fy * Xc(1) / Xc(2) + cy - kps[i].pt.y;
                    bad = (du*du + dv*dv) > reprojPriorGate * reprojPriorGate;
                }
                if(bad)
                {
                    current_frame_.pts.at(ft)[i] = static_cast<Pt>(nullptr);
                    nGated++;
                    nmatches--;
                }
            }
        }
        if(nGated > 0)
            AF_INFO("TrackReferenceKeyFrame: prior-consistency gate dropped " << nGated
                    << " matches | frame=" << current_frame_.mnId);
    }

    current_frame_.set_pose(posePrior);
    Optimizer::PoseOptimization(&current_frame_);

    const auto countMapInliers = [this]() {
        int n = 0;
        for (auto& [ft, N_ft] : current_frame_.N)
            for(int i = 0; i < N_ft; i++)
            {
                const Pt& pt = current_frame_.pts.at(ft)[i];
                if(pt && !current_frame_.mvbOutlier.at(ft)[i] && pt->number_of_observations() > 0)
                    n++;
            }
        return n;
    };
    int nmatchesMap = countMapInliers();

    // Divergence rescue: a collapse to (almost) zero inliers despite plentiful raw matches
    // means the optimizer left the basin, not that the matches are bad. Re-seed and
    // re-optimize once with the RGB-D depth channel disabled — pure 2D reprojection, the
    // configuration the 4-pass scheme was originally tuned for.
    if(nmatchesMap < TRACK_REFERENCE_KEYFRAME_MIN_MATCHES_LOW
       && nmatches >= 3 * TRACK_REFERENCE_KEYFRAME_MIN_MATCHES_HIGH)
    {
        AF_WARN("TrackReferenceKeyFrame: pose optimization collapsed (" << nmatchesMap
                << " inliers of " << nmatches << " raw matches) — retrying without depth channel"
                << " | frame=" << current_frame_.mnId);

        // Post-mortem dump: per-match reprojection residuals AT THE PRIOR POSE (before any
        // optimization), with each map point's provenance — enough to test offline whether
        // the match set is bimodal (two coherent populations with no common pose) and which
        // population (old vs freshly-created points, image region, depth) is inconsistent.
        if(!FrameDrawer::exp_folder.empty())
        {
            const mat3f Rcw_prior = posePrior.block<3,3>(0,0);
            const vec3f tcw_prior = posePrior.block<3,1>(0,3);
            const float fx = mK.at<float>(0,0), fy = mK.at<float>(1,1);
            const float cx = mK.at<float>(0,2), cy = mK.at<float>(1,2);
            std::ofstream dump(FrameDrawer::exp_folder + "/collapse_frame_"
                               + std::to_string(current_frame_.mnId) + ".csv");
            dump << "ft,kpIdx,u_kp,v_kp,u_proj,v_proj,z_cam,invDepth_meas,ptId,firstKFid,nObs\n";
            for (auto& [ft, N_ft] : current_frame_.N)
            {
                const auto& kps = current_frame_.keypoints.at(ft);
                const auto& invD = current_frame_.invDepth.at(ft);
                for(int i = 0; i < N_ft; i++)
                {
                    const Pt& pt = current_frame_.pts.at(ft)[i];
                    if(!pt)
                        continue;
                    const vec3f Xc = Rcw_prior * pt->get_world_pos() + tcw_prior;
                    if(Xc(2) <= 0.0f)
                        continue;
                    dump << int(ft) << "," << i << ","
                         << kps[i].pt.x << "," << kps[i].pt.y << ","
                         << (fx * Xc(0) / Xc(2) + cx) << "," << (fy * Xc(1) / Xc(2) + cy) << ","
                         << Xc(2) << "," << invD[i] << ","
                         << pt->ptId << "," << pt->mnFirstKFid << ","
                         << pt->number_of_observations() << "\n";
                }
            }
        }

        for (auto& [ft, N_ft] : current_frame_.N)
            std::fill(current_frame_.mvbOutlier.at(ft).begin(), current_frame_.mvbOutlier.at(ft).end(), false);
        current_frame_.set_pose(posePrior);
        Optimizer::PoseOptimization(&current_frame_, /*useDepthChannel=*/false);
        nmatchesMap = countMapInliers();
    }

    // Discard outliers
    for (auto& [ft, N] : current_frame_.N)
    {
        for(int i = 0; i < N; i++)
        {
            Pt pt = current_frame_.pts.at(ft)[i];
            if(pt)
            {
                if(current_frame_.mvbOutlier.at(ft)[i])
                {
                    current_frame_.pts.at(ft)[i] = static_cast<Pt>(nullptr);
                    current_frame_.mvbOutlier.at(ft)[i] = false;
                    pt->mbTrackInView = false;
                    pt->idLastFrameSeen = current_frame_.mnId;
                }
            }
        }
    }

    #ifdef PROFILING_EXHAUSTIVE
    t_end = std::chrono::steady_clock::now();
    t_duration = std::chrono::duration_cast<std::chrono::duration<double> >(t_end - t_start).count();
    pose_opt_times[int(1000 * t_duration)]++;
#endif

    if(nmatchesMap < TRACK_REFERENCE_KEYFRAME_MIN_MATCHES_LOW)
    {
        std::ostringstream reason;
        reason << "TrackReferenceKeyFrame: insufficient inlier matches after pose optimization (nmatchesMap="
               << nmatchesMap << " < " << TRACK_REFERENCE_KEYFRAME_MIN_MATCHES_LOW << ")"
               << " | frame=" << current_frame_.mnId << " rawMatches=" << nmatches;
        throw TrackingLostException(reason.str());
    }

    return true;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    Keyframe pRef = last_frame_.refKeyframe;
    mat4f Tlr = mlRelativeFramePoses.back();

    last_frame_.set_pose(Tlr * pRef->GetPose());
}

bool Tracking::TrackLocalMap()
{

    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
    UpdateLocalMap();
    SearchLocalPoints();

    // Optimize Pose
    Optimizer::PoseOptimization(&current_frame_);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for (auto& [ft, pts] : current_frame_.pts) {
        for(size_t i = 0; i < pts.size(); i++)
        {
            if(current_frame_.pts.at(ft)[i])
            {
                if(!current_frame_.mvbOutlier.at(ft)[i])
                {
                    current_frame_.pts.at(ft)[i]->IncreaseFound();

                    if(current_frame_.pts.at(ft)[i]->number_of_observations() > 0)
                        mnMatchesInliers++;

                }
                else if(mSensor==System::STEREO)
                    current_frame_.pts.at(ft)[i] = static_cast<Pt>(nullptr);

            }
        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if(current_frame_.mnId < lastRelocFrameId + maxFrames && mnMatchesInliers < minMatches_trackLocalMap_high)
    {
        std::ostringstream reason;
        reason << "TrackLocalMap: insufficient inliers shortly after relocalization (mnMatchesInliers="
               << mnMatchesInliers << " < " << minMatches_trackLocalMap_high << ")"
               << " | frame=" << current_frame_.mnId
               << " framesSinceReloc=" << (current_frame_.mnId - lastRelocFrameId) << " maxFrames=" << maxFrames;
        throw TrackingLostException(reason.str());
    }

    if(mnMatchesInliers < minMatches_trackLocalMap_low)
    {
        std::ostringstream reason;
        reason << "TrackLocalMap: insufficient inliers against local map (mnMatchesInliers="
               << mnMatchesInliers << " < " << minMatches_trackLocalMap_low << ")"
               << " | frame=" << current_frame_.mnId << " localPts=" << localPts.size();
        throw TrackingLostException(reason.str());
    }

    return true;
}

    float Tracking::MedianFlowFromLastFrame() const
    {
        // Collect pixel positions of last frame's map points, then measure how far the
        // same points moved in the current frame. Scale-free (pure 2D), cheap (two
        // linear passes), and needs no extra bookkeeping in the matchers.
        std::unordered_map<const MapPoint*, cv::Point2f> lastPositions;
        for (const auto& [ft, ptsFt] : last_frame_.pts) {
            const auto& kpsFt = last_frame_.keypoints.at(ft);
            for (size_t i = 0; i < ptsFt.size(); i++)
                if (ptsFt[i])
                    lastPositions[ptsFt[i].get()] = kpsFt[i].pt;
        }

        std::vector<float> flows;
        for (const auto& [ft, ptsFt] : current_frame_.pts) {
            const auto& kpsFt = current_frame_.keypoints.at(ft);
            for (size_t i = 0; i < ptsFt.size(); i++) {
                if (!ptsFt[i])
                    continue;
                auto it = lastPositions.find(ptsFt[i].get());
                if (it != lastPositions.end())
                    flows.push_back(static_cast<float>(cv::norm(kpsFt[i].pt - it->second)));
            }
        }

        if (flows.size() < static_cast<size_t>(minSharedPtsForFlow))
            return -1.0f;

        auto mid = flows.begin() + flows.size() / 2;
        std::nth_element(flows.begin(), mid, flows.end());
        return *mid;
    }

    bool Tracking::NeedNewKeyFrame()
    {
        // If Local Mapping is freezed by a Loop Closure do not insert keyframes
        if(localMapper->isStopped() || localMapper->stopRequested())
            return false;

        const size_t numKeyframesInMap = map->KeyFramesInMap();

        // Do not insert keyframes if not enough frames have passed from last relocalisation —
        // unless tracking is already demonstrably strong again (>=2x the TrackLocalMap "high"
        // threshold): at driving speed the full embargo freezes the reference keyframe for
        // ~a second of travel, decaying its matches until tracking is lost AGAIN right after
        // a successful relocalization (observed echo losses 20-22 frames after reloc; #9).
        if((current_frame_.mnId < lastRelocFrameId + maxFrames) && (numKeyframesInMap > maxFrames)
           && mnMatchesInliers < 2 * minMatches_trackLocalMap_high)
            return false;

        // Tracked MapPoints in the reference keyframe
        int nMinObs = nMinObs_high;
        if(numKeyframesInMap <= size_t(minNKFs))
            nMinObs = nMinObs_low;
        int nRefMatches = refKeyframe->TrackedMapPoints(nMinObs);

        // Local Mapping accept keyframes?
        bool localMappingIdle = localMapper->AcceptKeyFrames();

        // Check how many "close" points are being tracked and how many could be potentially created.
        int nNonTrackedClose = 0;
        int nTrackedClose= 0;

        bool bNeedToInsertClose = (nTrackedClose < minTrackedClose) && (nNonTrackedClose > minNonTrackedClose);

        // Thresholds
        const bool c1 = ((mnMatchesInliers < nRefMatches * refRatio_high_needNewKey || bNeedToInsertClose) && mnMatchesInliers > minMatchesInliers);

        bool c2{false};
        #ifdef ALLFEATURE_EVALUATION
        c2 = ((int( current_frame_.mnId) % ALLFEATURE_EVALUATION) == 0);
        #endif

        bool c3{false};
        #ifdef ALLFEATURE_MAX_KEYFRAMES
        c3 = ((current_frame_.mnId % ALLFEATURE_MAX_KEYFRAMES) == 0);
        #endif

        float overlap = current_frame_.GetOverlap();
        bool c4 = (overlap < 0.7f);

        // Median inlier count over the recent tracked frames (reference for the
        // emergency trigger below) — computed before pushing the current frame's
        // count, so the current frame is compared against its predecessors.
        int medianRecentInliers = -1;
        if (recentInliersHistory.size() >= inliersHistorySize / 2) {
            std::vector<int> h(recentInliersHistory.begin(), recentInliersHistory.end());
            auto mid = h.begin() + h.size() / 2;
            std::nth_element(h.begin(), mid, h.end());
            medianRecentInliers = *mid;
        }
        recentInliersHistory.push_back(mnMatchesInliers);
        if (recentInliersHistory.size() > inliersHistorySize)
            recentInliersHistory.pop_front();

        // Stationarity gate: a static camera adds no viewpoint information — new
        // keyframes would only feed zero-baseline triangulation, which poisons the map
        // with ill-conditioned points (see CLAUDE.md, Stop-Induced Keyframe Runaway
        // Investigation). Evaluation-forced keyframes (c2/c3) bypass the gate.
        const float medianFlow = MedianFlowFromLastFrame();
        if (!c2 && !c3 && medianFlow >= 0.0f && medianFlow < minMedianFlow_needNewKey)
            return false;

        //if(c4)
        if(c1 || c2 || c3 || c4)
        {
            if(localMappingIdle)
            {
                return true;
            }
            else
            {
                // if(c2 || c3 || c4){
                //     std::cout << "\nEmergency keyframe triggered by evaluation condition at frame " << current_frame_.mnId << std::endl;
                //     emergencyKeyframe = true;
                //     return true;
                // }
                // Emergency keyframe: only on a genuine drop against the *recent frames'*
                // own inlier level (not refKeyframe->TrackedMapPoints(), which inflates
                // after every insertion as LocalMapping triangulates new points into the
                // keyframe, re-arming the trigger indefinitely), and with a refire
                // cooldown so a persistent low-inlier state can't chain insertions.
                if(medianRecentInliers > 0
                   && mnMatchesInliers < 0.5f * static_cast<float>(medianRecentInliers)
                   && current_frame_.mnId >= lastEmergencyKFId + static_cast<FrameId>(emergencyKFCooldown)){
                    AF_WARN("NeedNewKeyFrame: emergency keyframe (mnMatchesInliers=" << mnMatchesInliers
                            << " < 0.5*medianRecentInliers=" << medianRecentInliers
                            << ", medianFlow=" << medianFlow << ") | frame=" << current_frame_.mnId);
                    lastEmergencyKFId = current_frame_.mnId;
                    emergencyKeyframe = true;
                    return true;
                }
                return false;
            }
        }
        else
            return false;
    }

    void Tracking::CreateNewKeyFrame(){
        if(!localMapper->SetNotStop(true))
            return;

        Keyframe keyframe = make_shared<KeyFrame>(current_frame_,map,keyFrameDB);

        refKeyframe = keyframe;
        current_frame_.refKeyframe = keyframe;

        localMapper->InsertKeyFrame(keyframe);
        localMapper->SetNotStop(false);
        lastKeyFrameId = current_frame_.mnId;
        lastKeyFrame = keyframe;
    }

    void Tracking::SearchLocalPoints()
    {
        // Do not search map points already matched
        for (const auto& [ft, N] : current_frame_.N) {
            for(auto& pt: current_frame_.pts.at(ft)){
                if(pt && !pt->is_bad()){
                    pt->IncreaseVisible();
                    pt->idLastFrameSeen = current_frame_.mnId;
                    pt->mbTrackInView = false;
                }
                else
                    pt = nullptr;
            }
        }

        // Project points in frame and check its visibility
        int nToMatch=0;
        for(auto& pt: localPts){
            if(pt->idLastFrameSeen == current_frame_.mnId)
                continue;
            if(pt->is_bad())
                continue;

            // Project (this fills MapPoint variables for matching)
            if(current_frame_.isInFrustum(pt,viewingCosLimit_slp)){
                pt->IncreaseVisible();
                nToMatch++;
            }
        }

        if(nToMatch > 0){
            matcher_->match_map_points_to_frame(current_frame_, localPts);
        }
    }

    void Tracking::UpdateLocalMap()
    {
        // This is for visualization
        map->SetReferenceMapPoints(localPts);
        // Update
        UpdateLocalKeyFrames();
        UpdateLocalPoints();
    }

    void Tracking::UpdateLocalPoints()
    {
        localPts.clear();
        set<PtId> ptIds{};
        for(const auto& keyframe : localKeyframes){
            for(const auto& ft: feature_types_){
                const vector<Pt> pts = keyframe->get_map_point_matches(ft);
                for(const auto& pt : pts){
                    if(!pt)
                        continue;
                    if (ptIds.find(pt->ptId) != ptIds.end())
                        continue;
                    if(!pt->is_bad()){
                        localPts.push_back(pt);
                        ptIds.insert(pt->ptId);
                    }
                }
            }
        }
    }

    void Tracking::UpdateLocalKeyFrames()
    {
        // Each map point vote for the keyframes in which it has been observed
        set<KeyframeId> keyframeIds{};
        {
            int maxObs = 0;
            auto keyframeMaxObs = static_cast<Keyframe>(nullptr);

            std::map<KeyframeId,int> keyframeCounter;
            std::map<KeyframeId,Keyframe> keyframes;

            for (auto& [ft, pts] : current_frame_.pts) {
                for(auto& pt: pts){
                    if(pt && !pt->is_bad()) {
                        const std::map<KeyframeId, Obs> observations = pt->GetObservations();
                        for (const auto &[keyId, obs]: observations) {
                            keyframeCounter[keyId]++;
                            keyframes[keyId] = obs->projKeyframe;
                        }
                    }
                    else
                        pt = nullptr;
                }
            }
            if(keyframeCounter.empty())
                return;

            // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
            localKeyframes.clear();
            localKeyframes.reserve(scaleReserveKey * keyframeCounter.size());
            for (const auto &[keyId, keyframe]: keyframes) {
                if(keyframe->is_bad())
                    continue;

                int keyFrameCount = keyframeCounter[keyframe->keyId];
                if(keyFrameCount > maxObs){
                    maxObs = keyFrameCount;
                    keyframeMaxObs = keyframe;
                }

                localKeyframes.push_back(keyframe);
                keyframeIds.insert(keyframe->keyId);
            }

            if(keyframeMaxObs){
                refKeyframe = keyframeMaxObs;
                current_frame_.refKeyframe = refKeyframe;
            }
        }

        // Include also some not-already-included keyframes that are neighbors to already-included keyframes
        for(const auto& keyframe: localKeyframes){

            // Limit the number of keyframes
            if(int(localKeyframes.size()) > _maxNumKey_)
                break;

            const vector<Keyframe> neighbors = keyframe->GetBestCovisibilityKeyFrames(_bestCovKey_);
            for(const auto& neighbor: neighbors){
                if(!neighbor->is_bad()){
                    if (keyframeIds.find(neighbor->keyId) == keyframeIds.end()){
                        localKeyframes.push_back(neighbor);
                        keyframeIds.insert(neighbor->keyId);
                        break;
                    }
                }
            }

            const set<Keyframe> childs = keyframe->GetChilds();
            for(const auto& child: childs){
                if(!child->is_bad()){
                    if (keyframeIds.find(child->keyId) == keyframeIds.end()){
                        localKeyframes.push_back(child);
                        keyframeIds.insert(child->keyId);
                        break;
                    }
                }
            }

            Keyframe parent = keyframe->GetParent();
            if(parent and !parent->is_bad()){
                if (keyframeIds.find(parent->keyId) == keyframeIds.end()){
                    localKeyframes.push_back(parent);
                    keyframeIds.insert(parent->keyId);
                    break;
                }
            }
        }
    }

bool Tracking::Relocalization(const FeatureType& featureType)
{
    // Compute Bag of Words Vector
    current_frame_.ComputeBoW(featureType);

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<Keyframe> vpCandidateKFs = keyFrameDB->DetectRelocalizationCandidates(&current_frame_);

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we set up a PnP solver

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<std::map<FeatureType, vector<Pt>>> vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        Keyframe pKF = vpCandidateKFs[i];
        if(pKF->is_bad())
            vbDiscarded[i] = true;
        else
        {
            std::map<FeatureType, int> nmatches_ft = matcher_->match_keyframe_to_frame(pKF, current_frame_, vvpMapPointMatches[i], std::vector<FeatureType>{featureType});
            if(nmatches_ft[featureType] < minNmatches)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(current_frame_,vvpMapPointMatches[i][featureType], featureType);
                pSolver->SetRansacParameters(ransac_probability,ransac_minInliers,ransac_maxIterations,ransac_minSet,ransac_epsilon,ransac_th2);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw_tmp = pSolver->iterate(numItpSolver,bNoMore,vbInliers,nInliers);
            mat4f Tcw{mat4f::Zero()};
            if(!Tcw_tmp.empty())
                Tcw = Converter::toMatrix4f(Tcw_tmp);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(Tcw(3,3) == 1.0f)
            {
                current_frame_.Tcw = Tcw;
                set<Pt> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        current_frame_.pts.at(featureType)[j]=vvpMapPointMatches[i][featureType][j];
                        sFound.insert(vvpMapPointMatches[i][featureType][j]);
                    }
                    else
                        current_frame_.pts.at(featureType)[j]=nullptr;
                }

                int nGood = Optimizer::PoseOptimization(&current_frame_);

                if(nGood < nGood_low)
                    continue;

                for(int io =0; io<current_frame_.N.at(featureType); io++)
                    if(current_frame_.mvbOutlier.at(featureType)[io])
                        current_frame_.pts.at(featureType)[io]=static_cast<Pt>(nullptr);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood < nGood_high)
                {
                    int nadditional = matcher_->SearchByProjection(current_frame_,vpCandidateKFs[i],sFound,radiusTh_high_reloc, true, featureType);

                    if(nadditional+nGood >= nGood_high)
                    {
                        nGood = Optimizer::PoseOptimization(&current_frame_);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood > nGood_medium && nGood < nGood_high)
                        {
                            sFound.clear();
                            for(int ip =0; ip<current_frame_.N.at(featureType); ip++)
                                if(current_frame_.pts.at(featureType)[ip])
                                    sFound.insert(current_frame_.pts.at(featureType)[ip]);
                            nadditional =matcher_->SearchByProjection(current_frame_,vpCandidateKFs[i],sFound,radiusTh_low_reloc,false, featureType);

                            // Final optimization
                            if(nGood+nadditional >= nGood_high)
                            {
                                nGood = Optimizer::PoseOptimization(&current_frame_);

                                for(int io =0; io < current_frame_.N.at(featureType); io++)
                                    if(current_frame_.mvbOutlier.at(featureType)[io])
                                        current_frame_.pts.at(featureType)[io]=nullptr;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood >= nGood_high)
                {
                    AF_INFO("Relocalization succeeded | frame=" << current_frame_.mnId
                            << " feature=" << featureName(featureType)
                            << " matchedKeyframe=" << vpCandidateKFs[i]->keyId
                            << " inliers=" << nGood << " requiredInliers=" << nGood_high);
                    std::cout.flush(); // AF_INFO writes to std::cout, which — unlike std::cerr's
                                        // implicit unitbuf flush behind AF_WARN — is fully buffered
                                        // once stdout is redirected to a file (as VSLAM-LAB's runner
                                        // does), so without this the line can sit unflushed for a while.
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        lastRelocFrameId = current_frame_.mnId;
        return true;
    }

}

void Tracking::Reset()
{

    cout << "System Reseting" << endl;
    if(viewer)
    {
        viewer->RequestStop();
        while(!viewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    localMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    loopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    keyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    map->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(initializer_)
    {
        initializer_ = nullptr;
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    resize_times.clear();
    frame_times.clear();
    tracking_times.clear();
    track_ref_times.clear();
    pose_opt_times.clear();
    local_map_times.clear();
    grabImageMonocular_times.clear();

    if(viewer)
        viewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::loadCameraParameters(const string &strCalibrationPath, const string &strSettingPath){

    YAML::Node settings = YAML::LoadFile(strSettingPath);
    YAML::Node calibration = YAML::LoadFile(strCalibrationPath);
    const YAML::Node& cameras = calibration["cameras"];

    std::string cam_name;
    cam_name = settings["cam_mono"].as<std::string>();
    YAML::Node cam{};
    for (size_t i{0}; i < cameras.size(); ++i){
        if (cameras[i]["cam_name"].as<std::string>() == cam_name){
            cam = cameras[i];
            break;
        }
    }

    mK = (cv::Mat_<float>(3, 3) << cam["focal_length"][0].as<float>(), 0.0f, cam["principal_point"][0].as<float>(),
            0.0f, cam["focal_length"][1].as<float>(), cam["principal_point"][1].as<float>(),
            0.0f, 0.0f, 1.0f);

    w = cam["image_dimension"][0].as<int>();
    h = cam["image_dimension"][1].as<int>();

    if(fixImageSize){
        float ratio = float(w) / float(h);
        int new_h = (int) sqrt(307200.f / ratio);
        int new_w = (int) (float(new_h) * ratio);
        float conv_ratio_h = float(new_h)/float(h);
        float conv_ratio_w = float(new_w)/float(w);

        w = new_w;
        h = new_h;

        mK.at<float>(0,0) *= conv_ratio_w;
        mK.at<float>(1,1) *= conv_ratio_h;
        mK.at<float>(0,2) *= conv_ratio_w;
        mK.at<float>(1,2) *= conv_ratio_h;
    }

    // Distortion coefficients
    mDistCoef = cv::Mat::zeros(4,1,CV_32F);
    if (cam["distortion_type"] && cam["distortion_coefficients"]) {
        std::vector<float> dist_coeffs_vec = cam["distortion_coefficients"].as<std::vector<float>>();
        mDistCoef = cv::Mat(dist_coeffs_vec.size(), 1, CV_32F, dist_coeffs_vec.data()).clone();
    }

    // Camera frequence (hz)
    fps = cam["fps"].as<float>();
    if(fps == 0)
        fps = fps0;

    // RGB order
    bool mbRGB = cam["cam_type"].as<std::string>() != "bgr";

    // Load settings file
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    // Print camera parameters
    AF_CONFIG_BEGIN("Camera Parameters");
    AF_CONFIG_FIELD("cam_name:           ", cam["cam_name"].as<std::string>());
    AF_CONFIG_FIELD("cam_type:           ", cam["cam_type"].as<std::string>());
    AF_CONFIG_FIELD("cam_model:          ", cam["cam_model"].as<std::string>());
    if (cam["distortion_type"] && cam["distortion_coefficients"])
        AF_CONFIG_FIELD("distortion_type:    ", cam["distortion_type"].as<std::string>());
    AF_CONFIG_FIELD("fx:                 ", mK.at<float>(0,0));
    AF_CONFIG_FIELD("fy:                 ", mK.at<float>(1,1));
    AF_CONFIG_FIELD("cx:                 ", mK.at<float>(0,2));
    AF_CONFIG_FIELD("cy:                 ", mK.at<float>(1,2));
    if (cam["distortion_type"] && cam["distortion_coefficients"])
        AF_CONFIG_FIELD("distortion_coefficients: ", mDistCoef.t());
    AF_CONFIG_FIELD("fps:                ", cam["fps"].as<float>());
    if(mbRGB)        AF_CONFIG_FIELD("color order:        ", "RGB (ignored if grayscale)");
    else            AF_CONFIG_FIELD("color order:        ", "BGR (ignored if grayscale)");
    AF_CONFIG_END();
}

shared_ptr<FeatureExtractor> Tracking::getFeatureExtractor(const int& scaleNumFeaturesMonocular_,
                                                           const string &featureSettingsYamlFile,
                                                            const FeatureType& featureType){

    shared_ptr<FeatureExtractorSettings> extractorSettings = make_shared<FeatureExtractorSettings>(featureType, featureSettingsYamlFile);
    extractorSettings->maxNumFeatures *= scaleNumFeaturesMonocular_;

    const AF_VSLAM::Feature& ft = get_feature(featureType);
    return ft.createExtractor(extractorSettings);
}

void Tracking::getGrayImage(cv::Mat& im , const bool& rgb){
    if(im.channels() == 3)
    {
        if(rgb)
            cvtColor(im,im,CV_RGB2GRAY);
        else
            cvtColor(im,im,CV_BGR2GRAY);
    }
    else if(im.channels() == 4)
    {
        if(rgb)
            cvtColor(im,im,CV_RGBA2GRAY);
        else
            cvtColor(im,im,CV_BGRA2GRAY);
    }
}

} //namespace ORB_SLAM
