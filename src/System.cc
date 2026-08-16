#include "System.h"
#include "Converter.h"
#include "FeatureMatcher.h"
#include "Optimizer.h"

#include "DBoW2/Random.h"

#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>
#include <yaml-cpp/yaml.h>
#include "afvslam_log.hpp"

namespace AF_VSLAM
{

System::System(const string &vocabularyFolder,
               const string &strCalibrationFile, const string &strSettingsFile,
               const eSensor sensor,
               const bool activateVisualization,
               const vector<FeatureType>& featureTypes,
               const bool& fixImageSize):
               featureTypes(featureTypes), mSensor(sensor), viewer(static_cast<shared_ptr<Viewer>>(nullptr)), mbReset(false)
{
    // Output welcome message

    if(mSensor==MONOCULAR)
        AF_INFO("Monocular sensor selected");
    else if(mSensor==STEREO)
        AF_INFO("Stereo sensor selected");
    else if(mSensor==RGBD)
        AF_INFO("RGB-D sensor selected");

    //Check settings file
    cv::FileStorage fsCalibration(strCalibrationFile.c_str(), cv::FileStorage::READ);
    if(!fsCalibration.isOpened())
    {
       cerr << "Failed to open calibration file at: " << strCalibrationFile << endl;
       exit(-1);
    }

    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "Failed to open settings file at: " << strSettingsFile << endl;
        exit(-1);
    }

    Optimizer::LoadParameters(fsSettings);

    DUtils::Random::SeedRandOnce(0);

    //Load ORB Vocabulary
    vocabulary = make_shared<Vocabulary>(vocabularyFolder, featureTypes[0]);
    vocabulary->createVocabulary();
    bool vocabularyLoaded = vocabulary->loadFromTextFile();
    if(!vocabularyLoaded){
        AF_ERROR("[System] Vocabulary loading failed");
        terminate();
    }

    //Create KeyFrame Database
    mpKeyFrameDatabase = make_shared<KeyFrameDatabase>(vocabulary);

    //Create the Map
    mpMap = make_shared<Map>();

    //Create Drawers. These are used by the Viewer
    frameDrawer = make_shared<FrameDrawer>(mpMap,featureTypes);
    mapDrawer = make_shared<MapDrawer>(mpMap, strSettingsFile,featureTypes);

    // Feature settings yaml file
    std::map<FeatureType, string> feature_settings_yaml_file;
    for (const auto& featureType : featureTypes){
        const AF_VSLAM::Feature& ft = get_feature(featureType);
        feature_settings_yaml_file[featureType] = ft.getSettingsYamlFile();
    }

    // Initialize matching thresholds
    for (const auto& featureType : featureTypes){
         FeatureMatcher::setDescriptorDistanceThresholds(feature_settings_yaml_file.at(featureType), featureType);
    }

    //Initialize the Tracking thread
    //(it will live in the main thread of execution, the one that called this constructor)
    tracker = make_shared<Tracking>(this, vocabulary, frameDrawer, mapDrawer,
                             mpMap, mpKeyFrameDatabase,
                             strCalibrationFile, strSettingsFile,
                             feature_settings_yaml_file,
                             mSensor, featureTypes, fixImageSize);

    //Initialize the Local Mapping thread and launch
    localMapper = make_shared<LocalMapping>(mpMap, mSensor==MONOCULAR, featureTypes, tracker->get_image_width(), tracker->get_image_height());
    mptLocalMapping = make_shared<thread>(&AF_VSLAM::LocalMapping::Run, localMapper);

    //Initialize the Loop Closing thread and launch
    loopCloser =  make_shared<LoopClosing>(mpMap, mpKeyFrameDatabase, vocabulary, mSensor!=MONOCULAR,
        featureTypes[featureLoopClosure], featureTypes,
        tracker->get_image_width(), tracker->get_image_height());
    mptLoopClosing = make_shared<thread>(&AF_VSLAM::LoopClosing::Run, loopCloser);

    //Initialize the Viewer thread and launch
    if(activateVisualization)
    {
        viewer = make_shared<Viewer>(this, frameDrawer,mapDrawer,tracker,
                                    strCalibrationFile, strSettingsFile,
                                    featureTypes);
        mptViewer = make_shared<thread>(&Viewer::Run, viewer);
        tracker->SetViewer(viewer);
        localMapper->SetViewer(viewer);
    }

    //Set pointers between threads
    tracker->SetLocalMapper(localMapper);
    tracker->SetLoopClosing(loopCloser);

    localMapper->SetTracker(tracker);
    localMapper->SetLoopCloser(loopCloser);

    loopCloser->SetTracker(tracker);
    loopCloser->SetLocalMapper(localMapper);
    loopCloser->SetMapDrawer(mapDrawer);
}

mat4f System::TrackStereo(const cv::Mat &, const cv::Mat &, const double &)
{
    std::cout << "This function (System::TrackStereo) has not been modified yet to work with AnyFeature-VSLAM"<< endl;
    std::terminate();

    // if(mSensor!=STEREO)
    // {
    //     cerr << "ERROR: you called TrackStereo but input sensor was not set to STEREO." << endl;
    //     exit(-1);
    // }

    // // Check mode change
    // {
    //     unique_lock<mutex> lock(mMutexMode);
    //     if(mbActivateLocalizationMode)
    //     {
    //        localMapper->RequestStop();

    //         // Wait until Local Mapping has effectively stopped
    //         while(!localMapper->isStopped())
    //         {
    //             usleep(1000);
    //         }

    //         tracker->InformOnlyTracking(true);
    //         mbActivateLocalizationMode = false;
    //     }
    //     if(mbDeactivateLocalizationMode)
    //     {
    //         tracker->InformOnlyTracking(false);
    //         localMapper->Release();
    //         mbDeactivateLocalizationMode = false;
    //     }
    // }

    // // Check reset
    // {
    // unique_lock<mutex> lock(mMutexReset);
    // if(mbReset)
    // {
    //     tracker->Reset();
    //     mbReset = false;
    // }
    // }

    // mat4f Tcw = tracker->GrabImageStereo(imLeft,imRight,timestamp);

    // unique_lock<mutex> lock2(mMutexState);
    // mTrackingState = tracker->mState;
    // mTrackedMapPoints = tracker->currentFrame.pts;
    // mTrackedKeyPointsUn = tracker->currentFrame.keypoints;
    // return Tcw;
}

mat4f System::TrackRGBD(const cv::Mat &, const cv::Mat &, const double &)
{
    std::cout << "This function (System::TrackRGBD) has not been modified yet to work with AnyFeature-VSLAM"<< endl;
    std::terminate();
    // if(mSensor!=RGBD)
    // {
    //     cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
    //     exit(-1);
    // }

    // // Check mode change
    // {
    //     unique_lock<mutex> lock(mMutexMode);
    //     if(mbActivateLocalizationMode)
    //     {
    //         localMapper->RequestStop();

    //         // Wait until Local Mapping has effectively stopped
    //         while(!localMapper->isStopped())
    //         {
    //             usleep(1000);
    //         }

    //         tracker->InformOnlyTracking(true);
    //         mbActivateLocalizationMode = false;
    //     }
    //     if(mbDeactivateLocalizationMode)
    //     {
    //         tracker->InformOnlyTracking(false);
    //         localMapper->Release();
    //         mbDeactivateLocalizationMode = false;
    //     }
    // }

    // // Check reset
    // {
    // unique_lock<mutex> lock(mMutexReset);
    // if(mbReset)
    // {
    //     tracker->Reset();
    //     mbReset = false;
    // }
    // }

    // mat4f Tcw = tracker->GrabImageRGBD(im,depthmap,timestamp);

    // unique_lock<mutex> lock2(mMutexState);
    // mTrackingState = tracker->mState;
    // mTrackedMapPoints = tracker->currentFrame.pts;
    // mTrackedKeyPointsUn = tracker->currentFrame.keypoints;
    // return Tcw;
}

mat4f System::Track(Image &im, const double &timestamp)
{
    std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();

    if(mSensor!=MONOCULAR && mSensor!=RGBD)
    {
        cerr << "ERROR: you called Track but input sensor was not set to mono or RGBD." << endl;
        exit(-1);
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        tracker->Reset();
        mbReset = false;
    }
    }

    mat4f Tcw = tracker->GrabImageMonocular(im,timestamp);

    mnFramesProcessed.fetch_add(1);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = tracker->mState;
    mTrackedMapPoints = tracker->currentFrame.pts;
    mTrackedKeyPointsUn = tracker->currentFrame.keypoints;

    std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
    double t_duration = std::chrono::duration_cast<std::chrono::duration<double> >(t_end - t_start).count();
    trackingTime.push_back(t_duration);

    return Tcw;
}

bool System::MapChanged()
{
    static int n=0;
    int curn = mpMap->GetLastBigChangeIdx();
    if(n<curn)
    {
        n=curn;
        return true;
    }
    else
        return false;
}

void System::Reset()
{
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

void System::Shutdown()
{
    localMapper->RequestFinish();
    loopCloser->RequestFinish();
    if(viewer)
    {
        viewer->RequestFinish();
        while(!viewer->isFinished())
            usleep(5000);
    }

    // Wait until all thread have effectively stopped
    while(!localMapper->isFinished() || !loopCloser->isFinished() || loopCloser->isRunningGBA())
    {
        usleep(5000);
    }

    if(viewer)
        pangolin::BindToContext(viewer->GetWindowTitle());
}

void System::SetSequenceInfo(const size_t nImages, const bool useMasks)
{
    mnSequenceImages.store(nImages);
    mbUseMasks.store(useMasks);
}

std::string System::GetModalityDescription() const
{
    std::string modality;
    switch(mSensor)
    {
        case MONOCULAR: modality = "Monocular"; break;
        case STEREO:    modality = "Stereo";    break;
        case RGBD:      modality = "RGB-D";     break;
    }
    if(mbUseMasks.load())
        modality += " + masks";
    return modality;
}

void System::SaveTrajectoryTUM(const string &filename)
{
    AF_INFO("Saving camera trajectory to " << filename << " ...");
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
        return;
    }

    vector<Keyframe> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    mat4f Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<AF_VSLAM::Keyframe>::iterator lRit = tracker->mlpReferences.begin();
    list<double>::iterator lT = tracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = tracker->mlbLost.begin();
    for(list<mat4f>::iterator lit=tracker->mlRelativeFramePoses.begin(),
        lend=tracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        if(*lbL)
            continue;

        Keyframe pKF = *lRit;

        mat4f Trw{mat4f::Identity()};

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        while(pKF->is_bad())
        {
            Trw = Trw * pKF->Tcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        mat4f Tcw = (*lit) * Trw;
        mat3f Rwc = Tcw.block<3,3>(0,0).transpose();
        vec3f twc = -Rwc * Tcw.block<3,1>(0,3);
        Eigen::Quaternionf q(Rwc);

        f << setprecision(6) << *lT << " " <<  setprecision(9) <<
            twc(0) << " " << twc(1) << " " << twc(2) << " " <<
            q.x() << " " << q.y() << " " << q.z()<< " " << q.w() << endl;
    }
    f.close();
    AF_INFO("Trajectory saved!");
}


void System::SaveKeyFrameTrajectoryVSLAMLAB(const string &filename)
{
    AF_INFO("Saving keyframe trajectory to " << filename << " ...");

    vector<Keyframe> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    //cv::Mat Two = vpKFs[0]->GetPoseInverse();

    std::ofstream f(filename.c_str());
    f.imbue(std::locale::classic());

    // CSV header
    f << "ts (ns),tx (m),ty (m),tz (m),qx,qy,qz,qw\n";

    for(size_t i=0; i<vpKFs.size(); i++)
    {

        Keyframe pKF = vpKFs[i];
        #ifdef ALLFEATURE_EVALUATION
        if ((int(pKF->frame_id) % ALLFEATURE_EVALUATION) != 0){
            continue;
        }
        #endif
       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->is_bad())
            continue;

        mat3f R = pKF->get_rotation().transpose();
        Eigen::Quaternionf q(R);
        vec3f t = pKF->get_camera_center();

        long long ts_ns = static_cast<long long>(std::round(pKF->mTimeStamp * 1e9));
        f << std::fixed << std::setprecision(9) << ts_ns << ','
          << std::scientific << std::setprecision(7)
          << static_cast<double>(t(0)) << ','
          << static_cast<double>(t(1)) << ','
          << static_cast<double>(t(2)) << ','
          << static_cast<double>(q.x()) << ','
          << static_cast<double>(q.y()) << ','
          << static_cast<double>(q.z()) << ','
          << static_cast<double>(q.w()) << '\n';
    }

    f.close();
    AF_INFO("Keyframe trajectory saved!");
}

struct PointRGB {
    float x, y, z;
    uint8_t r, g, b;
};

bool write_ply_binary(const std::string& path, const std::vector<PointRGB>& pts) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;

    out << "ply\n";
    out << "format binary_little_endian 1.0\n";
    out << "element vertex " << pts.size() << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "property uchar red\n";
    out << "property uchar green\n";
    out << "property uchar blue\n";
    out << "end_header\n";

    for (const auto& p : pts) {
        out.write(reinterpret_cast<const char*>(&p.x), sizeof(float));
        out.write(reinterpret_cast<const char*>(&p.y), sizeof(float));
        out.write(reinterpret_cast<const char*>(&p.z), sizeof(float));
        out.write(reinterpret_cast<const char*>(&p.r), sizeof(uint8_t));
        out.write(reinterpret_cast<const char*>(&p.g), sizeof(uint8_t));
        out.write(reinterpret_cast<const char*>(&p.b), sizeof(uint8_t));
    }

    return static_cast<bool>(out);
}

auto to_u8 = [](double v) {
    v = std::clamp(v, 0.0, 1.0);
    return static_cast<uint8_t>(std::lround(v * 255.0));
};

void System::SavePointCloudVSLAMLAB(const string &filename, const vector<string>& imageFilenames)
{
    AF_INFO("Saving point cloud to " << filename << " ...");

    std::vector<PointRGB> pts;
    auto mapPoints = mpMap->GetAllMapPoints();
    int numPoints = 0;
    std::map<int, cv::Mat> imageCache;
    float alpha = 0.0f;
    for (auto& mp: mapPoints) {
        if (mp->is_bad()) continue;
        numPoints++;
        PointRGB p;
        vec3f pos = mp->get_world_pos();
        p.x = pos(0);
        p.y = pos(1);
        p.z = pos(2);

        FeatureType ft = mp->featureType;
        cv::Scalar color = AF_VSLAM::getFeatureColor(ft, 0, true);
        // p.r = 255;
        // p.g = 255;
        // p.b = 255;

        auto keyfram = mp->GetCurrentRefKeyframe();
        int idx = mp->GetIndexInKeyFrame(keyfram);
        int imgIdx = keyfram->frame_id;

        if (imageCache.find(imgIdx) == imageCache.end()) {
            cv::Mat cvimg = cv::imread(imageFilenames[imgIdx],cv::IMREAD_UNCHANGED);
            imageCache[imgIdx] = cvimg;
        }
        cv::Mat cvimg = imageCache[imgIdx];
        int u = static_cast<int>(keyfram->mvKeys.at(mp->featureType)[idx].pt.x);
        int v = static_cast<int>(keyfram->mvKeys.at(mp->featureType)[idx].pt.y);
        if (cvimg.channels() == 1) {
            uint8_t intensity = cvimg.at<uint8_t>(v, u);
            p.r = intensity;
            p.g = intensity;
            p.b = intensity;
        } else if (cvimg.channels() == 3) {
            cv::Vec3b bgr = cvimg.at<cv::Vec3b>(v, u);
            p.r = to_u8(alpha * (float(bgr[2]) / 255.0f) + (1.0f - alpha) * (float(color[0])));
            p.g = to_u8(alpha * (float(bgr[1]) / 255.0f) + (1.0f - alpha) * (float(color[1])));
            p.b = to_u8(alpha * (float(bgr[0]) / 255.0f) + (1.0f - alpha) * (float(color[2])));
        }
        pts.push_back(p);
    }
    AF_INFO("Point cloud saved! Number of points: " << numPoints);
    write_ply_binary(filename, pts);
}



void System::SaveTrajectoryKITTI(const string &filename)
{
    AF_INFO("Saving camera trajectory to " << filename << " ...");
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
        return;
    }

    vector<Keyframe> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    mat4f Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<AF_VSLAM::Keyframe>::iterator lRit = tracker->mlpReferences.begin();
    list<double>::iterator lT = tracker->mlFrameTimes.begin();
    for(list<mat4f>::iterator lit=tracker->mlRelativeFramePoses.begin(), lend=tracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
    {
        AF_VSLAM::Keyframe pKF = *lRit;

        mat4f Trw{mat4f::Identity()};

        while(pKF->is_bad())
        {
            Trw = Trw * pKF->Tcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw *pKF->GetPose()*Two;

        mat4f Tcw = (*lit) * Trw;
        mat3f Rwc = Tcw.block<3,3>(0,0).transpose();
        vec3f twc = -Rwc * Tcw.block<3,1>(0,3);

        f << setprecision(9) <<
             Rwc(0,0) << " " << Rwc(0,1)  << " " << Rwc(0,2) << " "  << twc(0) << " " <<
             Rwc(1,0) << " " << Rwc(1,1)  << " " << Rwc(1,2) << " "  << twc(1) << " " <<
             Rwc(2,0) << " " << Rwc(2,1)  << " " << Rwc(2,2) << " "  << twc(2) << endl;
    }
    f.close();
    AF_INFO("Camera trajectory saved!");
}

int System::GetTrackingState()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackingState;
}

std::map<FeatureType,std::vector<Pt>> System::GetTrackedMapPoints()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedMapPoints;
}

std::map<FeatureType, std::vector<cv::KeyPoint>> System::GetTrackedKeyPointsUn()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedKeyPointsUn;
}

void System::SaveStatistics(const std::string &filename){
    // auto keyframes = mpMap->GetAllKeyFrames();
    // auto pts = mpMap->GetAllMapPoints();

    // size_t numKeyframes{0};
    // for (auto& keyframe: keyframes) {
    //     if(keyframe->is_bad())
    //         continue;
    //     numKeyframes++;
    // }

    // size_t numPts{0};
    // size_t numObservations{0};
    // for (auto& pt: pts) {
    //     if(pt->is_bad())
    //         continue;
    //     numPts++;
    //     numObservations += pt->number_of_observations();
    // }
    // float numObservationsPerPt = float(numObservations)/float(numPts);

    // double median_tracking_time{};
    // AF_VSLAM::vectorMedian(median_tracking_time,trackingTime);

    // double medianLocalMapppingTime{};
    // AF_VSLAM::vectorMedian(medianLocalMapppingTime,localMapper->localMappingTime);

    // double medianLoopClosingTime{};
    // AF_VSLAM::vectorMedian(medianLoopClosingTime,loopCloser->loopClosingTime);

    // long long finalVirtualMemUsed{virtualMemUsed.back()};
    // long long firstVirtualMemUsed{virtualMemUsed.front()};
    // long long maxVirtualMemUsed{0};
    // AF_VSLAM::vectorMax(maxVirtualMemUsed,virtualMemUsed);

    // ofstream f;
    // string statisticsFile = filename + ".txt";
    // f.open(statisticsFile.c_str());
    // f << fixed;
    // f << setprecision(0) << numKeyframes << " " << numPts << " " << numObservations << " " << setprecision(3) << numObservationsPerPt  <<
    // " " << setprecision(9) << median_tracking_time << " " << medianLocalMapppingTime << " " << medianLoopClosingTime <<
    // " " << tracker->numTrackedFrames <<" " << loopCloser->numOfLoopClosures << setprecision(0) <<
    // " " << firstVirtualMemUsed <<" " << maxVirtualMemUsed << " " << finalVirtualMemUsed <<
    // endl;

    // // Create statistics yaml file
    string statisticsFile_yaml = filename + ".yaml";
    // YAML::Node node;

    // node["graph"]["numKeyframes"] = numKeyframes;
    // node["graph"]["numPts"] = numPts;
    // node["graph"]["numObservations"] = numObservations;
    // node["graph"]["numObservationsPerPt"] = numObservationsPerPt;

    // node["profiling"]["median_tracking_time"] = median_tracking_time;
    // node["profiling"]["medianLocalMapppingTime"] = medianLocalMapppingTime;
    // node["profiling"]["medianLoopClosingTime"] = medianLoopClosingTime;

    // node["recall"]["numTrackedFrames"] = tracker->numTrackedFrames;
    // node["recall"]["loopCloser->numOfLoopClosures"] = loopCloser->numOfLoopClosures;

    // node["memory"]["firstVirtualMemUsed"] = firstVirtualMemUsed;
    // node["memory"]["maxVirtualMemUsed"] = maxVirtualMemUsed;
    // node["memory"]["finalVirtualMemUsed"] = finalVirtualMemUsed;

    // std::ofstream fout(statisticsFile_yaml);

    // fout << node;
    // fout.close();
    AF_INFO(statisticsFile_yaml + " file written successfully!");

}

void System::setImageSize(const int width, const int height){
    image_width = width;
    image_height = height;
}

void System::GBA(){
    AF_INFO("Starting Global Bundle Adjustment...");
    Optimizer::GlobalBundleAdjustemnt(mpMap, 100);
    AF_INFO("Global Bundle Adjustment finished.");
}

}
//namespace ORB_SLAM
