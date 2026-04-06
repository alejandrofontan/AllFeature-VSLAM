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



#include "System.h"
#include "Converter.h"
#include "MathFunctions.h"
#include "FeatureMatcher.h"
#include "Optimizer.h"

#include "DBoW2/Random.h"

#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>
#include <yaml-cpp/yaml.h>

namespace AF_VSLAM
{

System::System(const string &vocabularyFolder,
               const string &strCalibrationFile, const string &strSettingsFile,
               const eSensor sensor,
               const bool activateVisualization,
               const vector<FeatureType>& featureTypes,
               const bool& fixImageSize):
               mSensor(sensor), viewer(static_cast<shared_ptr<Viewer>>(nullptr)), mbReset(false), featureTypes(featureTypes)
{
    // Output welcome message
    cout << "Any-Feature V-SLAM 2024, Alejandro Fontan Villacampa, Queensland University of Technology\n"
    "    Acknowledgments to: Javier Civera and Michael Milford (Any-Feature V-SLAM)\n"
    "    Raul Mur-Artal, Juan D. Tardos, J. M. M. Montiel (ORB-SLAM2), Dorian Galvez-Lopez (DBoW2),\n    Carlos Campos, Richard Elvira and Juan J. Gómez Rodríguez (ORB-SLAM3)."
    "\n\nThis program comes with ABSOLUTELY NO WARRANTY;" << endl  <<
    "This is free software, and you are welcome to redistribute it under certain conditions. See LICENSE.txt." << endl << endl;

    cout << "Input sensor was set to: ";

    if(mSensor==MONOCULAR)
        cout << "mono" << endl;
    else if(mSensor==STEREO)
        cout << "Stereo" << endl;
    else if(mSensor==RGBD)
        cout << "RGB-D" << endl;

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

    DUtils::Random::SeedRandOnce(0);

    //Load ORB Vocabulary
    vocabulary = make_shared<Vocabulary>(vocabularyFolder, featureTypes[0]);
    vocabulary->createVocabulary();
    bool vocabularyLoaded = vocabulary->loadFromTextFile();
    if(!vocabularyLoaded){
        std::cout <<"[System] Vocabulary loading failed" << std::endl;
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
        std::unique_ptr<AF_VSLAM::Feature> ft = get_feature(featureType);
        feature_settings_yaml_file[featureType] = ft->getSettingsYamlFile();
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
    loopCloser =  make_shared<LoopClosing>(mpMap, mpKeyFrameDatabase, vocabulary, mSensor!=MONOCULAR, featureTypes[featureLoopClosure],
        tracker->get_image_width(), tracker->get_image_height());
    mptLoopClosing = make_shared<thread>(&AF_VSLAM::LoopClosing::Run, loopCloser);

    //Initialize the Viewer thread and launch
    if(activateVisualization)
    {
        viewer = make_shared<Viewer>(static_cast<shared_ptr<System>>(this), frameDrawer,mapDrawer,tracker,
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

mat4f System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp)
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

mat4f System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp)
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

mat4f System::TrackMonocular(Image &im, const double &timestamp)
{
    std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();

    if(mSensor!=MONOCULAR)
    {
        cerr << "ERROR: you called TrackMonocular but input sensor was not set to mono." << endl;
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
        pangolin::BindToContext("ORB-SLAM2: Map Viewer");
}

void System::SaveTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
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
    cout << endl << "trajectory saved!" << endl;
}


void System::SaveKeyFrameTrajectoryVSLAMLAB(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

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
    cout << endl << "trajectory saved!" << endl;
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
    cout << endl << "Saving point cloud to " << filename << " ..." << endl;

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
            // std::cout << " - Color from image. ";
            // std::cout << " - bgr[0]: " << static_cast<int>(bgr[0]) <<  std::endl;
            // std::cout << " - bgr[1]: " << static_cast<int>(bgr[1]) <<  std::endl;
            // std::cout << " - bgr[2]: " << static_cast<int>(bgr[2]) <<  std::endl;
            // std::cout << " - color[0]: " << color[0] <<  std::endl;
            // std::cout << " - color[1]: " << color[1] <<  std::endl;
            // std::cout << " - color[2]: " << color[2] <<  std::endl;
            // p.b = uint8_t(alpha * float(bgr[2]) + (1.0f - alpha) * float(color[0]));
            // p.g = uint8_t(alpha * float(bgr[1]) + (1.0f - alpha) * float(color[1]));
            // p.r = uint8_t(alpha * float(bgr[0]) + (1.0f - alpha) * float(color[2]));
            p.r = to_u8(alpha * (float(bgr[2]) / 255.0f) + (1.0f - alpha) * (float(color[0])));
            p.g = to_u8(alpha * (float(bgr[1]) / 255.0f) + (1.0f - alpha) * (float(color[1])));
            p.b = to_u8(alpha * (float(bgr[0]) / 255.0f) + (1.0f - alpha) * (float(color[2])));
            // std::cout << " - p.r: " << static_cast<int>(p.r) <<  std::endl;
            // std::cout << " - p.g: " << static_cast<int>(p.g) <<  std::endl;
            // std::cout << " - p.b: " << static_cast<int>(p.b) <<  std::endl;
        }

        // p.r = to_u8(color[0]); // R
        // p.g = to_u8(color[1]); // G
        // p.b = to_u8(color[2]); // B

        pts.push_back(p);
    }
    cout << endl << "point cloud saved! Number of points: " << numPoints << endl;
    write_ply_binary(filename, pts);
}



void System::SaveTrajectoryKITTI(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
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
          //  cout << "bad parent" << endl;
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
    cout << endl << "trajectory saved!" << endl;
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

    // double medianTrackingTime{};
    // AF_VSLAM::vectorMedian(medianTrackingTime,trackingTime);

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
    // " " << setprecision(9) << medianTrackingTime << " " << medianLocalMapppingTime << " " << medianLoopClosingTime <<
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

    // node["profiling"]["medianTrackingTime"] = medianTrackingTime;
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
    std::cout << statisticsFile_yaml + " file written successfully!" << std::endl;

}

void System::setImageSize(const int width, const int height){
    imageWidth = width;
    imageHeight = height;
}

void System::GBA(){
    std::cout << "Starting Global Bundle Adjustment..." <<std::endl;
    Optimizer::GlobalBundleAdjustemnt(mpMap, 100);
    std::cout << "Global Bundle Adjustment finished." <<std::endl;
}

}
//namespace ORB_SLAM
