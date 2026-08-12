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


#ifndef VIEWER_H
#define VIEWER_H

#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Tracking.h"
#include "System.h"

#include <mutex>

namespace AF_VSLAM
{

class Tracking;
class FrameDrawer;
class MapDrawer;
class System;

class Viewer
{
public:
    // 'system' is a non-owning pointer: System owns the Viewer (and joins its thread in
    // Shutdown()), so it always outlives it.
    Viewer(System* system, std::shared_ptr<FrameDrawer> frameDrawer, std::shared_ptr<MapDrawer> mapDrawer,
           std::shared_ptr<Tracking> tracker,
           const string &strCalibrationPath, const string &strSettingPath,
           const vector<FeatureType>& featureTypes);

    // Main thread function. Draw points, keyframes, the current camera pose and the last processed
    // frame. Drawing is refreshed according to the camera fps. We use Pangolin.
    void Run();

    void RequestFinish();

    void RequestStop();

    bool isFinished();

    bool isStopped();

    void Release();

    const std::string& GetWindowTitle() const { return windowTitle; }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Profiling
    std::mutex mutexProfileStats;
    void set_grabImageMonocular_time_median(double time){
        std::unique_lock<std::mutex> lock(mutexProfileStats);
        grabImageMonocular_time_median = time;
    };

    void set_runLocalMapping_time_median(double time){
        std::unique_lock<std::mutex> lock(mutexProfileStats);
        runLocalMapping_time_median = time;
    };

    double grabImageMonocular_time_median{0.0};
    double runLocalMapping_time_median{0.0};

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////

private:

    bool Stop();

    System* system;
    std::shared_ptr<FrameDrawer> frameDrawer;
    std::shared_ptr<MapDrawer> mapDrawer;
    std::shared_ptr<Tracking> tracker;

    // 1/fps in ms
    double mT{1e3/30.0};
    float image_width{640.0}, image_height{480.0};

    float mViewpointX{0.0f}, mViewpointY{-0.7f}, mViewpointZ{-1.8f}, mViewpointF{500.0f};

    std::string windowTitle{};

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    bool mbStopped;
    bool mbStopRequested;
    std::mutex mMutexStop;

    vector<FeatureType> featureTypes;

};

}


#endif // VIEWER_H
