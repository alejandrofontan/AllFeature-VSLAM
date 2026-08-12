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

#include "Viewer.h"
#include "Utils.h"
#include <pangolin/pangolin.h>
#include <yaml-cpp/yaml.h>
#include "afvslam_log.hpp"

#include <iomanip>
#include <mutex>
#include <sstream>

namespace AF_VSLAM
{

Viewer::Viewer(System* system, std::shared_ptr<FrameDrawer> frameDrawer,
               std::shared_ptr<MapDrawer> mapDrawer, std::shared_ptr<Tracking> tracker,
               const string &strCalibrationPath, const string &strSettingPath,
               const vector<FeatureType>& featureTypes):
        system(system), frameDrawer(frameDrawer),mapDrawer(mapDrawer), tracker(tracker),
    mbFinishRequested(false), mbFinished(true), mbStopped(true), mbStopRequested(false),
    featureTypes(featureTypes)
{

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

    float fps = cam["fps"].as<float>();
    if(fps<1)
        fps=30;
    mT = 1e3/fps;

    image_width = cam["image_dimension"][0].as<int>();
    image_height = cam["image_dimension"][1].as<int>();

    AF_INFO("Camera Parameters: " << strCalibrationPath);
    AF_CONFIG_BEGIN("Camera Parameters");
     AF_CONFIG_FIELD("Camera Name:        ", cam_name);
     AF_CONFIG_FIELD("FPS:                ", fps);
     AF_CONFIG_FIELD("Image Width:        ", image_width);
     AF_CONFIG_FIELD("Image Height:       ", image_height);
    AF_CONFIG_END();

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    mViewpointX = fSettings["Viewer.ViewpointX"];
    mViewpointY = fSettings["Viewer.ViewpointY"];
    mViewpointZ = fSettings["Viewer.ViewpointZ"];
    mViewpointF = fSettings["Viewer.ViewpointF"];

    AF_INFO("Viewer Parameters: " << strSettingPath);
    AF_CONFIG_BEGIN("Viewer Parameters");
     AF_CONFIG_FIELD("Viewpoint X:        ", mViewpointX);
     AF_CONFIG_FIELD("Viewpoint Y:        ", mViewpointY);
     AF_CONFIG_FIELD("Viewpoint Z:        ", mViewpointZ);
     AF_CONFIG_FIELD("Viewpoint F:        ", mViewpointF);
    AF_CONFIG_END();

    windowTitle = "VSLAM-LAB | AllFeature-VSLAM (";
    for(size_t i=0; i<featureTypes.size(); i++){
        windowTitle += " " + featureName(featureTypes[i]);
        if (i != featureTypes.size() - 1){
            windowTitle += ",";
        }
    }
    windowTitle += " )";
}

namespace
{
std::string trackingStateName(const int state)
{
    switch(state)
    {
        case Tracking::SYSTEM_NOT_READY: return "NOT READY";
        case Tracking::NO_IMAGES_YET:    return "WAITING";
        case Tracking::NOT_INITIALIZED:  return "INITIALIZING";
        case Tracking::OK:               return "TRACKING";
        case Tracking::LOST:             return "LOST";
        default:                         return "?";
    }
}
}

void Viewer::Run()
{
    mbFinished = false;
    mbStopped = false;

    const float scaleFactor{1.25};
    const float w = scaleFactor * 1280.0f;
    const float h = scaleFactor * 720.0f;

    // Left panel width (fraction of window); the 3D view fills the rest, with the
    // camera image rendered as an overlay in the 3D view's top-right corner.
    const float panelWidth{0.28f};
    const float imgWidthFrac{0.40f};

    pangolin::CreateWindowAndBind(windowTitle, w, h);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Anti-aliased, alpha-blended primitives (round points, smooth lines)
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix((1.0f - panelWidth) * w, h ,mViewpointF,mViewpointF, (1.0f - panelWidth) * w/2.0f, h/2.0f,0.1,1000),
        pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0)
    );

    pangolin::CreatePanel("menu").SetBounds(0.0f, 1.0f, 0.0f, panelWidth);

    const ViewerStyle defaults = mapDrawer->GetDefaultStyle();

    // Status
    pangolin::Var<std::string> menuModality("menu.Modality", system->GetModalityDescription());
    pangolin::Var<std::string> menuState("menu.Status", "WAITING");
    pangolin::Var<std::string> menuFrames("menu.Frames Tracked", "0");
    pangolin::Var<float> menuProgress("menu.Progress %", 0.0f, 0.0f, 100.0f);

    // Section headers: Pangolin has no header widget; an empty string var renders its label
    // in the same plain style as the status entries above.
    pangolin::Var<std::string> headerPerformance("menu.PERFORMANCE", "");

    // Profiling
    pangolin::Var<float> menuTrackingTime("menu.Tracking (ms)", 0.0f, 0.0f, 100.0f, false);
    pangolin::Var<float> menuLocalMappingTime("menu.Local Mapping (ms)", 0.0f, 0.0f, 400.0f, false);

    pangolin::Var<std::string> headerVisualization("menu.VISUALIZATION", "");

    // View
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",true,true);
    pangolin::Var<bool> menuAerialCamera("menu.Aerial View",false,true);
    pangolin::Var<bool> menuDarkTheme("menu.Dark Theme",true,true);

    // Elements
    pangolin::Var<bool> menuShowPoints("menu.Show Map Points",true,true);
    pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames",true,true);
    pangolin::Var<bool> menuShowGraph("menu.Show Graph",true,true);
    pangolin::Var<bool> menuShowTrajectory("menu.Show Trajectory",true,true);

    // Thickness
    pangolin::Var<float> menuPointSize("menu.Point Size", defaults.pointSize, 1.0f, 10.0f);
    pangolin::Var<float> menuTrajWidth("menu.Trajectory Width", defaults.trajectoryLineWidth, 0.5f, 10.0f);
    pangolin::Var<float> menuKFLineWidth("menu.KeyFrame Width", defaults.keyFrameLineWidth, 0.5f, 5.0f);
    pangolin::Var<float> menuGraphLineWidth("menu.Graph Width", defaults.graphLineWidth, 0.5f, 5.0f);
    pangolin::Var<float> menuCamLineWidth("menu.Camera Width", defaults.cameraLineWidth, 0.5f, 8.0f);

    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, panelWidth ,1.0)
        .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();

    // Camera image overlaid on the 3D view's top-right corner (created after d_cam, so it
    // renders on top). The texture is kept at the annotated frame's native resolution with
    // linear sampling — text and keypoint markers stay sharp at any overlay size. It is
    // (re)initialised lazily because the frame size changes once real images replace
    // FrameDrawer's 640x480 placeholder.
    cv::Mat im;
    const float imgW = imgWidthFrac * (1.0f - panelWidth);
    const float margin{0.006f};
    pangolin::View& d_img = pangolin::Display("img");
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    pangolin::GlTexture imageTexture;
    int texW{0}, texH{0};

    bool bAerial = false;
    while(1)
    {
        if(menuDarkTheme)
            glClearColor(vslamlab_colors::kDarkBg[0], vslamlab_colors::kDarkBg[1], vslamlab_colors::kDarkBg[2], 1.0f);
        else
            glClearColor(vslamlab_colors::kLightBg[0], vslamlab_colors::kLightBg[1], vslamlab_colors::kLightBg[2], 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        ////////////////////////////////////////////////////////////////////////////////////
        {
            std::unique_lock<std::mutex> lock(mutexProfileStats);
            menuTrackingTime = static_cast<float>(grabImageMonocular_time_median);
            menuLocalMappingTime = static_cast<float>(runLocalMapping_time_median);
        }

        // Status panel (map-size query locks the map, so refresh it sparingly)
        menuState = trackingStateName(tracker->mState);
        {
            const size_t nTotal = system->GetSequenceImageCount();
            const size_t nProcessed = system->GetFramesProcessedCount();
            std::ostringstream oss;
            oss << tracker->numTrackedFrames << " / "
                << (nTotal > 0 ? std::to_string(nTotal) : std::string("?"));
            menuFrames = oss.str();
            menuProgress = nTotal > 0 ? 100.0f * float(nProcessed) / float(nTotal) : 0.0f;
        }
        ////////////////////////////////////////////////////////////////////////////////////
        ViewerStyle style;
        style.darkTheme = menuDarkTheme;
        style.pointSize = menuPointSize;
        style.trajectoryLineWidth = menuTrajWidth;
        style.keyFrameLineWidth = menuKFLineWidth;
        style.graphLineWidth = menuGraphLineWidth;
        style.cameraLineWidth = menuCamLineWidth;
        style.keyFrameSize = defaults.keyFrameSize;
        style.cameraSize = defaults.cameraSize;

        mapDrawer->GetCurrentOpenGLCameraMatrix(Twc);

        if(menuAerialCamera != bAerial)
        {
            if(menuAerialCamera)
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt( 0, mViewpointY, 0, 0, 0, 0, 0, 0, 1 ));
            else
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
            bAerial = menuAerialCamera;
        }
        if(menuFollowCamera)
            s_cam.Follow(Twc);

        d_cam.Activate(s_cam);
        mapDrawer->DrawCurrentCamera(Twc, style);
        if(menuShowKeyFrames || menuShowGraph)
            mapDrawer->DrawKeyFrames(menuShowKeyFrames, menuShowGraph, style);
        if(menuShowPoints)
            mapDrawer->DrawMapPoints(style);
        if(menuShowTrajectory)
            mapDrawer->DrawTrajectory(style);

        im = frameDrawer->DrawFrame();
        if(!im.empty() && im.isContinuous())
        {
            if(im.cols != texW || im.rows != texH)
            {
                imageTexture.Reinitialise(im.cols, im.rows, GL_RGB, true, 0, GL_RGB, GL_UNSIGNED_BYTE);
                texW = im.cols;
                texH = im.rows;
                const float imgH = (float(texH) / float(texW)) * (imgW * w) / h;
                d_img.SetBounds(1.0f - imgH - margin, 1.0f - margin, 1.0f - imgW - margin, 1.0f - margin);
            }
            d_img.Activate();
            glColor3f(1.0,1.0,1.0);
            imageTexture.Upload(im.data,GL_RGB,GL_UNSIGNED_BYTE);
            imageTexture.RenderToViewportFlipY();
        }

        pangolin::FinishFrame();

        if(Stop())
        {
            while(isStopped())
            {
                usleep(3000);
            }
        }

        if(CheckFinish())
            break;
    }

    SetFinish();
}

void Viewer::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool Viewer::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void Viewer::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool Viewer::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void Viewer::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(!mbStopped)
        mbStopRequested = true;
}

bool Viewer::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool Viewer::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);

    if(mbFinishRequested)
        return false;
    else if(mbStopRequested)
    {
        mbStopped = true;
        mbStopRequested = false;
        return true;
    }

    return false;

}

void Viewer::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
}

}
