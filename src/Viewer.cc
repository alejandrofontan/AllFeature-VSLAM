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

#include <mutex>

namespace AF_VSLAM
{

Viewer::Viewer(std::shared_ptr<System> system, std::shared_ptr<FrameDrawer> frameDrawer,
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
    for (int i{0}; i < cameras.size(); ++i){
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
}

void Viewer::Run()
{
    mbFinished = false;
    mbStopped = false;

    const float scaleFactor{1.25};
    const float w = scaleFactor * 1280.0f;
    const float h = scaleFactor * 720.0f;
    const float wS{0.35f};
    const float wS_inv{1.0f - wS};
    const float w_pixel = wS * w;
    const float h_pixel = (image_height / float(image_width)) * w_pixel;
    const float hS = h_pixel / h;

    string title = "AllFeature-VSLAM : Map Viewer (";
    for(size_t i=0; i<featureTypes.size(); i++){
        title += " " + featureName(featureTypes[i]);
        if (i != featureTypes.size() - 1){
            title += ",";
        }
    }
    title += " )";
    pangolin::CreateWindowAndBind(title, w, h);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(wS_inv * w, h ,mViewpointF,mViewpointF, wS_inv * w/2.0f, h/2.0f,0.1,1000),
        pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0)
    );

    const float left_l = 0.01f;
    const float left_r = wS;
    const float left_t = 0.99f;
    const float left_b = 0.01f;
    pangolin::View& d_menu = pangolin::CreatePanel("menu").SetBounds(left_b, left_t - hS - 0.01f, left_l, left_r);
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",true,true);
    pangolin::Var<bool> menuAerialCamera("menu.Aerial Camera",false,true);

    pangolin::Var<float> menuTrackingTime("menu.Tracking (ms)", 0.0f, 0.0f, 100.0f, false);
    pangolin::Var<float> menuLocalMappingTime("menu.Local Mapping (ms)", 0.0f, 0.0f, 400.0f, false);

    pangolin::View& d_img = pangolin::Display("img")
            .SetBounds(left_t - hS,  left_t, left_l, left_r);

    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, left_r ,1.0)
        .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();

    pangolin::OpenGlMatrix Twc_aerial;
    Twc_aerial.SetIdentity();

    vec3f trajectoryCenter0{vec3f::Zero()};
    float cameraHeight0{1.0f};

    cv::Mat im = frameDrawer->DrawFrame();
    pangolin::GlTexture imageTexture  = pangolin::GlTexture(int(w_pixel) ,int(h_pixel), GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);

    int numIt{-1};
    bool bFollow = true;
    bool bAerial = false;
    while(1)
    {
        numIt++;

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f,1.0f,1.0f,1.0f);

        ////////////////////////////////////////////////////////////////////////////////////
        menuTrackingTime = static_cast<float>(grabImageMonocular_time_median);
        menuLocalMappingTime = static_cast<float>(runLocalMapping_time_median);

        ////////////////////////////////////////////////////////////////////////////////////
        mapDrawer->GetCurrentOpenGLCameraMatrix(Twc);

        d_cam.Activate(s_cam);
        mapDrawer->DrawCurrentCamera(Twc);
        mapDrawer->DrawKeyFrames(true,true);
        mapDrawer->DrawMapPoints();

        if(menuFollowCamera && bFollow)
        {
            if(menuAerialCamera && !bAerial){
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt( 0, mViewpointY, 0, 0, 0, 0, 0, 0, 1 ));
            }else if(!menuAerialCamera && bAerial){
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
            }
            s_cam.Follow(Twc);
        }else if(menuFollowCamera && !bFollow)
        {
            if(menuAerialCamera && !bAerial){
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt( 0, mViewpointY, 0, 0, 0, 0, 0, 0, 1 ));
            }else if(!menuAerialCamera && bAerial){
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
            }
            s_cam.Follow(Twc);
        }
        else if(!menuFollowCamera && bFollow)
        {
            if(menuAerialCamera && !bAerial){
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt( 0, mViewpointY, 0, 0, 0, 0, 0, 0, 1 ));
                s_cam.Follow(Twc);
            }else if(!menuAerialCamera && bAerial){
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
                s_cam.Follow(Twc);
            }
        }else if(!menuFollowCamera && !bFollow)
        {
            if(menuAerialCamera && !bAerial){
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt( 0, mViewpointY, 0, 0, 0, 0, 0, 0, 1 ));
                s_cam.Follow(Twc);
            }else if(!menuAerialCamera && bAerial){
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
                s_cam.Follow(Twc);
            }
        }

        if(menuFollowCamera && !bFollow)      bFollow = true;
        else if(!menuFollowCamera && bFollow) bFollow = false;

        if(menuAerialCamera && !bAerial)      bAerial = true;
        else if(!menuAerialCamera && bAerial) bAerial = false;


        im = frameDrawer->DrawFrame();
        cv::Mat imResized;
        cv::resize(im,imResized,cv::Size(int(w_pixel), int(h_pixel)));

        /////////////////////////////////////////////////////////
        cv::flip(imResized.clone(),imResized,0);

        d_img.Activate();
        glColor3f(1.0,1.0,1.0);
        imageTexture.Upload(imResized.data,GL_RGB,GL_UNSIGNED_BYTE);
        imageTexture.RenderToViewport();

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
