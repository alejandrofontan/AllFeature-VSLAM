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

// placecell visualizer (Thirdparty/placecell, placecell::viz): renders the kernel heatmap,
// the unexplained-information history and the alive-information strip to cv::Mat; this
// viewer uploads them as textures into Pangolin panels (PlaceCell.Visualize).
#include <placecell/viz.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <iomanip>
#include <memory>
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
        windowTitle += " " + feature_name(featureTypes[i]);
        if (i != featureTypes.size() - 1){
            windowTitle += ",";
        }
    }
    windowTitle += " )";
}

namespace
{
std::string trackingStateName(const TrackingState state)
{
    switch(state)
    {
        case TrackingState::SYSTEM_NOT_READY: return "NOT READY";
        case TrackingState::NO_IMAGES_YET:    return "WAITING";
        case TrackingState::NOT_INITIALIZED:  return "INITIALIZING";
        case TrackingState::OK:               return "TRACKING";
        case TrackingState::LOST:             return "LOST";
        default:                              return "?";
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
    // Manual reset (clears the map and re-initializes): the flag is consumed by
    // System::Track on the main thread before the next frame is processed.
    pangolin::Var<bool> menuReset("menu.Reset", false, false);

    // Section headers: Pangolin has no header widget; an empty string var renders its label
    // in the same plain style as the status entries above.
    pangolin::Var<std::string> headerPerformance("menu.PERFORMANCE", "");

    // Profiling
    pangolin::Var<float> menuTrackingTime("menu.Tracking (ms)", 0.0f, 0.0f, 100.0f, false);
    pangolin::Var<float> menuLocalMappingTime("menu.Local Mapping (ms)", 0.0f, 0.0f, 400.0f, false);

    pangolin::Var<std::string> headerLocalMapping("menu.LOCAL MAPPING", "");

    // Keyframe information budget tau (LocalMapping.KeyframeCullingMaxUnexplained), editable
    // online: seeded from the loaded parameter, written back every frame (atomic). It drives
    // both ends of the keyframe lifecycle: LocalMapping culls a keyframe only if every keyframe
    // stays at most tau unexplained, Tracking inserts a keyframe whose view is more than tau
    // unexplained by the local map. 0 = never cull / insert everything, 1 = cull anything / never insert for novelty.
    pangolin::Var<float> menuCullMaxUnexplained("menu.KF Max Unexplained",
                                                LocalMapping::params.keyframe_culling_max_unexplained.load(), 0.0f, 1.0f);

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
    pangolin::Var<bool> menuShowTopView("menu.Show Top View",true,true);

    // placecell window (PlaceCell.Visualize): the visualizer's three images in a second
    // Pangolin window, see the block after d_top below. Only offered when a placecell store
    // exists (vpr: megaloc).
    const placecell::PlaceCell* placeCell = system->GetPlaceCell();
    const PlaceCellSettings& placeCellSettings = system->GetPlaceCellSettings();
    std::unique_ptr<pangolin::Var<bool>> menuPlaceCellWindow;
    if(placeCell)
        menuPlaceCellWindow = std::make_unique<pangolin::Var<bool>>("menu.PlaceCell Window", placeCellSettings.visualize, true);

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

    // Top-down minimap of the complete trajectory, overlaid on the 3D view's bottom-right
    // corner (created after d_cam for the same on-top reason as d_img). The orthographic
    // fit to the trajectory bounds happens inside DrawTrajectoryTopView, so the bounds
    // here only pick the inset's screen rectangle (square in pixels).
    const float topW = 0.22f * (1.0f - panelWidth);
    const float topH = topW * w / h;
    pangolin::View& d_top = pangolin::Display("topview")
        .SetBounds(margin, margin + topH, 1.0f - topW - margin, 1.0f - margin);

    // placecell window: a SECOND Pangolin window driven from this same thread. Pangolin keeps
    // one GL context per named window, so each loop iteration renders the main window,
    // binds the placecell one, renders it and binds back. The window (and its GL textures
    // and views, which belong to its context) is created lazily when the menu toggle is on
    // and destroyed when it is off or the user closes it. Layout, 1:1 pixels (no resampling
    // of the plot text): kernel heatmap on the left, information history above the alive-
    // information strip on the right; the kernel canvas is (side + title) tall and (side +
    // colourbar) wide, so the side is derived from the window height and the plots get the
    // remaining width, stacked 64/36.
    const std::string placeCellWindowTitle = windowTitle + " | placecell";
    const int pcW{1500}, pcH{600}, pcMargin{8};
    constexpr int kPlaceCellTitleHeight{26};   // placecell::viz title strip (kTitleHeight)
    constexpr int kPlaceCellColorbarWidth{56}; // placecell::viz colourbar strip
    const int pcBandH = pcH - 2 * pcMargin;
    placecell::viz::Visualizer::Options placeCellVizOptions{};
    placeCellVizOptions.windows = false;   // rendered into Pangolin, never cv::imshow (headless OpenCV)
    placeCellVizOptions.max_hz = 0.0;      // throttled here (see the loop) to avoid its change-check copies
    placeCellVizOptions.kernel.centred = placeCellSettings.visualize_centred;
    placeCellVizOptions.kernel.target_size = std::max(64, pcBandH - kPlaceCellTitleHeight);
    placeCellVizOptions.history.last_n = static_cast<size_t>(std::max(0, placeCellSettings.visualize_history_last_n));
    placeCellVizOptions.history.width = std::max(200, pcW - (placeCellVizOptions.kernel.target_size + kPlaceCellColorbarWidth) - 3 * pcMargin);
    placeCellVizOptions.history.height = int(0.64f * pcBandH);
    placeCellVizOptions.alive.width = placeCellVizOptions.history.width;
    placeCellVizOptions.alive.height = pcBandH - placeCellVizOptions.history.height;
    std::unique_ptr<placecell::viz::Visualizer> placeCellViz;
    pangolin::View* d_pc_kernel{nullptr};
    pangolin::View* d_pc_information{nullptr};
    pangolin::View* d_pc_alive{nullptr};
    std::unique_ptr<pangolin::GlTexture> texPcKernel, texPcInformation, texPcAlive;
    int texPcKernelW{0}, texPcKernelH{0}, texPcInformationW{0}, texPcInformationH{0}, texPcAliveW{0}, texPcAliveH{0};
    bool placeCellWindowOpen{false};
    bool placeCellWindowFailed{false};
    const double placeCellMinPeriod = placeCellSettings.visualize_max_hz > 0.0 ? 1.0 / placeCellSettings.visualize_max_hz : 0.0;
    auto placeCellLastUpdate = std::chrono::steady_clock::time_point::min();

    // Upload a BGR render into its texture (re-created on a size change) and place its view
    // at 1:1 pixels in the placecell window: bounds are (bottom, top, left, right) as
    // fractions of that window. Call with the placecell context bound.
    auto uploadPlaceCellPanel = [&](pangolin::View& view, pangolin::GlTexture& texture, int& texW, int& texH,
                                    const cv::Mat& image, const int bottomPx, const int leftPx)
    {
        if(image.empty() || !image.isContinuous() || image.type() != CV_8UC3)
            return;
        if(image.cols != texW || image.rows != texH)
        {
            texture.Reinitialise(image.cols, image.rows, GL_RGB, true, 0, GL_BGR, GL_UNSIGNED_BYTE);
            texW = image.cols;
            texH = image.rows;
        }
        view.SetBounds(float(bottomPx) / pcH, float(bottomPx + texH) / pcH, float(leftPx) / pcW, float(leftPx + texW) / pcW);
        texture.Upload(image.data, GL_BGR, GL_UNSIGNED_BYTE);
    };

    // Tear the placecell window down (GL objects first, with its context bound) and return
    // to the main context. Safe to call when it is not open.
    auto closePlaceCellWindow = [&]()
    {
        if(!placeCellWindowOpen)
            return;
        pangolin::BindToContext(placeCellWindowTitle);
        texPcKernel.reset();
        texPcInformation.reset();
        texPcAlive.reset();
        texPcKernelW = texPcKernelH = texPcInformationW = texPcInformationH = texPcAliveW = texPcAliveH = 0;
        d_pc_kernel = d_pc_information = d_pc_alive = nullptr;
        pangolin::DestroyWindow(placeCellWindowTitle);
        pangolin::BindToContext(windowTitle);
        placeCellWindowOpen = false;
    };

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
            menuTrackingTime = static_cast<float>(grab_image_time_median_);
            menuLocalMappingTime = static_cast<float>(runLocalMapping_time_median);
        }

        // Live parameters
        LocalMapping::params.keyframe_culling_max_unexplained.store(static_cast<float>(menuCullMaxUnexplained));

        // Status panel (map-size query locks the map, so refresh it sparingly)
        menuState = trackingStateName(tracker->state_);
        {
            const size_t nTotal = system->GetSequenceImageCount();
            const size_t nProcessed = system->GetFramesProcessedCount();
            std::ostringstream oss;
            oss << tracker->num_tracked_frames_ << " / "
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

        if(menuReset)
        {
            system->reset();
            menuReset = false;
        }

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

        if(menuShowTopView)
        {
            d_top.Activate();
            mapDrawer->DrawTrajectoryTopView(style);
        }

        pangolin::FinishFrame();

        // placecell window: create it on demand, (re)render through the visualizer at most
        // visualize_max_hz, upload what changed, draw the three textures, and return to
        // the main context. Closing the window from its title bar unticks the menu toggle.
        const bool wantPlaceCellWindow = menuPlaceCellWindow && *menuPlaceCellWindow && !placeCellWindowFailed;
        if(wantPlaceCellWindow)
        {
            try
            {
                const bool justOpened = !placeCellWindowOpen;
                if(justOpened)
                {
                    pangolin::CreateWindowAndBind(placeCellWindowTitle, pcW, pcH);
                    placeCellWindowOpen = true;
                    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                    d_pc_kernel = &pangolin::Display("placecell_kernel");
                    d_pc_information = &pangolin::Display("placecell_information");
                    d_pc_alive = &pangolin::Display("placecell_alive");
                    texPcKernel = std::make_unique<pangolin::GlTexture>();
                    texPcInformation = std::make_unique<pangolin::GlTexture>();
                    texPcAlive = std::make_unique<pangolin::GlTexture>();
                    if(!placeCellViz)
                        placeCellViz = std::make_unique<placecell::viz::Visualizer>(*placeCell, placeCellVizOptions);
                }
                else
                    pangolin::BindToContext(placeCellWindowTitle);

                if(pangolin::ShouldQuit())
                {
                    // The user closed the placecell window
                    closePlaceCellWindow();
                    *menuPlaceCellWindow = false;
                }
                else
                {
                    glClearColor(vslamlab_colors::kDarkBg[0], vslamlab_colors::kDarkBg[1], vslamlab_colors::kDarkBg[2], 1.0f);
                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                    const auto now = std::chrono::steady_clock::now();
                    const bool due = justOpened || placeCellLastUpdate == std::chrono::steady_clock::time_point::min()
                                     || std::chrono::duration<double>(now - placeCellLastUpdate).count() >= placeCellMinPeriod;
                    if(due)
                    {
                        placeCellLastUpdate = now;
                        // A fresh window has no textures yet: force a render even if nothing changed
                        if(placeCellViz->update(justOpened) || justOpened)
                        {
                            uploadPlaceCellPanel(*d_pc_kernel, *texPcKernel, texPcKernelW, texPcKernelH,
                                                 placeCellViz->kernel_image(), pcMargin, pcMargin);
                            const int plotsLeft = pcMargin + texPcKernelW + pcMargin;
                            uploadPlaceCellPanel(*d_pc_alive, *texPcAlive, texPcAliveW, texPcAliveH,
                                                 placeCellViz->alive_image(), pcMargin, plotsLeft);
                            uploadPlaceCellPanel(*d_pc_information, *texPcInformation, texPcInformationW, texPcInformationH,
                                                 placeCellViz->information_image(), pcMargin + texPcAliveH, plotsLeft);
                        }
                    }

                    glColor3f(1.0, 1.0, 1.0);
                    if(texPcKernelW > 0)      { d_pc_kernel->Activate();      texPcKernel->RenderToViewportFlipY(); }
                    if(texPcInformationW > 0) { d_pc_information->Activate(); texPcInformation->RenderToViewportFlipY(); }
                    if(texPcAliveW > 0)       { d_pc_alive->Activate();       texPcAlive->RenderToViewportFlipY(); }
                    pangolin::FinishFrame();
                    pangolin::BindToContext(windowTitle);
                }
            }
            catch(const std::exception& e)
            {
                AF_WARN("[Viewer] placecell window disabled: " << e.what());
                placeCellWindowFailed = true;
                closePlaceCellWindow();
                placeCellViz.reset();
            }
        }
        else if(placeCellWindowOpen)
            closePlaceCellWindow();

        if(Stop())
        {
            while(is_stopped())
            {
                usleep(3000);
            }
        }

        if(CheckFinish())
            break;
    }

    // System::Shutdown binds back to the main window's context afterwards
    closePlaceCellWindow();

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

void Viewer::request_stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(!mbStopped)
        mbStopRequested = true;
}

bool Viewer::is_stopped()
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

void Viewer::release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
}

}
