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

#ifndef MAPDRAWER_H
#define MAPDRAWER_H

#include"Map.h"
#include"MapPoint.h"
#include"KeyFrame.h"
#include<pangolin/pangolin.h>

#include<array>
#include<chrono>
#include<cstdint>
#include<map>
#include<memory>
#include<mutex>
#include<vector>

namespace AF_VSLAM
{

// VSLAM-LAB brand palette (sampled from the project logo), RGB in [0,1].
namespace vslamlab_colors
{
    constexpr float kCyan[3]       = {0.710f, 0.953f, 0.976f};
    constexpr float kSky[3]        = {0.659f, 0.863f, 0.980f};
    constexpr float kPeriwinkle[3] = {0.506f, 0.584f, 0.984f};
    constexpr float kLavender[3]   = {0.647f, 0.616f, 0.875f};

    // Saturated variants of the same hues — for the trajectory and camera, which need to
    // stand out against the pastel points and muted wireframes.
    constexpr float kCyanVivid[3]       = {0.20f, 0.85f, 0.92f};
    constexpr float kSkyVivid[3]        = {0.28f, 0.62f, 0.95f};
    constexpr float kPeriwinkleVivid[3] = {0.38f, 0.45f, 0.98f};
    constexpr float kLavenderVivid[3]   = {0.62f, 0.44f, 0.92f};

    constexpr float kDarkBg[3]     = {0.043f, 0.051f, 0.078f};  // deep navy
    constexpr float kLightBg[3]    = {0.985f, 0.985f, 0.995f};

    // Interpolate the 4-stop logo gradient (cyan -> sky -> periwinkle -> lavender), t in [0,1].
    void gradient(float t, float rgb[3]);
    // Same gradient over the vivid stops.
    void gradientVivid(float t, float rgb[3]);
}

// How DrawMapPoints colors each map point. Integer-valued so the Viewer can expose it as a
// pangolin::Var<int> selector ("Point Color") and new modes can be appended without a UI change.
enum class PointColorMode : int
{
    Feature = 0,  // the feature type's palette color (getFeatureColor)
    RGB = 1,      // the point's own image color, sampled at creation (MapPoint::color)
};
constexpr int kPointColorModeCount = 2;

// Settings-yaml spelling ("feature" | "rgb") <-> enum; unknown names return false and leave `mode` untouched.
bool parsePointColorMode(const std::string& name, PointColorMode& mode);

// All appearance knobs the Viewer UI can adjust at runtime, passed to every draw call.
struct ViewerStyle
{
    bool darkTheme{true};
    PointColorMode pointColorMode{PointColorMode::Feature};
    float pointSize{2.0f};
    // Linear depth fog on the map points, towards the theme background. The range is anchored
    // at the SLAM camera and scaled by the local scene depth D (median point distance to that
    // camera, measured at snapshot time): fully clear up to fogStart*D beyond the camera,
    // fully faded from fogEnd*D. Viewer checkbox "Depth Fog"; Viewer.DepthFog/FogStart/FogEnd.
    bool depthFog{true};
    float fogStart{1.5f};
    float fogEnd{6.0f};
    float keyFrameSize{0.05f};
    float keyFrameLineWidth{1.0f};
    float graphLineWidth{0.9f};
    float trajectoryLineWidth{3.0f};
    float cameraSize{0.08f};
    float cameraLineWidth{3.0f};
};

class MapDrawer
{
public:
    MapDrawer(shared_ptr<Map> pMap, const string &strSettingPath, const vector<FeatureType>& featureTypes);

    shared_ptr<Map> mpMap;

    void DrawMapPoints(const ViewerStyle& style);
    void DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph, const ViewerStyle& style);
    // Polyline linking sequential (keyId-ordered) keyframes, plus loop-closure markers.
    void DrawTrajectory(const ViewerStyle& style);
    // Minimap for an overlay view: the full trajectory projected onto the ground plane
    // (x-z, matching the aerial view's orientation), auto-fitted to the currently active
    // viewport with its own orthographic projection. Draws nothing before 2 keyframes.
    void DrawTrajectoryTopView(const ViewerStyle& style);
    void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc, const ViewerStyle& style);
    void set_current_camera_pose(const mat4f &Tcw_);
    void SetReferenceKeyFrame(Keyframe pKF);
    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M);
    void SetCurrentOpenGLCameraMatrix(const mat3f& Rwc,const vec3f& twc, pangolin::OpenGlMatrix &M);
    void AddLoopClosureKeyframe(const mat4f &Tcw_);

    // Defaults loaded from the settings yaml, used to seed the Viewer UI controls.
    ViewerStyle GetDefaultStyle() const { return mDefaultStyle; }

    // Free the GL objects owned by the drawer (the map-point vertex buffer). Must run on the
    // viewer thread while its GL context is still current: the Viewer calls it at the end of
    // Run(), before the window goes away; the destructor then has nothing GL-side to do.
    void ReleaseGL();

private:

    ViewerStyle mDefaultStyle{};

    // --- Map-point snapshot -------------------------------------------------------------
    // DrawMapPoints used to walk the live map every viewer frame, taking every MapPoint's
    // position mutex and copying the map's point set under the map mutex — a lock storm that
    // competes with Tracking and LocalMapping. Instead, the points are snapshotted into a
    // vertex buffer at most `pointRefreshPeriod_` apart (Viewer.MapPointsRefreshHz) and every
    // frame just draws the buffer. Both color sets are stored so the "Point Color" selector
    // switches instantly without a refresh.
    struct PointVertex
    {
        float x, y, z;
        std::uint8_t feat[4];   // feature-type palette color, RGBA
        std::uint8_t rgb[4];    // image color at creation (MapPoint::color, BGR -> RGB), RGBA
    };
    static_assert(sizeof(PointVertex) == 20, "PointVertex must stay tightly packed for the GL strides");

    std::vector<PointVertex> pointSnapshot_{};
    std::unique_ptr<pangolin::GlBufferData> pointBuffer_{};
    std::size_t pointBufferCount_{0};       // vertices currently in the GL buffer
    std::size_t pointBufferCapacity_{0};    // vertices the GL buffer can hold before a Reinitialise
    std::chrono::steady_clock::duration pointRefreshPeriod_{std::chrono::milliseconds(200)};
    std::chrono::steady_clock::time_point lastPointRefresh_{};
    bool pointSnapshotEverTaken_{false};

    // Fog anchor, measured at snapshot time: the SLAM camera center (world) and the median
    // distance of the points to it (local scene depth). sceneDepth 0 = no fog this snapshot.
    vec3f pointSceneCenter_{vec3f::Zero()};
    float pointSceneDepth_{0.0f};

    // Rebuild pointSnapshot_ from the live map (the only place the map is locked for drawing
    // points) and upload it to pointBuffer_. Viewer thread only.
    void RefreshPointSnapshot();

    std::mutex mMutexCamera;
    mat4f mCameraPose{mat4f::Zero()};
    std::mutex mutexLoopClosures;
    std::vector<vec3f>loopClosures{};

    vector<FeatureType> featureTypes{};

    // Feature color lookup cached once at construction: getFeatureColor() heap-allocates a
    // Feature instance per call, far too expensive to call per point in RefreshPointSnapshot.
    std::map<FeatureType, std::array<float,3>> featureColors{};
};

} //namespace ORB_SLAM

#endif // MAPDRAWER_H
