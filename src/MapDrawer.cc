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

#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "Converter.h"
#include "Utils.h"

#include <pangolin/pangolin.h>
#include <algorithm>
#include <mutex>

namespace AF_VSLAM
{

namespace vslamlab_colors
{
namespace
{
void gradientOver(const float* stops[4], float t, float rgb[3])
{
    t = std::clamp(t, 0.0f, 1.0f);
    const float scaled = t * 3.0f;
    const int i = std::min(2, int(scaled));
    const float f = scaled - float(i);
    for (int c = 0; c < 3; c++)
        rgb[c] = (1.0f - f) * stops[i][c] + f * stops[i+1][c];
}
}

void gradient(float t, float rgb[3])
{
    const float* stops[4] = {kCyan, kSky, kPeriwinkle, kLavender};
    gradientOver(stops, t, rgb);
}

void gradientVivid(float t, float rgb[3])
{
    const float* stops[4] = {kCyanVivid, kSkyVivid, kPeriwinkleVivid, kLavenderVivid};
    gradientOver(stops, t, rgb);
}
}

namespace
{
// Wide GL_LINE_SMOOTH lines render broken (clamped or stippled) on common drivers even when
// the reported smooth-width range claims support, so smooth only 1px lines and draw anything
// wider aliased — that keeps the thickness controls honest everywhere.
void applyLineWidth(const float width)
{
    if (width > 1.001f)
        glDisable(GL_LINE_SMOOTH);
    else
        glEnable(GL_LINE_SMOOTH);
    glLineWidth(width);
}
}

MapDrawer::MapDrawer(shared_ptr<Map> pMap, const string &strSettingPath, const vector<FeatureType>& featureTypes):
    mpMap(pMap),featureTypes(featureTypes)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    if (!fSettings["Viewer.KeyFrameSize"].empty())
        mDefaultStyle.keyFrameSize = fSettings["Viewer.KeyFrameSize"];
    if (!fSettings["Viewer.KeyFrameLineWidth"].empty())
        mDefaultStyle.keyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];
    if (!fSettings["Viewer.GraphLineWidth"].empty())
        mDefaultStyle.graphLineWidth = fSettings["Viewer.GraphLineWidth"];
    if (!fSettings["Viewer.PointSize"].empty())
        mDefaultStyle.pointSize = fSettings["Viewer.PointSize"];
    if (!fSettings["Viewer.CameraSize"].empty())
        mDefaultStyle.cameraSize = fSettings["Viewer.CameraSize"];
    if (!fSettings["Viewer.CameraLineWidth"].empty())
        mDefaultStyle.cameraLineWidth = fSettings["Viewer.CameraLineWidth"];
    if (!fSettings["Viewer.TrajectoryLineWidth"].empty())
        mDefaultStyle.trajectoryLineWidth = fSettings["Viewer.TrajectoryLineWidth"];

    for (size_t i = 0; i < featureTypes.size(); i++)
    {
        const cv::Scalar c = getFeatureColor(featureTypes[i], 0, true);
        featureColors[featureTypes[i]] = {float(c[0]), float(c[1]), float(c[2])};
    }
}

const std::array<float,3>& MapDrawer::pointColor(const FeatureType ft) const
{
    const auto it = featureColors.find(ft);
    if (it != featureColors.end())
        return it->second;
    static const std::array<float,3> fallback{0.5f, 0.5f, 0.5f};
    return fallback;
}

void MapDrawer::DrawMapPoints(const ViewerStyle& style)
{
    const vector<Pt> &vpMPs = mpMap->get_all_map_points();

    if(vpMPs.empty())
        return;

    glPointSize(style.pointSize);
    glBegin(GL_POINTS);

    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
    {
        if(vpMPs[i]->is_bad())
             continue;

        const std::array<float,3>& c = pointColor(vpMPs[i]->featureType);
        vec3f pos = vpMPs[i]->get_world_pos();
        glColor4f(c[0], c[1], c[2], 0.9f);
        glVertex3f(pos(0),pos(1),pos(2));
    }
    glEnd();
}

void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph, const ViewerStyle& style)
{
    const float &w = style.keyFrameSize;
    const float h = w*0.75f;
    const float z = w*0.6f;

    const vector<Keyframe> vpKFs = mpMap->GetAllKeyFrames();

    if(bDrawKF)
    {
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            Keyframe pKF = vpKFs[i];
            mat4f Twc_tmp = pKF->get_pose_inverse().transpose();
            cv::Mat Twc = Converter::toCvMat(Twc_tmp);

            glPushMatrix();

            glMultMatrixf(Twc.ptr<GLfloat>(0));

            applyLineWidth(style.keyFrameLineWidth);
            if (style.darkTheme)
                glColor4f(0.42f, 0.42f, 0.56f, 0.55f);   // muted lavender-grey
            else
                glColor4f(0.55f, 0.55f, 0.66f, 0.65f);   // muted slate
            glBegin(GL_LINES);
            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            glEnd();

            glPopMatrix();
        }
    }

    if(bDrawGraph)
    {
        applyLineWidth(style.graphLineWidth);
        using namespace vslamlab_colors;
        glColor4f(kSky[0], kSky[1], kSky[2], style.darkTheme ? 0.25f : 0.4f);
        glBegin(GL_LINES);

        for(size_t i=0; i<vpKFs.size(); i++)
        {
            // Covisibility Graph
            const vector<Keyframe > vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
            vec3f Ow = vpKFs[i]->get_camera_center();
            if(!vCovKFs.empty())
            {
                for(vector<Keyframe >::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
                {
                    if((*vit)->keyId < vpKFs[i]->keyId)
                        continue;
                    vec3f Ow2 = (*vit)->get_camera_center();
                    glVertex3f(Ow(0),Ow(1),Ow(2));
                    glVertex3f(Ow2(0),Ow2(1),Ow2(2));
                }
            }

            // Spanning tree
            Keyframe pParent = vpKFs[i]->GetParent();
            if(pParent)
            {
                vec3f Owp = pParent->get_camera_center();
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owp(0),Owp(1),Owp(2));
            }

            // Loops
            set<Keyframe > sLoopKFs = vpKFs[i]->GetLoopEdges();
            for(set<Keyframe>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
            {
                if((*sit)->keyId < vpKFs[i]->keyId)
                    continue;
                vec3f Owl = (*sit)->get_camera_center();
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owl(0),Owl(1),Owl(2));
            }
        }

        glEnd();
    }
}

void MapDrawer::DrawTrajectory(const ViewerStyle& style)
{
    vector<Keyframe> vpKFs = mpMap->GetAllKeyFrames();
    vpKFs.erase(std::remove_if(vpKFs.begin(), vpKFs.end(),
                               [](const Keyframe& kf){ return kf->is_bad(); }),
                vpKFs.end());
    if (vpKFs.size() < 2)
        return;
    std::sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

    // Sequential keyframe polyline, tinted with the logo gradient along its length.
    applyLineWidth(style.trajectoryLineWidth);
    glBegin(GL_LINE_STRIP);
    const float denom = float(vpKFs.size() - 1);
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        float rgb[3];
        vslamlab_colors::gradientVivid(float(i) / denom, rgb);
        glColor4f(rgb[0], rgb[1], rgb[2], 1.0f);
        const vec3f Ow = vpKFs[i]->get_camera_center();
        glVertex3f(Ow(0), Ow(1), Ow(2));
    }
    glEnd();

    // Loop-closure markers.
    {
        unique_lock<mutex> lock(mutexLoopClosures);
        if (!loopClosures.empty())
        {
            using namespace vslamlab_colors;
            glPointSize(std::max(6.0f, 3.0f * style.trajectoryLineWidth));
            glColor4f(kLavenderVivid[0], kLavenderVivid[1], kLavenderVivid[2], 1.0f);
            glBegin(GL_POINTS);
            for (const auto& twc : loopClosures)
                glVertex3f(twc(0), twc(1), twc(2));
            glEnd();
        }
    }
}

void MapDrawer::DrawTrajectoryTopView(const ViewerStyle& style)
{
    vector<Keyframe> vpKFs = mpMap->GetAllKeyFrames();
    vpKFs.erase(std::remove_if(vpKFs.begin(), vpKFs.end(),
                               [](const Keyframe& kf){ return kf->is_bad(); }),
                vpKFs.end());
    if (vpKFs.size() < 2)
        return;
    std::sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

    std::vector<vec3f> centers;
    centers.reserve(vpKFs.size());
    for (const auto& kf : vpKFs)
        centers.push_back(kf->get_camera_center());

    bool hasCam = false;
    vec3f twc = vec3f::Zero();
    {
        unique_lock<mutex> lock(mMutexCamera);
        if (mCameraPose(3,3) == 1.0f)
        {
            twc = -mCameraPose.block<3,3>(0,0).transpose() * mCameraPose.block<3,1>(0,3);
            hasCam = true;
        }
    }

    // Ground-plane (x-z) bounds of the whole trajectory plus the current camera.
    float minX = centers[0](0), maxX = minX, minZ = centers[0](2), maxZ = minZ;
    for (const auto& Ow : centers)
    {
        minX = std::min(minX, Ow(0)); maxX = std::max(maxX, Ow(0));
        minZ = std::min(minZ, Ow(2)); maxZ = std::max(maxZ, Ow(2));
    }
    if (hasCam)
    {
        minX = std::min(minX, twc(0)); maxX = std::max(maxX, twc(0));
        minZ = std::min(minZ, twc(2)); maxZ = std::max(maxZ, twc(2));
    }

    // Aspect-preserving orthographic fit with a margin, so the trajectory never
    // stretches whatever rectangle the overlay viewport has.
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    const float aspect = viewport[3] > 0 ? float(viewport[2]) / float(viewport[3]) : 1.0f;
    const float cx = 0.5f * (minX + maxX);
    const float cz = 0.5f * (minZ + maxZ);
    float halfW = std::max(0.5f * (maxX - minX), 1e-4f) * 1.15f;
    float halfH = std::max(0.5f * (maxZ - minZ), 1e-4f) * 1.15f;
    if (halfW / halfH > aspect)
        halfH = halfW / aspect;
    else
        halfW = halfH * aspect;

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(cx - halfW, cx + halfW, cz - halfH, cz + halfH, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    // The 3D scene behind the overlay already filled the depth buffer this frame.
    const GLboolean depthWasEnabled = glIsEnabled(GL_DEPTH_TEST);
    glDisable(GL_DEPTH_TEST);

    // Background panel + border so the inset reads against the map behind it.
    using namespace vslamlab_colors;
    const float* bg = style.darkTheme ? kDarkBg : kLightBg;
    glColor4f(bg[0], bg[1], bg[2], 0.85f);
    glBegin(GL_QUADS);
    glVertex2f(cx - halfW, cz - halfH);
    glVertex2f(cx + halfW, cz - halfH);
    glVertex2f(cx + halfW, cz + halfH);
    glVertex2f(cx - halfW, cz + halfH);
    glEnd();

    const float bx = 0.995f * halfW;
    const float bz = 0.995f * halfH;
    applyLineWidth(1.0f);
    if (style.darkTheme)
        glColor4f(0.42f, 0.42f, 0.56f, 0.8f);
    else
        glColor4f(0.55f, 0.55f, 0.66f, 0.8f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(cx - bx, cz - bz);
    glVertex2f(cx + bx, cz - bz);
    glVertex2f(cx + bx, cz + bz);
    glVertex2f(cx - bx, cz + bz);
    glEnd();

    // Full-trajectory polyline, same gradient as the 3D trajectory.
    applyLineWidth(std::max(1.5f, 0.6f * style.trajectoryLineWidth));
    glBegin(GL_LINE_STRIP);
    const float denom = float(centers.size() - 1);
    for (size_t i = 0; i < centers.size(); i++)
    {
        float rgb[3];
        gradientVivid(float(i) / denom, rgb);
        glColor4f(rgb[0], rgb[1], rgb[2], 1.0f);
        glVertex2f(centers[i](0), centers[i](2));
    }
    glEnd();

    // Current camera position marker.
    if (hasCam)
    {
        glPointSize(std::max(5.0f, 2.5f * style.trajectoryLineWidth));
        glColor4f(kCyanVivid[0], kCyanVivid[1], kCyanVivid[2], 1.0f);
        glBegin(GL_POINTS);
        glVertex2f(twc(0), twc(2));
        glEnd();
    }

    if (depthWasEnabled)
        glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc, const ViewerStyle& style)
{
    const float &w = style.cameraSize;
    const float h = w*0.75f;
    const float z = w*0.6f;

    glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif

    applyLineWidth(style.cameraLineWidth);
    using namespace vslamlab_colors;
    glColor4f(kPeriwinkleVivid[0], kPeriwinkleVivid[1], kPeriwinkleVivid[2], 1.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}


void MapDrawer::set_current_camera_pose(const mat4f &Tcw_)
{
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw_;
}

void MapDrawer::AddLoopClosureKeyframe(const mat4f &Tcw_)
{
    unique_lock<mutex> lock(mutexLoopClosures);
    loopClosures.emplace_back(Tcw_.block<3,1>(0,3));
}

void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
{
    mat4f Tcw{};
    {
        unique_lock<mutex> lock(mMutexCamera);
        Tcw = mCameraPose;
    }

    if(Tcw(3,3) == 1.0f)
    {
        const mat3f Rwc = Tcw.block<3,3>(0,0).transpose();
        const vec3f twc = -Rwc * Tcw.block<3,1>(0,3);
        SetCurrentOpenGLCameraMatrix(Rwc,twc, M);
    }
    else
        M.SetIdentity();
}
void MapDrawer::SetCurrentOpenGLCameraMatrix(const mat3f& Rwc,const vec3f& twc, pangolin::OpenGlMatrix &M){
    M.m[0] = Rwc(0,0);
    M.m[1] = Rwc(1,0);
    M.m[2] = Rwc(2,0);
    M.m[3]  = 0.0;

    M.m[4] = Rwc(0,1);
    M.m[5] = Rwc(1,1);
    M.m[6] = Rwc(2,1);
    M.m[7]  = 0.0;

    M.m[8] = Rwc(0,2);
    M.m[9] = Rwc(1,2);
    M.m[10] = Rwc(2,2);
    M.m[11]  = 0.0;

    M.m[12] = twc(0);
    M.m[13] = twc(1);
    M.m[14] = twc(2);
    M.m[15]  = 1.0;
}

} //namespace ORB_SLAM
