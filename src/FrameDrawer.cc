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

#include "FrameDrawer.h"
#include "Tracking.h"
#include "Utils.h"
#include <filesystem>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include<mutex>

namespace AF_VSLAM
{

FrameDrawer::FrameDrawer(shared_ptr<Map> pMap, const vector<FeatureType>& feature_types):
    mpMap(pMap), feature_types(feature_types)
{
    mState=TrackingState::SYSTEM_NOT_READY;
    // mIm stays empty until update() copies the first frame in: no placeholder with a made-up size
    // (a 640x480 one once sized the viewer's texture wrongly, issue #7)
}

// The ellipse of a 2x2 covariance at k sigmas; k = sqrt(5.991) is the 95% contour, the inlier gate of pose
// optimisation. Semi-axes are floored at 3 px so sub-pixel covariances stay visible around the keypoint dot.
// Drawn on the raw (distorted) keypoint with the undistorted keypoint's covariance.
static cv::RotatedRect covariance_ellipse(const cv::Point2f& center, const mat2f& sigma2, float k)
{
    constexpr float min_semi_axis = 3.0f;
    const float a = sigma2(0,0), b = sigma2(0,1), c = sigma2(1,1);
    const float mean = 0.5f * (a + c);
    const float radius = std::sqrt(0.25f * (a - c) * (a - c) + b * b);
    const float major = std::max(k * std::sqrt(std::max(mean + radius, 0.0f)), min_semi_axis);
    const float minor = std::max(k * std::sqrt(std::max(mean - radius, 0.0f)), min_semi_axis);
    const float angle_deg = 0.5f * std::atan2(2.0f * b, a - c) * 180.0f / float(CV_PI);
    return {center, cv::Size2f(2.0f * major, 2.0f * minor), angle_deg};
}

cv::Mat FrameDrawer::DrawFrame(float ellipse_scale)
{
    cv::Mat im;
    cv::Mat mask;
    std::map<FeatureType, vector<cv::KeyPoint>> vIniKeys; // Initialization: KeyPoints in reference frame
    std::map<FeatureType, vector<int>> vMatches; // Initialization: correspondeces with reference keypoints
    std::map<FeatureType, vector<cv::KeyPoint>> vCurrentKeys; // KeyPoints in current frame
    std::map<FeatureType, vector<mat2f>> vCurrentSigma2; // their position covariances
    std::map<FeatureType, vector<bool>> vbVO, vbMap; // Tracked MapPoints in current frame
    TrackingState state; // Tracking state

    //Copy variables within scoped mutex
    {
        unique_lock<mutex> lock(mMutex);
        state=mState;
        if(mState==TrackingState::SYSTEM_NOT_READY)
            mState=TrackingState::NO_IMAGES_YET;

        mIm.copyTo(im);
        mMask.copyTo(mask);

        if(mState==TrackingState::NOT_INITIALIZED)
        {
            vCurrentKeys = mvCurrentKeys;
            vIniKeys = mvIniKeys;
            for (const auto& [ft, matches] : matches_per_feature_) {
                vMatches[ft] = matches;
            }
        }
        else if(mState==TrackingState::OK)
        {
            vCurrentKeys = mvCurrentKeys;
            vbVO = mvbVO;
            vbMap = mvbMap;
            vCurrentSigma2 = mvCurrentSigma2;
        }
        else if(mState==TrackingState::LOST)
        {
            vCurrentKeys = mvCurrentKeys;
        }
    } // destroy scoped mutex -> release mutex

    // No frame yet: nothing to draw on. The viewer skips an empty image.
    if(im.empty())
        return {};

    // The viewer uploads this buffer as GL_RGB. The loaded image is BGR (cv::imread order,
    // same assumption as MapDrawer's point colors) or gray.
    if(im.depth() != CV_8U)
        cv::normalize(im, im, 0, 255, cv::NORM_MINMAX, CV_8U);
    if(im.channels() == 1)
        cv::cvtColor(im, im, cv::COLOR_GRAY2RGB);
    else if(im.channels() == 4)
        cv::cvtColor(im, im, cv::COLOR_BGRA2RGB);
    else
        cv::cvtColor(im, im, cv::COLOR_BGR2RGB);

    // Tint the dynamic (mask == 0) region red, same style as the export self-check
    // overlay. The viewer uploads this buffer as GL_RGB, so red is channel 0.
    if(!mask.empty() && mask.size() == im.size())
    {
        cv::Mat tinted;
        cv::addWeighted(im, 0.4, cv::Mat(im.size(), im.type(), cv::Scalar(153,0,0)), 1.0, 0.0, tinted);
        tinted.copyTo(im, mask == 0);
    }

    //Draw
    if(state==TrackingState::NOT_INITIALIZED) //INITIALIZING
    {

        for (auto const& [feature_type, N_] : N) {
            cv::Scalar color = getFeatureColor(feature_type, 0);
            const auto& ini = vIniKeys[feature_type];
            const auto& cur = vCurrentKeys[feature_type];
            for(unsigned int i=0; i<vMatches[feature_type].size(); i++)
            {
                // The matches must index this snapshot's two keypoint sets; skip any that do not
                if(vMatches[feature_type][i]>=0 && i < ini.size() && size_t(vMatches[feature_type][i]) < cur.size())
                {
                    cv::line(im,vIniKeys[feature_type][i].pt,vCurrentKeys[feature_type][vMatches[feature_type][i]].pt,
                            cv::Scalar(color));
                }
            }
        }
    }
    else if(state==TrackingState::OK) //TRACKING
    {
        mnTracked=0;
        mnTrackedVO=0;
        const float r = 5;
        for (auto const& [feature_type, N_] : N) {
            const int n = vCurrentKeys[feature_type].size();
            cv::Scalar color = getFeatureColor(feature_type, 0);
            // Ellipses only when every keypoint has a covariance; otherwise this feature type falls back to boxes
            const vector<mat2f>* sigma2 = nullptr;
            const auto it = vCurrentSigma2.find(feature_type);
            if(it != vCurrentSigma2.end() && int(it->second.size()) == n)
                sigma2 = &it->second;
            for(int i=0;i<n;i++)
            {
                if(vbVO.at(feature_type)[i] || vbMap.at(feature_type)[i])
                {
                    const cv::Point2f& pt = vCurrentKeys[feature_type][i].pt;

                    // Map point match in the feature colour, visual-odometry match (last frame's point) in blue
                    const bool in_map = vbMap[feature_type][i];
                    const cv::Scalar c = in_map ? color : cv::Scalar(255,0,0);
                    if(sigma2)
                    {
                        if(ellipse_scale > 0.0f)
                            cv::ellipse(im, covariance_ellipse(pt, (*sigma2)[i], ellipse_scale), c, 1, cv::LINE_AA);
                    }
                    else
                        cv::rectangle(im, cv::Point2f(pt.x-r, pt.y-r), cv::Point2f(pt.x+r, pt.y+r), c);
                    cv::circle(im,pt,2,c,-1);
                    if(in_map)
                        mnTracked++;
                    else
                        mnTrackedVO++;
                }
            }
        }
    }

    cv::Mat imWithInfo;
    DrawTextInfo(im,state, imWithInfo);

    return imWithInfo;
}


void FrameDrawer::DrawTextInfo(cv::Mat &im, TrackingState nState, cv::Mat &imText)
{
    stringstream s;
    if(nState==TrackingState::NO_IMAGES_YET)
        s << " WAITING FOR IMAGES";
    else if(nState==TrackingState::NOT_INITIALIZED)
        s << " TRYING TO INITIALIZE ";
    else if(nState==TrackingState::OK)
    {
        s << "SLAM MODE |  ";

        int nKFs = mpMap->keyframes_in_map();
        int nMPs = mpMap->map_points_in_map();
        s << "KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;
        if(mnTrackedVO>0)
            s << ", + VO matches: " << mnTrackedVO;
    }
    else if(nState==TrackingState::LOST)
    {
        s << " TRACK LOST. TRYING TO RELOCALIZE ";
    }
    else if(nState==TrackingState::SYSTEM_NOT_READY)
    {
        s << " INITIALIZING SYSTEM. PLEASE WAIT...";
    }

    int baseline=0;
    double fontScale = 1.5;
    int thickness = 2.0;
    cv::Size textSize = cv::getTextSize(s.str(),cv::FONT_HERSHEY_PLAIN,fontScale,thickness,&baseline);
    imText = cv::Mat(im.rows+textSize.height+10,im.cols,im.type());
    im.copyTo(imText.rowRange(0,im.rows).colRange(0,im.cols));
    imText.rowRange(im.rows,imText.rows) = cv::Mat::zeros(textSize.height+10,im.cols,im.type());
    cv::putText(imText,s.str(),cv::Point(5,imText.rows-5),cv::FONT_HERSHEY_PLAIN,fontScale,cv::Scalar(255, 255, 255),thickness,8);

}

void FrameDrawer::update(Tracking *pTracker)
{
    unique_lock<mutex> lock(mMutex);
    pTracker->color_image_.copyTo(mIm);
    pTracker->mask_image_.copyTo(mMask);
    mvCurrentKeys = pTracker->current_frame_.raw_keypoints;
    mvCurrentSigma2.clear();
    for (const auto& [feature_type, data] : pTracker->current_frame_.features.per_type)
        mvCurrentSigma2[feature_type] = data.sigma2;
    for (auto const& [feature_type, raw_keypoints] : mvCurrentKeys) {
        N[feature_type] = raw_keypoints.size();
        mvbVO[feature_type] = vector<bool>(N[feature_type],false);
        mvbMap[feature_type] = vector<bool>(N[feature_type],false);
    }


    if(pTracker->last_processed_state_==TrackingState::NOT_INITIALIZED)
    {
        mvIniKeys = pTracker->initial_frame_.raw_keypoints;
        matches_per_feature_ = pTracker->matches_per_feature_;
    }
    else if(pTracker->last_processed_state_==TrackingState::OK)
    {
        for (auto const& [feature_type, N_] : N) {
            for(int i = 0; i < N_; i++)
            {
                Pt pMP = pTracker->current_frame_.pts.at(feature_type)[i];
                if(pMP)
                {
                    if(!pTracker->current_frame_.outliers.at(feature_type)[i])
                    {
                        if(pMP->number_of_observations() > 0)
                            mvbMap[feature_type][i] = true;
                        else
                            mvbVO[feature_type][i] = true;
                    }
                }
            }
        }
    }
    mState = pTracker->last_processed_state_;
}

} //namespace ORB_SLAM
