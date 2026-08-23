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

// Defined here (not per-executable) so every binary linking the library gets
// it; entry points that want collapse dumps assign it from their exp_folder.
std::string FrameDrawer::exp_folder{};

FrameDrawer::FrameDrawer(shared_ptr<Map> pMap, const vector<FeatureType>& featureTypes):
    mpMap(pMap), featureTypes(featureTypes)
{
    mState=TrackingState::SYSTEM_NOT_READY;
    mIm = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
}

cv::Mat FrameDrawer::DrawFrame()
{
    cv::Mat im;
    cv::Mat mask;
    std::map<FeatureType, vector<cv::KeyPoint>> vIniKeys; // Initialization: KeyPoints in reference frame
    std::map<FeatureType, vector<int>> vMatches; // Initialization: correspondeces with reference keypoints
    std::map<FeatureType, vector<cv::KeyPoint>> vCurrentKeys; // KeyPoints in current frame
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
        }
        else if(mState==TrackingState::LOST)
        {
            vCurrentKeys = mvCurrentKeys;
        }
    } // destroy scoped mutex -> release mutex

    if(im.channels()<3) //this should be always true
        cvtColor(im,im,cv::COLOR_GRAY2BGR);

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

        for (auto const& [featType, N_] : N) {
            cv::Scalar color = getFeatureColor(featType, 0);
            for(unsigned int i=0; i<vMatches[featType].size(); i++)
            {
                if(vMatches[featType][i]>=0)
                {
                    cv::line(im,vIniKeys[featType][i].pt,vCurrentKeys[featType][vMatches[featType][i]].pt,
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
        for (auto const& [featType, N_] : N) {
            const int n = vCurrentKeys[featType].size();
            cv::Scalar color = getFeatureColor(featType, 0);
            for(int i=0;i<n;i++)
            {
                if(vbVO.at(featType)[i] || vbMap.at(featType)[i])
                {
                    cv::Point2f pt1,pt2;
                    pt1.x=vCurrentKeys[featType][i].pt.x-r;
                    pt1.y=vCurrentKeys[featType][i].pt.y-r;
                    pt2.x=vCurrentKeys[featType][i].pt.x+r;
                    pt2.y=vCurrentKeys[featType][i].pt.y+r;

                    // This is a match to a MapPoint in the map
                    if(vbMap[featType][i])
                    {
                        cv::rectangle(im,pt1,pt2,color);
                        cv::circle(im,vCurrentKeys[featType][i].pt,2,color,-1);
                        mnTracked++;
                    }
                    else // This is match to a "visual odometry" MapPoint created in the last frame
                    {
                        cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,0));
                        cv::circle(im,vCurrentKeys[featType][i].pt,2,cv::Scalar(255,0,0),-1);
                        mnTrackedVO++;
                    }
                }
            }
        }
    }

    cv::Mat imWithInfo;
    DrawTextInfo(im,state, imWithInfo);

    // int compression = 9; // PNG compression 0..9
    // std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, compression};
    // static string lastSavedImage = "";
    // if ((imName!="") && (imName != lastSavedImage)) {
    //     std::cout << "Saving frame: " << imName << std::endl;
    //     std::filesystem::path full = std::filesystem::path(AF_VSLAM::FrameDrawer::exp_folder) / imName;
    //     std::string path = full.string();
    //     cv::Mat out = im;
    //     cv::cvtColor(out, out, cv::COLOR_RGB2BGR);
    //     bool ok = cv::imwrite(path, out, params);
    //     lastSavedImage = imName;
    // }

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
        s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
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
    pTracker->gray_image_.copyTo(mIm);
    pTracker->mask_image_.copyTo(mMask);
    imName = pTracker->image_name_;
    mvCurrentKeys = pTracker->current_frame_.mvKeys;
    for (auto const& [featType, mvKeys] : mvCurrentKeys) {
        N[featType] = mvKeys.size();
        mvbVO[featType] = vector<bool>(N[featType],false);
        mvbMap[featType] = vector<bool>(N[featType],false);
    }


    if(pTracker->last_processed_state_==TrackingState::NOT_INITIALIZED)
    {
        mvIniKeys = pTracker->initial_frame_.mvKeys;
        matches_per_feature_ = pTracker->matches_per_feature_;
    }
    else if(pTracker->last_processed_state_==TrackingState::OK)
    {
        for (auto const& [featType, N_] : N) {
            for(int i = 0; i < N_; i++)
            {
                Pt pMP = pTracker->current_frame_.pts.at(featType)[i];
                if(pMP)
                {
                    if(!pTracker->current_frame_.outliers.at(featType)[i])
                    {
                        if(pMP->number_of_observations() > 0)
                            mvbMap[featType][i] = true;
                        else
                            mvbVO[featType][i] = true;
                    }
                }
            }
        }
    }
    mState = pTracker->last_processed_state_;
}

} //namespace ORB_SLAM
