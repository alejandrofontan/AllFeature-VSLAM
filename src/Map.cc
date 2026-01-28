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

#include "Map.h"

#include<mutex>

namespace ANYFEATURE_VSLAM
{

Map::Map():mnMaxKFid(0),mnBigChangeIdx(0)
{
}

void Map::AddKeyFrame(Keyframe pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.insert(pKF);
    if(pKF->keyId > mnMaxKFid)
        mnMaxKFid = pKF->keyId;
}

void Map::AddMapPoint(Pt pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}

void Map::EraseMapPoint(Pt pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void Map::EraseKeyFrame(Keyframe pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void Map::SetReferenceMapPoints(const vector<Pt > &vpMPs)
{
    unique_lock<mutex> lock(mMutexMap);
    mvpReferenceMapPoints = vpMPs;
}

void Map::InformNewBigChange()
{
    unique_lock<mutex> lock(mMutexMap);
    mnBigChangeIdx++;
}

int Map::GetLastBigChangeIdx()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnBigChangeIdx;
}

vector<Keyframe> Map::GetAllKeyFrames()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<Keyframe>(mspKeyFrames.begin(),mspKeyFrames.end());
}

vector<Pt> Map::GetAllMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<Pt>(mspMapPoints.begin(),mspMapPoints.end());
}

size_t Map::MapPointsInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspMapPoints.size();
}

size_t Map::KeyFramesInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}

vector<Pt> Map::GetReferenceMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return mvpReferenceMapPoints;
}

long unsigned int Map::GetMaxKFid()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnMaxKFid;
}

void Map::clear()
{
    mspMapPoints.clear();
    mspKeyFrames.clear();
    mnMaxKFid = 0;
    mvpReferenceMapPoints.clear();
    mvpKeyFrameOrigins.clear();
}

float Map::NormalizeMap()
{
    //unique_lock<mutex> lock(mMutexMap);
    std::vector<float> depths;
    depths.reserve(mspKeyFrames.size());
    for(const auto& keyframe: mspKeyFrames)
    {
        float medianDepth = keyframe->ComputeSceneMedianDepth(2);
        depths.push_back(medianDepth);
    }
    float medianDepth = 0.0f;
    if(!depths.empty())
    {
        std::sort(depths.begin(), depths.end());
        medianDepth = depths[0.5f * (depths.size() - 1)];
    }       

    for(const auto& keyframe: mspKeyFrames)
    {
        mat4f Tc2w = keyframe->GetPose();
        Tc2w.block<3,1>(0,3) /= medianDepth;
        keyframe->SetPose(Tc2w);

    }

    std::vector<Pt> mspMapPoints_ = GetAllMapPoints();
    for(size_t iMP=0; iMP<mspMapPoints_.size(); iMP++)
    {   
        Pt pMP = mspMapPoints_[iMP];
        if(pMP)
        {
            pMP->SetWorldPos(pMP->GetWorldPos()/medianDepth);
        }
    }

    std::cout << "Normalizing map with median depth = " << medianDepth << std::endl;
    return medianDepth;
} 
}
//namespace ORB_SLAM
