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

namespace AF_VSLAM
{

Map::Map():mnMaxKFid(0),mnBigChangeIdx(0)
{
}

void Map::add_keyframe(Keyframe pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.insert(pKF);
    if(pKF->keyId > mnMaxKFid)
        mnMaxKFid = pKF->keyId;
}

void Map::add_map_point(Pt pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}

Pt Map::make_map_point(const vec3f& world_pos, const Keyframe& ref_keyframe, KeypointIndex ref_index,
                       FeatureType feature_type,
                       const Keyframe& proj_keyframe, KeypointIndex proj_index)
{
    auto pt = make_shared<MapPoint>(world_pos, ref_keyframe, this_map(), feature_type,
                                    ref_keyframe->features.at(feature_type).colors.at(ref_index));

    std::vector<std::pair<Keyframe, KeypointIndex>> observations{{ref_keyframe, ref_index}};
    if(proj_keyframe && proj_keyframe != ref_keyframe)   // twice: one observation, two slots written
        observations.emplace_back(proj_keyframe, proj_index);

    pt->add_observations(observations);

    // Claim the slots one at a time; on a loss, release the ones already taken and let pt die
    // unreferenced. It never reached this map, and erase_map_point_match clears a slot only if
    // it still holds pt, so nothing another thread put there is disturbed.
    std::vector<std::pair<Keyframe, KeypointIndex>> published;
    published.reserve(observations.size());
    for(const auto& [keyframe, index] : observations)
    {
        if(!keyframe->add_map_point(pt, index))
        {
            for(const auto& [taken_keyframe, taken_index] : published)
                taken_keyframe->erase_map_point_match(pt, taken_index);
            return nullptr;
        }
        published.emplace_back(keyframe, index);
    }

    add_map_point(pt);
    return pt;
}

Pt Map::create_monocular_map_point(const vec3f& world_pos,
                                   const Keyframe& ref_keyframe, KeypointIndex ref_index,
                                   const Keyframe& proj_keyframe, KeypointIndex proj_index,
                                   FeatureType feature_type)
{
    return make_map_point(world_pos, ref_keyframe, ref_index, feature_type, proj_keyframe, proj_index);
}

Pt Map::create_map_point(const vec3f& world_pos,
                         const Keyframe& ref_keyframe, KeypointIndex ref_index,
                         FeatureType feature_type)
{
    return make_map_point(world_pos, ref_keyframe, ref_index, feature_type, nullptr, 0);
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

void Map::set_reference_map_points(const vector<Pt > &vpMPs)
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

vector<Pt> Map::get_all_map_points()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<Pt>(mspMapPoints.begin(),mspMapPoints.end());
}

size_t Map::map_points_in_map()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspMapPoints.size();
}

size_t Map::keyframes_in_map()
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
    // Under the map lock like every other accessor; the contents are released after it (the last references to
    // keyframes and points may go here, and their destruction must not run under mMutexMap)
    decltype(mspMapPoints) map_points;
    decltype(mspKeyFrames) keyframes;
    decltype(mvpReferenceMapPoints) reference_points;
    decltype(keyframe_origins_) origins;
    {
        unique_lock<mutex> lock(mMutexMap);
        map_points.swap(mspMapPoints);
        keyframes.swap(mspKeyFrames);
        reference_points.swap(mvpReferenceMapPoints);
        origins.swap(keyframe_origins_);
        mspMapPoints.clear();
        mspKeyFrames.clear();
        mnMaxKFid = 0;
        mvpReferenceMapPoints.clear();
        keyframe_origins_.clear();
    }
}
}
//namespace ORB_SLAM
