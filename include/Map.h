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

#ifndef MAP_H
#define MAP_H

#include "MapPoint.h"
#include "KeyFrame.h"
#include <set>

#include <mutex>



namespace AF_VSLAM
{

class MapPoint;
class KeyFrame;

class Map
{
public:
    Map();

    void add_keyframe(Keyframe pKF);
    void add_map_point(Pt pMP);
    void EraseMapPoint(Pt pMP);
    void EraseKeyFrame(Keyframe pKF);
    void set_reference_map_points(const std::vector<Pt> &vpMPs);
    void InformNewBigChange();
    int GetLastBigChangeIdx();

    std::vector<Keyframe> GetAllKeyFrames();
    std::vector<Pt> get_all_map_points();
    std::vector<Pt> GetReferenceMapPoints();

    size_t map_points_in_map();
    size_t keyframes_in_map();

    long unsigned int GetMaxKFid();

    void clear();
    float NormalizeMap();

    vector<Keyframe> keyframe_origins_;

    std::mutex map_update_mutex_;

    // This avoid that two points are created simultaneously in separate threads (id conflict)
    std::mutex mMutexPointCreation;

protected:
    std::set<Pt> mspMapPoints;
    std::set<Keyframe> mspKeyFrames;

    std::vector<Pt> mvpReferenceMapPoints;

    long unsigned int mnMaxKFid;

    // Index related to a big change in the map (loop closure, global BA)
    int mnBigChangeIdx;

    std::mutex mMutexMap;
};

} //namespace ORB_SLAM

#endif // MAP_H
