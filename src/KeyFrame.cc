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

#include "KeyFrame.h"
#include "Converter.h"
#include<mutex>
#include <memory>
#include <limits>

namespace AF_VSLAM
{

static bool KeyframeComparison(const pair<int, Keyframe>& a, const pair<int, Keyframe>& b){
    return (a.first != b.first) ? (a.first < b.first) : (a.second->keyId < b.second->keyId);
}

long unsigned int KeyFrame::next_id=0;

KeyFrame::KeyFrame(Frame &F, shared_ptr<Map> pMap, shared_ptr<PlaceRecognition> place_recognition):
    featureTypes(F.featureTypes), cache_matched_pairs(F.cache_matched_pairs), cache_matched_pairs_feat_type(F.cache_matched_pairs_feat_type),
    frame_id(F.frame_id),  timestamp(F.timestamp), grid_cols(FRAME_GRID_COLS), grid_rows(FRAME_GRID_ROWS),
    grid_element_width_inv(F.grid_element_width_inv), grid_element_height_inv(F.grid_element_height_inv),
    fuse_target_for_keyframe(0), ba_local_for_keyframe(0), ba_fixed_for_keyframe(0), ba_global_for_keyframe(0),
    fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
    mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), raw_keypoints(F.raw_keypoints), keypoints(F.keypoints),
    inv_depth(F.inv_depth), sigma2_inv_depth(F.sigma2_inv_depth),
    keypoint_colors(F.keypoint_colors), image(F.image), global_descriptor(F.global_descriptor), size_tolerance(F.size_tolerance),
    keyPtsSigma2(F.keyPtsSigma2),keyPtsInf(F.keyPtsInf),keyPtsSize(F.keyPtsSize),
    maxKeyPtSize(F.maxKeyPtSize),maxKeyPtSigma(F.maxKeyPtSigma),
    min_x(F.min_x), min_y(F.min_y), max_x(F.max_x), max_y(F.max_y),
    mK(F.mK), map_points_(F.pts), place_recognition_(std::move(place_recognition)),
    first_connection_(true), parent_(NULL), not_erase_(false),
    to_be_erased_(false), bad_(false), map_(pMap)
{
    keyId = next_id++;
    for(auto& [ft, N_] : N){
        grid_[ft].resize(grid_cols);
        for(int i=0; i<grid_cols;i++)
        {
            grid_[ft][i].resize(grid_rows);
            for(int j=0; j<grid_rows; j++)
                grid_[ft][i][j] = F.grid[ft][i][j];
        }
    }
    set_pose(F.Tcw);

    for(auto& [ft, N_] : N)
        if (N_ > 0){
            descriptors[ft] = F.descriptors.at(ft).clone();
        }
}

void KeyFrame::compute_global_descriptor()
{
    if(place_recognition_)
        place_recognition_->compute(*this);
}

float KeyFrame::vpr_similarity(const Keyframe& other) const
{
    if(!place_recognition_ || !place_recognition_->is_active() || !other)
        return std::numeric_limits<float>::quiet_NaN();
    return place_recognition_->score(*this, *other);
}

void KeyFrame::set_pose(const mat4f &Tcw_)
{
    unique_lock<mutex> lock(pose_mutex_);

    Tcw = Tcw_;

    mat3f Rcw = Tcw.block<3,3>(0,0);
    vec3f tcw = Tcw.block<3,1>(0,3);
    mat3f Rwc = Rcw.transpose();
    twc = -Rwc * tcw;

    Twc = mat4f::Identity();
    Twc.block<3,3>(0,0) = Rwc;
    Twc.block<3,1>(0,3) = twc;
}

mat4f KeyFrame::get_pose() const
{
    unique_lock<mutex> lock(pose_mutex_);
    return Tcw;
}

mat4f KeyFrame::get_pose_inverse() const
{
    unique_lock<mutex> lock(pose_mutex_);
    return Twc;
}

vec3f KeyFrame::get_camera_center() const
{
    unique_lock<mutex> lock(pose_mutex_);
    return twc;
}

mat3f KeyFrame::get_rotation() const
{
    unique_lock<mutex> lock(pose_mutex_);
    return Tcw.block<3,3>(0,0);
}

vec3f KeyFrame::get_translation() const
{
    unique_lock<mutex> lock(pose_mutex_);
    return Tcw.block<3,1>(0,3);
}

void KeyFrame::add_connection(const Keyframe& keyframe, const int &weight)
{
    {
        unique_lock<mutex> lock(connections_mutex_);
        if(!connected_keyframe_weights_.count(keyframe->keyId)){
            connected_keyframe_weights_[keyframe->keyId] = weight;
            connected_keyframes_[keyframe->keyId] = keyframe;
        }
        else if(connected_keyframe_weights_[keyframe->keyId] != weight)
            connected_keyframe_weights_[keyframe->keyId] = weight;
        else
            return;
    }

    update_best_covisibles();
}

void KeyFrame::update_best_covisibles()
{
    unique_lock<mutex> lock(connections_mutex_);
    vector<pair<int,Keyframe> > vPairs;
    vPairs.reserve(connected_keyframe_weights_.size());

    for(auto& keyframeWeight: connected_keyframe_weights_)
        vPairs.push_back(make_pair(keyframeWeight.second,connected_keyframes_[keyframeWeight.first]));

    sort(vPairs.begin(),vPairs.end(),KeyframeComparison);
    list<Keyframe> keyframes;
    list<int> weights;
    for(size_t i = 0, iend = vPairs.size(); i < iend;i++)
    {
        keyframes.push_front(vPairs[i].second);
        weights.push_front(vPairs[i].first);
    }

    ordered_connected_keyframes_ = vector<Keyframe>(keyframes.begin(),keyframes.end());
    ordered_weights_ = vector<int>(weights.begin(), weights.end());
}

map<KeyframeId,Keyframe> KeyFrame::get_connected_keyframes() const
{
    unique_lock<mutex> lock(connections_mutex_);
    map<KeyframeId,Keyframe> connectedKeyFrames_tmp;
    for(auto connectedKeyFrame: connected_keyframes_)
        connectedKeyFrames_tmp[connectedKeyFrame.first] = connectedKeyFrame.second;
    return connectedKeyFrames_tmp;
}

vector<Keyframe> KeyFrame::get_covisible_keyframes() const
{
    unique_lock<mutex> lock(connections_mutex_);
    return ordered_connected_keyframes_;
}

vector<Keyframe> KeyFrame::get_best_covisibility_keyframes(const int &N) const
{
    unique_lock<mutex> lock(connections_mutex_);
    if((int)ordered_connected_keyframes_.size()<N)
        return ordered_connected_keyframes_;
    else
        return vector<Keyframe>(ordered_connected_keyframes_.begin(),ordered_connected_keyframes_.begin()+N);

}

vector<Keyframe> KeyFrame::get_covisibles_by_weight(const int &w) const
{
    unique_lock<mutex> lock(connections_mutex_);

    if(ordered_connected_keyframes_.empty())
        return vector<Keyframe>();

    const auto it = upper_bound(ordered_weights_.begin(),ordered_weights_.end(),w,KeyFrame::weight_greater);
    if(it == ordered_weights_.end())
        return vector<Keyframe>();
    else
    {
        int n = it - ordered_weights_.begin();
        return vector<Keyframe>(ordered_connected_keyframes_.begin(), ordered_connected_keyframes_.begin()+n);
    }
}

int KeyFrame::get_weight(const Keyframe& keyframe) const
{
    unique_lock<mutex> lock(connections_mutex_);
    const auto it = connected_keyframe_weights_.find(keyframe->keyId);
    return it != connected_keyframe_weights_.end() ? it->second : 0;
}

Pt KeyFrame::create_monocular_map_point(const vec3f& worldPos,
                            const KeypointIndex& refIndex,
                            const Keyframe& projKeyframe, const KeypointIndex& projIndex,
                            const FeatureType& featureType)
{
    auto pt = make_shared<MapPoint>(worldPos,this_keyframe(),map_, featureType,
                                    keypoint_colors.at(featureType)[refIndex]);

    add_map_point(pt,refIndex);
    pt->add_observation(this_keyframe(), refIndex);

    projKeyframe->add_map_point(pt,projIndex);
    pt->add_observation(projKeyframe, projIndex);

    map_->add_map_point(pt);
    return pt;
}

Pt KeyFrame::create_map_point(const vec3f& worldPos, const KeypointIndex& refIndex, const FeatureType& featureType){
        auto pt = make_shared<MapPoint>(worldPos,this_keyframe(),map_,featureType,
                                        keypoint_colors.at(featureType)[refIndex]);

        add_map_point(pt,refIndex);
        pt->add_observation(this_keyframe(), refIndex);
        map_->add_map_point(pt);
        return pt;
}

void KeyFrame::add_map_point(const Pt& pt, const KeypointIndex& index)
{
    unique_lock<mutex> lock(features_mutex_);
    map_points_[pt->featureType][index] = pt;
}

void KeyFrame::erase_map_point_match(const size_t &idx, const FeatureType& featType)
{
    unique_lock<mutex> lock(features_mutex_);
    map_points_[featType][idx] = nullptr;
}

void KeyFrame::erase_map_point_match(const Pt& pMP)
{
    // The point's own lock is taken by get_index_in_keyframe, before ours (keyframe -> point
    // is the lock order everywhere else in this class)
    const int idx = pMP->get_index_in_keyframe(this_keyframe());
    if(idx < 0)
        return;
    unique_lock<mutex> lock(features_mutex_);
    map_points_[pMP->featureType][idx] = nullptr;
}

void KeyFrame::replace_map_point_match(const size_t &idx, const Pt& pMP)
{
    // Runs on the local-mapping thread (MapPoint::replace, fusion) while Tracking reads the
    // same vectors: a torn shared_ptr write racing a copy corrupts the refcount
    unique_lock<mutex> lock(features_mutex_);
    map_points_[pMP->featureType][idx] = pMP;
}

set<Pt> KeyFrame::get_map_points(const FeatureType& featType) const
{
    unique_lock<mutex> lock(features_mutex_);
    set<Pt> s;
    for(size_t i=0, iend=map_points_.at(featType).size(); i<iend; i++)
    {
        if(!map_points_.at(featType)[i])
            continue;
        Pt pMP = map_points_.at(featType)[i];
        if(!pMP->is_bad())
            s.insert(pMP);
    }
    return s;
}

int KeyFrame::tracked_map_points(const int &minObs) const
{
    unique_lock<mutex> lock(features_mutex_);

    int nPoints=0;
    const bool bCheckObs = minObs>0;
    for(const auto& [ft, pts] : map_points_){
        for(int i = 0; i < N.at(ft); i++)
        {
            Pt pMP = pts[i];
            if(pMP)
            {
                if(!pMP->is_bad())
                {
                    if(bCheckObs)
                    {
                        if(pts[i]->number_of_observations() >= minObs)
                            nPoints++;
                    }
                    else
                        nPoints++;
                }
            }
        }
    }
    return nPoints;
}

vector<Pt> KeyFrame::get_map_point_matches(const FeatureType& feat_type) const
{
    unique_lock<mutex> lock(features_mutex_);
    if (map_points_.count(feat_type) == 0)
        return vector<Pt>();
    return map_points_.at(feat_type);
}

Pt KeyFrame::get_map_point(const size_t &idx, const FeatureType& featType) const
{
    unique_lock<mutex> lock(features_mutex_);
    return map_points_.at(featType)[idx];
}

void KeyFrame::update_connections()
{
    map<KeyframeId ,int> KFweightsCounter;
    map<KeyframeId ,Keyframe> KFcounter;

    std::map<FeatureType, vector<Pt>> pts;
    {
        unique_lock<mutex> lockMPs(features_mutex_);
        pts = map_points_;
    }

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    for (auto const& [ft, pts_] : pts) {
        for(auto& pt: pts_)
        {
            if(!pt)
                continue;

            if(pt->is_bad())
                continue;

            map<KeyframeId , Obs> observations = pt->get_observations();

            for(auto& obs: observations)
            {
                if(obs.first == keyId)
                    continue;
                KFweightsCounter[obs.first]++;
                KFcounter[obs.first] = obs.second->projKeyframe;
            }
        }
    }

    // This should not happen
    if(KFcounter.empty())
        return;

    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    int nmax = 0;
    Keyframe keyframeMaxObs = nullptr;
    int th = 15;

    vector<pair<int,Keyframe> > vPairs;
    vPairs.reserve(KFcounter.size());
    for(auto& weightCount: KFweightsCounter)
    {
        if(weightCount.second > nmax)
        {
            nmax = weightCount.second;
            keyframeMaxObs = KFcounter[weightCount.first];
        }
        if(weightCount.second >= th)
        {
            vPairs.push_back(make_pair(weightCount.second,KFcounter[weightCount.first]));
            KFcounter[weightCount.first]->add_connection(this_keyframe(),weightCount.second);
        }
    }

    if(vPairs.empty())
    {
        vPairs.push_back(make_pair(nmax,keyframeMaxObs));
        keyframeMaxObs->add_connection(this_keyframe(),nmax);
    }

    sort(vPairs.begin(),vPairs.end(),KeyframeComparison);
    list<Keyframe> keyframes;
    list<int> weights;
    for(size_t i = 0; i < vPairs.size(); i++)
    {
        keyframes.push_front(vPairs[i].second);
        weights.push_front(vPairs[i].first);
    }

    {
        unique_lock<mutex> lockCon(connections_mutex_);

        connected_keyframe_weights_ = KFweightsCounter;
        connected_keyframes_ = KFcounter;
        ordered_connected_keyframes_ = vector<Keyframe>(keyframes.begin(),keyframes.end());
        ordered_weights_ = vector<int>(weights.begin(), weights.end());

        if(first_connection_ && keyId!=0)
        {
            parent_ = ordered_connected_keyframes_.front();
            parent_->add_child(this_keyframe());
            first_connection_ = false;
        }

    }
}

void KeyFrame::add_child(const Keyframe& pKF)
{
    unique_lock<mutex> lockCon(connections_mutex_);
    children_.insert(pKF);
}

void KeyFrame::erase_child(const Keyframe& pKF)
{
    unique_lock<mutex> lockCon(connections_mutex_);
    children_.erase(pKF);
}

void KeyFrame::change_parent(const Keyframe& pKF)
{
    unique_lock<mutex> lockCon(connections_mutex_);
    parent_ = pKF;
    pKF->add_child(this_keyframe());
}

KeyframeIdSet KeyFrame::get_children() const
{
    unique_lock<mutex> lockCon(connections_mutex_);
    return children_;
}

Keyframe KeyFrame::get_parent() const
{
    unique_lock<mutex> lockCon(connections_mutex_);
    return parent_;
}

bool KeyFrame::has_child(const Keyframe& pKF) const
{
    unique_lock<mutex> lockCon(connections_mutex_);
    return children_.count(pKF);
}

void KeyFrame::add_loop_edge(const Keyframe& pKF)
{
    unique_lock<mutex> lockCon(connections_mutex_);
    not_erase_ = true;
    loop_edges_.insert(pKF);
}

KeyframeIdSet KeyFrame::get_loop_edges() const
{
    unique_lock<mutex> lockCon(connections_mutex_);
    return loop_edges_;
}

void KeyFrame::set_not_erase()
{
    unique_lock<mutex> lock(connections_mutex_);
    not_erase_ = true;
}

void KeyFrame::set_erase()
{
    {
        unique_lock<mutex> lock(connections_mutex_);
        if(loop_edges_.empty())
        {
            not_erase_ = false;
        }
    }

    if(to_be_erased_)
    {
        set_bad_flag();
    }
}

void KeyFrame::set_bad_flag()
{
    {
        unique_lock<mutex> lock(connections_mutex_);
        if(keyId==0)
            return;
        else if(not_erase_)
        {
            to_be_erased_ = true;
            return;
        }
    }

    // Snapshots: other threads mutate both containers (update_connections, add_map_point)
    // while the erase calls below run without our locks -- and those calls lock the other
    // keyframes / the points, so they cannot run under ours
    std::map<KeyframeId, Keyframe> connected;
    std::map<FeatureType, std::vector<Pt>> map_points;
    {
        unique_lock<mutex> lock(connections_mutex_);
        connected = connected_keyframes_;
    }
    {
        unique_lock<mutex> lock(features_mutex_);
        map_points = map_points_;
    }

    for(const auto& [id, keyframe] : connected)
        keyframe->erase_connection(this_keyframe());

    for(const auto& [ft, points] : map_points)
        for(const Pt& point : points)
            if(point)
                point->erase_observation(this_keyframe());

    {
        unique_lock<mutex> lock(connections_mutex_);
        unique_lock<mutex> lock1(features_mutex_);

        // connected_keyframes_ used to be left behind (the weights map was cleared twice
        // instead), pinning every neighbour of a culled keyframe
        connected_keyframe_weights_.clear();
        connected_keyframes_.clear();
        ordered_connected_keyframes_.clear();
        ordered_weights_.clear();

        // Update Spanning Tree. A keyframe culled before its first update_connections has no
        // parent (this fork culls far more aggressively than stock); its children then keep
        // their parent pointer to this keyframe instead of being re-hung
        KeyframeIdSet sParentCandidates;
        if(parent_)
            sParentCandidates.insert(parent_);

        // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest
        while(!children_.empty())
        {
            bool bContinue = false;

            int max = -1;
            Keyframe pC;
            Keyframe pP;

            for(auto sit=children_.begin(), send=children_.end(); sit!=send; sit++)
            {
                Keyframe pKF = *sit;
                if(pKF->is_bad())
                    continue;

                // Check if a parent candidate is connected to the keyframe
                vector<Keyframe> vpConnected = pKF->get_covisible_keyframes();
                for(size_t i=0, iend=vpConnected.size(); i<iend; i++)
                {
                    for(auto spcit=sParentCandidates.begin(), spcend=sParentCandidates.end(); spcit!=spcend; spcit++)
                    {
                        if(vpConnected[i]->keyId == (*spcit)->keyId)
                        {
                            int w = pKF->get_weight(vpConnected[i]);
                            if(w>max)
                            {
                                pC = pKF;
                                pP = vpConnected[i];
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }

            if(bContinue)
            {
                pC->change_parent(pP);
                sParentCandidates.insert(pC);
                children_.erase(pC);
            }
            else
                break;
        }

        // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
        if(parent_)
        {
            for(const Keyframe& child : children_)
                child->change_parent(parent_);
            parent_->erase_child(this_keyframe());
        }
        bad_ = true;
    }

    map_->EraseKeyFrame(this_keyframe());
    place_recognition_->erase(this_keyframe());
}

bool KeyFrame::is_bad() const
{
    unique_lock<mutex> lock(connections_mutex_);
    return bad_;
}

void KeyFrame::erase_connection(const Keyframe& keyframe)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(connections_mutex_);
        if(connected_keyframe_weights_.count(keyframe->keyId))
        {
            connected_keyframe_weights_.erase(keyframe->keyId);
            connected_keyframes_.erase(keyframe->keyId);
            bUpdate=true;
        }
    }

    if(bUpdate)
        update_best_covisibles();
}

vector<size_t> KeyFrame::get_features_in_area(const float &x, const float &y, const float &r, const FeatureType& featType) const
{
    auto it1 = N.find(featType);
    if (it1 == N.end() )
        return vector<size_t> ();

    vector<size_t> vIndices;
    vIndices.reserve(N.at(featType));

    const int nMinCellX = max(0,(int)floor((x-min_x-r)*grid_element_width_inv));
    if(nMinCellX>=grid_cols)
        return vIndices;

    const int nMaxCellX = min((int)grid_cols-1,(int)ceil((x-min_x+r)*grid_element_width_inv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-min_y-r)*grid_element_height_inv));
    if(nMinCellY>=grid_rows)
        return vIndices;

    const int nMaxCellY = min((int)grid_rows-1,(int)ceil((y-min_y+r)*grid_element_height_inv));
    if(nMaxCellY<0)
        return vIndices;

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t>& vCell = grid_.at(featType)[ix][iy];
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = keypoints.at(featType)[vCell[j]];
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool KeyFrame::is_in_image(const float &x, const float &y) const
{
    return (x>=min_x && x<max_x && y>=min_y && y<max_y);
}

float KeyFrame::compute_scene_median_depth(const int q) const
{
    std::map<FeatureType, std::vector<Pt>> vpMapPoints;
    mat4f Tcw_;
    {
        unique_lock<mutex> lock(features_mutex_);
        unique_lock<mutex> lock2(pose_mutex_);
        vpMapPoints = map_points_;
        Tcw_ = Tcw;
    }

    // Read the snapshot, not the member (the loop used to re-read map_points_ unlocked)
    vector<float> vDepths;
    const vec3f Rcw2 = Tcw_.block<1,3>(2,0).transpose();
    const float zcw = Tcw_(2,3);
    for(const auto& [ft, points] : vpMapPoints)
        for(const Pt& pMP : points)
            if(pMP)
                vDepths.push_back(Rcw2.dot(pMP->get_world_pos()) + zcw);
    // No map points (a fresh or heavily culled keyframe): -1, which every caller treats as
    // "skip" -- (size-1)/q would underflow and index out of bounds
    if(vDepths.empty())
        return -1.0f;

    // nth_element places exactly the element sort() would put at this position — O(n) not
    // O(n log n), and this runs once per covisible neighbor in CreateNewMapPoints.
    const size_t nth = (vDepths.size()-1)/q;
    nth_element(vDepths.begin(), vDepths.begin() + nth, vDepths.end());

    return vDepths[nth];
}

    float KeyFrame::get_keypoint_size(const KeypointIndex &keyPtIdx, const FeatureType& featType) const {
        return keyPtsSize.at(featType)[keyPtIdx];
    }

    float KeyFrame::get_keypoint_sigma2(const KeypointIndex &keyPtIdx, const FeatureType& featType) const
    {
        return 0.5f * (keyPtsSigma2.at(featType)[keyPtIdx](0,0) + keyPtsSigma2.at(featType)[keyPtIdx](1,1));
    }

    float KeyFrame::get_keypoint_information_1d(const KeypointIndex &keyPtIdx, const FeatureType& featType) const
    {
        return 0.5f * (keyPtsInf.at(featType)[keyPtIdx](0,0) + keyPtsInf.at(featType)[keyPtIdx](1,1));
    }

    mat2f KeyFrame::get_keypoint_information_2d(const KeypointIndex &keyPtIdx, const FeatureType& featType) const
    {
        return keyPtsInf.at(featType)[keyPtIdx];
    }

    float KeyFrame::get_keypoint_sigma(const KeypointIndex &keyPtIdx, const FeatureType& featType) const
    {
        return sqrtf(get_keypoint_sigma2(keyPtIdx, featType));
    }

    KeyFrame::PoseMatrices KeyFrame::get_pose_matrices() const
    {
        unique_lock<mutex> lock(pose_mutex_);
        return PoseMatrices{Tcw, Twc, Tcw.block<3,3>(0,0), Twc.block<3,3>(0,0), Tcw.block<3,1>(0,3), twc};
    }

} //namespace ORB_SLAM
