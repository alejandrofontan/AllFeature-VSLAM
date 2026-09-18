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

#include "LoopClosing.h"

#include "Sim3Solver.h"

#include "Converter.h"

#include "Optimizer.h"

#include "FeatureMatcher.h"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <thread>


namespace AF_VSLAM
{

    LoopConnections::LoopConnections(const KeyframeId & keyframeId, const Keyframe& keyframe, const map<KeyframeId,Keyframe>& connections):
            keyframeId(keyframeId), keyframe(keyframe), connections(connections){}

LoopClosing::LoopClosing(shared_ptr<Map>pMap, shared_ptr<PlaceRecognition> place_recognition,
    std::shared_ptr<LocalMapping> local_mapper, std::shared_ptr<MapDrawer> map_drawer,
    const bool bFixScale,
    const std::vector<FeatureType>& feat_types,
    int image_width, int image_height):
    featureType(place_recognition->verification_feature()), feat_types(feat_types), mpMap(pMap),
    map_drawer_(std::move(map_drawer)),
    place_recognition(std::move(place_recognition)), local_mapper_(std::move(local_mapper)),
    mpMatchedKF(NULL), mbRunningGBA(false), mbFinishedGBA(true),
    mbStopGBA(false), mpThreadGBA(NULL), mbFixScale(bFixScale), mnFullBAIdx(0),
    image_width(image_width), image_height(image_height)
{
    matcher = std::make_shared<FeatureMatcher>(image_width, image_height, feat_types, "LoopClosing", 0.8, true);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// # Parameters
LoopClosingParameters LoopClosing::params{};

void LoopClosing::LoadParameters(const cv::FileStorage& fSettings)
{
    auto read_if_present = [&fSettings](const char* key, auto& field)
    {
        const cv::FileNode node = fSettings[key];
        if(!node.empty())
            node >> field;
    };

    read_if_present("LoopClosing.CovisibilityConsistencyThreshold", params.covisibility_consistency_threshold);
    read_if_present("LoopClosing.MinKeyframesBetweenLoops", params.min_keyframes_between_loops);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void LoopClosing::Run()
{
    {
        std::lock_guard<std::mutex> lock(finish_mutex_);
        finished_ = false;
    }

    while(1)
    {
        // Check if there are keyframes in the queue
        if(has_new_keyframes())
        {
            // Detect loop candidates and check covisibility consistency

            if(detect_loop())
            {
               // Compute similarity transformation [sR|t]
               // In the stereo/RGBD case s=1
               if(ComputeSim3())
               {
                   std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();

                   // Perform loop fusion and pose graph optimization
                   CorrectLoop();

                   ++numOfLoopClosures;
                   map_drawer_->AddLoopClosureKeyframe(current_keyframe_->get_pose_inverse());

                   std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
                   double t_duration = std::chrono::duration_cast<std::chrono::duration<double> >(t_end - t_start).count();
                   loopClosingTime.push_back(t_duration);
               }
            }
        }

        reset_if_requested();

        if(is_finish_requested())
            break;

        usleep(5000);
    }

    set_finished();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// # Keyframe queue
void LoopClosing::insert_keyframe(const Keyframe& keyframe)
{
    // Without an active VPR backend this thread never runs (System's constructor): a
    // queued keyframe would never be drained and would stay alive, culled or not.
    if(!place_recognition->is_active())
        return;

    std::lock_guard<std::mutex> lock(new_keyframes_mutex_);
    new_keyframes_.push_back(keyframe);
}

bool LoopClosing::has_new_keyframes() const
{
    std::lock_guard<std::mutex> lock(new_keyframes_mutex_);
    return !new_keyframes_.empty();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// # Loop detection
bool LoopClosing::detect_loop()
{
    {
        std::lock_guard<std::mutex> lock(new_keyframes_mutex_);
        current_keyframe_ = new_keyframes_.front();
        new_keyframes_.pop_front();
        // Keyframe culling must not delete the keyframe while this thread works on it.
        // SetErase lifts the guard (a culling deferred meanwhile is applied then); on
        // success ComputeSim3/CorrectLoop keep it until the loop is closed or rejected.
        current_keyframe_->SetNotErase();
    }

    // Candidates retrieved by the VPR backend (covisible keyframes excluded), kept only
    // when their covisibility group was also retrieved by the previous keyframes. No
    // detection right after a loop closure, nor for the first keyframes of a map (both
    // gates share the counter: keyIds start at 0); the consistency groups are left as
    // they are meanwhile.
    loop_candidates_.clear();
    const KeyframeId min_keyframes_between_loops = static_cast<KeyframeId>(params.min_keyframes_between_loops);
    if(current_keyframe_->keyId >= last_loop_keyframe_id_ + min_keyframes_between_loops)
        loop_candidates_ = consistent_loop_candidates(place_recognition->detect_loop_candidates(current_keyframe_));

    // Into the database only after its own query
    place_recognition->add(current_keyframe_);

    if(loop_candidates_.empty())
    {
        current_keyframe_->SetErase();
        return false;
    }
    return true;
}

// True when the two sorted id lists share a keyframe
static bool share_keyframe(const std::vector<KeyframeId>& a, const std::vector<KeyframeId>& b)
{
    auto ia = a.begin();
    auto ib = b.begin();
    while(ia != a.end() && ib != b.end())
    {
        if(*ia == *ib)
            return true;
        if(*ia < *ib)
            ++ia;
        else
            ++ib;
    }
    return false;
}

// Covisibility-consistency vote (ORB-SLAM2). Each candidate expands to its covisibility
// group: the candidate plus the keyframes connected to it. A group is consistent with a
// group kept from the previous keyframe when the two share a keyframe; it then inherits
// that group's consistency count plus one, and a count reaching
// params.covisibility_consistency_threshold makes the candidate a loop candidate. The
// groups built here replace consistent_groups_ for the next keyframe (no candidates ->
// no groups). Rules kept from the original: a previous group passes its count to the
// first candidate consistent with it only, so the same group is never carried twice;
// a candidate consistent with several previous groups is stored once per group, and
// enters the result once; a candidate consistent with no previous group starts a new
// group with count 0.
std::vector<Keyframe> LoopClosing::consistent_loop_candidates(const std::vector<Keyframe>& candidates)
{
    std::vector<Keyframe> loop_candidates;
    std::vector<ConsistentGroup> current_groups;
    std::vector<bool> carried(consistent_groups_.size(), false);   // previous group already passed on

    for(const Keyframe& candidate : candidates)
    {
        // Covisibility group as sorted ids (GetConnectedKeyFrames is ordered by id)
        ConsistentGroup group;
        for(const auto& [id, keyframe] : candidate->GetConnectedKeyFrames())
            group.keyframes.push_back(id);
        group.keyframes.insert(std::lower_bound(group.keyframes.begin(), group.keyframes.end(), candidate->keyId),
                               candidate->keyId);

        bool consistent_with_some_group = false;
        bool enough_consistent = false;
        for(size_t i = 0; i < consistent_groups_.size(); i++)
        {
            const ConsistentGroup& previous = consistent_groups_[i];
            if(!share_keyframe(group.keyframes, previous.keyframes))
                continue;
            consistent_with_some_group = true;

            const int consistency = previous.consistency + 1;
            if(!carried[i])
            {
                current_groups.push_back(ConsistentGroup{group.keyframes, consistency});
                carried[i] = true;
            }
            if(consistency >= params.covisibility_consistency_threshold && !enough_consistent)
            {
                loop_candidates.push_back(candidate);
                enough_consistent = true;
            }
        }

        if(!consistent_with_some_group)
            current_groups.push_back(std::move(group));
    }

    consistent_groups_ = std::move(current_groups);
    return loop_candidates;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool LoopClosing::ComputeSim3()
{
    // For each consistent loop candidate we try to compute a Sim3

    const int nInitialCandidates = loop_candidates_.size();

    // We compute first ORB matches for each candidate
    // If enough matches are found, we setup a Sim3Solver

    vector<Sim3Solver*> vpSim3Solvers;
    vpSim3Solvers.resize(nInitialCandidates);

    vector<vector<Pt>> vvpMapPointMatches;
    vvpMapPointMatches.resize(nInitialCandidates);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nInitialCandidates);

    int nCandidates=0; //candidates with enough matches

    for(int i=0; i<nInitialCandidates; i++)
    {
        Keyframe pKF = loop_candidates_[i];

        // avoid that local mapping erase it while it is being processed in this thread
        pKF->SetNotErase();

        if(pKF->is_bad())
        {
            vbDiscarded[i] = true;
            continue;
        }

        std::map<FeatureType, vector<pair<size_t,size_t>>> matched_pairs;
        matcher->match_keyframes_for_compute_sim3(current_keyframe_, pKF, matched_pairs, feat_types);

        // TO DO
        // Currently we use only one feature type downstream
        int nmatches=0;
        vvpMapPointMatches[i] = vector<Pt>(current_keyframe_->get_map_point_matches(featureType).size(),static_cast<Pt>(NULL));
        for (const auto& matches : matched_pairs[featureType]) {
            vvpMapPointMatches[i][matches.first] = pKF->get_map_point_matches(featureType)[matches.second];
            nmatches++;
        }

        if(nmatches<20)
        {
            vbDiscarded[i] = true;
            continue;
        }
        else
        {
            Sim3Solver* pSolver = new Sim3Solver(current_keyframe_,pKF,vvpMapPointMatches[i],featureType, mbFixScale);
            pSolver->set_ransac_parameters(0.99,20,300);
            vpSim3Solvers[i] = pSolver;
        }

        nCandidates++;
    }

    bool bMatch = false;

    // Perform alternatively RANSAC iterations for each candidate
    // until one is succesful or all fail
    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nInitialCandidates; i++)
        {
            if(vbDiscarded[i])
                continue;

            Keyframe pKF = loop_candidates_[i];

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            Sim3Solver* pSolver = vpSim3Solvers[i];
            cv::Mat Scm  = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If RANSAC returns a Sim3 optimize with all correspondences
            if(!Scm.empty())
            {
                vector<Pt> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<Pt>(NULL));
                for(size_t j=0, jend=vbInliers.size(); j<jend; j++)
                {
                    if(vbInliers[j])
                       vpMapPointMatches[j]=vvpMapPointMatches[i][j];
                }

                mat3f R = pSolver->GetEstimatedRotation();
                vec3f t = pSolver->GetEstimatedTranslation();
                const float s = pSolver->GetEstimatedScale();

                g2o::Sim3 gScm(R.cast<double>(),t.cast<double>(),s);
                const int nInliers = Optimizer::OptimizeSim3(current_keyframe_, pKF, vpMapPointMatches, gScm, 10, mbFixScale, featureType);

                // If optimization is succesful stop ransacs and continue
                if(nInliers>=20)
                {
                    bMatch = true;
                    mpMatchedKF = pKF;
                    g2o::Sim3 gSmw(pKF->get_rotation().cast<double>(),pKF->get_translation().cast<double>(),1.0);
                    mg2oScw = gScm*gSmw;
                    mScw = Converter::to_matrix4f(mg2oScw);

                    mvpCurrentMatchedPoints = vpMapPointMatches;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        for(int i=0; i<nInitialCandidates; i++)
             loop_candidates_[i]->SetErase();
        current_keyframe_->SetErase();
        return false;
    }

    // Retrieve MapPoints seen in Loop Keyframe and neighbors
    vector<Keyframe> vpLoopConnectedKFs = mpMatchedKF->get_covisible_keyframes();
    vpLoopConnectedKFs.push_back(mpMatchedKF);
    mvpLoopMapPoints.clear();
    map<PtId, Keyframe> pt_to_keyframe_id;
    for(vector<Keyframe>::iterator vit=vpLoopConnectedKFs.begin(); vit!=vpLoopConnectedKFs.end(); vit++)
    {
        Keyframe pKF = *vit;
        map<FeatureType, vector<pair<size_t,size_t>>> matched_pairs = matcher->match_keyframes(current_keyframe_, pKF, feat_types);

        vector<Pt> vpMapPoints = pKF->get_map_point_matches(featureType);
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            Pt pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->is_bad() && pMP->mnLoopPointForKF!=current_keyframe_->keyId)
                {
                    mvpLoopMapPoints.push_back(pMP);
                    pMP->mnLoopPointForKF=current_keyframe_->keyId;
                    pt_to_keyframe_id[pMP->ptId] = pKF;
                }
            }
        }
    }

    // Find more matches projecting with the computed Sim3
    matcher->search_by_projection_for_compute_sim3(current_keyframe_, mScw,
        mvpLoopMapPoints, mvpCurrentMatchedPoints, pt_to_keyframe_id);

    // If enough matches accept Loop
    int nTotalMatches = 0;
    for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
    {
        if(mvpCurrentMatchedPoints[i])
            nTotalMatches++;
    }

    if(nTotalMatches>=40)
    {
        for(int i=0; i<nInitialCandidates; i++)
            if(loop_candidates_[i]!=mpMatchedKF)
                loop_candidates_[i]->SetErase();
        return true;
    }
    else
    {
        for(int i=0; i<nInitialCandidates; i++)
            loop_candidates_[i]->SetErase();
        current_keyframe_->SetErase();
        return false;
    }

}

void LoopClosing::CorrectLoop()
{
    cout << "Loop detected!" << endl;

    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    local_mapper_->request_stop();

    // If a Global Bundle Adjustment is running, abort it
    if(isRunningGBA())
    {
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx = true;

        if(mpThreadGBA)
        {
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
    }

    // Wait until Local Mapping has effectively stopped
    while(!local_mapper_->is_stopped())
    {
        usleep(1000);
    }

    // Ensure current keyframe is updated
    current_keyframe_->update_connections();

    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    mvpCurrentConnectedKFs = current_keyframe_->get_covisible_keyframes();
    mvpCurrentConnectedKFs.push_back(current_keyframe_);

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    CorrectedSim3[current_keyframe_]=mg2oScw;
    mat4f Twc = current_keyframe_->get_pose_inverse();


    {
        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->map_update_mutex_);

        for(vector<Keyframe>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
            Keyframe pKFi = *vit;

            mat4f Tiw = pKFi->get_pose();

            if(pKFi!=current_keyframe_)
            {
                mat4f Tic = Tiw * Twc;
                mat3f Ric = Tic.block<3,3>(0,0);
                vec3f tic = Tic.block<3,1>(0,3);
                g2o::Sim3 g2oSic(Ric.cast<double>(),tic.cast<double>(),1.0);
                g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oScw;
                //Pose corrected with the Sim3 of the loop closure
                CorrectedSim3[pKFi]=g2oCorrectedSiw;
            }

            mat3f Riw = Tiw.block<3,3>(0,0);
            vec3f tiw = Tiw.block<3,1>(0,3);
            g2o::Sim3 g2oSiw(Riw.cast<double>(),tiw.cast<double>(),1.0);
            //Pose without correction
            NonCorrectedSim3[pKFi]=g2oSiw;
        }

        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
        {
            Keyframe pKFi = mit->first;
            g2o::Sim3 g2oCorrectedSiw = mit->second;
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

            g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi];

            vector<Pt> vpMPsi = pKFi->get_map_point_matches(featureType);
            for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
            {
                Pt pMPi = vpMPsi[iMP];
                if(!pMPi)
                    continue;
                if(pMPi->is_bad())
                    continue;
                if(pMPi->mnCorrectedByKF==current_keyframe_->keyId)
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                vec3f P3Dw = pMPi->get_world_pos();
                Eigen::Matrix<double,3,1> eigP3Dw = P3Dw.cast<double>();
                Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

                pMPi->set_world_pos(eigCorrectedP3Dw.cast<float>());
                pMPi->mnCorrectedByKF = current_keyframe_->keyId;
                pMPi->mnCorrectedReference = pKFi->keyId;
                pMPi->UpdateNormalAndDepth();
            }

            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
            Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
            double s = g2oCorrectedSiw.scale();

            eigt *=(1./s); //[R t/s;0 1]

            mat4f correctedTiw = Converter::to_matrix4f(eigR,eigt);

            pKFi->set_pose(correctedTiw);

            // Make sure connections are updated
            pKFi->update_connections();
        }

        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
        {
            if(mvpCurrentMatchedPoints[i])
            {
                Pt pLoopMP = mvpCurrentMatchedPoints[i];
                Pt pCurMP = current_keyframe_->get_map_point(i, featureType);
                if(pCurMP)
                    pCurMP->replace(pLoopMP);
                else
                {
                    current_keyframe_->add_map_point(pLoopMP,KeypointIndex (i));
                    pLoopMP->add_observation(current_keyframe_,KeypointIndex (i));
                }
            }
        }

    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    SearchAndFuse(CorrectedSim3);


    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    map<KeyframeId,LoopConnections> loopConnections{};

    for(const auto& pKFi: mvpCurrentConnectedKFs)
    {
        vector<Keyframe> vpPreviousNeighbors = pKFi->get_covisible_keyframes();

        // Update connections. Detect new links.
        pKFi->update_connections();
        loopConnections.insert(std::make_pair(pKFi->keyId, LoopConnections(pKFi->keyId, pKFi, pKFi->GetConnectedKeyFrames())));

        for(auto& vit_prev: vpPreviousNeighbors)
        {
            loopConnections[pKFi->keyId].connections.erase(vit_prev->keyId);
        }
        for(auto& vit2: mvpCurrentConnectedKFs)
        {
            loopConnections[pKFi->keyId].connections.erase(vit2->keyId);
        }
    }

    // Optimize graph
    Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, current_keyframe_, NonCorrectedSim3, CorrectedSim3, loopConnections, mbFixScale);

    mpMap->InformNewBigChange();

    // Add loop edge
    mpMatchedKF->AddLoopEdge(current_keyframe_);
    current_keyframe_->AddLoopEdge(mpMatchedKF);

    // Launch a new thread to perform Global Bundle Adjustment
    mbRunningGBA = true;
    mbFinishedGBA = false;
    mbStopGBA = false;
    mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment,this,current_keyframe_->keyId);

    // Loop closed. release Local Mapping.
    local_mapper_->release();

    last_loop_keyframe_id_ = current_keyframe_->keyId;
}

void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
{

    for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
    {
        Keyframe pKF = mit->first;

        g2o::Sim3 g2oScw = mit->second;
        mat4f cvScw = Converter::to_matrix4f(g2oScw);

        vector<Pt> vpReplacePoints(mvpLoopMapPoints.size(),static_cast<Pt>(NULL));
        matcher->fuse_map_points_to_keyframe(pKF,cvScw,mvpLoopMapPoints,4.0f,vpReplacePoints, featureType);

        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->map_update_mutex_);
        const int nLP = mvpLoopMapPoints.size();
        for(int i=0; i<nLP;i++)
        {
            Pt pRep = vpReplacePoints[i];
            if(pRep)
            {
                pRep->replace(mvpLoopMapPoints[i]);
            }
        }
    }
}


void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
{
    cout << "Starting Global Bundle Adjustment" << endl;

    int idx =  mnFullBAIdx;
    Optimizer::global_bundle_adjustment(mpMap,10,&mbStopGBA,nLoopKF,false);

    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        unique_lock<mutex> lock(mMutexGBA);
        if(idx!=mnFullBAIdx)
            return;

        if(!mbStopGBA)
        {
            cout << "Global Bundle Adjustment finished" << endl;
            cout << "Updating map ..." << endl;
            local_mapper_->request_stop();
            // Wait until Local Mapping has effectively stopped

            while(!local_mapper_->is_stopped() && !local_mapper_->is_finished())
            {
                usleep(1000);
            }

            // Get Map Mutex
            unique_lock<mutex> lock(mpMap->map_update_mutex_);

            // Correct keyframes starting at map first keyframe
            list<Keyframe> lpKFtoCheck(mpMap->keyframe_origins_.begin(),mpMap->keyframe_origins_.end());

            while(!lpKFtoCheck.empty())
            {
                Keyframe pKF = lpKFtoCheck.front();
                const KeyframeIdSet sChilds = pKF->get_children();
                mat4f Twc = pKF->get_pose_inverse();
                for(auto sit=sChilds.begin();sit!=sChilds.end();sit++)
                {
                    Keyframe pChild = *sit;
                    if(pChild->mnBAGlobalForKF!=nLoopKF)
                    {
                        mat4f Tchildc = pChild->get_pose() * Twc;
                        pChild->TcwGBA = Tchildc * pKF->TcwGBA;//*Tcorc*pKF->mTcwGBA;
                        pChild->mnBAGlobalForKF=nLoopKF;

                    }
                    lpKFtoCheck.push_back(pChild);
                }

                pKF->TcwBefGBA = pKF->get_pose();
                pKF->set_pose(pKF->TcwGBA);
                lpKFtoCheck.pop_front();
            }

            // Correct MapPoints
            const vector<Pt> vpMPs = mpMap->get_all_map_points();

            for(size_t i=0; i<vpMPs.size(); i++)
            {
                Pt pMP = vpMPs[i];

                if(pMP->is_bad())
                    continue;

                if(pMP->mnBAGlobalForKF==nLoopKF)
                {
                    // If optimized by Global BA, just update
                    pMP->set_world_pos(pMP->PosGBA);
                }
                else
                {
                    // Update according to the correction of its reference keyframe
                    Keyframe pRefKF = pMP->GetReferenceKeyFrame();

                    if(pRefKF->mnBAGlobalForKF!=nLoopKF)
                        continue;

                    // Map to non-corrected camera
                    mat3f Rcw = pRefKF->TcwBefGBA.block<3,3>(0,0);
                    vec3f tcw = pRefKF->TcwBefGBA.block<3,1>(0,3);
                    vec3f Xc = Rcw*pMP->get_world_pos()+tcw;

                    // Backproject using corrected camera
                    mat4f Twc = pRefKF->get_pose_inverse();
                    mat3f Rwc = Twc.block<3,3>(0,0);
                    vec3f twc = Twc.block<3,1>(0,3);

                    pMP->set_world_pos(Rwc*Xc+twc);
                }
            }

            mpMap->InformNewBigChange();

            local_mapper_->release();

            cout << "Map updated!" << endl;
        }

        mbFinishedGBA = true;
        mbRunningGBA = false;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// # Reset protocol
void LoopClosing::request_reset()
{
    // Without an active VPR backend this thread never runs, so nobody would ever clear
    // the request (the caller would spin forever); there is nothing to reset either.
    if(!place_recognition->is_active())
        return;

    {
        std::lock_guard<std::mutex> lock(reset_mutex_);
        reset_requested_ = true;
    }
    // Block until the Loop Closing thread has performed the reset (reset_if_requested)
    while(is_reset_requested())
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
}

bool LoopClosing::is_reset_requested() const
{
    std::lock_guard<std::mutex> lock(reset_mutex_);
    return reset_requested_;
}

void LoopClosing::reset_if_requested()
{
    std::lock_guard<std::mutex> lock(reset_mutex_);
    if(!reset_requested_)
        return;

    {
        std::lock_guard<std::mutex> queue_lock(new_keyframes_mutex_);
        new_keyframes_.clear();
    }
    last_loop_keyframe_id_ = 0;

    // Loop-detection state of the wiped map. Keyframe ids restart at 0 after a reset, so
    // stale consistency groups (keyed by id) could vote for the new map's first
    // candidates; the vectors would also keep the old map's keyframes and points alive.
    consistent_groups_.clear();
    loop_candidates_.clear();
    mvpCurrentConnectedKFs.clear();
    mvpCurrentMatchedPoints.clear();
    mvpLoopMapPoints.clear();
    current_keyframe_.reset();
    mpMatchedKF.reset();

    reset_requested_ = false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// # Finish protocol
void LoopClosing::request_finish()
{
    std::lock_guard<std::mutex> lock(finish_mutex_);
    finish_requested_ = true;
}

bool LoopClosing::is_finish_requested() const
{
    std::lock_guard<std::mutex> lock(finish_mutex_);
    return finish_requested_;
}

void LoopClosing::set_finished()
{
    std::lock_guard<std::mutex> lock(finish_mutex_);
    finished_ = true;
}

bool LoopClosing::is_finished() const
{
    std::lock_guard<std::mutex> lock(finish_mutex_);
    return finished_;
}

} //namespace ORB_SLAM
