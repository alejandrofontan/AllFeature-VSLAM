#include "LocalMapping.h"
#include "LoopClosing.h"
#include "FeatureMatcher.h"
#include "Optimizer.h"
#include "Converter.h"
#include "Utils.h"
#include "afvslam_log.hpp"

#include <mutex>
#include <Eigen/Core>
#include <Eigen/SVD>

namespace AF_VSLAM
{

LocalMapping::LocalMapping(shared_ptr<Map> pMap, const float bMonocular, const vector<FeatureType>& featureTypes, const int& image_width, const int& image_height):
    featureTypes(featureTypes), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true),
    mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true),
    image_width(image_width), image_height(image_height)
{
    matcher = std::make_shared<FeatureMatcher>(image_width, image_height, featureTypes, "LocalMapping");
}

void LocalMapping::Run()
{

    mbFinished = false;

    while(1)
    {
        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames())
        {

            std::chrono::steady_clock::time_point t_start_0 = std::chrono::steady_clock::now();
            //////////////////////////////////////////////////////////////////////////////////////////////////////
            //////////////////////////////////////////////////////////////////////////////////////////////////////

            // BoW conversion and insertion in Map
            ProcessNewKeyFrame();

            // Check recent MapPoints

            MapPointCulling();

            //////////////////////////////////////////////////////////////////////////////////////////////////////
            // Triangulate new MapPoints
            std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();
            CreateNewMapPoints();
            std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
            double t_duration = std::chrono::duration_cast<std::chrono::duration<double> >(t_end - t_start).count();
            createNewMapPoints_times[int(1000 * t_duration)]++;
            //////////////////////////////////////////////////////////////////////////////////////////////////////

            if(!CheckNewKeyFrames())
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                t_start = std::chrono::steady_clock::now();
                for (const auto& [feat, N_] : mpCurrentKeyFrame->N) {
                    SearchInNeighbors(feat);
                }
                t_end = std::chrono::steady_clock::now();
                t_duration = std::chrono::duration_cast<std::chrono::duration<double> >(t_end - t_start).count();
                searchInNeighbors_times[int(1000 * t_duration)]++;
            }
            mbAbortBA = false;
            //if(!CheckNewKeyFrames() && !stopRequested())
            if(!CheckNewKeyFrames())
            {
                // Local BA
                if(mpMap->keyframes_in_map()>2){
                    t_start = std::chrono::steady_clock::now();
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpMap);
                    t_end = std::chrono::steady_clock::now();
                    t_duration = std::chrono::duration_cast<std::chrono::duration<double> >(t_end - t_start).count();
                    localbundleadjustment_times[int(1000 * t_duration)]++;
                }
                // Check redundant local Keyframes
                KeyFrameCulling();
            }
            loopCloser->insert_keyframe(mpCurrentKeyFrame);

            if(viewer)
                viewer->set_runLocalMapping_time_median(map_median(localMapping_times));

            //////////////////////////////////////////////////////////////////////////////////////////////////////
            //////////////////////////////////////////////////////////////////////////////////////////////////////
            t_end = std::chrono::steady_clock::now();
            t_duration = std::chrono::duration_cast<std::chrono::duration<double> >(t_end - t_start_0).count();
            localMapping_times[int(1000 * t_duration)]++;

            #ifdef PROFILING_EXHAUSTIVE
            AF_PROFILE_BEGIN("Local Mapping Profiling");
            AF_PROFILE_FIELD(createNewMapPoints_times,          "  Create NewMap Points");
            AF_PROFILE_FIELD(searchInNeighbors_times,          "  Search in Neighbors");
            AF_PROFILE_FIELD(localbundleadjustment_times,          "  Local Bundle Adjustment");
            AF_PROFILE_FIELD(localMapping_times, "Local Mapping");
            AF_PROFILE_END();
            #endif
        }
        else if(Stop())
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            if(CheckFinish())
                break;
        }
        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true);

        if(CheckFinish())
            break;

        usleep(3000);
    }

    SetFinish();
}

void LocalMapping::insert_keyframe(Keyframe pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}


bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame()
{
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    mpCurrentKeyFrame->compute_global_descriptor();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    for(const auto& feat: mpCurrentKeyFrame->featureTypes){
        const vector<Pt> vpMapPointMatches = mpCurrentKeyFrame->get_map_point_matches(feat);

        for(size_t i=0; i<vpMapPointMatches.size(); i++)
        {
            Pt pMP = vpMapPointMatches[i];
            if(pMP)
            {
                if(!pMP->is_bad())
                {
                    if(!pMP->is_in_keyframe(mpCurrentKeyFrame))
                    {
                        pMP->add_observation(mpCurrentKeyFrame, i);
                    }
                    else // this can only happen for new stereo points inserted by the Tracking
                    {
                        mlpRecentAddedMapPoints.push_back(pMP);
                    }
                }
            }
        }
    }
    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->update_connections();

    // Insert Keyframe in Map
    mpMap->add_keyframe(mpCurrentKeyFrame);

}

void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<Pt>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->keyId;

    while(lit!=mlpRecentAddedMapPoints.end())
    {
        Pt pMP = *lit;
        if(pMP->is_bad())
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if((pMP->GetFoundRatio() < 0.25f ) && (pMP->featureType == mpCurrentKeyFrame->featureTypes[0]))
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->number_of_observations() <= MAP_POINT_CULLING_MIN_NUM_OBSERVATIONS)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}

mat3f LocalMapping::SkewSymmetricMatrix(const vec3f &v)
{
    mat3f M;
    M <<     0.0f   , -v(2),  v(1),
          v(2),    0.0f    , -v(0),
         -v(1), v(0) ,     0.0f;

    return M;
}

mat3f LocalMapping::ComputeF12(Keyframe &pKF1, Keyframe &pKF2)
{
    mat3f R1w = pKF1->get_rotation();
    vec3f t1w = pKF1->get_translation();
    mat3f R2w = pKF2->get_rotation();
    vec3f t2w = pKF2->get_translation();

    mat3f R12 = R1w * R2w.transpose();
    vec3f t12 = -R1w * R2w.transpose() * t2w + t1w;

    mat3f t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;


    return Converter::toMatrix3f(K1.t().inv()) * t12x * R12 * Converter::toMatrix3f(K2.inv());
}

void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    const vector <Keyframe> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(CREATE_NEW_MAP_POINTS_BEST_COVISIBILITY_KEYFRAMES);

    // Init Current Keyframe
    mat4f Twc1, Tcw1;
    mat3f Rwc1, Rcw1;
    vec3f twc1, tcw1;
    mpCurrentKeyFrame->getFullPose(Twc1, Rwc1, twc1, Tcw1, Rcw1, tcw1);

    float fx1, fy1, cx1, cy1, invfx1, invfy1;
    mpCurrentKeyFrame->getFullIntrinsics(fx1, fy1, cx1, cy1, invfx1, invfy1);

    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////
    const auto& fts = mpCurrentKeyFrame->featureTypes;
    const int NK = (int)vpNeighKFs.size();
    const int NF = (int)fts.size();

    std::vector<char> skipK(NK, 0);
    skipK.shrink_to_fit(); // optional

    // Decide which neighbors are cached (single-thread)
    for (int k = 0; k < NK; ++k) {
        auto pKF2 = vpNeighKFs[k];

        auto it = mpCurrentKeyFrame->cache_matched_pairs.find(pKF2->frame_id);
        if (it != mpCurrentKeyFrame->cache_matched_pairs.end() && !it->second.empty()) {
            skipK[k] = 1;
        }
    }

    std::vector<std::vector<std::vector<cv::DMatch>>> out(
        NK, std::vector<std::vector<cv::DMatch>>(NF)
    );

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int k = 0; k < NK; ++k) {
        for (int i = 0; i < NF; ++i) {
            if (skipK[k]) continue;

            auto pKF2 = vpNeighKFs[k];
            FeatureType ft = fts[i];

            // If some ft might be missing, you'd want find() guards here instead of at()
            out[k][i] = matcher->serialFeatureMatching(
                mpCurrentKeyFrame->descriptors.at(ft), pKF2->descriptors.at(ft),
                mpCurrentKeyFrame->keypoints.at(ft),     pKF2->keypoints.at(ft),
                ft
            );
        }
    }

    // Build stacked matches and store (serial)
    for (int k = 0; k < NK; ++k) {
        if (skipK[k]) continue;

        auto pKF2 = vpNeighKFs[k];
        std::vector<cv::DMatch> allMatches;

        int queryOffset = 0;
        int trainOffset = 0;

        auto& it1 = mpCurrentKeyFrame->cache_matched_pairs_feat_type[pKF2->frame_id];
        auto& it2 = pKF2->cache_matched_pairs_feat_type[mpCurrentKeyFrame->frame_id];

        for (int i = 0; i < NF; ++i) {
            FeatureType ft = fts[i];

            const int nq = (int)mpCurrentKeyFrame->keypoints.at(ft).size();
            const int nt = (int)pKF2->keypoints.at(ft).size();

            auto& m = out[k][i];
            allMatches.reserve(allMatches.size() + m.size());

            for (const auto& d : m) {
                cv::DMatch dd = d;
                dd.queryIdx += queryOffset;
                dd.trainIdx += trainOffset;

                it1[ft].push_back(d);
                it2[ft].push_back(cv::DMatch(d.trainIdx, d.queryIdx, d.distance));
                allMatches.push_back(dd);
            }
            queryOffset += nq;
            trainOffset += nt;
        }

        pKF2->cache_matched_pairs.insert_or_assign(mpCurrentKeyFrame->frame_id, FeatureMatcher::swap_match_direction(allMatches));
        mpCurrentKeyFrame->cache_matched_pairs.insert_or_assign(pKF2->frame_id, std::move(allMatches));
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////
    // Search matches with epipolar restriction and triangulate
    std::map<FeatureType, int> newMapPoints;
    int nFromSensor{0};
    int nFromTriangulation{0};
    int j{0};
    std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();
    for(size_t i{0}; i < vpNeighKFs.size(); i++)
    {
        // Init Second Keyframe
        Keyframe  pKF2 = vpNeighKFs[i];
        mat4f Twc2, Tcw2;
        mat3f Rwc2, Rcw2;
        vec3f twc2, tcw2;
        pKF2->getFullPose(Twc2, Rwc2, twc2, Tcw2, Rcw2, tcw2);

        float fx2, fy2, cx2, cy2, invfx2, invfy2;
        pKF2->getFullIntrinsics(fx2, fy2, cx2, cy2, invfx2, invfy2);

        // Check first that baseline is not too short
        vec3f vBaseline = twc2 - twc1;
        const float baseline = vBaseline.norm();

        const float medianDepthKF2 = pKF2->compute_scene_median_depth(2);
        const float ratioBaselineDepth = baseline/medianDepthKF2;

        if(ratioBaselineDepth < CREATE_NEW_MAP_POINTS_RATIO_BASELINE_DEPTH)
            continue;

        // Search matches that fullfil epipolar constraint
        std::map<FeatureType, vector<pair<size_t,size_t>>> vMatchedIndices;
        std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
        double t_duration = std::chrono::duration_cast<std::chrono::duration<double> >(t_end - t_start).count();

        #ifdef ALLFEATURE_REAL_TIME
        if ((j <= 1) || (t_duration < 10.05))
            matcher->match_keyframes_for_triangulation(mpCurrentKeyFrame, pKF2, vMatchedIndices, mpCurrentKeyFrame->featureTypes);
        ++j;
        #else
        matcher->match_keyframes_for_triangulation(mpCurrentKeyFrame, pKF2, vMatchedIndices, mpCurrentKeyFrame->featureTypes);
        #endif

        for(auto& [featureType, N_]: pKF2->N){
            // Triangulate each
            auto it = vMatchedIndices.find(featureType);
            if(it == vMatchedIndices.end())
                continue;
            const int nmatches = vMatchedIndices.at(featureType).size();
            for(int ikp{0}; ikp < nmatches; ikp++)
            {
                const int &idx1 = vMatchedIndices.at(featureType)[ikp].first;
                const int &idx2 = vMatchedIndices.at(featureType)[ikp].second;

                const cv::KeyPoint &kp1 = mpCurrentKeyFrame->keypoints.at(featureType)[idx1];
                const cv::KeyPoint &kp2 = pKF2->keypoints.at(featureType)[idx2];

                // Check parallax between rays
                vec3f xn1{(kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0f};
                vec3f xn2{(kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0f};

                vec3f ray1 = Rwc1 * xn1;
                vec3f ray2 = Rwc2 * xn2;
                const float cosParallaxRays = ray1.dot(ray2)/(ray1.norm() * ray2.norm());
                //const float sinParallaxRays = ray1.cross(ray2).norm() / (ray1.norm() * ray2.norm());

                vec3f x3D;
                bool fromSensorDepth = false;
                const float invDepth1 = mpCurrentKeyFrame->inv_depth.at(featureType)[idx1];
                const float invDepth2 = pKF2->inv_depth.at(featureType)[idx2];
                const bool haveDepth1 = invDepth1 > 0.0f;
                const bool haveDepth2 = invDepth2 > 0.0f;
                //if(true)
                //if(cosParallaxRays > 0 && (sinParallaxRays > sinThr))
                if(haveDepth1 && haveDepth2)
                {
                    // Two independent sensor depth readings of the same point: back-project
                    // each from its own keyframe and average. The reprojection-error checks
                    // below now validate agreement between them in both views.
                    vec3f x3D_1 = Rwc1 * (xn1 * (1.0f / invDepth1)) + twc1;
                    vec3f x3D_2 = Rwc2 * (xn2 * (1.0f / invDepth2)) + twc2;
                    x3D = 0.5f * (x3D_1 + x3D_2);
                    fromSensorDepth = true;
                }
                else if(haveDepth1)
                {
                    // Back-project from KF1's own measured depth: no second-view triangulation,
                    // no parallax requirement. The reprojection-error check against KF2 below
                    // still cross-validates it.
                    x3D = Rwc1 * (xn1 * (1.0f / invDepth1)) + twc1;
                    fromSensorDepth = true;
                }
                else if(haveDepth2)
                {
                    // Symmetric case: back-project from KF2's depth, cross-validated against KF1.
                    x3D = Rwc2 * (xn2 * (1.0f / invDepth2)) + twc2;
                    fromSensorDepth = true;
                }
                else if(cosParallaxRays > 0 && (cosParallaxRays < CREATE_NEW_MAP_POINTS_MIN_COS))
                {
                    Eigen::Matrix<float, 4, 4> A;
                        A.row(0) = xn1(0) * Tcw1.row(2) - Tcw1.row(0);
                        A.row(1) = xn1(1) * Tcw1.row(2) - Tcw1.row(1);
                        A.row(2) = xn2(0) * Tcw2.row(2) - Tcw2.row(0);
                        A.row(3) = xn2(1) * Tcw2.row(2) - Tcw2.row(1);

                    Eigen::JacobiSVD<Eigen::Matrix<float,4,4>> svd(
                        A, Eigen::ComputeFullV
                    );

                    const Eigen::Matrix<float,4,4>& V = svd.matrixV();
                    Eigen::Vector4f x_h = V.col(3);

                    const float w = x_h(3);
                    if (std::abs(w) < 1e-12f)
                        continue;

                    x3D = x_h.head<3>() / w;
                }
                else
                    continue; //No depth, and no stereo / very low parallax

                //Check triangulation in front of cameras
                float z1 = Rcw1.row(2).dot(x3D) + tcw1(2);
                if(z1<=0)
                    continue;

                float z2 = Rcw2.row(2).dot(x3D) + tcw2(2);
                if(z2<=0)
                    continue;

                //Check reprojection error in first keyframe
                const float &sigmaSquare1 = mpCurrentKeyFrame->GetKeyPt1DSigma2(idx1, featureType);
                const float x1 = Rcw1.row(0).dot(x3D) + tcw1(0);
                const float y1 = Rcw1.row(1).dot(x3D) + tcw1(1);
                const float invz1 = 1.0f / z1;


                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if((errX1*errX1+errY1*errY1) > CHI2_2DOF * sigmaSquare1)
                    continue;

                // Check reprojection error in second keyframe
                const float sigmaSquare2 = pKF2->GetKeyPt1DSigma2(idx2, featureType);
                const float x2 = Rcw2.row(0).dot(x3D) + tcw2(0);
                const float y2 = Rcw2.row(1).dot(x3D) + tcw2(1);
                const float invz2 = 1.0f / z2;

                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2) > CHI2_2DOF * sigmaSquare2)
                    continue;

                // Check scale consistency
                vec3f normal1 = x3D - twc1;
                float dist1 = normal1.norm();

                vec3f normal2 = x3D - twc2;
                float dist2 = normal2.norm();

                if(dist1 == 0 || dist2 == 0)
                    continue;

                // Triangulation is succesfull
                newMapPoints[featureType]++;
                if(fromSensorDepth)
                    nFromSensor++;
                else
                    nFromTriangulation++;
                Pt pMP = mpCurrentKeyFrame->create_monocular_map_point(x3D, KeypointIndex(idx1),
                                                                    pKF2,  KeypointIndex(idx2),
                                                                    featureType);
                mlpRecentAddedMapPoints.push_back(pMP);
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////
    // Back-project close, still-unmatched keypoints straight from mpCurrentKeyFrame's own depth
    // reading. The loop above only creates points for keypoints that found a 2D feature match
    // against a covisible neighbor (match_keyframes_for_triangulation) -- a keypoint with a
    // perfectly good sensor-depth reading but no such match (textureless region, repeated
    // pattern, fast motion) was previously silently dropped, even though depth alone already
    // places it in the map with no matching required at all. Matches the existing haveDepth1/
    // haveDepth2 branch above: any inv_depth>0 is trusted, no mThDepth range gate.
    for(const auto& featureType : fts)
    {
        const auto& invDepthFt = mpCurrentKeyFrame->inv_depth.at(featureType);
        const auto& keypointsFt = mpCurrentKeyFrame->keypoints.at(featureType);

        for(size_t idx = 0; idx < invDepthFt.size(); idx++)
        {
            if(invDepthFt[idx] <= 0.0f)
                continue; // no valid sensor depth at this keypoint

            if(mpCurrentKeyFrame->get_map_point(idx, featureType))
                continue; // already has a map point (from tracking, or the matched-pairs loop above)

            const cv::KeyPoint& kp1 = keypointsFt[idx];
            vec3f xn1{(kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0f};
            vec3f x3D = Rwc1 * (xn1 * (1.0f / invDepthFt[idx])) + twc1;

            newMapPoints[featureType]++;
            nFromSensor++;
            Pt pMP = mpCurrentKeyFrame->CreateMapPoint(x3D, KeypointIndex(idx), featureType);
            mlpRecentAddedMapPoints.push_back(pMP);
        }
    }

    //cout << "[LocalMapping::CreateNewMapPoints] New map points: " << (nFromSensor + nFromTriangulation)
         //<< " (sensor: " << nFromSensor << ", triangulation: " << nFromTriangulation << ")" << endl;
}

void LocalMapping::SearchInNeighbors(const FeatureType& featureType)
{
    // Retrieve neighbor keyframes
    const vector<Keyframe > vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(SEARCH_IN_NEIGHBORS_NUM_KEYFRAMES);
    vector<Keyframe > vpTargetKFs;
    for(vector<Keyframe >::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        Keyframe  pKFi = *vit;
        if(pKFi->is_bad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->keyId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->keyId;

        // Extend to some second neighbors
        const vector<Keyframe > vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(SEARCH_IN_NEIGHBORS_NUM_KEYFRAMES_SECOND);
        for(vector<Keyframe >::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            Keyframe  pKFi2 = *vit2;
            if(pKFi2->is_bad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->keyId || pKFi2->keyId == mpCurrentKeyFrame->keyId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }

    // Search matches by projection from current KF in target KFs
    //FeatureMatcher matcher;
    vector<Pt> vpMapPointMatches = mpCurrentKeyFrame->get_map_point_matches(featureType);
    for(vector<Keyframe >::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        Keyframe  pKFi = *vit;
        matcher->fuse_map_points_to_keyframe(pKFi,vpMapPointMatches, SEARCH_IN_NEIGHBORS_RADIUS_TH, featureType);
    }

    // Search matches by projection from target KFs in current KF
    vector<Pt> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    for(vector<Keyframe >::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        Keyframe  pKFi = *vitKF;

        vector<Pt> vpMapPointsKFi = pKFi->get_map_point_matches(featureType);

        for(vector<Pt>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            Pt pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->is_bad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->keyId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->keyId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher->fuse_map_points_to_keyframe(mpCurrentKeyFrame,vpFuseCandidates, SEARCH_IN_NEIGHBORS_RADIUS_TH, featureType);

    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->get_map_point_matches(featureType);
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        Pt pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->is_bad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }
    // Update connections in covisibility graph
    mpCurrentKeyFrame->update_connections();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::accepts_keyframes()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling()
{

    // vector<Keyframe > vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();
    // for(vector<Keyframe >::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    // {
    //     Keyframe  pKF = *vit;
    //     if(pKF->keyId == 0)
    //         continue;
    //     if ((int(pKF->frame_id) % ALLFEATURE_EVALUATION) == 0)
    //         continue;
    //     if ((int(pKF->frame_id) % ALLFEATURE_KEYFRAMES) == 0)
    //         continue;
    //     if(mpCurrentKeyFrame->keyId - pKF->keyId < 3)
    //         continue;
    //     pKF->SetBadFlag();
    // }


    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points

    vector<Keyframe > vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    for(vector<Keyframe >::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        Keyframe  pKF = *vit;
        for(const auto feat: pKF->featureTypes){
            if(pKF->keyId == 0)
                continue;
            const vector<Pt> vpMapPoints = pKF->get_map_point_matches(feat);

            int nObs = KEYFRAME_CULLING_MIN_NUM_OBSERVATIONS;
            const int thObs=nObs;
            int nRedundantObservations=0;
            int nMPs=0;
            for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
            {
                Pt pMP = vpMapPoints[i];

                if(pMP)
                {
                    if(!pMP->is_bad())
                    {
                        nMPs++;
                        if(pMP->number_of_observations() > thObs)
                        {
                            const map<KeyframeId , Obs> observations = pMP->GetObservations();
                            int nObs=0;
                            for(auto& obs: observations)
                            {
                                Keyframe keyframe_i = obs.second->projKeyframe;
                                if(keyframe_i->keyId == pKF->keyId)
                                    continue;
                                if(keyframe_i->is_bad())
                                    continue;

                                nObs++;
                                if(nObs>=thObs)
                                    break;
                            }
                            if(nObs>=thObs)
                            {
                                nRedundantObservations++;
                            }
                        }
                    }
                }
            }

            if(nRedundantObservations > KEYFRAME_CULLING_COVISIBILITY_THRESHOLD * nMPs){
                #ifdef ALLFEATURE_EVALUATION

                    if ((int(pKF->frame_id) % ALLFEATURE_EVALUATION) != 0){
                        if ((int(pKF->frame_id) % ALLFEATURE_MAX_KEYFRAMES) != 0)
                            pKF->SetBadFlag();
                    }
                #else
                    pKF->SetBadFlag();
                #endif

            }
    }
    }
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
}

void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested=false;

        localMapping_times.clear();
        createNewMapPoints_times.clear();
        searchInNeighbors_times.clear();
        localbundleadjustment_times.clear();
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} //namespace ORB_SLAM
