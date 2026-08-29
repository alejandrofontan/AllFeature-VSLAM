#include "LocalMapping.h"
#include "LoopClosing.h"
#include "FeatureMatcher.h"
#include "Optimizer.h"
#include "Converter.h"
#include "Utils.h"
#include "afvslam_log.hpp"

#include <mutex>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <unordered_set>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

namespace AF_VSLAM
{

LocalMappingParameters LocalMapping::params{};

void LocalMapping::LoadParameters(const cv::FileStorage &fSettings)
{
    auto read_if_present = [&fSettings](const char* key, auto& field)
    {
        const cv::FileNode node = fSettings[key];
        if(!node.empty())
            node >> field;
    };

    read_if_present("LocalMapping.KeyframeCullingMethod", params.keyframe_culling_method);
    if(params.keyframe_culling_method != "heuristic" && params.keyframe_culling_method != "information")
        AF_ERROR("[LocalMapping] Unknown LocalMapping.KeyframeCullingMethod '" + params.keyframe_culling_method + "' (options: heuristic, information)");

    read_if_present("LocalMapping.KeyframeCullingRedundancyRatio", params.keyframe_culling_redundancy_ratio);
    read_if_present("LocalMapping.KeyframeCullingMinObservations", params.keyframe_culling_min_observations);

    float max_unexplained = params.keyframe_culling_max_unexplained.load();
    read_if_present("LocalMapping.KeyframeCullingMaxUnexplained", max_unexplained);
    params.keyframe_culling_max_unexplained.store(max_unexplained);
    read_if_present("LocalMapping.KeyframeCullingMinAge", params.keyframe_culling_min_age);
    read_if_present("LocalMapping.KeyframeCullingMinKeyframes", params.keyframe_culling_min_keyframes);
    read_if_present("LocalMapping.KeyframeCullingScope", params.keyframe_culling_scope);
    if(params.keyframe_culling_scope != "map" && params.keyframe_culling_scope != "local")
        AF_ERROR("[LocalMapping] Unknown LocalMapping.KeyframeCullingScope '" + params.keyframe_culling_scope + "' (options: map, local)");
    read_if_present("LocalMapping.KeyframeCullingMaxPerCall", params.keyframe_culling_max_per_call);
    int centred = params.keyframe_culling_centred ? 1 : 0;   // cv::FileStorage has no bool reader
    read_if_present("LocalMapping.KeyframeCullingCentred", centred);
    params.keyframe_culling_centred = (centred != 0);
}

LocalMapping::LocalMapping(shared_ptr<Map> pMap, const float bMonocular, const vector<FeatureType>& featureTypes, const int& image_width, const int& image_height):
    featureTypes(featureTypes), mpMap(pMap),
    abort_ba_(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), accept_keyframes_(true),
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
        set_accept_keyframes(false);

        // Check if there are keyframes in the queue
        if(has_new_keyframes())
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

            if(!has_new_keyframes())
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                t_start = std::chrono::steady_clock::now();
                for (const auto& [feat, N_] : current_keyframe_->N) {
                    SearchInNeighbors(feat);
                }
                t_end = std::chrono::steady_clock::now();
                t_duration = std::chrono::duration_cast<std::chrono::duration<double> >(t_end - t_start).count();
                searchInNeighbors_times[int(1000 * t_duration)]++;
            }
            abort_ba_ = false;
            //if(!has_new_keyframes() && !is_stop_requested())
            if(!has_new_keyframes())
            {
                // Local BA
                if(mpMap->keyframes_in_map()>2){
                    t_start = std::chrono::steady_clock::now();
                    Optimizer::LocalBundleAdjustment(current_keyframe_,&abort_ba_, mpMap);
                    t_end = std::chrono::steady_clock::now();
                    t_duration = std::chrono::duration_cast<std::chrono::duration<double> >(t_end - t_start).count();
                    localbundleadjustment_times[int(1000 * t_duration)]++;
                }
                // Check redundant local Keyframes
                cull_keyframes();
            }
            loopCloser->insert_keyframe(current_keyframe_);

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
            while(is_stopped() && !CheckFinish())
            {
                usleep(3000);
            }
            if(CheckFinish())
                break;
        }
        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        set_accept_keyframes(true);

        if(CheckFinish())
            break;

        usleep(3000);
    }

    SetFinish();
}

void LocalMapping::insert_keyframe(const Keyframe& keyframe)
{
    std::lock_guard<std::mutex> lock(new_keyframes_mutex_);
    new_keyframes_.push_back(keyframe);
    abort_ba_ = true;
}

bool LocalMapping::has_new_keyframes() const
{
    std::lock_guard<std::mutex> lock(new_keyframes_mutex_);
    return !new_keyframes_.empty();
}

void LocalMapping::ProcessNewKeyFrame()
{
    {
        unique_lock<mutex> lock(new_keyframes_mutex_);
        current_keyframe_ = new_keyframes_.front();
        new_keyframes_.pop_front();
    }

    // Compute Bags of Words structures
    current_keyframe_->compute_global_descriptor();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    for(const auto& feat: current_keyframe_->featureTypes){
        const vector<Pt> vpMapPointMatches = current_keyframe_->get_map_point_matches(feat);

        for(size_t i=0; i<vpMapPointMatches.size(); i++)
        {
            Pt pMP = vpMapPointMatches[i];
            if(pMP)
            {
                if(!pMP->is_bad())
                {
                    if(!pMP->is_in_keyframe(current_keyframe_))
                    {
                        pMP->add_observation(current_keyframe_, i);
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
    current_keyframe_->update_connections();

    // Insert Keyframe in Map
    mpMap->add_keyframe(current_keyframe_);

    // Extend the online keyframe VPR matrix with the new keyframe (no-op without a descriptor)
    grow_keyframe_vpr_matrix(current_keyframe_);
}

// ------------------------------------------------------------------------------------------
// Online keyframe VPR similarity matrix
// ------------------------------------------------------------------------------------------

Eigen::MatrixXf LocalMapping::keyframe_vpr_matrix() const
{
    unique_lock<mutex> lock(vpr_mutex_);
    return keyframe_vpr_matrix_;
}

std::vector<Keyframe> LocalMapping::keyframe_vpr_order() const
{
    unique_lock<mutex> lock(vpr_mutex_);
    return vpr_keyframes_;
}

bool LocalMapping::has_keyframe_vpr_matrix() const
{
    unique_lock<mutex> lock(vpr_mutex_);
    return !vpr_keyframes_.empty();
}

void LocalMapping::grow_keyframe_vpr_matrix(const Keyframe& keyframe)
{
    // The descriptor is computed by compute_global_descriptor() at the top of
    // ProcessNewKeyFrame; it is empty when the VPR backend is not image-based (bow/none).
    const std::vector<float>& descriptor = keyframe->global_descriptor;
    if(descriptor.empty())
        return;
    unique_lock<mutex> lock(vpr_mutex_);
    const int k = int(vpr_keyframes_.size());
    keyframe_vpr_matrix_.conservativeResize(k + 1, k + 1);
    for(int i = 0; i < k; i++){
        const std::vector<float>& other = vpr_keyframes_[i]->global_descriptor;
        float s = std::numeric_limits<float>::quiet_NaN();
        if(other.size() == descriptor.size()){
            double dot = 0.0;
            for(size_t d = 0; d < descriptor.size(); d++)
                dot += double(descriptor[d]) * other[d];
            s = float(dot);   // unit descriptors: dot product == cosine
        }
        keyframe_vpr_matrix_(i, k) = s;
        keyframe_vpr_matrix_(k, i) = s;
    }
    keyframe_vpr_matrix_(k, k) = 1.0f;
    vpr_keyframes_.push_back(keyframe);
}

void LocalMapping::print_keyframe_vpr_matrix() const
{
    std::vector<Keyframe> keyframes;
    Eigen::MatrixXf matrix;
    {
        unique_lock<mutex> lock(vpr_mutex_);
        keyframes = vpr_keyframes_;
        matrix = keyframe_vpr_matrix_;
    }
    const int k = int(keyframes.size());
    if(k == 0)
        return;
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    out << "[VPR] keyframe similarity matrix (cosine): " << k << " keyframes (rows/cols in insertion order, * = culled keyframe)\n";
    out << "        ";
    for(int j = 0; j < k; j++)
        out << std::setw(7) << (std::string(keyframes[j]->is_bad() ? "*" : "") + std::to_string(keyframes[j]->keyId));
    out << "\n";
    for(int i = 0; i < k; i++){
        out << std::setw(7) << (std::string(keyframes[i]->is_bad() ? "*" : "") + std::to_string(keyframes[i]->keyId)) << " ";
        for(int j = 0; j < k; j++){
            const float d = matrix(i, j);
            if(std::isnan(d)) out << std::setw(7) << "nan";
            else out << std::setw(7) << d;
        }
        out << "\n";
    }
    std::cout << out.str() << std::flush;
}

void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<Pt>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = current_keyframe_->keyId;

    while(lit!=mlpRecentAddedMapPoints.end())
    {
        Pt pMP = *lit;
        if(pMP->is_bad())
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if((pMP->GetFoundRatio() < 0.25f ) && (pMP->featureType == current_keyframe_->featureTypes[0]))
        {
            pMP->set_bad_flag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->number_of_observations() <= MAP_POINT_CULLING_MIN_NUM_OBSERVATIONS)
        {
            pMP->set_bad_flag();
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


    return Converter::to_matrix3f(K1.t().inv()) * t12x * R12 * Converter::to_matrix3f(K2.inv());
}

void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    const vector <Keyframe> vpNeighKFs = current_keyframe_->get_best_covisibility_keyframes(CREATE_NEW_MAP_POINTS_BEST_COVISIBILITY_KEYFRAMES);

    // Init Current Keyframe
    mat4f Twc1, Tcw1;
    mat3f Rwc1, Rcw1;
    vec3f twc1, tcw1;
    current_keyframe_->getFullPose(Twc1, Rwc1, twc1, Tcw1, Rcw1, tcw1);

    float fx1, fy1, cx1, cy1, invfx1, invfy1;
    current_keyframe_->getFullIntrinsics(fx1, fy1, cx1, cy1, invfx1, invfy1);

    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////
    const auto& fts = current_keyframe_->featureTypes;
    const int NK = (int)vpNeighKFs.size();
    const int NF = (int)fts.size();

    std::vector<char> skipK(NK, 0);
    skipK.shrink_to_fit(); // optional

    // Decide which neighbors are cached (single-thread)
    for (int k = 0; k < NK; ++k) {
        auto pKF2 = vpNeighKFs[k];

        auto it = current_keyframe_->cache_matched_pairs.find(pKF2->frame_id);
        if (it != current_keyframe_->cache_matched_pairs.end() && !it->second.empty()) {
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
                current_keyframe_->descriptors.at(ft), pKF2->descriptors.at(ft),
                current_keyframe_->keypoints.at(ft),     pKF2->keypoints.at(ft),
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

        auto& it1 = current_keyframe_->cache_matched_pairs_feat_type[pKF2->frame_id];
        auto& it2 = pKF2->cache_matched_pairs_feat_type[current_keyframe_->frame_id];

        for (int i = 0; i < NF; ++i) {
            FeatureType ft = fts[i];

            const int nq = (int)current_keyframe_->keypoints.at(ft).size();
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

        pKF2->cache_matched_pairs.insert_or_assign(current_keyframe_->frame_id, FeatureMatcher::swap_match_direction(allMatches));
        current_keyframe_->cache_matched_pairs.insert_or_assign(pKF2->frame_id, std::move(allMatches));
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
            matcher->match_keyframes_for_triangulation(current_keyframe_, pKF2, vMatchedIndices, current_keyframe_->featureTypes);
        ++j;
        #else
        matcher->match_keyframes_for_triangulation(current_keyframe_, pKF2, vMatchedIndices, current_keyframe_->featureTypes);
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

                const cv::KeyPoint &kp1 = current_keyframe_->keypoints.at(featureType)[idx1];
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
                const float invDepth1 = current_keyframe_->inv_depth.at(featureType)[idx1];
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
                const float &sigmaSquare1 = current_keyframe_->GetKeyPt1DSigma2(idx1, featureType);
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
                Pt pMP = current_keyframe_->create_monocular_map_point(x3D, KeypointIndex(idx1),
                                                                    pKF2,  KeypointIndex(idx2),
                                                                    featureType);
                mlpRecentAddedMapPoints.push_back(pMP);
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////
    // Back-project close, still-unmatched keypoints straight from current_keyframe_'s own depth
    // reading. The loop above only creates points for keypoints that found a 2D feature match
    // against a covisible neighbor (match_keyframes_for_triangulation) -- a keypoint with a
    // perfectly good sensor-depth reading but no such match (textureless region, repeated
    // pattern, fast motion) was previously silently dropped, even though depth alone already
    // places it in the map with no matching required at all. Matches the existing haveDepth1/
    // haveDepth2 branch above: any inv_depth>0 is trusted, no mThDepth range gate.
    for(const auto& featureType : fts)
    {
        const auto& invDepthFt = current_keyframe_->inv_depth.at(featureType);
        const auto& keypointsFt = current_keyframe_->keypoints.at(featureType);

        for(size_t idx = 0; idx < invDepthFt.size(); idx++)
        {
            if(invDepthFt[idx] <= 0.0f)
                continue; // no valid sensor depth at this keypoint

            if(current_keyframe_->get_map_point(idx, featureType))
                continue; // already has a map point (from tracking, or the matched-pairs loop above)

            const cv::KeyPoint& kp1 = keypointsFt[idx];
            vec3f xn1{(kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0f};
            vec3f x3D = Rwc1 * (xn1 * (1.0f / invDepthFt[idx])) + twc1;

            newMapPoints[featureType]++;
            nFromSensor++;
            Pt pMP = current_keyframe_->create_map_point(x3D, KeypointIndex(idx), featureType);
            mlpRecentAddedMapPoints.push_back(pMP);
        }
    }

    //cout << "[LocalMapping::CreateNewMapPoints] New map points: " << (nFromSensor + nFromTriangulation)
         //<< " (sensor: " << nFromSensor << ", triangulation: " << nFromTriangulation << ")" << endl;
}

void LocalMapping::SearchInNeighbors(const FeatureType& featureType)
{
    // Retrieve neighbor keyframes
    const vector<Keyframe > vpNeighKFs = current_keyframe_->get_best_covisibility_keyframes(SEARCH_IN_NEIGHBORS_NUM_KEYFRAMES);
    vector<Keyframe > vpTargetKFs;
    for(vector<Keyframe >::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        Keyframe  pKFi = *vit;
        if(pKFi->is_bad() || pKFi->mnFuseTargetForKF == current_keyframe_->keyId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = current_keyframe_->keyId;

        // Extend to some second neighbors
        const vector<Keyframe > vpSecondNeighKFs = pKFi->get_best_covisibility_keyframes(SEARCH_IN_NEIGHBORS_NUM_KEYFRAMES_SECOND);
        for(vector<Keyframe >::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            Keyframe  pKFi2 = *vit2;
            if(pKFi2->is_bad() || pKFi2->mnFuseTargetForKF==current_keyframe_->keyId || pKFi2->keyId == current_keyframe_->keyId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }

    // Search matches by projection from current KF in target KFs
    //FeatureMatcher matcher;
    vector<Pt> vpMapPointMatches = current_keyframe_->get_map_point_matches(featureType);
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
            if(pMP->is_bad() || pMP->mnFuseCandidateForKF == current_keyframe_->keyId)
                continue;
            pMP->mnFuseCandidateForKF = current_keyframe_->keyId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher->fuse_map_points_to_keyframe(current_keyframe_,vpFuseCandidates, SEARCH_IN_NEIGHBORS_RADIUS_TH, featureType);

    // Update points
    vpMapPointMatches = current_keyframe_->get_map_point_matches(featureType);
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
    current_keyframe_->update_connections();
}

void LocalMapping::request_stop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(new_keyframes_mutex_);
    abort_ba_ = true;
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

bool LocalMapping::is_stopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::is_stop_requested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    new_keyframes_.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::accepts_keyframes() const
{
    std::lock_guard<std::mutex> lock(accept_mutex_);
    return accept_keyframes_;
}

void LocalMapping::set_accept_keyframes(const bool accept)
{
    std::lock_guard<std::mutex> lock(accept_mutex_);
    accept_keyframes_ = accept;
}

bool LocalMapping::set_insertion_lock(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    abort_ba_ = true;
}

void LocalMapping::cull_keyframes()
{
    if(params.keyframe_culling_method == "information"){
        // The information method needs the online kernel, i.e. an image-embedding VPR backend
        // (vpr: megaloc). Without descriptors, degrade to the heuristic once, loudly.
        if(current_keyframe_->global_descriptor.empty() && !has_keyframe_vpr_matrix()){
            static bool warned{false};
            if(!warned){
                AF_WARN("[LocalMapping] KeyframeCullingMethod: information requested but keyframes carry no global "
                        "descriptor (vpr is not megaloc) — falling back to the heuristic culling; set vpr: megaloc to use it");
                warned = true;
            }
            cull_keyframes_heuristic();
            return;
        }
        cull_keyframes_information();
        return;
    }
    cull_keyframes_heuristic();
}

void LocalMapping::cull_keyframes_heuristic()
{
    // A covisible keyframe is redundant when more than
    // params.keyframe_culling_redundancy_ratio of its (non-bad) map points are each
    // observed by at least params.keyframe_culling_min_observations OTHER non-bad
    // keyframes. The test runs per feature type, and a keyframe redundant for ANY
    // of its feature types is culled. (Stock ORB-SLAM2's extra requirement that the
    // other observations be at the same or a finer scale level is not applied.)

    // By-value snapshot: get_covisible_keyframes copies under the keyframe's mutex
    const std::vector<Keyframe> local_keyframes = current_keyframe_->get_covisible_keyframes();
    for(const Keyframe& keyframe : local_keyframes){
        if(keyframe->keyId == 0)
            continue; // the first keyframe anchors the map and is never culled

        for(const FeatureType ft : keyframe->featureTypes){
            // By-value snapshot: get_map_point_matches copies under the keyframe's mutex
            const std::vector<Pt> map_points = keyframe->get_map_point_matches(ft);

            int num_points = 0;
            int num_redundant = 0;
            for(const Pt& pt : map_points){
                if(!pt || pt->is_bad())
                    continue;
                num_points++;

                // number_of_observations counts depth-verified observations twice
                if(pt->number_of_observations() <= params.keyframe_culling_min_observations)
                    continue;

                // By-value snapshot: get_observations copies under the point's mutex
                const std::map<KeyframeId, Obs> observations = pt->get_observations();
                int num_other_observers = 0;
                for(const auto& [key_id, obs] : observations){
                    if(key_id == keyframe->keyId || obs->projKeyframe->is_bad())
                        continue;
                    if(++num_other_observers >= params.keyframe_culling_min_observations)
                        break;
                }
                if(num_other_observers >= params.keyframe_culling_min_observations)
                    num_redundant++;
            }

            if(num_redundant <= params.keyframe_culling_redundancy_ratio * static_cast<float>(num_points))
                continue;

            bool protected_keyframe{false};
#ifdef ALLFEATURE_EVALUATION
            // Evaluation builds keep the evaluation-sampled and the cadence keyframes
            protected_keyframe = (keyframe->frame_id % ALLFEATURE_EVALUATION) == 0
                              || (keyframe->frame_id % ALLFEATURE_MAX_KEYFRAMES) == 0;
#endif
            if(!protected_keyframe)
                keyframe->set_bad_flag();
        }
    }
}

void LocalMapping::cull_keyframes_information()
{
    // Show the online keyframe VPR matrix as it grows (debugging aid)
    // print_keyframe_vpr_matrix();

    // Joint-information culling. Let K be the (cosine, PSD) similarity kernel over keyframes,
    // A the alive keyframes and H the keyframes culled earlier (rows kept in the online matrix).
    // Under a Gaussian model the information of keyframe x NOT explained by the alive set is its
    // conditional variance  v_x = K_xx - k_xA K_AA^-1 k_Ax  (in [0,1]; exp(-2 I(x; A))).
    // For an alive keyframe i that is v_i = 1 / (K_AA^-1)_ii, its unique information given the
    // other alive keyframes; removing i raises every v_h by W_hi^2 / M_ii with W = K_HA K_AA^-1
    // and M = K_AA^-1. Greedy rule: cull the alive keyframe with the smallest v_i among those
    // for which, after the cull, every keyframe ever inserted (the culled ones AND i itself)
    // stays at most tau = keyframe_culling_max_unexplained unexplained; stop when none qualifies.
    // M and W are updated by rank-one Schur downdates, so a cull costs O(|H||A| + |A|^2).
    // KF0, the current keyframe, keyframes younger than keyframe_culling_min_age and (in
    // evaluation builds) the sampled keyframes are never culled but still act as explainers.
    //
    // KERNEL (keyframe_culling_centred): MegaLoc descriptors share a large common-mode
    // component (unrelated places still score ~0.35-0.4), which compresses every v_i and makes
    // tau over-sensitive. Double-centring the Gram matrix over all keyframes inserted so far,
    //     K_c = J S J,  J = I - 11^T/n,   C_ij = K_c_ij / sqrt(K_c_ii K_c_jj)
    // is exactly the correlation of the mean-centred descriptors: unrelated pairs move to ~0,
    // near-duplicates stay ~0.85, the kernel stays PSD (rank n-1, hence the diagonal jitter).
    // The centring set grows with the map, so the kernel drifts slightly as keyframes arrive.
    //
    // ONLINE THRESHOLD CHANGES: culling is irreversible, so the history invariant v_h <= tau
    // only holds for the tau in force when h was culled. If tau is LOWERED afterwards, rows
    // with v_h > tau would make every candidate infeasible under a plain "v_h + price <= tau"
    // test and jam the culler. The constraint is therefore relative: a cull may not raise any
    // history row by more than max(tau - v_h, slack), i.e. rows within budget behave as before
    // and rows already over budget only protect their actual explainers (slack absorbs the
    // dense-W numerical dust of unrelated candidates). RAISING tau would otherwise cull
    // everything newly feasible in one burst; keyframe_culling_max_per_call spreads that over
    // successive calls.
    //
    // SCOPE (keyframe_culling_scope): "map" marginalises over every alive keyframe of the map;
    // "local" over the current keyframe and its covisible keyframes only (the classic local
    // window), with candidates drawn from that window and the history reduced to the culled
    // keyframes whose best alive explainer lies in it (so far-away history cannot veto a local
    // cull, and far-away keyframes cannot explain a local one).

    std::vector<Keyframe> keyframes;
    Eigen::MatrixXf similarity;
    {
        unique_lock<mutex> lock(vpr_mutex_);
        keyframes = vpr_keyframes_;
        similarity = keyframe_vpr_matrix_;
    }
    const int n = int(keyframes.size());
    if(n < 3)
        return;

    // Usable keyframes: those with a complete similarity row (a descriptor-size mismatch leaves NaN)
    std::vector<int> alive, history;
    std::vector<char> usable(n, 1);
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            if(std::isnan(similarity(i, j))) { usable[i] = 0; break; }

    if(params.keyframe_culling_centred){
        // Double-centre over the usable set, then renormalise to unit diagonal (in double)
        std::vector<int> u;
        for(int i = 0; i < n; i++) if(usable[i]) u.push_back(i);
        const int m = int(u.size());
        if(m >= 3){
            Eigen::MatrixXd S(m, m);
            for(int a = 0; a < m; a++)
                for(int b = 0; b < m; b++)
                    S(a, b) = double(similarity(u[a], u[b]));
            const Eigen::VectorXd row_mean = S.rowwise().mean();
            const double total_mean = row_mean.mean();
            Eigen::MatrixXd C = S;
            C.colwise() -= row_mean;
            C.rowwise() -= row_mean.transpose();
            C.array() += total_mean;
            Eigen::VectorXd d = C.diagonal().cwiseMax(1e-9).cwiseSqrt();
            for(int a = 0; a < m; a++)
                for(int b = 0; b < m; b++)
                    similarity(u[a], u[b]) = float(C(a, b) / (d(a) * d(b)));
        }
    }

    for(int i = 0; i < n; i++){
        if(!usable[i]) continue;
        if(keyframes[i]->is_bad()) history.push_back(i);
        else alive.push_back(i);
    }
    const bool local_scope = (params.keyframe_culling_scope == "local");
    if(local_scope){
        std::unordered_set<KeyframeId> window{current_keyframe_->keyId};
        for(const Keyframe& keyframe : current_keyframe_->get_covisible_keyframes())
            window.insert(keyframe->keyId);
        // history rows stay only if their best alive explainer (over the whole map) is local
        std::vector<int> local_history;
        for(int h : history){
            int best = -1; float best_similarity = -std::numeric_limits<float>::infinity();
            for(int a : alive)
                if(similarity(h, a) > best_similarity){ best_similarity = similarity(h, a); best = a; }
            if(best >= 0 && window.count(keyframes[best]->keyId))
                local_history.push_back(h);
        }
        history.swap(local_history);
        std::vector<int> local_alive;
        for(int a : alive)
            if(window.count(keyframes[a]->keyId))
                local_alive.push_back(a);
        alive.swap(local_alive);
    }
    const int na = int(alive.size());
    if(na <= params.keyframe_culling_min_keyframes)
        return;

    const KeyframeId current_id = current_keyframe_->keyId;
    auto is_protected = [&](const Keyframe& keyframe) -> bool {
        if(keyframe->keyId == 0 || keyframe->keyId == current_id)
            return true;
        if(keyframe->keyId + KeyframeId(params.keyframe_culling_min_age) > current_id)
            return true;
#ifdef ALLFEATURE_EVALUATION
        if((keyframe->frame_id % ALLFEATURE_EVALUATION) == 0 || (keyframe->frame_id % ALLFEATURE_MAX_KEYFRAMES) == 0)
            return true;
#endif
        return false;
    };
    std::vector<char> candidate(na, 0);
    int num_candidates = 0;
    for(int a = 0; a < na; a++){
        candidate[a] = !is_protected(keyframes[alive[a]]);
        num_candidates += candidate[a];
    }
    if(num_candidates == 0)
        return;

    const double tau = double(params.keyframe_culling_max_unexplained.load());   // live: Viewer slider
    auto fmt = [](double x) { std::ostringstream s; s << std::fixed << std::setprecision(3) << x; return s.str(); };
    constexpr double jitter = 1e-6;
    constexpr double over_budget_slack = 0.01;   // max deterioration allowed for history rows already above tau
    const int max_per_call = params.keyframe_culling_max_per_call;

    // K_AA (double precision, jittered diagonal) and its inverse M; W = K_HA M and v_h for the history
    Eigen::MatrixXd K_AA(na, na);
    for(int a = 0; a < na; a++)
        for(int b = 0; b < na; b++)
            K_AA(a, b) = double(similarity(alive[a], alive[b])) + (a == b ? jitter : 0.0);
    Eigen::MatrixXd M = K_AA.ldlt().solve(Eigen::MatrixXd::Identity(na, na));

    // History rows are stored in a growable list; a culled keyframe joins it during the loop
    std::vector<Eigen::VectorXd> W_rows;      // W_h over the alive columns (stale columns masked by `removed`)
    std::vector<double> v_h;                  // unexplained information of each history keyframe
    auto add_history_row = [&](int idx, const Eigen::MatrixXd& M_now, double v) {
        Eigen::VectorXd k(na);
        for(int a = 0; a < na; a++) k(a) = double(similarity(idx, alive[a]));
        W_rows.push_back(M_now * k);           // M symmetric: W_h = k^T M
        v_h.push_back(v);
    };
    for(int h : history){
        Eigen::VectorXd k(na);
        for(int a = 0; a < na; a++) k(a) = double(similarity(h, alive[a]));
        const Eigen::VectorXd w = M * k;
        W_rows.push_back(w);
        v_h.push_back(std::max(0.0, double(similarity(h, h)) - w.dot(k)));
    }

    std::vector<char> removed(na, 0);
    int num_alive = na;
    int num_culled = 0;
    while(num_alive > params.keyframe_culling_min_keyframes && (max_per_call <= 0 || num_culled < max_per_call)){
        // Score every candidate: unique information v_i and the worst unexplained keyframe after culling it.
        // Feasible iff v_i <= tau and no history row is raised by more than max(tau - v_h, slack).
        int best = -1;
        double best_v = std::numeric_limits<double>::infinity();
        double best_worst = 0.0;
        for(int a = 0; a < na; a++){
            if(removed[a] || !candidate[a]) continue;
            const double M_aa = M(a, a);
            if(M_aa <= 0.0) continue;
            const double v_i = 1.0 / M_aa;
            if(v_i > tau || v_i >= best_v) continue;
            bool feasible = true;
            double worst = v_i;
            for(size_t h = 0; h < W_rows.size(); h++){
                const double w = W_rows[h](a);
                const double price = w * w / M_aa;
                if(price > std::max(tau - v_h[h], over_budget_slack)){ feasible = false; break; }
                worst = std::max(worst, v_h[h] + price);
            }
            if(feasible){
                best = a; best_v = v_i; best_worst = worst;
            }
        }
        if(best < 0)
            break;

        Keyframe keyframe = keyframes[alive[best]];
        keyframe->set_bad_flag();
        if(!keyframe->is_bad()){
            // Deferred by SetNotErase (loop closing holds it): leave it alive, never retry this call
            candidate[best] = 0;
            continue;
        }
        num_culled++;
        num_alive--;
        AF_INFO("[VPR] culled keyframe " << keyframe->keyId << " (frame " << keyframe->frame_id
                << "): unique information " << fmt(best_v)
                << ", worst unexplained keyframe after the cull " << fmt(best_worst)
                << ", alive keyframes " << num_alive);

        // Rank-one downdate: M' = M - m m^T / M_ii ; W' = W - W_:,i m^T / M_ii ; v_h += W_hi^2 / M_ii
        const double M_ii = M(best, best);
        const Eigen::VectorXd m = M.col(best);
        for(size_t h = 0; h < W_rows.size(); h++){
            const double w = W_rows[h](best);
            v_h[h] += w * w / M_ii;
            W_rows[h] -= (w / M_ii) * m;
        }
        M -= (m * m.transpose()) / M_ii;
        M.row(best).setZero();
        M.col(best).setZero();
        removed[best] = 1;
        // The culled keyframe joins the history with v_i = 1/M_ii (Schur identity)
        add_history_row(alive[best], M, best_v);
    }

    double worst = 0.0;
    int over_budget = 0;
    for(double v : v_h){ worst = std::max(worst, v); over_budget += (v > tau); }
    if(num_culled > 0){
        AF_INFO("[VPR] cull_keyframes[" << params.keyframe_culling_scope << (params.keyframe_culling_centred ? ", centred" : ", raw")
                << "]: culled " << num_culled << " of " << num_candidates
                << " candidates (" << num_alive << " alive in scope, " << n << " keyframes ever); worst unexplained keyframe "
                << fmt(worst) << " (tau " << fmt(tau) << ")"
                << (max_per_call > 0 && num_culled >= max_per_call ? " [per-call limit reached]" : ""));
    } else {
        // Threshold lowered below earlier culls: report the over-budget history once per change
        static int last_over_budget{-1};
        if(over_budget != last_over_budget){
            if(over_budget > 0)
                AF_INFO("[VPR] cull_keyframes: " << over_budget << " culled keyframes are above tau (max unexplained " << fmt(worst)
                        << " > tau " << fmt(tau) << ") — only keyframes that do not explain them can be culled");
            last_over_budget = over_budget;
        }
    }
}

void LocalMapping::request_reset()
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
        new_keyframes_.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested=false;
        {
            unique_lock<mutex> lockVpr(vpr_mutex_);
            vpr_keyframes_.clear();
            keyframe_vpr_matrix_.resize(0, 0);
        }

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
