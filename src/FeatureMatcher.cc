
#include<stdint-gcc.h>
#include <memory>
#include<limits.h>

#include "FeatureMatcher.h"
#include "Converter.h"
#include "MathFunctions.h"
#include <cmath>

#include "Feature_orb32.h"
#include "Feature_akaze61.h"
#include "Feature_brisk48.h"
#include "Feature_surf64.h"
#include "Feature_kaze64.h"
#include "Feature_sift128.h"
#include "Feature_r2d2_128.h"
#include "Feature_anyFeatBin.h"
#include "Feature_anyFeatNonBin.h"
#include "Feature_aliked128.h"
#include "Feature_superpoint256.h"


#include<opencv2/core/core.hpp>
//#include<opencv2/features2d/features2d.hpp>

#include "DBoW2/FeatureVector.h"

using namespace std;

namespace AF_VSLAM
{

std::map<FeatureType, Descriptor_Distance_Type> FeatureMatcher::TH_HIGH = {};
std::map<FeatureType, Descriptor_Distance_Type> FeatureMatcher::TH_LOW = {};
std::map<FeatureType, Descriptor_Distance_Type> FeatureMatcher::descDistTh_high_reloc = {};
std::map<FeatureType, Descriptor_Distance_Type> FeatureMatcher::descDistTh_low_reloc = {};

VerbosityLevel FeatureMatcher::verbosity{MEDIUM};

const int FeatureMatcher::HISTO_LENGTH = 30;
float FeatureMatcher::radiusScale{1.15f};

FeatureMatcher::FeatureMatcher(const int& imageWidth, const int& imageHeight, float nnratio, bool checkOri):
    mfNNratio(nnratio), mbCheckOrientation(checkOri), imageWidth(imageWidth), imageHeight(imageHeight)
{
    std::cout << "Initializing SiftMatchGPU..." << std::endl;

    sift_match_gpu_ = SiftMatchGPU();
    sift_match_gpu_.SetLanguage(SiftMatchGPU::SIFTMATCH_CUDA);
    if (sift_match_gpu_.VerifyContextGL() == 0) {
       std::cout << "Initialization failed!" << std::endl;
    }
    int max_supported = 4000;
    sift_match_gpu_ .Allocate(max_supported, 1);
    std::cout << "Finished initializing SiftMatchGPU." << std::endl;

    std::cout << "Initializing LightGlue..." << std::endl;
    torch_device = std::make_shared<torch::Device>(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    matcher_lightglue = std::make_shared<matcher::LightGlue>();
    matcher_lightglue->to(*torch_device);
    std::cout << "Finished initializing LightGlue." << std::endl;

    std::cout << "Initializing SuperPoint-LightGlue..." << std::endl;
    std::string config_path = "Thirdparty/SuperPoint-LightGlue-TensorRT/config/config.yaml";
    std::string model_dir = "Thirdparty/SuperPoint-LightGlue-TensorRT/weights/";
    Configs configs(config_path, model_dir);
    matcher_lightglue_superpoint = std::make_shared<SuperPointLightGlue>(configs.superpoint_lightglue_config);
    std::cout << "[SuperPoint-LightGlue config dump]\n";
    std::cout << "  onnx_file:   " << configs.superpoint_lightglue_config.onnx_file << "\n";
    std::cout << "  engine_file: " << configs.superpoint_lightglue_config.engine_file << "\n";
    std::cout << "  input_tensor_names size: " << configs.superpoint_lightglue_config.input_tensor_names.size() << "\n";
    std::cout << "  output_tensor_names size: " << configs.superpoint_lightglue_config.output_tensor_names.size() << "\n";
    if (!matcher_lightglue_superpoint->build()) {
        throw std::runtime_error(
                "SuperPoint-LightGlue: failed to build/load TensorRT engine.\n"
                "  config: " + config_path + "\n"
                "  weights: " + model_dir);
    }
    std::cout << "SuperPoint-LightGlue engine loaded from: " << model_dir << "\n";
    std::cout << "Finished initializing SuperPoint-LightGlue." << std::endl;

}

static inline std::vector<cv::DMatch> swap_match_direction(const std::vector<cv::DMatch>& in)
{
    std::vector<cv::DMatch> out = in;
    for (auto& m : out)
        std::swap(m.queryIdx, m.trainIdx);
    return out;
}

// Matches features from a keyframe to a frame across all requested feature types,
// then associates the surviving matches to 3D map points.
//
// The matching proceeds in three stages:
//   1. For each feature type, brute-force NN descriptor matching is run between
//      the keyframe and the frame. All matches are pooled into a single list with
//      globally offset indices.
//   2. Outliers are jointly filtered across all feature types using MAGSAC on the
//      fundamental matrix (cv::USAC_MAGSAC).
//   3. Each inlier match is resolved to a map point via the keyframe's map point
//      associations. Only valid, non-bad map points are written to map_pts_matches.
//
// The matched keypoint pairs are also cached in both the keyframe and the frame
// for downstream use (e.g. visualization, loop closing).
//
// Returns the number of valid map point matches per feature type.
// Used in: Tracking::TrackReferenceKeyFrame, Tracking::Relocalization
map<FeatureType, int> FeatureMatcher::match_keyframe_to_frame(Keyframe& keyframe, Frame& frame,
    map<FeatureType, vector<Pt>>& map_pts_matches, const vector<FeatureType>& feat_types)
{
    vector<cv::KeyPoint> kps1, kps2;
    vector<int> kps1_indexes, kps2_indexes;
    vector<cv::DMatch> all_matches;
    vector<FeatureType> used_feature_types;
    map<FeatureType, vector<Pt>> map_pts_kf;

    for (const auto& ft : feat_types) {
        // Skip feature types not present in both keyframe and frame
        auto it1 = keyframe->descriptors.find(ft);
        auto it2 = frame.descriptors.find(ft);

        if (it1 == keyframe->descriptors.end() || it2 == frame.descriptors.end())
            continue;

        // Get map points associated to the keyframe for this feature type
        map_pts_kf[ft] = keyframe->get_map_point_matches(ft);

        // Brute-force NN matching between keyframe and frame descriptors
        const auto& v1 = keyframe->keypoints.at(ft);
        const auto& v2 = frame.keypoints.at(ft);
        vector<cv::DMatch> matches = match_descriptors(
            it1->second, it2->second,
            v1, v2, ft);

        // Offset match indices to account for previously appended feature types
        const size_t size_kpts1 = kps1.size();
        const size_t size_kpts2 = kps2.size();
        for (auto& m : matches) {
            m.queryIdx += size_kpts1;
            m.trainIdx += size_kpts2;
        }

        // Accumulate matches and keypoints across feature types
        all_matches.insert(all_matches.end(), matches.begin(), matches.end());
        kps1.insert(kps1.end(), v1.begin(), v1.end());
        kps2.insert(kps2.end(), v2.begin(), v2.end());

        // Track per-feature-type original keypoint indices for later map point lookup
        const size_t base1 = kps1_indexes.size();
        kps1_indexes.resize(base1 + v1.size());
        std::iota(kps1_indexes.begin() + base1, kps1_indexes.end(), 0);

        const size_t base2 = kps2_indexes.size();
        kps2_indexes.resize(base2 + v2.size());
        std::iota(kps2_indexes.begin() + base2, kps2_indexes.end(), 0);

        used_feature_types.insert(used_feature_types.end(), v1.size(), ft);

        // Initialize output map point matches for this feature type
        map_pts_matches[ft] = vector<Pt>(frame.N.at(ft), nullptr);
    }

    // Filter outliers using MAGSAC across all feature types jointly
    vector<cv::DMatch> robust_matches = filter_matches_by_fundamental(all_matches, kps1, kps2, cv::USAC_MAGSAC);

    // Associate surviving matches to map points
    map<FeatureType, int> matches_count;
    for (const auto& m : robust_matches) {
        const int query_idx = kps1_indexes[m.queryIdx];
        const int train_idx = kps2_indexes[m.trainIdx];
        const FeatureType feat_type = used_feature_types[m.queryIdx];
        const Pt pt = map_pts_kf[feat_type][query_idx];
        if (!pt || pt->isBad())
            continue;
        map_pts_matches[feat_type][train_idx] = pt;
        matches_count[feat_type]++;
    }

    // Cache matched pairs in both keyframe and frame for downstream use
    frame.cache_matched_pairs.insert_or_assign(keyframe->frame_id, swap_match_direction(robust_matches));
    keyframe->cache_matched_pairs.insert_or_assign(frame.mnId, std::move(robust_matches));

    return matches_count;
}

// For each feature type, matches unmatched frame keypoints against projected map points
// by descriptor similarity, then verifies each match via a spatial projection check.
// Returns the number of new map point assignments made to the frame.
// Used in: Tracking::TrackLocalMap
int FeatureMatcher::match_map_points_to_frame(Frame& frame, const vector<Pt>& map_pts)
{
    map<FeatureType, vector<size_t>> pt_idx, frame_idx;
    map<FeatureType, cv::Mat> desc_pts, desc_frame;

    // Collect valid map points grouped by feature type
    for (size_t idx = 0; idx < map_pts.size(); idx++) {
        const auto& pt = map_pts[idx];
        if (!pt || pt->isBad())
            continue;

        FeatureType ft = pt->featureType;
        pt_idx[ft].push_back(idx);
        desc_pts[ft].push_back(pt->get_descriptor());
    }

    int num_matches = 0;
    for (const auto& [ft, N]: frame.N) {
        // Collect unmatched frame keypoints grouped by feature type
        const auto& pts_ft  = frame.pts.at(ft);
        const auto& desc_ft = frame.descriptors.at(ft);
        frame_idx[ft].reserve(N);
        for (size_t i = 0; i < N; i++) {
            // Skip keypoints already assigned to an observed map point
            if (pts_ft[i] && pts_ft[i]->number_of_observations() > 0)
                continue;

            frame_idx[ft].push_back(i);
            desc_frame[ft].push_back(desc_ft.row(i));
        }

        // Skip feature types with no descriptors on either side
        auto it1 = desc_frame.find(ft);
        auto it2 = desc_pts.find(ft);
        if (it1 == desc_frame.end() || it2 == desc_pts.end())
            continue;

        // Match frame keypoint descriptors against map point descriptors
        vector<cv::DMatch> matches = match_descriptors_only(it1->second, it2->second, ft);

        // Validate each match: accept only if the map point projects near the matched keypoint
        const auto& f_idx = frame_idx.at(ft);
        const auto& p_idx = pt_idx.at(ft);
        for (const auto& m : matches) {
            size_t query_idx = f_idx[m.queryIdx];
            size_t train_idx = p_idx[m.trainIdx];
            Pt pt = map_pts[train_idx];
            const vector<size_t> area_indices = frame.get_features_in_area(pt->track_proj_x, pt->track_proj_y, projection_match_radius_th, ft);
            if (area_indices.empty())
                continue;

            if (std::find(area_indices.begin(), area_indices.end(), query_idx) != area_indices.end()) {
                frame.pts.at(ft)[query_idx] = pt;
                num_matches++;
            }
        }
    }
    return num_matches;
}

// Matches two keyframes across all feature types to find candidate pairs for triangulation.
// Reuses cached matches from a prior tracking step if available; otherwise runs parallel
// descriptor matching followed by LMEDS fundamental matrix filtering.
// Outputs only pairs where neither keyframe has an existing 3D map point for those keypoints.
// Used in: LocalMapping::CreateNewMapPoints
void FeatureMatcher::match_keyframes_for_triangulation(const Keyframe& keyframe1, const Keyframe& keyframe2,
                                std::map<FeatureType, vector<pair<size_t,size_t>>>& matched_pairs,
                                const std::vector<FeatureType>& feat_types){

    // Reuse cached matches if keyframe1 already has matches stored for keyframe2
    auto cache_it = keyframe1->cache_matched_pairs.find(keyframe2->frame_id);
    bool cached = (cache_it != keyframe1->cache_matched_pairs.end() && !cache_it->second.empty());

    // No cache — run parallel descriptor matching across all feature types
    map<FeatureType, vector<cv::DMatch>> matches_by_type;
    if (!cached) {
        matches_by_type = match_descriptors_parallel(feat_types,
            keyframe1->descriptors, keyframe2->descriptors, keyframe1->keypoints, keyframe2->keypoints);
    }

    // Offset match indices to account for previously appended feature types
    vector<cv::KeyPoint> kps1, kps2;
    vector<size_t> kps1_indexes, kps2_indexes;
    vector<cv::DMatch> all_matches;
    vector<FeatureType> used_feature_types;
    for (const auto& ft : feat_types) {
        matched_pairs[ft].clear();

        // Ensure both frames contain the requested feature type
        auto it1 = keyframe1->descriptors.find(ft);
        auto it2 = keyframe2->descriptors.find(ft);
        if (it1 == keyframe1->descriptors.end() || it2 == keyframe2->descriptors.end())
            continue;

        // Accumulate keypoints and track per-feature-type original indices for later lookup
        auto const& v1 = keyframe1->keypoints.at(ft);
        auto const& v2 = keyframe2->keypoints.at(ft);

        if (!cached) {
            auto matches = std::move(matches_by_type[ft]);
            size_t size_kpts1 = kps1.size();
            size_t size_kpts2 = kps2.size();
            for (auto& m : matches) {
                m.queryIdx += size_kpts1;
                m.trainIdx += size_kpts2;
            }
            all_matches.insert(all_matches.end(),
                std::make_move_iterator(matches.begin()),
                std::make_move_iterator(matches.end()));

            kps1.insert(kps1.end(), v1.begin(), v1.end());
            kps2.insert(kps2.end(), v2.begin(), v2.end());
        }

        const size_t base1 = kps1_indexes.size();
        kps1_indexes.resize(base1 + v1.size());
        std::iota(kps1_indexes.begin() + base1, kps1_indexes.end(), 0);
        const size_t base2 = kps2_indexes.size();
        kps2_indexes.resize(base2 + v2.size());
        std::iota(kps2_indexes.begin() + base2, kps2_indexes.end(), 0);
        used_feature_types.insert(used_feature_types.end(), v1.size(), ft);
    }

    // Filter outliers jointly across all feature types, or retrieve from cache
    vector<cv::DMatch> computed_matches;
    if (!cached) {
        if (all_matches.size() < min_matches_for_triangulation)
            return;

        computed_matches = filter_matches_by_fundamental(all_matches, kps1, kps2, cv::FM_LMEDS);
    }
    const vector<cv::DMatch>& filtered_matches = cached ? cache_it->second : computed_matches;

    for (const auto& m : filtered_matches) {
        const size_t query_idx = kps1_indexes[m.queryIdx];
        const size_t train_idx = kps2_indexes[m.trainIdx];
        const FeatureType feat_type = used_feature_types[m.queryIdx];

        // Only triangulate points that don't already have a 3D MapPoint
        if (!keyframe1->get_map_point(query_idx, feat_type) && !keyframe2->get_map_point(train_idx, feat_type)) {
            matched_pairs[feat_type].emplace_back(query_idx, train_idx);
        }
    }
}

// Projects each map point into the keyframe and attempts to fuse it with an existing keypoint.
// For each map point, applies visibility checks (depth, image bounds, scale, viewing angle),
// then finds the closest keypoint by descriptor distance within radius_th.
// If a match is found: replaces the weaker map point if one already exists, or adds a new observation.
// Returns the number of fused map points.
// Used in: LocalMapping::SearchInNeighbors
int FeatureMatcher::fuse_map_points_to_keyframe(Keyframe& keyframe, const vector<Pt>& map_pts, const float& radius_th, const FeatureType& feat_type)
{
    const mat3f Rcw = keyframe->get_rotation();
    const vec3f tcw = keyframe->get_translation();
    const float fx = keyframe->fx;
    const float fy = keyframe->fy;
    const float cx = keyframe->cx;
    const float cy = keyframe->cy;

    const Descriptor_Distance_Type th_low = TH_LOW[feat_type];

    const vec3f cam_center = keyframe->get_camera_center();

    int n_fused = 0;
    const auto& kps_ft  = keyframe->keypoints.at(feat_type);
    const auto& desc_ft = keyframe->descriptors.at(feat_type);
    for (const auto& pt : map_pts) {
        if (!pt)
            continue;

        if (pt->isBad() || pt->is_in_keyframe(keyframe))
            continue;

        const vec3f pos_world = pt->get_world_pos();

        // Project map point into keyframe
        const vec3f pos_cam = Rcw * pos_world + tcw;

        // Depth must be positive
        if (pos_cam(2) < 0.0f)
            continue;

        const float invz = 1.0f / pos_cam(2);

        // Compute projected image coordinates
        const float u = fx * pos_cam(0) * invz + cx;
        const float v = fy * pos_cam(1) * invz + cy;

        // Point must be inside the image
        if (!keyframe->is_in_image(u, v))
            continue;

        const float max_distance = pt->get_max_distance_invariance();
        const float min_distance = pt->get_min_distance_invariance();
        const vec3f dir_to_pt = pos_world - cam_center;
        const float dist3D = dir_to_pt.squaredNorm();

        // Depth must be inside the scale pyramid of the image
        if (dist3D < min_distance * min_distance || dist3D > max_distance * max_distance)
            continue;

        // Viewing angle must be less than 60 deg
        const vec3f normal = pt->get_normal();

        if (dir_to_pt.dot(normal) < cos_viewing_angle_th * dist3D)
            continue;

        // Search in a radius
        const vector<size_t> area_indices = keyframe->get_features_in_area(u, v, radius_th, feat_type);
        if (area_indices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat ref_descriptor = pt->get_descriptor();
        Descriptor_Distance_Type best_dist{highest_possible_distance};
        int best_idx{-1};
        for (const size_t idx : area_indices) {
            // Skip reprojection outliers
            const float ex = u - kps_ft[idx].pt.x;
            const float ey = v - kps_ft[idx].pt.y;
            const float e2 = ex * ex + ey * ey;

            if (e2 * keyframe->get_keypt_1Dinf(KeypointIndex(idx), feat_type) > 5.99f)
                continue;

            const cv::Mat &descriptor = desc_ft.row(idx);
            const Descriptor_Distance_Type desc_dist = DescriptorDistance(ref_descriptor, descriptor, pt->featureType);

            // Keep only the closest descriptor match
            if (desc_dist < best_dist) {
                best_dist = desc_dist;
                best_idx = idx;
            }
        }

        // Fuse: replace weaker map point or add new observation
        if (best_dist <= th_low) {
            Pt existing_pt = keyframe->get_map_point(best_idx, feat_type);
            if (existing_pt) {
                if (!existing_pt->isBad()) {
                    if (existing_pt->number_of_observations() > pt->number_of_observations())
                        pt->replace(existing_pt);
                    else
                        existing_pt->replace(pt);
                }
            }
            else {
                pt->add_observation(keyframe, best_idx);
                keyframe->add_map_point(pt, best_idx);
            }
            n_fused++;
        }
    }

    return n_fused;
}

// SearchForInitialization Frame-Frame
// Tracking::MonocularInitialization
int FeatureMatcher::SearchForInitialization(const Frame &F1, const Frame &F2,
    vector<cv::Point2f> &pointsPrevMatched, vector<int> &matches12, const FeatureType& featType)
{
    matches12.clear();

    // Ensure both frames contain the requested feature type
    auto it1 = F1.descriptors.find(featType);
    auto it2 = F2.descriptors.find(featType);
    if (it1 == F1.descriptors.end() || it2 == F2.descriptors.end())
        return 0;

    std::vector<cv::DMatch> matches = featureMatching_2(F1.descriptors.at(featType), F2.descriptors.at(featType),
         F1.keypoints.at(featType), F2.keypoints.at(featType), featType, sFI_ff_outlierMethod);

    int numMatches = 0;
    matches12 = vector<int>(F1.keypoints.at(featType).size(),-1);
    for(const auto& m : matches) {
        if (F1.keyPtsSize.at(featType)[m.queryIdx] > 1.0)
            continue;
        if (F2.keyPtsSize.at(featType)[m.trainIdx] > 1.0)
            continue;
        if (F1.keyPtsSize.at(featType)[m.queryIdx] != F2.keyPtsSize.at(featType)[m.trainIdx])
            continue;

        matches12[m.queryIdx] = m.trainIdx;
        pointsPrevMatched[m.queryIdx] = F2.keypoints.at(featType)[m.trainIdx].pt;
        numMatches++;
    }
    return numMatches;
}

float FeatureMatcher::RadiusByViewingCos(const float &viewCos)
{
    if(viewCos>0.998)
        return 2.5;
    else
        return 4.0;
}


bool FeatureMatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const mat3f& F12 , const Keyframe pKF2, const float& sigma2_kp2)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    const float a = kp1.pt.x * F12(0,0) + kp1.pt.y * F12(1,0) + F12(2,0);
    const float b = kp1.pt.x * F12(0,1) + kp1.pt.y * F12(1,1) + F12(2,1);
    const float c = kp1.pt.x * F12(0,2) + kp1.pt.y * F12(1,2) + F12(2,2);

    const float num = a*kp2.pt.x+b*kp2.pt.y+c;

    const float den = a*a+b*b;

    if(den==0)
        return false;

    const float dsqr = num*num/den;

    return dsqr < 3.84f * sigma2_kp2;
}

// SearchByProjection 2
// Compute Sim3
int FeatureMatcher::SearchByProjection(Keyframe pKF, const mat4f& Scw, const vector<Pt> &vpPoints, vector<Pt> &vpMatched,
    const float& radiusTh, const FeatureType& featType)
{

    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    mat3f sRcw = Scw.block<3,3>(0,0);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    mat3f Rcw = sRcw / scw;
    vec3f tcw = Scw.block<3,1>(0,3);
    vec3f Ow = -Rcw.transpose() * tcw;

    // Set of MapPoints already found in the KeyFrame
    set<Pt> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<Pt>(NULL));

    int nmatches=0;

    // For each Candidate MapPoint Project and Match
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        Pt pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        vec3f p3Dw = pMP->get_world_pos();

        // Transform into Camera Coords.
        vec3f p3Dc = Rcw * p3Dw + tcw;

        // Depth must be positive
        if(p3Dc(2) < 0.0f)
            continue;

        // Project into Image
        const float invz = 1.0f / p3Dc(2);
        const float x = p3Dc(0) * invz;
        const float y = p3Dc(1) * invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->is_in_image(u,v))
            continue;

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->get_max_distance_invariance();
        const float minDistance = pMP->get_min_distance_invariance();
        vec3f PO = p3Dw - Ow;
        const float dist3D = PO.norm();

        if(dist3D < minDistance || dist3D > maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        vec3f Pn = pMP->get_normal();

        if(PO.dot(Pn) < 0.5f * dist3D)
            continue;


        // Search in a radius
        float predictedSize = pMP->PredictSize(dist3D);
        const float radius = radiusScale * radiusTh * predictedSize;
        const vector<size_t> vIndices = pKF->get_features_in_area(u,v,radius, featType);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat refDescriptor = pMP->get_descriptor();
        Descriptor_Distance_Type bestDist{highest_possible_distance};
        int bestIdx{-1};

        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            if(vpMatched[idx])
                continue;

            //const float keyPtSize = pKF->GetKeyPtSize(KeypointIndex (idx), featType);
            //if((keyPtSize < predictedSize / pKF->sizeTolerance) || (keyPtSize > predictedSize * pKF->sizeTolerance))
            //    continue;

            const cv::Mat &descriptor = pKF->descriptors.at(featType).row(idx);
            const Descriptor_Distance_Type descDist = DescriptorDistance(refDescriptor,descriptor,pMP->featureType);

            if(descDist < bestDist)
            {
                bestDist = descDist;
                bestIdx = idx;
            }
        }

        if(bestDist <= TH_LOW[featType])
        {
            vpMatched[bestIdx]=pMP;
            nmatches++;
        }

    }

    return nmatches;
}

// SearchByBoW 2
// ComputeSim3
int FeatureMatcher::SearchByBoW(Keyframe pKF1, Keyframe pKF2, vector<Pt > &vpMatches12, const FeatureType&  featType)
{

    const vector<cv::KeyPoint> &vKeysUn1 = pKF1->keypoints.at(featType);
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const vector<Pt> vpMapPoints1 = pKF1->get_map_point_matches(featType);
    const cv::Mat &Descriptors1 = pKF1->descriptors.at(featType);

    const vector<cv::KeyPoint> &vKeysUn2 = pKF2->keypoints.at(featType);
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
    const vector<Pt> vpMapPoints2 = pKF2->get_map_point_matches(featType);
    const cv::Mat &Descriptors2 = pKF2->descriptors.at(featType);

    vpMatches12 = vector<Pt>(vpMapPoints1.size(),static_cast<Pt>(NULL));
    vector<bool> vbMatched2(vpMapPoints2.size(),false);

    int nMatches{0};
    float rotFactor{};
    vector<vector<int>> rotHist = initRotationHistogram(rotFactor,HISTO_LENGTH);

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while(f1it != f1end && f2it != f2end)
    {
        if(f1it->first == f2it->first)
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];

                Pt pMP1 = vpMapPoints1[idx1];
                if(!pMP1)
                    continue;
                if(pMP1->isBad())
                    continue;

                const cv::Mat &refDescriptor = Descriptors1.row(idx1);
                Descriptor_Distance_Type bestDist1{highest_possible_distance},bestDist2{highest_possible_distance};
                int bestIdx2{-1};

                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    const size_t idx2 = f2it->second[i2];

                    Pt pMP2 = vpMapPoints2[idx2];

                    if(vbMatched2[idx2] || !pMP2)
                        continue;

                    if(pMP2->isBad())
                        continue;

                    const cv::Mat &descriptor = Descriptors2.row(idx2);
                    Descriptor_Distance_Type descDist = DescriptorDistance(refDescriptor,descriptor,pMP2->featureType);

                    if(descDist < bestDist1)
                    {
                        bestDist2 = bestDist1;
                        bestDist1 = descDist;
                        bestIdx2 = idx2;
                    }
                    else if(descDist < bestDist2)
                    {
                        bestDist2 = descDist;
                    }
                }

                if(bestDist1 < TH_LOW[featType])
                {
                    if(static_cast<float>(bestDist1) < mfNNratio * static_cast<float>(bestDist2))
                    {
                        vpMatches12[idx1] = vpMapPoints2[bestIdx2];
                        vbMatched2[bestIdx2] = true;
                        nMatches++;
                        if(mbCheckOrientation)
                            updateRotationHistogram(rotHist,idx1,vKeysUn1[idx1],vKeysUn2[bestIdx2],rotFactor,HISTO_LENGTH);
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if(mbCheckOrientation)
        filterMatchesWithOrientation(rotHist,vpMatches12,nMatches);

    return nMatches;
}

// Fuse 2
// Loop Closing
int FeatureMatcher::Fuse(Keyframe pKF, const mat4f& Scw, const vector<Pt> &vpPoints, const float& radiusTh, vector<Pt> &vpReplacePoint, const FeatureType& featType)
{

    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    mat3f sRcw = Scw.block<3,3>(0,0);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    mat3f Rcw = sRcw / scw;
    vec3f tcw = Scw.block<3,1>(0,3);
    vec3f Ow = -Rcw.transpose() * tcw;

    // Set of MapPoints already found in the KeyFrame
    const set<Pt> spAlreadyFound = pKF->get_map_points(featType);

    int nFused=0;

    const int nPoints = vpPoints.size();

    // For each candidate MapPoint project and match
    for(int iMP=0; iMP<nPoints; iMP++)
    {
        Pt pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        vec3f p3Dw = pMP->get_world_pos();

        // Transform into Camera Coords.
        vec3f p3Dc = Rcw * p3Dw + tcw;

        // Depth must be positive
        if(p3Dc(2) < 0.0f)
            continue;

        // Project into Image
        const float invz = 1.0f / p3Dc(2);
        const float x = p3Dc(0) * invz;
        const float y = p3Dc(1) * invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->is_in_image(u,v))
            continue;

        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->get_max_distance_invariance();
        const float minDistance = pMP->get_min_distance_invariance();
        vec3f PO = p3Dw-Ow;
        const float dist3D = PO.norm();

        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        vec3f Pn = pMP->get_normal();

        if(PO.dot(Pn) < 0.5f * dist3D)
            continue;

        // Search in a radius
        const float predictedSize = pMP->PredictSize(dist3D);
        const float radius = radiusScale * radiusTh * predictedSize;

        const vector<size_t> vIndices = pKF->get_features_in_area(u,v,radius, featType);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat refDescriptor = pMP->get_descriptor();
        Descriptor_Distance_Type bestDist{highest_possible_distance};
        int bestIdx{-1};

        for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
        {
            const size_t idx = *vit;

            //const float keyPtSize = pKF->GetKeyPtSize(KeypointIndex (idx), featType);
            //if((keyPtSize < predictedSize / pKF->sizeTolerance) || (keyPtSize > predictedSize * pKF->sizeTolerance))
            //    continue;

            const cv::Mat &descriptor = pKF->descriptors.at(featType).row(idx);
            Descriptor_Distance_Type descDist = DescriptorDistance(refDescriptor,descriptor,pMP->featureType);

            if(descDist < bestDist)
            {
                bestDist = descDist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist <= TH_LOW[featType])
        {
            Pt pMPinKF = pKF->get_map_point(bestIdx, featType);
            if(pMPinKF)
            {
                if(!pMPinKF->isBad())
                    vpReplacePoint[iMP] = pMPinKF;
            }
            else
            {
                pMP->add_observation(pKF,bestIdx);
                pKF->add_map_point(pMP,bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

int FeatureMatcher::SearchBySim3(Keyframe pKF1, Keyframe pKF2, vector<Pt> &vpMatches12,
                                 const float &s12, const mat3f  &R12, const vec3f &t12, const float& radiusTh, const FeatureType& featType)
{

    const float &fx = pKF1->fx;
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;

    // Camera 1 from world
    mat3f R1w = pKF1->get_rotation();
    vec3f t1w = pKF1->get_translation();

    //Camera 2 from world
    mat3f R2w = pKF2->get_rotation();
    vec3f t2w = pKF2->get_translation();

    //Transformation between cameras
    mat3f sR12 = s12 * R12;
    mat3f sR21 = (1.0/s12) * R12.transpose();
    vec3f t21 = -sR21 * t12;

    const vector<Pt> vpMapPoints1 = pKF1->get_map_point_matches(featType);
    const int N1 = vpMapPoints1.size();

    const vector<Pt> vpMapPoints2 = pKF2->get_map_point_matches(featType);
    const int N2 = vpMapPoints2.size();

    vector<bool> vbAlreadyMatched1(N1,false);
    vector<bool> vbAlreadyMatched2(N2,false);

    for(int i=0; i<N1; i++)
    {
        Pt pMP = vpMatches12[i];
        if(pMP)
        {
            vbAlreadyMatched1[i]=true;
            int idx2 = pMP->GetIndexInKeyFrame(pKF2);
            if(idx2>=0 && idx2<N2)
                vbAlreadyMatched2[idx2]=true;
        }
    }

    vector<int> vnMatch1(N1,-1);
    vector<int> vnMatch2(N2,-1);

    // Transform from KF1 to KF2 and search
    for(int i1=0; i1<N1; i1++)
    {
        Pt pMP = vpMapPoints1[i1];

        if(!pMP || vbAlreadyMatched1[i1])
            continue;

        if(pMP->isBad())
            continue;

        vec3f p3Dw = pMP->get_world_pos();
        vec3f p3Dc1 = R1w * p3Dw + t1w;
        vec3f p3Dc2 = sR21 * p3Dc1 + t21;

        // Depth must be positive
        if(p3Dc2(2) < 0.0f)
            continue;

        const float invz = 1.0f / p3Dc2(2);
        const float x = p3Dc2(0) * invz;
        const float y = p3Dc2(1) * invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF2->is_in_image(u,v))
            continue;

        const float maxDistance = pMP->get_max_distance_invariance();
        const float minDistance = pMP->get_min_distance_invariance();
        const float dist3D = p3Dc2.norm();

        // Depth must be inside the scale invariance region
        if(dist3D < minDistance || dist3D > maxDistance )
            continue;

        // Search in a radius
        const float predictedSize = pMP->PredictSize(dist3D);
        const float radius = radiusScale * radiusTh * predictedSize;

        const vector<size_t> vIndices = pKF2->get_features_in_area(u,v,radius, featType);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat refDescriptor = pMP->get_descriptor();
        Descriptor_Distance_Type bestDist{highest_possible_distance};
        int bestIdx{-1};

        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF2->keypoints.at(featType)[idx];

            //const float keyPtSize = pKF2->GetKeyPtSize(KeypointIndex (idx), featType);
            //if((keyPtSize < predictedSize / pKF2->sizeTolerance) || (keyPtSize > predictedSize * pKF2->sizeTolerance))
            //    continue;

            const cv::Mat &descriptor = pKF2->descriptors.at(featType).row(idx);
            const Descriptor_Distance_Type descDist = DescriptorDistance(refDescriptor,descriptor,pMP->featureType);

            if(descDist < bestDist)
            {
                bestDist = descDist;
                bestIdx = idx;
            }
        }

        if(bestDist <= TH_HIGH[featType])
        {
            vnMatch1[i1] = bestIdx;
        }
    }

    // Transform from KF2 to KF2 and search
    for(int i2=0; i2<N2; i2++)
    {
        Pt pMP = vpMapPoints2[i2];

        if(!pMP || vbAlreadyMatched2[i2])
            continue;

        if(pMP->isBad())
            continue;

        vec3f p3Dw = pMP->get_world_pos();
        vec3f p3Dc2 = R2w * p3Dw + t2w;
        vec3f p3Dc1 = sR12 * p3Dc2 + t12;

        // Depth must be positive
        if(p3Dc1(2) < 0.0f)
            continue;

        const float invz = 1.0f / p3Dc1(2);
        const float x = p3Dc1(0) * invz;
        const float y = p3Dc1(1) * invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF1->is_in_image(u,v))
            continue;

        const float maxDistance = pMP->get_max_distance_invariance();
        const float minDistance = pMP->get_min_distance_invariance();
        const float dist3D = p3Dc1.norm();

        // Depth must be inside the scale pyramid of the image
        if(dist3D < minDistance || dist3D > maxDistance)
            continue;

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const float predictedSize = pMP->PredictSize(dist3D);
        const float radius = radiusScale * radiusTh * predictedSize;

        const vector<size_t> vIndices = pKF1->get_features_in_area(u,v,radius, featType);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat refDescriptor = pMP->get_descriptor();
        Descriptor_Distance_Type bestDist{highest_possible_distance};
        int bestIdx{-1};

        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF1->keypoints.at(featType)[idx];

            //const float keyPtSize = pKF1->GetKeyPtSize(KeypointIndex (idx), featType);
            //if((keyPtSize < predictedSize / pKF1->sizeTolerance) || (keyPtSize > predictedSize * pKF1->sizeTolerance))
            //    continue;

            const cv::Mat &descriptor = pKF1->descriptors.at(featType).row(idx);
            const Descriptor_Distance_Type descDist = DescriptorDistance(refDescriptor,descriptor,pMP->featureType);

            if(descDist < bestDist)
            {
                bestDist = descDist;
                bestIdx = idx;
            }
        }

        if(bestDist <= TH_HIGH[featType])
        {
            vnMatch2[i2]=bestIdx;
        }
    }

    // Check agreement
    int nFound = 0;

    for(int i1=0; i1<N1; i1++)
    {
        int idx2 = vnMatch1[i1];

        if(idx2>=0)
        {
            int idx1 = vnMatch2[idx2];
            if(idx1==i1)
            {
                vpMatches12[i1] = vpMapPoints2[idx2];
                nFound++;
            }
        }
    }

    return nFound;
}

// SearchByProjection 4
// Relocalization
int FeatureMatcher::SearchByProjection(Frame &CurrentFrame, Keyframe pKF, const set<Pt> &sAlreadyFound, const float& radiusTh,
     const bool& useHighMatchingThreshold, const FeatureType& featType)
{
    Descriptor_Distance_Type descDistanceTh = descDistTh_low_reloc[featType];
    if(useHighMatchingThreshold)
        descDistanceTh = descDistTh_high_reloc[featType];

    const mat3f Rcw = CurrentFrame.Tcw.block<3,3>(0,0);
    const vec3f tcw = CurrentFrame.Tcw.block<3,1>(0,3);
    const vec3f Ow = -Rcw.transpose() * tcw;

    // Rotation Histogram (to check rotation consistency)
    int nMatches{0};
    float rotFactor{};
    vector<vector<int>> rotHist = initRotationHistogram(rotFactor,HISTO_LENGTH);

    const vector<Pt> vpMPs = pKF->get_map_point_matches(featType);

    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        Pt pMP = vpMPs[i];

        if(pMP)
        {
            if(!pMP->isBad() && !sAlreadyFound.count(pMP))
            {
                //Project
                vec3f x3Dw = pMP->get_world_pos();
                vec3f x3Dc = Rcw * x3Dw + tcw;

                const float xc = x3Dc(0);
                const float yc = x3Dc(1);
                const float invzc = 1.0f / x3Dc(2);

                const float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
                const float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;

                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;

                // Compute predicted scale level
                vec3f PO = x3Dw - Ow;
                float dist3D = PO.norm();

                const float maxDistance = pMP->get_max_distance_invariance();
                const float minDistance = pMP->get_min_distance_invariance();

                // Depth must be inside the scale pyramid of the image
                if(dist3D<minDistance || dist3D>maxDistance)
                    continue;

                // Search in a window
                float predictedSize = pMP->PredictSize(dist3D);
                const float radius = radiusScale * radiusTh * predictedSize;

                const vector<size_t> vIndices2 = CurrentFrame.get_features_in_area(u, v, radius, featType);

                if(vIndices2.empty())
                    continue;

                const cv::Mat refDescriptor = pMP->get_descriptor();
                Descriptor_Distance_Type bestDist{highest_possible_distance};
                int bestIdx2{-1};

                for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
                {
                    const size_t i2 = *vit;
                    if(CurrentFrame.pts.at(featType)[i2])
                        continue;

                    const cv::Mat &descriptor = CurrentFrame.descriptors.at(featType).row(i2);
                    const Descriptor_Distance_Type descDist = DescriptorDistance(refDescriptor,descriptor,pMP->featureType);

                    if(descDist < bestDist)
                    {
                        bestDist = descDist;
                        bestIdx2 = i2;
                    }
                }

                if(bestDist <= descDistanceTh)
                {
                    CurrentFrame.pts.at(featType)[bestIdx2]=pMP;
                    nMatches++;

                    if(mbCheckOrientation)
                        updateRotationHistogram(rotHist,bestIdx2, pKF->keypoints.at(featType)[i],CurrentFrame.keypoints.at(featType)[bestIdx2],rotFactor,HISTO_LENGTH);
                }
            }
        }
    }

    if(mbCheckOrientation)
        filterMatchesWithOrientation(rotHist,CurrentFrame.pts.at(featType),nMatches);

    return nMatches;
}

Descriptor_Distance_Type FeatureMatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b, const FeatureType& featureType_)
{
    std::unique_ptr<AF_VSLAM::Feature> ft = get_feature(featureType_);
    return ft->DescriptorDistance(a,b);
}

void FeatureMatcher::setDescriptorDistanceThresholds(const string &feature_settings_yaml_file, const FeatureType& featureType) {

    cv::FileStorage fSettings(feature_settings_yaml_file, cv::FileStorage::READ);
    cout << endl  << "Loading Feature Matcher Settings from : " << feature_settings_yaml_file << endl;
    FeatureMatcher::TH_LOW[featureType] = fSettings["FeatureMatcher.TH_LOW"];
    FeatureMatcher::TH_HIGH[featureType] = fSettings["FeatureMatcher.TH_HIGH"];
    FeatureMatcher::descDistTh_low_reloc[featureType] = fSettings["FeatureMatcher.descDistTh_high_reloc"];
    FeatureMatcher::descDistTh_high_reloc[featureType] = fSettings["FeatureMatcher.descDistTh_low_reloc"];
    cout <<  "- TH_LOW: " << FeatureMatcher::TH_LOW[featureType] << endl;
    cout <<  "- TH_HIGH: " << FeatureMatcher::TH_HIGH[featureType] << endl;
    cout <<  "- descDistTh_low_reloc: " << FeatureMatcher::descDistTh_low_reloc[featureType] << endl;
    cout <<  "- descDistTh_high_reloc: " << FeatureMatcher::descDistTh_high_reloc[featureType] << endl;
}

vector<vector<int>> FeatureMatcher::initRotationHistogram(float& rotFactor, const int& histLength){
    vector<vector<int>> rotHist;
    rotHist.resize(histLength);
    for(int i = 0; i < histLength; i++)
        rotHist[i].reserve(500);
    rotFactor = 1.0f / float(histLength);return rotHist;
}

    void FeatureMatcher::updateRotationHistogram(vector<vector<int>>& rotHist,
                                                     const KeypointIndex& idx,
                                                     const cv::KeyPoint& keyPt, const cv::KeyPoint& refKeyPt,
                                                     const float& rotFactor, const int& histLength){
        float rot = keyPt.angle - refKeyPt.angle;
        if(rot < 0.0)
            rot += 360.0f;
        int bin = (int) round(rot * rotFactor);
        if(bin == histLength)
            bin = 0;
        assert(bin >= 0 && bin < histLength);
        rotHist[bin].push_back(idx);
    }

    void FeatureMatcher::filterMatchesWithOrientation(vector<vector<int>>& rotHist, vector<Pt>& points, int& nMatches){
        int ind1{-1}, ind2{-1}, ind3{-1};
        computeThreeMaxima(rotHist,ind1,ind2,ind3);

        for(int i = 0; i < rotHist.size(); i++){
            if(i == ind1 || i == ind2 || i == ind3)
                continue;
            for(int j : rotHist[i]){
                points[j] = static_cast<Pt>(nullptr);
                nMatches--;
            }
        }
    }

    void FeatureMatcher::filterMatchesWithOrientation(vector<vector<int>>& rotHist, vector<int>& matches, int& nMatches){
        int ind1{-1}, ind2{-1}, ind3{-1};
        computeThreeMaxima(rotHist,ind1,ind2,ind3);

        for(int i = 0; i < rotHist.size(); i++){
            if(i == ind1 || i == ind2 || i == ind3)
                continue;
            for(int idx1 : rotHist[i]){
                if(matches[idx1] >= 0){
                    nMatches--;
                    matches[idx1] =-1;
                }
            }
        }
    }

    void FeatureMatcher::computeThreeMaxima(vector<vector<int>>& rotHist, int &ind1, int &ind2, int &ind3){
        int max1{0}, max2{0}, max3{0};
        for(int i = 0; i < rotHist.size(); i++)
        {
            const int s = (int) rotHist[i].size();
            if(s > max1)
            {
                max3=max2;
                max2=max1;
                max1=s;
                ind3=ind2;
                ind2=ind1;
                ind1=i;
            }
            else if(s > max2)
            {
                max3=max2;
                max2=s;
                ind3=ind2;
                ind2=i;
            }
            else if(s > max3)
            {
                max3=s;
                ind3=i;
            }
        }

        if(max2 < 0.1f*(float)max1)
        {
            ind2=-1;
            ind3=-1;
        }
        else if(max3 < 0.1f*(float)max1)
        {
            ind3=-1;
        }
    }

    std::vector<cv::DMatch> FeatureMatcher::match_descriptors_only(
        const cv::Mat& desc1, const cv::Mat& desc2, const FeatureType& featType_){

        std::unique_ptr<AF_VSLAM::Feature> ft = get_feature(featType_);
        MatcherType matcherType = ft->getMatcherType();

        std::vector<cv::DMatch> matches;
        switch(matcherType) {
            case LIGHTGLUE_SUPERPOINT:
            case LIGHTGLUE_ALIKED:
            case BF_L2:{
                bf_matcher_L2.match(desc1, desc2, matches);
                break;
            }
            case LIGHTGLUE_SIFT:
            {
                sift_match_gpu_.SetDescriptors(0, desc1.rows, desc1.ptr<float>());
                sift_match_gpu_.SetDescriptors(1, desc2.rows, desc2.ptr<float>());
                const int max_out = 4000;
                uint32_t (*match_buffer)[2] = new uint32_t[max_out][2];
                int num_matches = sift_match_gpu_.GetSiftMatch(max_out, match_buffer, 0.7f, 0.8f, 1);
                matches.clear();
                matches.reserve(num_matches);
                for (int i = 0; i < num_matches; ++i) {
                    matches.emplace_back(
                        static_cast<int>(match_buffer[i][0]),
                        static_cast<int>(match_buffer[i][1]),
                        0.0f);
                }
                delete[] match_buffer;
                break;
            }
            case BF_HAMMING:
            {
                bf_matcher_hamming.match(desc1, desc2, matches);
                break;
            }
        }

        return matches;
    }

    std::vector<cv::DMatch> FeatureMatcher::match_descriptors(
        const cv::Mat& desc1, const cv::Mat& desc2,
        const std::vector<cv::KeyPoint>& kps1, const std::vector<cv::KeyPoint>& kps2,
        const FeatureType& featType_){

        std::unique_ptr<AF_VSLAM::Feature> ft = get_feature(featType_);
        MatcherType matcherType = ft->getMatcherType();

        std::vector<cv::DMatch> matches;
        switch(matcherType) {
            case LIGHTGLUE_SIFT:
            case LIGHTGLUE_ALIKED:
            {
                matches = lightglueMatching(kps1, desc1, kps2, desc2, 0.0f);
                break;
            }
            case LIGHTGLUE_SUPERPOINT:
            {
                matches = matcherLightglueSuperpoint(kps1, desc1, kps2, desc2, 0.0f);
                break;
            }
            case BF_L2:
            {
                bf_matcher_L2.match(desc1, desc2, matches);
                break;
            }
            case BF_HAMMING:
            {
                bf_matcher_hamming.match(desc1, desc2, matches);
                break;
            }
        }

        return matches;
    }

    std::vector<cv::DMatch> FeatureMatcher::featureMatching_2(const cv::Mat& desc1, const cv::Mat& desc2,
        const std::vector<cv::KeyPoint>& kps1, const std::vector<cv::KeyPoint>& kps2, const FeatureType& featType_,
        int outlierMehod){

        std::unique_ptr<AF_VSLAM::Feature> ft = get_feature(featType_);
        MatcherType matcherType = ft->getMatcherType();

        std::vector<cv::DMatch> matches;

        switch(matcherType) {
            case LIGHTGLUE_SIFT:
            case LIGHTGLUE_ALIKED:
            {
                matches = lightglueMatching(kps1, desc1, kps2, desc2, 0.0f);
                break;
            }
            case LIGHTGLUE_SUPERPOINT:
            {
                matches = matcherLightglueSuperpoint(kps1, desc1, kps2, desc2, 0.0f);
                break;
            }
            case BF_L2:
            {
                bf_matcher_L2.match(desc1, desc2, matches);
                break;
            }
            case BF_HAMMING:
            {
                bf_matcher_hamming.match(desc1, desc2, matches);
                break;
            }
        }

        return filter_matches_by_fundamental(matches, kps1, kps2, outlierMehod);
    }

    std::vector<cv::DMatch> FeatureMatcher::filter_matches_by_fundamental(std::vector<cv::DMatch>& matches,
        const std::vector<cv::KeyPoint>& kps1, const std::vector<cv::KeyPoint>& kps2,
        int outlierMethod, const int maxForRansac){

        std::sort(matches.begin(), matches.end(),
            [](const cv::DMatch& a, const cv::DMatch& b) { return a.distance < b.distance; });

        if (matches.size() > maxForRansac) matches.resize(maxForRansac);

        // Build point correspondences ---
        std::vector<cv::Point2f> pts1; pts1.reserve(matches.size());
        std::vector<cv::Point2f> pts2; pts2.reserve(matches.size());
        for (const auto& m : matches) {
            // Safety: ensure indices are valid
            if (m.queryIdx < 0 || m.queryIdx >= (int)kps1.size()) continue;
            if (m.trainIdx < 0 || m.trainIdx >= (int)kps2.size()) continue;

            pts1.push_back(kps1[m.queryIdx].pt);
            pts2.push_back(kps2[m.trainIdx].pt);
        }

        if (pts1.size() < 8) return matches; // not enough after filtering

        const double reprojThreshold = 3.0;  // pixels
        const double confidence      = 0.95;

        std::vector<uchar> inlierMask;
        cv::Mat F = cv::findFundamentalMat(
            pts1, pts2,
            outlierMethod,
            reprojThreshold,
            confidence,
            inlierMask
        );

        if (F.empty() || inlierMask.size() != pts1.size())
            return matches;

        std::vector<cv::DMatch> inlierMatches;
        inlierMatches.reserve(matches.size());

        // Rebuild matchesUsed aligned with pts1/pts2:
        std::vector<cv::DMatch> matchesUsed;
        matchesUsed.reserve(matches.size());

        for (const auto& m : matches) {
            if (m.queryIdx < 0 || m.queryIdx >= (int)kps1.size()) continue;
            if (m.trainIdx < 0 || m.trainIdx >= (int)kps2.size()) continue;
            matchesUsed.push_back(m);
        }

        if (matchesUsed.size() != inlierMask.size())
            return matches; // alignment mismatch fallback

        for (size_t i = 0; i < inlierMask.size(); ++i) {
            if (inlierMask[i]) inlierMatches.push_back(matchesUsed[i]);
        }

        // std::cout << "FeatureMatcher::featureMatching: "
        //           << inlierMatches.size() << " inliers found out of "
        //           << matches.size() << " matches." << std::endl;

        return inlierMatches;
    }

    std::map<FeatureType, std::vector<cv::DMatch>> FeatureMatcher::match_descriptors_parallel(
        const std::vector<FeatureType>& featureTypes,
        const std::map<FeatureType, cv::Mat> &desc1_, const std::map<FeatureType, cv::Mat> &desc2_,
        const std::map<FeatureType, std::vector<cv::KeyPoint>> &kps1_, const std::map<FeatureType, std::vector<cv::KeyPoint>> &kps2_
    ){

        struct MatchOut {
            FeatureType ft{};
            bool valid = false;
            std::vector<cv::DMatch> matches;
            size_t frameN = 0;
        };

        std::vector<FeatureType> jobs;
        jobs.reserve(featureTypes.size());
        for (const auto ft : featureTypes) {
            auto it1 = desc1_.find(ft);
            auto it2 = desc2_.find(ft);
            if (it1 == desc1_.end() || it2 == desc2_.end())
                continue;
            jobs.push_back(ft);
        }

            std::vector<MatchOut> outs(jobs.size());

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)jobs.size(); ++i) {
            auto ft = jobs[i];
            auto& o = outs[i];
            o.ft = ft;
            o.valid = true;

            const auto& desc1 = desc1_.at(ft);
            const auto& desc2 = desc2_.at(ft);
            const auto& kps1  = kps1_.at(ft);
            const auto& kps2  = kps2_.at(ft);

            //o.frameN = (size_t)frame.N.at(ft);
            o.matches = match_descriptors(desc1, desc2, kps1, kps2, ft);
        }

        std::map<FeatureType, std::vector<cv::DMatch>> matches_by_type;
        for (auto& o : outs) {
            if (!o.valid) continue;
            matches_by_type[o.ft] = std::move(o.matches);
        }
        return matches_by_type;
    }

    std::vector<cv::DMatch> FeatureMatcher::serialFeatureMatching(
        const cv::Mat& desc1_, const cv::Mat& desc2_,
        const std::vector<cv::KeyPoint>& kps1_, const std::vector<cv::KeyPoint>& kps2_,
        const FeatureType& ft
    ){

        std::vector<cv::DMatch> matches = match_descriptors(desc1_, desc2_, kps1_, kps2_, ft);
        return matches;
    }

    } //namespace ORB_SLAM
