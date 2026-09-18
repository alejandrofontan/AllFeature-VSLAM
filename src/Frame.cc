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

#include "Frame.h"
#include "Converter.h"
#include <thread>

#include <vector>
#include <utility>
#include <type_traits>
#include <opencv2/features2d.hpp> // for cv::KeyPoint
#ifdef _OPENMP
#include <omp.h>
#endif

namespace AF_VSLAM
{

long unsigned int Frame::next_id=0;
bool Frame::initial_computations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::min_x, Frame::min_y, Frame::max_x, Frame::max_y;
float Frame::grid_element_width_inv, Frame::grid_element_height_inv;

Frame::Frame(const Image & img, const double &timeStamp,
             const std::map<FeatureType, shared_ptr<FeatureExtractor>>& extractor,
             shared_ptr<PlaceRecognition> place_recognition, const cv::Mat &K, const cv::Mat &distCoef, const float &bf, const float &thDepth)
    :place_recognition(place_recognition),
    feature_extractors(extractor),
    timestamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    frame_id = next_id++;
    w = img.img.cols;
    h = img.img.rows;

    // Image-embedding VPR backends (MegaLoc) embed keyframes after creation, off the
    // tracking thread: keep a shared header of the image (no copy) for that purpose.
    if(place_recognition && place_recognition->needs_image())
        image = img.img;

    // Scale Level Info (assumes every feature type shares the same pyramid scale factor)
    size_tolerance = feature_extractors.begin()->second->GetScaleFactor();

    // Feature extraction
    extract_features(0, img);
    if(num_keypoints_total == 0)
        return;

    undistort_keypoints();
    get_depth(img);
    get_colors(img);

    // No map-point associations yet
    for(auto& [ft, N_] : N){
        pts[ft] = vector<Pt>(N_, static_cast<Pt>(nullptr));
        outliers[ft] = vector<bool>(N_, false);
    }

    // This is done only for the first Frame (or after a change in the calibration)
    if(initial_computations)
    {
        compute_image_bounds(img.grayImg);

        grid_element_width_inv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(max_x - min_x);
        grid_element_height_inv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(max_y - min_y);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f / fx;
        invfy = 1.0f / fy;

        initial_computations = false;
    }

    mb = mbf / fx;
    assign_features_to_grid();
}

void Frame::assign_features_to_grid()
{
    for(const auto [ft, N_]: N){
        int nReserve = 0.5f * N_ / (FRAME_GRID_COLS*FRAME_GRID_ROWS);


        for(unsigned int i=0; i<FRAME_GRID_COLS;i++){
            for (unsigned int j=0; j<FRAME_GRID_ROWS;j++){
                grid[ft][i][j].reserve(nReserve);
            }
        }

        for(int i = 0; i < N_; i++)
        {
            const cv::KeyPoint &kp = keypoints.at(ft)[i];
            int nGridPosX, nGridPosY;
            if(pos_in_grid(kp,nGridPosX,nGridPosY)){
                grid[ft][nGridPosX][nGridPosY].push_back(i);
            }
        }
    }
}

void Frame::extract_features(int, const Image& img)
{
    featureTypes.clear();
    num_keypoints_total = 0;

    // 1) Stable job list
    std::vector<std::pair<FeatureType, shared_ptr<FeatureExtractor>>> jobs;
    jobs.reserve(feature_extractors.size());
    for (auto& [ft, extractor] : feature_extractors) {
        jobs.emplace_back(ft, extractor);
    }

    // 2) Per-job outputs (no shared writes here)
    struct Out {
        std::vector<cv::KeyPoint>   keys;
        cv::Mat   desc;
        vector<mat2f> sigma2;
        vector<mat2f>    inf;
        vector<float>   size;
        int n = 0;
    };

    std::vector<Out> outs(jobs.size());

    // 3) Parallel extraction (OpenMP if enabled; otherwise runs serially)
    #pragma omp parallel for
    for (int i = 0; i < (int)jobs.size(); ++i) {
        auto& extractor = jobs[i].second;
        auto& o = outs[i];

        (*extractor)(img, o.keys, o.desc, o.sigma2, o.inf, o.size);
        o.n = (int)o.keys.size();
    }

    // 4) Serial merge into your maps/vectors (safe)
    featureTypes.reserve(jobs.size());
    for (size_t i = 0; i < jobs.size(); ++i) {
        const auto ft = jobs[i].first;
        auto& o = outs[i];

        raw_keypoints[ft]        = std::move(o.keys);
        descriptors[ft]  = std::move(o.desc);
        keyPtsSigma2[ft]  = std::move(o.sigma2);
        keyPtsInf[ft]     = std::move(o.inf);
        keyPtsSize[ft]    = std::move(o.size);

        N[ft] = o.n;
        num_keypoints_total += o.n;
        featureTypes.push_back(ft);
    }

    maxKeyPtSize  = feature_extractors.begin()->second->GetMaxKeyPtSize();
    maxKeyPtSigma = feature_extractors.begin()->second->GetMaxKeyPtSigma();
}

void Frame::set_pose(const mat4f& Tcw_)
{
    Tcw = Tcw_;
    update_pose_matrices();
}

void Frame::update_pose_matrices()
{
    Rcw = Tcw.block<3,3>(0,0);
    tcw = Tcw.block<3,1>(0,3);
    Rwc = Rcw.transpose();
    twc = -Rwc * tcw;
}

bool Frame::is_in_frustum(const Pt& pMP, float viewingCosLimit) const
{
    pMP->track_in_view = false;

    // 3D in absolute coordinates
    vec3f P = pMP->get_world_pos();

    // 3D in camera coordinates
    const vec3f Pc = Rcw * P + tcw;
    const float &PcX = Pc(0);
    const float &PcY = Pc(1);
    const float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ < 0.0f)
        return false;

    // Project in image and check it is not outside
    const float invz = 1.0f / PcZ;
    const float u = fx * PcX * invz + cx;
    const float v = fy * PcY * invz + cy;

    if(u < min_x || u > max_x)
        return false;
    if(v < min_y || v > max_y)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->get_max_distance_invariance();
    const float minDistance = pMP->get_min_distance_invariance();
    const vec3f PO = P - twc;
    const float dist = PO.norm();

    if(dist < minDistance || dist > maxDistance)
        return false;

    // Check viewing angle
    vec3f Pn = pMP->get_normal();

    const float viewCos = PO.dot(Pn) / dist;

    if(viewCos < viewingCosLimit)
        return false;

    // Data used by the tracking
    pMP->track_in_view = true;
    pMP->track_proj_x = u;
    pMP->track_proj_y = v;

    return true;
}

vector<size_t> Frame::get_features_in_area(const float &x, const float  &y, const float  &r, const FeatureType& featType) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N.at(featType));

    const int nMinCellX = max(0,(int)floor((x-min_x-r)*grid_element_width_inv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-min_x+r)*grid_element_width_inv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-min_y-r)*grid_element_height_inv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-min_y+r)*grid_element_height_inv));
    if(nMaxCellY<0)
        return vIndices;

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t>& vCell = grid.at(featType)[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = keypoints.at(featType)[vCell[j]];

                // if(keyPtsSize.at(featType)[vCell[j]] < minSize)
                //     continue;
                // if(keyPtsSize.at(featType)[vCell[j]] > maxSize)
                //     continue;

                const float distx = kpUn.pt.x - x;
                const float disty = kpUn.pt.y - y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::pos_in_grid(const cv::KeyPoint &kp, int &posX, int &posY) const
{
    posX = round((kp.pt.x-min_x)*grid_element_width_inv);
    posY = round((kp.pt.y-min_y)*grid_element_height_inv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void Frame::drop_unobserved_points()
{
    for (auto& [ft, num_keypoints] : N) {
        for(int i = 0; i < num_keypoints; i++)
        {
            const Pt& map_point = pts.at(ft)[i];
            if(map_point && map_point->number_of_observations() < 1)
            {
                outliers.at(ft)[i] = false;
                pts.at(ft)[i] = nullptr;
            }
        }
    }
}

int Frame::count_inlier_map_points() const
{
    int count = 0;
    for (const auto& [ft, num_keypoints] : N)
        for(int i = 0; i < num_keypoints; i++)
        {
            const Pt& map_point = pts.at(ft)[i];
            if(map_point && !outliers.at(ft)[i] && map_point->number_of_observations() > 0)
                count++;
        }
    return count;
}

void Frame::drop_outlier_points()
{
    for (auto& [ft, num_keypoints] : N) {
        for(int i = 0; i < num_keypoints; i++)
        {
            if(pts.at(ft)[i] && outliers.at(ft)[i])
                pts.at(ft)[i] = nullptr;
        }
    }
}

void Frame::compute_global_descriptor()
{
    if(place_recognition)
        place_recognition->compute(*this);
}

void Frame::undistort_keypoints()
{
    for(auto& [ft,extractor] : feature_extractors)
    {
        if(mDistCoef.at<float>(0)==0.0)
        {
            keypoints[ft] = raw_keypoints[ft];
            continue;
        }

        // Fill matrix with points
        cv::Mat mat(N.at(ft),2,CV_32F);
        for(int i=0; i<N.at(ft); i++)
        {
            mat.at<float>(i,0) = raw_keypoints[ft][i].pt.x;
            mat.at<float>(i,1) = raw_keypoints[ft][i].pt.y;
        }

        // Undistort points
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        // Fill undistorted keypoint vector
        keypoints[ft].resize(N.at(ft));
        for(int i = 0; i < N.at(ft); i++)
        {
            cv::KeyPoint kp = raw_keypoints[ft][i];
            kp.pt.x=mat.at<float>(i,0);
            kp.pt.y=mat.at<float>(i,1);
            keypoints[ft][i]=kp;
        }
    }
}

void Frame::get_depth(const Image& img)
{
    for(auto& [ft, N_] : N)
    {
        inv_depth[ft] = vector<float>(N_, 0.0f);
        sigma2_inv_depth[ft] = vector<float>(N_, 0.0f);

        if(img.depthImg.empty())
            continue;

        const vector<cv::KeyPoint>& kps = raw_keypoints.at(ft);
        for(int i = 0; i < N_; i++)
        {
            // Sample at the keypoint's distorted pixel coordinates: the depth image
            // is indexed the same way as the original (pre-undistortion) RGB image.
            const int u = cvRound(kps[i].pt.x);
            const int v = cvRound(kps[i].pt.y);

            if(u < 0 || v < 0 || u >= img.depthImg.cols || v >= img.depthImg.rows)
                continue;

            float depth;
            switch(img.depthImg.type())
            {
                case CV_16U:
                    depth = static_cast<float>(img.depthImg.at<uint16_t>(v, u));
                    break;
                case CV_32F:
                    depth = img.depthImg.at<float>(v, u);
                    break;
                default:
                    throw std::runtime_error("Frame::get_depth: unsupported depthImg type: " + std::to_string(img.depthImg.type()));
            }

            if(depth > 0.0f)
            {
                inv_depth[ft][i] = 1.0f / depth;
                sigma2_inv_depth[ft][i] = depthNoiseCoeff * depthNoiseCoeff;
            }
        }
    }
}

void Frame::get_colors(const Image& img)
{
    // Prefer the color image; fall back to the gray one (both share the keypoints' resize/crop).
    const cv::Mat& im = img.img.empty() ? img.grayImg : img.img;
    const int channels = im.empty() ? 0 : im.channels();
    if(!im.empty() && im.depth() != CV_8U)
        throw std::runtime_error("Frame::get_colors: unsupported image depth: " + std::to_string(im.depth()));

    for(auto& [ft, N_] : N)
    {
        std::vector<cv::Vec3b>& colors = keypoint_colors[ft];
        colors.assign(N_, cv::Vec3b(0, 0, 0));
        if(im.empty())
            continue;

        const vector<cv::KeyPoint>& kps = raw_keypoints.at(ft);
        for(int i = 0; i < N_; i++)
        {
            // Distorted pixel coordinates: the image is indexed like the depth image in get_depth.
            const int u = cvRound(kps[i].pt.x);
            const int v = cvRound(kps[i].pt.y);
            if(u < 0 || v < 0 || u >= im.cols || v >= im.rows)
                continue;

            switch(channels)
            {
                case 1: {
                    const uchar g = im.at<uchar>(v, u);
                    colors[i] = cv::Vec3b(g, g, g);
                    break;
                }
                case 3:
                    colors[i] = im.at<cv::Vec3b>(v, u);
                    break;
                case 4: {
                    const cv::Vec4b bgra = im.at<cv::Vec4b>(v, u);
                    colors[i] = cv::Vec3b(bgra[0], bgra[1], bgra[2]);
                    break;
                }
                default:
                    throw std::runtime_error("Frame::get_colors: unsupported number of channels: " + std::to_string(channels));
            }
        }
    }
}

void Frame::compute_image_bounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        min_x = min(mat.at<float>(0,0),mat.at<float>(2,0));
        max_x = max(mat.at<float>(1,0),mat.at<float>(3,0));
        min_y = min(mat.at<float>(0,1),mat.at<float>(1,1));
        max_y = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else
    {
        min_x = 0.0f;
        max_x = imLeft.cols;
        min_y = 0.0f;
        max_y = imLeft.rows;
    }
}

    float Frame::get_keypoint_size(const KeypointIndex &keyPtIdx, const FeatureType& featType) const {
        return keyPtsSize.at(featType)[keyPtIdx];
    }

    float Frame::get_keypoint_sigma2(const KeypointIndex &keyPtIdx, const FeatureType& featType) const
    {
        return 0.5f * (keyPtsSigma2.at(featType)[keyPtIdx](0,0) + keyPtsSigma2.at(featType)[keyPtIdx](1,1));
    }

    float Frame::get_keypoint_information_1d(const KeypointIndex &keyPtIdx, const FeatureType& featType) const
    {
        return 0.5f * (keyPtsInf.at(featType)[keyPtIdx](0,0) + keyPtsInf.at(featType)[keyPtIdx](1,1));
    }

    mat2f Frame::get_keypoint_information_2d(const KeypointIndex &keyPtIdx, const FeatureType& featType) const
    {
        return keyPtsInf.at(featType)[keyPtIdx];
    }

    float Frame::get_overlap() const
    {
        cv::Mat1b mask(h, w, uchar(0));
        cv::Mat1b mask_0(h, w, uchar(0));
        const int radius = 30; // pixels

        for(const auto& [ft, pts]: pts)
        {
            for(const auto& pt : pts)
            {
                if((!pt) || pt->is_bad())
                    continue;

                const int x = (int) pt->track_proj_x;
                const int y = (int) pt->track_proj_y;

                if (0 <= x && x < w && 0 <= y && y < h)
                {
                    // filled circle of 1s (non-zero) around (x,y)
                    cv::circle(mask, cv::Point(x, y), radius, cv::Scalar(1), cv::FILLED, cv::LINE_8);
                }
            }
        }

        for(const auto& [ft, kpts]: keypoints)
        {
            for(const auto& kpt : kpts)
            {

                const int x = (int) kpt.pt.x;
                const int y = (int) kpt.pt.y;

                if (0 <= x && x < w && 0 <= y && y < h)
                {
                    // filled circle of 1s (non-zero) around (x,y)
                    cv::circle(mask_0, cv::Point(x, y), radius, cv::Scalar(1), cv::FILLED, cv::LINE_8);
                }
            }
        }
        // --- overlap metrics ---
        cv::Mat1b inter, uni;
        cv::bitwise_and(mask, mask_0, inter);
        cv::bitwise_or(mask, mask_0, uni);

        // const int A = cv::countNonZero(mask);
        const int B = cv::countNonZero(mask_0);
        const int I = cv::countNonZero(inter);
        // const int U = cv::countNonZero(uni);

        // const float iou  = (U > 0) ? (float)I / (float)U : 0.0f;              // |A∩B|/|A∪B|
        const float covB = (B > 0) ? (float)I / (float)B : 0.0f;              // coverage of mask_0 by mask
        // const float covA = (A > 0) ? (float)I / (float)A : 0.0f;              // coverage of mask by mask_0
        // const float dice = (A + B > 0) ? (2.0f * (float)I) / (float)(A + B) : 0.0f;

        //std::cout << "Frame " << frame_id << " cov(B)=" << covB << std::endl;

        // std::cout << "Frame " << frame_id
        //         << " A=" << A << " B=" << B << " I=" << I << " U=" << U
        //         << " IoU=" << iou << " Dice=" << dice
        //         << " cov(B)=" << covB << " cov(A)=" << covA
        //         << std::endl;

        return covB; // or return c
    }

} //namespace ORB_SLAM
