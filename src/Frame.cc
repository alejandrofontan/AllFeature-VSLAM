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

#include <vector>
#include <utility>
#include <type_traits>
#include <opencv2/features2d.hpp>

namespace AF_VSLAM
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :vocabulary(frame.vocabulary), featureExtractorLeft(frame.featureExtractorLeft), featureExtractorRight(frame.featureExtractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     pts(frame.pts), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId), refKeyframe(frame.refKeyframe),
     sizeTolerance(frame.sizeTolerance),invSizeTolerance(frame.invSizeTolerance),
     keyPtsSigma2(frame.keyPtsSigma2),keyPtsInf(frame.keyPtsInf),keyPtsSize(frame.keyPtsSize),
     maxKeyPtSize(frame.maxKeyPtSize),maxKeyPtSigma(frame.maxKeyPtSigma), featureTypes(frame.featureTypes), w(frame.w), h(frame.h)
{
    for (FeatureType ft : featureTypes)
        for(int i = 0;i<FRAME_GRID_COLS;i++)
            for(int j=0; j<FRAME_GRID_ROWS; j++)
                mGrid[ft][i][j] = frame.mGrid.at(ft)[i][j];

    for (auto const& [featType, desc] : frame.descriptors){
            desc.copyTo(descriptors[featType]);
    }
    for (auto const& [featType, desc] : frame.descriptorsRight){
            desc.copyTo(descriptorsRight[featType]);
    }

    if(frame.Tcw(3,3) == 1.0f)
        SetPose(frame.Tcw);
}

Frame::Frame(const Image & img, const double &timeStamp,
             std::map<FeatureType, shared_ptr<FeatureExtractor>>& extractor,
             shared_ptr<Vocabulary> vocabulary, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :vocabulary(vocabulary),
    featureExtractorLeft(extractor), featureExtractorRight(),
    mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId = nNextId++;
    w = img.img.cols;
    h = img.img.rows;

    // Scale Level Info
    sizeTolerance = featureExtractorLeft.begin()->second->GetScaleFactor();
    invSizeTolerance = 1.0f / sizeTolerance;

    // Feature extraction
    ExtractFeatures(0,img);
    if(Ntotal == 0)
        return;

    UndistortKeyPoints();

    // Set no stereo information
    for(auto& [ft, N_] : N){
        mvuRight[ft] = vector<float>(N_,-1);
        mvDepth[ft] = vector<float>(N_,-1);
        pts[ft] = vector<Pt>(N_, static_cast<Pt>(nullptr));
        mvbOutlier[ft] = vector<bool>(N_, false);
    }

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(img.grayImg);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;
    AssignFeaturesToGrid();
}

void Frame::AssignFeaturesToGrid()
{
    for(const auto [ft, N_]: N){
        int nReserve = 0.5f * N_ / (FRAME_GRID_COLS*FRAME_GRID_ROWS);


        for(unsigned int i=0; i<FRAME_GRID_COLS;i++){
            for (unsigned int j=0; j<FRAME_GRID_ROWS;j++){
                mGrid[ft][i][j].reserve(nReserve);
            }
        }

        for(int i = 0; i < N_; i++)
        {
            const cv::KeyPoint &kp = mvKeysUn.at(ft)[i];
            int nGridPosX, nGridPosY;
            if(PosInGrid(kp,nGridPosX,nGridPosY)){
                mGrid[ft][nGridPosX][nGridPosY].push_back(i);
            }
        }
    }
}

void Frame::ExtractFeatures(int flag, const Image& img)
{
    featureTypes.clear();
    Ntotal = 0;

    // 1) Stable job list
    std::vector<std::pair<FeatureType, shared_ptr<FeatureExtractor>>> jobs;
    jobs.reserve(featureExtractorLeft.size());
    for (auto& [ft, extractor] : featureExtractorLeft) {
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

        mvKeys[ft]        = std::move(o.keys);
        descriptors[ft]  = std::move(o.desc);
        keyPtsSigma2[ft]  = std::move(o.sigma2);
        keyPtsInf[ft]     = std::move(o.inf);
        keyPtsSize[ft]    = std::move(o.size);

        N[ft] = o.n;
        Ntotal += o.n;
        featureTypes.push_back(ft);
    }

    maxKeyPtSize  = featureExtractorLeft.begin()->second->GetMaxKeyPtSize();
    maxKeyPtSigma = featureExtractorLeft.begin()->second->GetMaxKeyPtSigma();
}

void Frame::SetPose(const mat4f& Tcw_)
{
    Tcw = Tcw_;
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{
    Rcw = Tcw.block<3,3>(0,0);
    tcw = Tcw.block<3,1>(0,3);
    Rwc = Rcw.transpose();
    twc = -Rwc * tcw;
}

bool Frame::isInFrustum(Pt pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    vec3f P = pMP->GetWorldPos();

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

    if(u < mnMinX || u > mnMaxX)
        return false;
    if(v < mnMinY || v > mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const vec3f PO = P - twc;
    const float dist = PO.norm();

    if(dist < minDistance || dist > maxDistance)
        return false;

    // Check viewing angle
    vec3f Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn) / dist;

    if(viewCos < viewingCosLimit)
        return false;

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf*invz;
    pMP->mTrackProjY = v;

    pMP->trackSigma = pMP->PredictSigma(dist);
    pMP->trackSize = pMP->PredictSize(dist);
    pMP->trackViewCos = viewCos;

    return true;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const FeatureType& featType) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N.at(featType));

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid.at(featType)[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn.at(featType)[vCell[j]];

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

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void Frame::ComputeBoW(const FeatureType& featType)
{
    if(mBowVec.empty())
        vocabulary->transform(descriptors[featType], mBowVec,mFeatVec);
}

void Frame::UndistortKeyPoints()
{
    for(auto& [ft,extractor] : featureExtractorLeft)
    {
        if(mDistCoef.at<float>(0)==0.0)
        {
            mvKeysUn[ft] = mvKeys[ft];
            continue;
        }

        // Fill matrix with points
        cv::Mat mat(N.at(ft),2,CV_32F);
        for(int i=0; i<N.at(ft); i++)
        {
            mat.at<float>(i,0) = mvKeys[ft][i].pt.x;
            mat.at<float>(i,1) = mvKeys[ft][i].pt.y;
        }

        // Undistort points
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        // Fill undistorted keypoint vector
        mvKeysUn[ft].resize(N.at(ft));
        for(int i = 0; i < N.at(ft); i++)
        {
            cv::KeyPoint kp = mvKeys[ft][i];
            kp.pt.x=mat.at<float>(i,0);
            kp.pt.y=mat.at<float>(i,1);
            mvKeysUn[ft][i]=kp;
        }
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
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

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

    void Frame::ComputeStereoMatches(const FeatureType& featureType_)
    {
        std::cout << "This function (Frame::ComputeStereoMatches) has not been modified yet to work with AnyFeature-VSLAM"<< endl;
        std::terminate();
    }


    void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
    {
        std::cout << "This function (Frame::ComputeStereoFromRGBD) has not been modified yet to work with AnyFeature-VSLAM"<< endl;
        std::terminate();
    }

    vec3f Frame::UnprojectStereo(const int &i)
    {
        std::cout << "This function (Frame::UnprojectStereo) has not been modified yet to work with AnyFeature-VSLAM"<< endl;
        std::terminate();
    }

    float Frame::GetKeyPtSize(const KeypointIndex &keyPtIdx, const FeatureType& featType) const {
        return keyPtsSize.at(featType)[keyPtIdx];
    }

    float Frame::GetKeyPt1DSigma2(const KeypointIndex &keyPtIdx, const FeatureType& featType) const
    {
        return 0.5f * (keyPtsSigma2.at(featType)[keyPtIdx](0,0) + keyPtsSigma2.at(featType)[keyPtIdx](1,1));
    }

    mat2f Frame::GetKeyPt2DSigma2(const KeypointIndex &keyPtIdx, const FeatureType& featType) const
    {
        return keyPtsSigma2.at(featType)[keyPtIdx];
    }

    mat3f Frame::GetKeyPt3DSigma2(const KeypointIndex &keyPtIdx, const FeatureType& featType) const
    {
        mat3f sigma2Matrix{mat3f::Zero()};
        sigma2Matrix.block<2,2>(0,0) = keyPtsSigma2.at(featType)[keyPtIdx];
        sigma2Matrix(2,2) = GetKeyPt1DSigma2(keyPtIdx, featType);
        return sigma2Matrix;
    }

    float Frame::GetKeyPt1DInf(const KeypointIndex &keyPtIdx, const FeatureType& featType) const
    {
        return 0.5f * (keyPtsInf.at(featType)[keyPtIdx](0,0) + keyPtsInf.at(featType)[keyPtIdx](1,1));
    }

    mat2f Frame::GetKeyPt2DInf(const KeypointIndex &keyPtIdx, const FeatureType& featType) const
    {
        return keyPtsInf.at(featType)[keyPtIdx];
    }

    mat3f Frame::GetKeyPt3DInf(const KeypointIndex &keyPtIdx, const FeatureType& featType) const
    {
        mat3f infMatrix{mat3f::Zero()};
        infMatrix.block<2,2>(0,0) = keyPtsInf.at(featType)[keyPtIdx];
        infMatrix(2,2) = GetKeyPt1DInf(keyPtIdx, featType);
        return infMatrix;
    }

    float Frame::GetOverlap()
    {
        cv::Mat1b mask(h, w, uchar(0));
        cv::Mat1b mask_0(h, w, uchar(0));
        const int radius = 30; // pixels

        for(const auto& [ft, pts]: pts)
        {
            for(const auto& pt : pts)
            {
                if((!pt) || pt->isBad())
                    continue;

                const int x = (int) pt->mTrackProjX;
                const int y = (int) pt->mTrackProjY;

                if (0 <= x && x < w && 0 <= y && y < h)
                {
                    // filled circle of 1s (non-zero) around (x,y)
                    cv::circle(mask, cv::Point(x, y), radius, cv::Scalar(1), cv::FILLED, cv::LINE_8);
                }
            }
        }

        for(const auto& [ft, kpts]: mvKeysUn)
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

        //std::cout << "Frame " << mnId << " cov(B)=" << covB << std::endl;

        // std::cout << "Frame " << mnId
        //         << " A=" << A << " B=" << B << " I=" << I << " U=" << U
        //         << " IoU=" << iou << " Dice=" << dice
        //         << " cov(B)=" << covB << " cov(A)=" << covA
        //         << std::endl;

        return covB; // or return c
    }

} //namespace ORB_SLAM
