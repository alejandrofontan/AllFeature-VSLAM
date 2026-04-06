#include "Feature_orb32.h"

float AF_VSLAM::Orb32::descriptor_distance(const cv::Mat &a, const cv::Mat &b) const {
    // Bit set count operation from
    // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return Descriptor_Distance_Type(dist);
}

AF_VSLAM::FeatureExtractor_orb32::FeatureExtractor_orb32(std::shared_ptr<FeatureExtractorSettings> &settings_):
        FeatureExtractor(settings_){

    iniThFAST = settings->detectTh;

    mvScaleFactor.resize(settings->nOctaves);
    mvLevelSigma2.resize(settings->nOctaves);
    mvScaleFactor[0]=1.0f;
    mvLevelSigma2[0]=1.0f;
    for(int i=1; i<settings->nOctaves; i++)
    {
        mvScaleFactor[i]=mvScaleFactor[i-1]*settings->scaleFactor;
        mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
    }

    mvInvScaleFactor.resize(settings->nOctaves);
    mvInvLevelSigma2.resize(settings->nOctaves);
    for(int i=0; i<settings->nOctaves; i++)
    {
        mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
        mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
    }

    mvImagePyramid.resize(settings->nOctaves);

    mnFeaturesPerLevel.resize(settings->nOctaves);
    float factor = 1.0f / settings->scaleFactor;
    float nDesiredFeaturesPerScale = settings->maxNumFeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)settings->nOctaves));

    int sumFeatures = 0;
    for( int level = 0; level < settings->nOctaves-1; level++ )
    {
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= factor;
    }
    mnFeaturesPerLevel[settings->nOctaves-1] = std::max(settings->maxNumFeatures - sumFeatures, 0);

    const int npoints = 512;
    const cv::Point* pattern0 = (const cv::Point*)bit_pattern_31_;
    std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));

    //This is for orientation
    // pre-compute the end of a row in a circular patch
    umax.resize(HALF_PATCH_SIZE + 1);

    int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));

    // Make sure we are symmetric
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }
}

void AF_VSLAM::FeatureExtractor_orb32::detectAndCompute(const Image& img, std::vector<cv::KeyPoint>& _keypoints, cv::Mat& _descriptors){
    ComputePyramid(img.grayImg);
    std::vector < std::vector<cv::KeyPoint> > allKeypoints;
    ComputeKeyPointsOctTree(allKeypoints);
    int nkeypoints = 0;
    for (int level = 0; level < settings->nOctaves; ++level)
        nkeypoints += (int)allKeypoints[level].size();

    _descriptors.create(nkeypoints, 32, CV_8U);
    int offset = 0;
    for (int level = 0; level < settings->nOctaves; ++level)
    {
        std::vector<cv::KeyPoint>& keypoints = allKeypoints[level];
        int nkeypointsLevel = (int)keypoints.size();

        if(nkeypointsLevel==0)
            continue;

        cv::Mat workingMat = mvImagePyramid[level].clone();
        GaussianBlur(workingMat, workingMat, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);
        cv::Mat desc = _descriptors.rowRange(offset, offset + nkeypointsLevel);
        computeDescriptors(workingMat, keypoints, desc, pattern);
        offset += nkeypointsLevel;
        if (level != 0)
        {
            float scale = mvScaleFactor[level];
            for (std::vector<cv::KeyPoint>::iterator keypoint = keypoints.begin(),
                 keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
                keypoint->pt *= scale;
        }
        _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
    }
}

int AF_VSLAM::FeatureExtractor_orb32::GetKeypointOctave(const cv::KeyPoint& keypoint) const{
    return keypoint.octave;
}

float AF_VSLAM::FeatureExtractor_orb32::GetKeypointSize(const cv::KeyPoint& keypoint) const{
    return powf(settings->GetDetectorScaleFactor(), float(GetKeypointOctave(keypoint)));
}

float AF_VSLAM::DescriptorDistance_orb32(const cv::Mat &a, const cv::Mat &b){
    // Bit set count operation from
    // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return Descriptor_Distance_Type(dist);
}

void AF_VSLAM::FeatureExtractor_orb32::ComputePyramid(cv::Mat image)
{


    for (int level = 0; level < settings->nOctaves; ++level)
    {
        float scale = mvInvScaleFactor[level];
        cv::Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));
        cv::Size wholeSize(sz.width + EDGE_THRESHOLD*2, sz.height + EDGE_THRESHOLD*2);
        cv::Mat temp(wholeSize, image.type()), masktemp;
        mvImagePyramid[level] = temp(cv::Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

        // Compute the resized image
        if( level != 0 )
        {
            cv::resize(mvImagePyramid[level-1], mvImagePyramid[level], sz, 0, 0, cv::INTER_LINEAR);

            cv::copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           cv::BORDER_REFLECT_101+cv::BORDER_ISOLATED);
        }
        else
        {
            cv::copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           cv::BORDER_REFLECT_101);
        }
    }

}

void AF_VSLAM::FeatureExtractor_orb32::ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints)
{
    allKeypoints.resize(settings->nOctaves);

    const float W = 30;

    for (int level = 0; level < settings->nOctaves; ++level)
    {
        const int minBorderX = EDGE_THRESHOLD-3;
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD+3;
        const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD+3;

        std::vector<cv::KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(settings->maxNumFeatures*10);

        const float width = (maxBorderX-minBorderX);
        const float height = (maxBorderY-minBorderY);

        const int nCols = width/W;
        const int nRows = height/W;
        const int wCell = ceil(width/nCols);
        const int hCell = ceil(height/nRows);

        for(int i=0; i<nRows; i++)
        {
            const float iniY =minBorderY+i*hCell;
            float maxY = iniY+hCell+6;

            if(iniY>=maxBorderY-3)
                continue;
            if(maxY>maxBorderY)
                maxY = maxBorderY;

            for(int j=0; j<nCols; j++)
            {
                const float iniX =minBorderX+j*wCell;
                float maxX = iniX+wCell+6;
                if(iniX>=maxBorderX-6)
                    continue;
                if(maxX>maxBorderX)
                    maxX = maxBorderX;

                std::vector<cv::KeyPoint> vKeysCell;
                FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                     vKeysCell, iniThFAST,true);

                if(vKeysCell.empty())
                {
                    FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                         vKeysCell, minThFAST,true);
                }

                if(!vKeysCell.empty())
                {
                    for(std::vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end();vit++)
                    {
                        (*vit).pt.x+=j*wCell;
                        (*vit).pt.y+=i*hCell;
                        vToDistributeKeys.push_back(*vit);
                    }
                }

            }
        }

        std::vector<cv::KeyPoint> & keypoints = allKeypoints[level];
        keypoints.reserve(settings->maxNumFeatures);

        keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX,
                                      minBorderY, maxBorderY,mnFeaturesPerLevel[level], level);

        const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

        // Add border to coordinates and scale information
        const int nkps = keypoints.size();
        for(int i=0; i<nkps ; i++)
        {
            keypoints[i].pt.x+=minBorderX;
            keypoints[i].pt.y+=minBorderY;
            keypoints[i].octave=level;
            keypoints[i].size = scaledPatchSize;
        }
    }

    // compute orientations
    for (int level = 0; level < settings->nOctaves; ++level)
        computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
}