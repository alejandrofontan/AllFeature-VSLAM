#include "Feature_kaze64.h"

AF_VSLAM::FeatureExtractor_kaze64::FeatureExtractor_kaze64(std::shared_ptr<FeatureExtractorSettings> &settings_):
        FeatureExtractor(settings_){

        kaze = cv::KAZE::create();
        // kaze->setNOctaves(settings->nOctaves/4);
        // kaze->setNOctaveLayers(settings->nOctaves/2);
        // kaze->setThreshold(settings->detectTh);
}

void AF_VSLAM::FeatureExtractor_kaze64::detectAndCompute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors){

    // KAZE feature detection and description
    std::vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;
    kaze->detectAndCompute(img.grayImg, cv::Mat(), keypoints_, descriptors_);

    // Normalize octave and size
    for (int i = 0; i < (int)keypoints_.size(); ++i){
        keypoints_[i].size = 1.0;
        keypoints_[i].octave = 0;
    }

    // Retain only if not too many keypoints
    if (keypoints_.size() < settings->OVERSIZE_KEYPOINT_FACTOR * settings->maxNumFeatures){
        keypoints = keypoints_;
        descriptors = descriptors_;
        return;
    }

    // Retain best keypoints using octree distribution
    keypoints =  DistributeOctTree(keypoints_, 0, img.grayImg.cols-1, 0, img.grayImg.rows-1, settings->maxNumFeatures, 0);
    descriptors.create((int)keypoints.size(), descriptors_.cols, descriptors_.type());
    for (int i = 0; i < (int)keypoints.size(); ++i)
        descriptors_.row(keypoints[i].class_id).copyTo(descriptors.row(i));
}

int AF_VSLAM::FeatureExtractor_kaze64::GetKeypointOctave(const cv::KeyPoint& keypoint) const{
    return 0;
}

float AF_VSLAM::FeatureExtractor_kaze64::GetKeypointSize(const cv::KeyPoint& keypoint) const{
    return 1.0f;
}

float AF_VSLAM::DescriptorDistance_kaze64(const cv::Mat &a, const cv::Mat &b){
    return (Descriptor_Distance_Type) cv::norm(a,b,cv::NORM_L2);
}
