#include "Feature_brisk48.h"

AF_VSLAM::FeatureExtractor_brisk48::FeatureExtractor_brisk48(std::shared_ptr<FeatureExtractorSettings> &settings_):
        FeatureExtractor(settings_){

    //brisk_detector = new brisk::BriskFeatureDetector(int(settings->detectTh), settings->nOctaves / 2, true);
    brisk_detector = new brisk::BriskFeatureDetector(34, 4, true);
    brisk_extractor = new brisk::BriskDescriptorExtractor(true, true, brisk::BriskDescriptorExtractor::Version::briskV2);
}

void AF_VSLAM::FeatureExtractor_brisk48::detectAndCompute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors){

    // BRISK feature detection and description
    std::vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;

    brisk_detector->detect(img.grayImg, keypoints_);
    brisk_extractor->compute(img.grayImg, keypoints_, descriptors_);

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

int AF_VSLAM::FeatureExtractor_brisk48::GetKeypointOctave(const cv::KeyPoint& keypoint) const{
    return keypoint.octave;
}

float AF_VSLAM::FeatureExtractor_brisk48::GetKeypointSize(const cv::KeyPoint& keypoint) const{
    return powf(settings->GetDetectorScaleFactor(), float(GetKeypointOctave(keypoint)));
}

float AF_VSLAM::DescriptorDistance_brisk48(const cv::Mat &a, const cv::Mat &b){
    return (Descriptor_Distance_Type) cv::norm(a,b,cv::NORM_HAMMING);
}
