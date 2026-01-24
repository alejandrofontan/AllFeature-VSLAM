#include "Feature_akaze61.h"

ANYFEATURE_VSLAM::FeatureExtractor_akaze61::FeatureExtractor_akaze61(std::shared_ptr<FeatureExtractorSettings> &settings_):
        FeatureExtractor(settings_){

    akazeOptions = AKAZEOptions();
    akazeOptions.omax = settings->GetDetectorNumOctaves() / 4;
    akazeOptions.nsublevels = settings->GetDetectorNumOctaves() / 2;
    akazeOptions.dthreshold = float(settings->detectTh);
}

void ANYFEATURE_VSLAM::FeatureExtractor_akaze61::detectAndCompute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors){

    // AKAZE feature detection and description
    std::vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_; 
    evolution->Feature_Detection(keypoints_);
    evolution->Compute_Descriptors(keypoints_, descriptors_);
    
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

void ANYFEATURE_VSLAM::FeatureExtractor_akaze61::setupImage(const Image& img){
    cv::Mat img_32;
    img.grayImg.convertTo(img_32, CV_32F, 1.0/255.0,0);
    akazeOptions.img_width = img_32.cols;
    akazeOptions.img_height = img_32.rows;

    evolution.reset();
    evolution = std::make_shared<libAKAZE::AKAZE>(akazeOptions);
    evolution->Create_Nonlinear_Scale_Space(img_32);
}

int ANYFEATURE_VSLAM::FeatureExtractor_akaze61::GetKeypointOctave(const cv::KeyPoint& keypoint) const{
    return keypoint.class_id;
}

float ANYFEATURE_VSLAM::FeatureExtractor_akaze61::GetKeypointSize(const cv::KeyPoint& keypoint) const{
    return powf(settings->GetDetectorScaleFactor(), float(GetKeypointOctave(keypoint)));
}

float ANYFEATURE_VSLAM::DescriptorDistance_akaze61(const cv::Mat &a, const cv::Mat &b){
    return (Descriptor_Distance_Type) cv::norm(a,b,cv::NORM_HAMMING);
}