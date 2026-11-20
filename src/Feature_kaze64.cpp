#include "Feature_kaze64.h"

AF_VSLAM::Kaze64Extractor::Kaze64Extractor(std::shared_ptr<FeatureExtractorSettings> &settings_):
        FeatureExtractor(settings_){

        kaze = cv::KAZE::create();
}

void AF_VSLAM::Kaze64Extractor::detect_and_compute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors){

    // KAZE feature detection and description
    std::vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;
    kaze->detectAndCompute(img.grayImg, cv::Mat(), keypoints_, descriptors_);

    flatten_keypoint_scale(keypoints_);
    select_keypoints(img, keypoints_, descriptors_, keypoints, descriptors);
}

float AF_VSLAM::Kaze64Extractor::get_keypoint_size(const cv::KeyPoint&) const{
    return 1.0f;
}
