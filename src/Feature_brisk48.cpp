#include "Feature_brisk48.h"

AF_VSLAM::Brisk48Extractor::Brisk48Extractor(std::shared_ptr<FeatureExtractorSettings> &settings_):
        FeatureExtractor(settings_){

    brisk_detector = new brisk::BriskFeatureDetector(34, 4, true);
    brisk_extractor = new brisk::BriskDescriptorExtractor(true, true, brisk::BriskDescriptorExtractor::Version::briskV2);
}

void AF_VSLAM::Brisk48Extractor::detect_and_compute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors){

    // BRISK feature detection and description
    std::vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;

    brisk_detector->detect(img.grayImg, keypoints_);
    brisk_extractor->compute(img.grayImg, keypoints_, descriptors_);

    flatten_keypoint_scale(keypoints_);
    select_keypoints(img, keypoints_, descriptors_, keypoints, descriptors);
}

float AF_VSLAM::Brisk48Extractor::get_keypoint_size(const cv::KeyPoint&) const{
    // detect_and_compute sets every keypoint's octave to 0, so the old scaleFactor^octave was always 1
    return 1.0f;
}
