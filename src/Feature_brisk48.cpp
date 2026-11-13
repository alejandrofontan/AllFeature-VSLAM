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

    // Normalize octave and size
    for (int i = 0; i < (int)keypoints_.size(); ++i){
        keypoints_[i].size = 1.0;
        keypoints_[i].octave = 0;
    }

    // Discard keypoints outside the mask (if one was loaded for this frame)
    filter_keypoints_by_mask(keypoints_, descriptors_, img.mask);

    // Retain only if not too many keypoints
    if (keypoints_.size() < settings->oversize_keypoint_factor * settings->max_num_features){
        keypoints = keypoints_;
        descriptors = descriptors_;
        return;
    }

    // Retain best keypoints using octree distribution
    keypoints =  distribute_oct_tree(keypoints_, 0, img.grayImg.cols-1, 0, img.grayImg.rows-1, settings->max_num_features);
    descriptors.create((int)keypoints.size(), descriptors_.cols, descriptors_.type());
    for (int i = 0; i < (int)keypoints.size(); ++i)
        descriptors_.row(keypoints[i].class_id).copyTo(descriptors.row(i));
}

float AF_VSLAM::Brisk48Extractor::get_keypoint_size(const cv::KeyPoint&) const{
    // detect_and_compute sets every keypoint's octave to 0, so the old scaleFactor^octave was always 1
    return 1.0f;
}
