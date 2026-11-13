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

float AF_VSLAM::Kaze64Extractor::get_keypoint_size(const cv::KeyPoint&) const{
    return 1.0f;
}
