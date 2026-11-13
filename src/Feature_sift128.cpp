/**
 * Module: AllFeature-VSLAM - Feature_sift128.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.1)
 * - Version: 2.0 (cv::SIFT on the CPU)
 * - Created: 2024-06-07
 * - Updated: 2026-10-16
 * - License: GPLv3 License
 */

#include "Feature_sift128.h"

#include "afvslam_log.hpp"

AF_VSLAM::Sift128Extractor::Sift128Extractor(std::shared_ptr<FeatureExtractorSettings> &settings_):
        FeatureExtractor(settings_){
    // cv::SIFT with the LightGlue repository's SIFT configuration (nOctaveLayers 4, contrast 0.0066667, edge 10,
    // sigma 1.6 - the same values settings/sift128_settings.yaml has carried); the budget is cv::SIFT's nfeatures
    sift_ = std::make_unique<SiftCpu>(settings->max_num_features);
    AF_INFO("SIFT128 ready: cv::SIFT (CPU) + RootSIFT, budget " << settings->max_num_features << " keypoints");
}

void AF_VSLAM::Sift128Extractor::detect_and_compute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors){
    sift_->detect(img.grayImg, keypoints, descriptors);
    // Discard keypoints outside the mask (if one was loaded for this frame). Scale (size) and orientation (angle)
    // stay on the keypoints: LightGlue-SIFT takes them as inputs.
    filter_keypoints_by_mask(keypoints, descriptors, img.mask);
}

float AF_VSLAM::Sift128Extractor::get_keypoint_size(const cv::KeyPoint&) const{
    return 1.0f;
}
