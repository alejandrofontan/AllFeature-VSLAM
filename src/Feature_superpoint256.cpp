/**
 * Module: AllFeature-VSLAM - Feature_superpoint256.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.1)
 * - Version: 2.0 (ONNX Runtime)
 * - Created: 2024-06-07
 * - Updated: 2026-10-16
 * - License: GPLv3 License
 */

#include "Feature_superpoint256.h"

#include "afvslam_log.hpp"
#include "onnx/FeatureModels.h"

AF_VSLAM::Superpoint256Extractor::Superpoint256Extractor(std::shared_ptr<FeatureExtractorSettings> &settings_):
        FeatureExtractor(settings_){
    AF_INFO("Initializing SuperPoint256 feature extractor (ONNX Runtime)...");
    if (onnx::feature_models_config().static_shapes)
        AF_INFO("SuperPoint256: features_static_shapes is on - the fixed-size export for the frame size loads on the first frame");
    else
        load(onnx::feature_model_path(Superpoint256::s_model_file), 640, 480);
}

void AF_VSLAM::Superpoint256Extractor::load(const std::string& path, const int width, const int height)
{
    extractor_ = std::make_unique<onnx::SuperPointOnnx>(path, onnx::feature_models_config().ort);
    // Warm-up on a frame-sized blank: the first Run pays allocation and (CUDA / Core ML) kernel selection
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    extractor_->detect(cv::Mat::zeros(height, width, CV_8UC1), 1, kps, desc);
    AF_INFO("SuperPoint256 ready: " << extractor_->model_path() << " (" << extractor_->precision()
            << " weights) on onnxruntime/" << extractor_->provider() << ", budget " << settings->max_num_features
            << " keypoints, threshold " << extractor_->detection_threshold() << ", warm-up "
            << extractor_->last_graph_ms() << " ms");
}

AF_VSLAM::onnx::SuperPointOnnx& AF_VSLAM::Superpoint256Extractor::extractor_for(const int width, const int height)
{
    if (onnx::feature_models_config().static_shapes && (!extractor_ || width != loaded_width_ || height != loaded_height_))
    {
        if (extractor_)
            AF_WARN("SuperPoint256: frame size changed to " << width << "x" << height << " - loading its static export");
        load(onnx::static_model_path(Superpoint256::s_model_file, width, height, "superpoint"), width, height);
        loaded_width_ = width;
        loaded_height_ = height;
    }
    return *extractor_;
}

void AF_VSLAM::Superpoint256Extractor::detect_and_compute(
    const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    // The dense graph + the C++ tail (threshold, best max_num_features, descriptor sampling, L2); then the mask
    extractor_for(img.grayImg.cols, img.grayImg.rows).detect(img.grayImg, settings->max_num_features, keypoints, descriptors);
    filter_keypoints_by_mask(keypoints, descriptors, img.mask);
}

float AF_VSLAM::Superpoint256Extractor::get_keypoint_size(const cv::KeyPoint&) const{
    return 1.0f;
}
