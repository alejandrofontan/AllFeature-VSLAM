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

AF_VSLAM::FeatureExtractor_superpoint256::FeatureExtractor_superpoint256(std::shared_ptr<FeatureExtractorSettings> &settings_):
        FeatureExtractor(settings_){
    AF_INFO("Initializing SuperPoint256 feature extractor (ONNX Runtime)...");
    if (onnx::feature_models_config().static_shapes)
        AF_INFO("SuperPoint256: features_static_shapes is on - the fixed-size export for the frame size loads on the first frame");
    else
        load(onnx::feature_model_path(Superpoint256::kModelFile), 640, 480);
}

void AF_VSLAM::FeatureExtractor_superpoint256::load(const std::string& path, const int width, const int height)
{
    extractor_ = std::make_unique<onnx::SuperPointOnnx>(path, onnx::feature_models_config().ort);
    // Warm-up on a frame-sized blank: the first Run pays allocation and (CUDA / Core ML) kernel selection
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    extractor_->detect(cv::Mat::zeros(height, width, CV_8UC1), 1, kps, desc);
    AF_INFO("SuperPoint256 ready: " << extractor_->model_path() << " (" << extractor_->precision()
            << " weights) on onnxruntime/" << extractor_->provider() << ", budget " << settings->maxNumFeatures
            << " keypoints, threshold " << extractor_->detection_threshold() << ", warm-up "
            << extractor_->last_graph_ms() << " ms");
}

AF_VSLAM::onnx::SuperPointOnnx& AF_VSLAM::FeatureExtractor_superpoint256::extractor_for(const int width, const int height)
{
    if (onnx::feature_models_config().static_shapes && (!extractor_ || width != loaded_width_ || height != loaded_height_))
    {
        if (extractor_)
            AF_WARN("SuperPoint256: frame size changed to " << width << "x" << height << " - loading its static export");
        load(onnx::static_model_path(Superpoint256::kModelFile, width, height, "superpoint"), width, height);
        loaded_width_ = width;
        loaded_height_ = height;
    }
    return *extractor_;
}

void AF_VSLAM::FeatureExtractor_superpoint256::detectAndCompute(
    const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    // The dense graph + the C++ tail (threshold, best maxNumFeatures, descriptor sampling, L2); then the mask
    extractor_for(img.grayImg.cols, img.grayImg.rows).detect(img.grayImg, settings->maxNumFeatures, keypoints, descriptors);
    FilterKeypointsByMask(keypoints, descriptors, img.mask);
}

int AF_VSLAM::FeatureExtractor_superpoint256::GetKeypointOctave(const cv::KeyPoint&) const{
    return 0;
}

float AF_VSLAM::FeatureExtractor_superpoint256::GetKeypointSize(const cv::KeyPoint&) const{
    return 1.0f;
}

float AF_VSLAM::DescriptorDistance_superpoint256(const cv::Mat &a, const cv::Mat &b){
    return (Descriptor_Distance_Type) cv::norm(a, b, cv::NORM_L2);
}
