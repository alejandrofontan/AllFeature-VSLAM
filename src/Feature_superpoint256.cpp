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
    AF_INFO("Initializing SuperPoint256 feature extractor (ONNX Runtime)...");   // the session is created in prepare()
}

void AF_VSLAM::Superpoint256Extractor::prepare(const int width, const int height)
{
    extractor_for(width, height);
}

void AF_VSLAM::Superpoint256Extractor::load(const std::string& path, const int width, const int height)
{
    extractor_ = std::make_unique<onnx::SuperPointOnnx>(path, onnx::feature_models_config().ort);
    extractor_->set_subpixel(onnx::feature_models_config().superpoint_subpixel);
    // Warm-up on a frame-sized blank: the first Run pays allocation and (CUDA / Core ML) kernel selection
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    extractor_->detect(cv::Mat::zeros(height, width, CV_8UC1), 1, kps, desc);
    AF_INFO("SuperPoint256 ready: " << extractor_->model_path() << " (" << extractor_->precision()
            << " weights) on onnxruntime/" << extractor_->provider() << ", budget " << settings->max_num_features
            << " keypoints, threshold " << extractor_->detection_threshold()
            << (extractor_->subpixel() ? ", sub-pixel positions" : ", integer-pixel positions") << ", warm-up "
            << extractor_->last_graph_ms() << " ms");
}

AF_VSLAM::onnx::SuperPointOnnx& AF_VSLAM::Superpoint256Extractor::extractor_for(const int width, const int height)
{
    // The dynamic file takes any size; a fixed-size export only its own
    if (extractor_ && (!fixed_size_ || (width == loaded_width_ && height == loaded_height_)))
        return *extractor_;
    if (extractor_)
        AF_WARN("SuperPoint256: frame size changed to " << width << "x" << height << " - loading its fixed-size export");

    // Fixed-size exports are Core ML's need (it compiles only bounded shapes); the other providers gain nothing
    // from them, so they load the dynamic file whatever features_static_shapes says
    const onnx::FeatureModelsConfig& config = onnx::feature_models_config();
    const std::string dynamic_path = onnx::feature_model_path(Superpoint256::s_model_stem);
    const bool wants_fixed = config.static_shapes && onnx::resolve_provider(config.ort.provider, false) == "coreml";
    const std::string path = wants_fixed ? onnx::static_model_path(Superpoint256::s_model_stem, width, height, "superpoint")
                                         : dynamic_path;
    load(path, width, height);
    fixed_size_ = path != dynamic_path;   // false too when the fixed-size file is missing (static_model_path warned)
    loaded_width_ = width;
    loaded_height_ = height;
    return *extractor_;
}

void AF_VSLAM::Superpoint256Extractor::detect_and_compute(
    const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    // The dense graph + the C++ tail (threshold, mask, best max_num_features, descriptor sampling, L2). The mask goes
    // in before the budget is taken; the filter after it only acts on a mask detect() could not use (type or size)
    extractor_for(img.grayImg.cols, img.grayImg.rows).detect(img.grayImg, settings->max_num_features, keypoints, descriptors,
                                                             img.mask);
    filter_keypoints_by_mask(keypoints, descriptors, img.mask);
}

float AF_VSLAM::Superpoint256Extractor::get_keypoint_size(const cv::KeyPoint&) const{
    return 1.0f;
}
