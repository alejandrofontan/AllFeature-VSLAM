/**
 * Module: AllFeature-VSLAM - Feature_aliked128.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.1)
 * - Version: 2.0 (ONNX Runtime)
 * - Created: 2024-06-07
 * - Updated: 2026-10-16
 * - License: GPLv3 License
 */

#include "Feature_aliked128.h"

#include "afvslam_log.hpp"
#include "onnx/FeatureModels.h"

AF_VSLAM::Aliked128Extractor::Aliked128Extractor(std::shared_ptr<FeatureExtractorSettings> &settings_):
        FeatureExtractor(settings_){
    AF_INFO("Initializing ALIKED128 feature extractor (ONNX Runtime)...");
    if (onnx::feature_models_config().static_shapes)
        AF_INFO("ALIKED128: features_static_shapes is on - the fixed-size export for the frame size loads on the first frame");
    else
        load(onnx::feature_model_path(Aliked128::s_model_file), 640, 480);
}

void AF_VSLAM::Aliked128Extractor::load(const std::string& path, const int width, const int height)
{
    extractor_ = std::make_unique<onnx::AlikedOnnx>(path, onnx::feature_models_config().ort);
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    extractor_->detect(cv::Mat::zeros(height, width, CV_8UC3), kps, desc);   // warm-up
    AF_INFO("ALIKED128 ready: " << extractor_->model_path() << " (" << extractor_->precision()
            << " weights) on onnxruntime/" << extractor_->provider() << ", budget " << settings->max_num_features
            << " keypoints, threshold " << extractor_->detection_threshold() << ", warm-up "
            << extractor_->last_graph_ms() << " ms");
}

AF_VSLAM::onnx::AlikedOnnx& AF_VSLAM::Aliked128Extractor::extractor_for(const int width, const int height)
{
    if (onnx::feature_models_config().static_shapes && (!extractor_ || width != loaded_width_ || height != loaded_height_))
    {
        if (extractor_)
            AF_WARN("ALIKED128: frame size changed to " << width << "x" << height << " - loading its static export");
        load(onnx::static_model_path(Aliked128::s_model_file, width, height, "aliked"), width, height);
        loaded_width_ = width;
        loaded_height_ = height;
    }
    return *extractor_;
}

void AF_VSLAM::Aliked128Extractor::detect_and_compute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors){
    // The graph converts nothing: AlikedOnnx takes the BGR frame (cv::imread order) and feeds RGB, as the model expects
    std::vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;
    extractor_for(img.img.cols, img.img.rows).detect(img.img, keypoints_, descriptors_);

    // No flatten_keypoint_scale here: AlikedOnnx already builds every keypoint at size 1,
    // angle 0, octave 0 -- ALIKED is single-scale, so there is no scale to discard.
    select_keypoints(img, keypoints_, descriptors_, keypoints, descriptors);
}

float AF_VSLAM::Aliked128Extractor::get_keypoint_size(const cv::KeyPoint&) const{
    return 1.0f;
}
