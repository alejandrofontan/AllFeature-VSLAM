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

AF_VSLAM::FeatureExtractor_aliked128::FeatureExtractor_aliked128(std::shared_ptr<FeatureExtractorSettings> &settings_):
        FeatureExtractor(settings_){
    AF_INFO("Initializing ALIKED128 feature extractor (ONNX Runtime)...");
    if (onnx::feature_models_config().static_shapes)
        AF_INFO("ALIKED128: features_static_shapes is on - the fixed-size export for the frame size loads on the first frame");
    else
        load(onnx::feature_model_path(Aliked128::kModelFile), 640, 480);
}

void AF_VSLAM::FeatureExtractor_aliked128::load(const std::string& path, const int width, const int height)
{
    extractor_ = std::make_unique<onnx::AlikedOnnx>(path, onnx::feature_models_config().ort);
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    extractor_->detect(cv::Mat::zeros(height, width, CV_8UC3), kps, desc);   // warm-up
    AF_INFO("ALIKED128 ready: " << extractor_->model_path() << " (" << extractor_->precision()
            << " weights) on onnxruntime/" << extractor_->provider() << ", budget " << settings->maxNumFeatures
            << " keypoints, threshold " << extractor_->detection_threshold() << ", warm-up "
            << extractor_->last_graph_ms() << " ms");
}

AF_VSLAM::onnx::AlikedOnnx& AF_VSLAM::FeatureExtractor_aliked128::extractor_for(const int width, const int height)
{
    if (onnx::feature_models_config().static_shapes && (!extractor_ || width != loaded_width_ || height != loaded_height_))
    {
        if (extractor_)
            AF_WARN("ALIKED128: frame size changed to " << width << "x" << height << " - loading its static export");
        load(onnx::static_model_path(Aliked128::kModelFile, width, height, "aliked"), width, height);
        loaded_width_ = width;
        loaded_height_ = height;
    }
    return *extractor_;
}

void AF_VSLAM::FeatureExtractor_aliked128::detectAndCompute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors){
    // The graph converts nothing: AlikedOnnx takes the BGR frame (cv::imread order) and feeds RGB, as the model expects
    std::vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;
    extractor_for(img.img.cols, img.img.rows).detect(img.img, keypoints_, descriptors_);

    // Discard keypoints outside the mask (if one was loaded for this frame)
    FilterKeypointsByMask(keypoints_, descriptors_, img.mask);

    // Retain only if not too many keypoints
    if (keypoints_.size() < settings->OVERSIZE_KEYPOINT_FACTOR * settings->maxNumFeatures){
        keypoints = keypoints_;
        descriptors = descriptors_;
        return;
    }

    // Retain best keypoints using octree distribution
    keypoints = DistributeOctTree(keypoints_, 0, img.grayImg.cols-1, 0, img.grayImg.rows-1, settings->maxNumFeatures, 0);
    descriptors.create((int)keypoints.size(), descriptors_.cols, descriptors_.type());
    for (int i = 0; i < (int)keypoints.size(); ++i)
        descriptors_.row(keypoints[i].class_id).copyTo(descriptors.row(i));
}

int AF_VSLAM::FeatureExtractor_aliked128::GetKeypointOctave(const cv::KeyPoint&) const{
    return 0;
}

float AF_VSLAM::FeatureExtractor_aliked128::GetKeypointSize(const cv::KeyPoint&) const{
    return 1.0f;
}

float AF_VSLAM::DescriptorDistance_aliked128(const cv::Mat &a, const cv::Mat &b){
    return (Descriptor_Distance_Type) cv::norm(a, b, cv::NORM_L2);
}
