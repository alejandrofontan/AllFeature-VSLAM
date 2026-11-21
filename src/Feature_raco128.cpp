/**
 * Module: AllFeature-VSLAM - Feature_raco128.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.2)
 * - Version: 1.0
 * - Created: 2026-11-21
 * - License: GPLv3 License
 */

#include "Feature_raco128.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "afvslam_log.hpp"
#include "onnx/FeatureModels.h"

namespace
{
// A covariance the optimiser can invert: finite, positive variances and a positive determinant. The floor is the
// variance of FeatureExtractor's smallest sigma (0.1 px).
constexpr float min_variance{0.01f};

bool is_usable(const cv::Vec3f& c)
{
    const float xx = c[0], xy = c[1], yy = c[2];
    return std::isfinite(xx) && std::isfinite(xy) && std::isfinite(yy) && xx >= min_variance && yy >= min_variance &&
           xx * yy - xy * xy >= min_variance * min_variance;
}
}

AF_VSLAM::Raco128Extractor::Raco128Extractor(std::shared_ptr<FeatureExtractorSettings> &settings_):
        FeatureExtractor(settings_){
    AF_INFO("Initializing RaCo128 feature extractor (ONNX Runtime)...");   // the session is created in prepare()
}

void AF_VSLAM::Raco128Extractor::prepare(const int width, const int height)
{
    if (!extractor_)
        load(onnx::feature_model_path(Raco128::s_model_stem), width, height);
}

void AF_VSLAM::Raco128Extractor::load(const std::string& path, const int width, const int height)
{
    extractor_ = std::make_unique<onnx::RacoOnnx>(path, onnx::feature_models_config().ort);
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    std::vector<cv::Vec3f> cov;
    extractor_->detect(cv::Mat::zeros(height, width, CV_8UC3), kps, desc, cov);   // warm-up
    AF_INFO("RaCo128 ready: " << extractor_->model_path() << " (" << extractor_->precision()
            << " weights) on onnxruntime/" << extractor_->provider() << ", budget " << settings->max_num_features
            << " keypoints of the graph's top-" << extractor_->top_k() << " by ranker score, "
            << (onnx::feature_models_config().raco_covariance ? "its covariance per keypoint" : "sigma 1 px per keypoint")
            << ", warm-up " << extractor_->last_graph_ms() << " ms");
}

void AF_VSLAM::Raco128Extractor::detect_and_compute(const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors){
    // 1. The graph's keypoints (top-k by detection score), with ranker score, covariance and descriptor
    if (!extractor_)
        load(onnx::feature_model_path(Raco128::s_model_stem), img.img.cols, img.img.rows);   // prepare() was not called
    std::vector<cv::KeyPoint> detected;
    cv::Mat detected_descriptors;
    std::vector<cv::Vec3f> detected_covariances;
    extractor_->detect(img.img, detected, detected_descriptors, detected_covariances);

    // 2. The mask first (as filter_keypoints_by_mask reads it), so that the budget is filled from the unmasked area
    std::vector<int> candidates;
    candidates.reserve(detected.size());
    for (int i = 0; i < static_cast<int>(detected.size()); ++i)
    {
        if (!img.mask.empty())
        {
            const int x = cvRound(detected[i].pt.x), y = cvRound(detected[i].pt.y);
            if (x < 0 || y < 0 || x >= img.mask.cols || y >= img.mask.rows || img.mask.at<uchar>(y, x) == 0)
                continue;
        }
        candidates.push_back(i);
    }

    // 3. The budget by ranker score (cv::KeyPoint::response), RaCo's own criterion for a smaller set
    const size_t budget = static_cast<size_t>(std::max(settings->max_num_features, 0));
    if (candidates.size() > budget)
    {
        std::partial_sort(candidates.begin(), candidates.begin() + budget, candidates.end(), [&detected](int a, int b) {
            return (detected[a].response != detected[b].response) ? (detected[a].response > detected[b].response) : (a < b);
        });
        candidates.resize(budget);
    }

    // 4. The kept keypoints, their descriptors and their covariances (for compute_sigma), in one order
    const int n = static_cast<int>(candidates.size());
    keypoints.clear();
    keypoints.reserve(n);
    covariances_.clear();
    covariances_.reserve(n);
    descriptors.create(n, detected_descriptors.cols, detected_descriptors.type());
    for (int i = 0; i < n; ++i)
    {
        const int source = candidates[i];
        keypoints.push_back(detected[source]);
        keypoints.back().class_id = i;
        covariances_.push_back(detected_covariances[source]);
        detected_descriptors.row(source).copyTo(descriptors.row(i));
    }
}

void AF_VSLAM::Raco128Extractor::compute_sigma(KeypointData& data)
{
    if (!onnx::feature_models_config().raco_covariance || covariances_.size() != data.keypoints.size())
    {
        FeatureExtractor::compute_sigma(data);   // sigma from get_keypoint_size: 1 px
        return;
    }
    data.sigma2.clear();
    data.information.clear();
    data.sigma2.reserve(covariances_.size());
    data.information.reserve(covariances_.size());
    for (const cv::Vec3f& c : covariances_)
    {
        mat2f sigma2 = mat2f::Identity();   // an unusable covariance: the fixed 1 px of the other learned features
        if (is_usable(c))
            sigma2 << c[0], c[1],
                      c[1], c[2];
        data.sigma2.push_back(sigma2);
        data.information.emplace_back(sigma2.inverse());
    }
}

float AF_VSLAM::Raco128Extractor::get_keypoint_size(const cv::KeyPoint&) const{
    return 1.0f;
}
