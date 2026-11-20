#pragma once

#include <list>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "Types.h"
#include "Image.h"
#include "KeypointData.h"

namespace AF_VSLAM
{

class FeatureExtractor;
class ExtractorNode;

class ExtractorNode
{
public:
    ExtractorNode():no_more(false){}

    void divide_node(ExtractorNode &top_left, ExtractorNode &top_right, ExtractorNode &lower_left, ExtractorNode &lower_right);

    std::vector<cv::KeyPoint> keys;
    cv::Point2i upper_left, upper_right, bottom_left, bottom_right;
    std::list<ExtractorNode>::iterator self_in_list;  
    bool no_more;
};

class FeatureExtractorSettings
{
public:

    int max_num_features;   // keypoints per frame: Tracking's share of the feature's get_max_features(); 0: none

    FeatureExtractorSettings(const FeatureType& feature_type, int max_num_features);

    FeatureType feature_type;

    const float oversize_keypoint_factor{1.05f};
};

class FeatureExtractor
{
public:

    FeatureExtractor(std::shared_ptr<FeatureExtractorSettings>& settings_): settings(settings_){}
    virtual ~FeatureExtractor() = default;

    std::shared_ptr<FeatureExtractorSettings> settings{};

    void operator()(const Image& img, KeypointData& data);

    // Changes the keypoint budget (settings->max_num_features). An extractor that derives something from
    // it at construction redoes that here (orb32: the per-level split; sift128: cv::SIFT's nfeatures).
    virtual void set_budget(int max_num_features) { settings->max_num_features = max_num_features; }

protected:

    void compute_size(KeypointData& data);
    void compute_sigma(KeypointData& data);

    [[nodiscard]] virtual float get_keypoint_size(const cv::KeyPoint& keypoint) const = 0;

    virtual void detect_and_compute(const Image&, std::vector<cv::KeyPoint>&, cv::Mat&){}
    std::vector<cv::KeyPoint> distribute_oct_tree(std::vector<cv::KeyPoint>& keys_to_distribute, const int &min_x,
        const int &max_x, const int &min_y, const int &max_y, const int &max_keypoints);
    void filter_keypoints_by_mask(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, const cv::Mat& mask) const;

    static void flatten_keypoint_scale(std::vector<cv::KeyPoint>& keypoints);

    void select_keypoints(const Image& img,
                          std::vector<cv::KeyPoint>& detected, cv::Mat& detected_descriptors,
                          std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
};

}
