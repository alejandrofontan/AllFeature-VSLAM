#pragma once

#include <iostream>
#include <memory>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/core.hpp>

#include "Types.h"

namespace AF_VSLAM
{

class FeatureExtractor;
class FeatureExtractorSettings;

class Feature
{
public:
    virtual ~Feature() = default;

    virtual const std::string& get_feature_name() const = 0;
    virtual FeatureType get_type() const = 0;
    virtual MatcherType get_matcher_type() const = 0;
    virtual const std::string& get_settings_yaml_file() const = 0;
    virtual const Eigen::Matrix<float,3,1>& get_color() const = 0;
    virtual float descriptor_distance(const cv::Mat &, const cv::Mat &) const { return 0.0f; }
    virtual std::shared_ptr<FeatureExtractor> create_extractor(
        std::shared_ptr<FeatureExtractorSettings>) const { return nullptr; }
};

}
