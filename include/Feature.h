#ifndef AF_VSLAM_FEATURE_H
#define AF_VSLAM_FEATURE_H

#include <iostream>
#include <memory>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/core.hpp>

#include "Types.h"

namespace AF_VSLAM {

    class FeatureExtractor;
    class FeatureExtractorSettings;

    class Feature {
    public:
        virtual ~Feature() = default;

        virtual const std::string& getFeatureName()    const = 0;
        virtual const FeatureType  getType()           const = 0;
        virtual const MatcherType getMatcherType()     const = 0;
        virtual const std::string& getSettingsYamlFile()   const = 0;
        virtual const Eigen::Matrix<float,3,1>& getColor() const = 0;
        virtual float DescriptorDistance(const cv::Mat &a, const cv::Mat &b) const { return 0.0f; };
        virtual std::shared_ptr<FeatureExtractor> createExtractor(
            std::shared_ptr<FeatureExtractorSettings> settings) const { return nullptr; };
    };
}

#endif //AF_VSLAM_FEATURE_H
