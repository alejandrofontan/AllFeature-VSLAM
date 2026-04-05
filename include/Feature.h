#ifndef ANYFEATURE_VSLAM_FEATURE_H
#define ANYFEATURE_VSLAM_FEATURE_H

#include <iostream>
#include <eigen3/Eigen/Dense>

#include "Types.h"

namespace ANYFEATURE_VSLAM {

    class Feature {
    public:
        virtual ~Feature() = default;

        virtual const std::string& getFeatureName()    const = 0;
        virtual const FeatureType  getType()           const = 0;
        virtual const Eigen::Matrix<float,3,1>& getColor() const = 0;
    };
}

#endif //ANYFEATURE_VSLAM_FEATURE_H
