
#ifndef AF_VSLAM_FEATURE_FACTORY_H
#define AF_VSLAM_FEATURE_FACTORY_H

#include <memory>
#include "Feature_orb32.h"
#include "Feature_akaze61.h"
#include "Feature_brisk48.h"
#include "Feature_surf64.h"
#include "Feature_kaze64.h"
#include "Feature_sift128.h"
#include "Feature_r2d2_128.h"
#include "Feature_anyFeatBin.h"
#include "Feature_anyFeatNonBin.h"
#include "Feature_aliked128.h"
#include "Feature_superpoint256.h"

inline std::unique_ptr<AF_VSLAM::Feature> get_feature(const FeatureType& featureType) {
    switch (featureType) {
        case FEAT_ORB:
            return std::make_unique<AF_VSLAM::Orb32>();
        case FEAT_AKAZE61:
            return std::make_unique<AF_VSLAM::Akaze61>();
        case FEAT_BRISK:
            return std::make_unique<AF_VSLAM::Brisk48>();
        case FEAT_SURF64:
            return std::make_unique<AF_VSLAM::Surf64>();
        case FEAT_KAZE64:
            return std::make_unique<AF_VSLAM::Kaze64>();
        case FEAT_SIFT128:
            return std::make_unique<AF_VSLAM::Sift128>();
        case FEAT_R2D2:
            return std::make_unique<AF_VSLAM::R2d2_128>();
        case FEAT_ANYFEATBIN:
            return std::make_unique<AF_VSLAM::AnyFeatBin>();
        case FEAT_ANYFEATNONBIN:
            return std::make_unique<AF_VSLAM::AnyFeatNonBin>();
        case FEAT_ALIKED128:
            return std::make_unique<AF_VSLAM::Aliked128>();
        case FEAT_SUPERPOINT256:
            return std::make_unique<AF_VSLAM::Superpoint256>();
        default:{
            std::cout <<"get_feature"<< std::endl;
            std::terminate();
        }
    }
}

inline FeatureType get_feature_type(const std::string& str) {
    if (str == "orb32") {
        return FEAT_ORB;
    } else if (str == "akaze61") {
        return FEAT_AKAZE61;
    } else if (str == "brisk48") {
        return FEAT_BRISK;
    } else if (str == "surf64") {
        return FEAT_SURF64;
    } else if (str == "kaze64") {
        return FEAT_KAZE64;
    } else if (str == "sift128") {
        return FEAT_SIFT128;
    } else if (str == "r2d2_128") {
        return FEAT_R2D2;
    } else if (str == "anyfeatbin") {
        return FEAT_ANYFEATBIN;
    } else if (str == "anyfeatnonbin") {
        return FEAT_ANYFEATNONBIN;
    } else if (str == "aliked128") {
        return FEAT_ALIKED128;
    } else if (str == "superpoint256") {
        return FEAT_SUPERPOINT256;
    } else {
        std::terminate();
    }
}

#endif //AF_VSLAM_FEATURE_FACTORY_H