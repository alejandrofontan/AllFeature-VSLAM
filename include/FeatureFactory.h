
#ifndef AF_VSLAM_FEATURE_FACTORY_H
#define AF_VSLAM_FEATURE_FACTORY_H

#include <memory>
#include "Feature_orb32.h"
#include "Feature_akaze61.h"
#include "Feature_brisk48.h"
#include "Feature_surf64.h"
#include "Feature_kaze64.h"
#include "Feature_sift128.h"
#include "Feature_aliked128.h"
#include "Feature_superpoint256.h"

// Returns a stateless singleton per feature type (every Feature virtual is const).
// Previously returned a fresh unique_ptr per call — a heap allocation inside every
// descriptor_distance/match dispatch, i.e. per descriptor comparison in the fuse and
// ComputeDistinctiveDescriptors inner loops (issue #13 addendum, factory-singletons item).
inline const AF_VSLAM::Feature& get_feature(const FeatureType& featureType) {
    static const AF_VSLAM::Orb32 orb32;
    static const AF_VSLAM::Akaze61 akaze61;
    static const AF_VSLAM::Brisk48 brisk48;
    static const AF_VSLAM::Surf64 surf64;
    static const AF_VSLAM::Kaze64 kaze64;
    static const AF_VSLAM::Sift128 sift128;
    static const AF_VSLAM::Aliked128 aliked128;
    static const AF_VSLAM::Superpoint256 superpoint256;

    switch (featureType) {
        case FEAT_ORB32:
            return orb32;
        case FEAT_AKAZE61:
            return akaze61;
        case FEAT_BRISK48:
            return brisk48;
        case FEAT_SURF64:
            return surf64;
        case FEAT_KAZE64:
            return kaze64;
        case FEAT_SIFT128:
            return sift128;
        case FEAT_ALIKED128:
            return aliked128;
        case FEAT_SUPERPOINT256:
            return superpoint256;
        default:{
            std::cout <<"get_feature"<< std::endl;
            std::terminate();
        }
    }
}

inline FeatureType get_feature_type(const std::string& str) {
    if (str == "orb32") {
        return FEAT_ORB32;
    } else if (str == "akaze61") {
        return FEAT_AKAZE61;
    } else if (str == "brisk48") {
        return FEAT_BRISK48;
    } else if (str == "surf64") {
        return FEAT_SURF64;
    } else if (str == "kaze64") {
        return FEAT_KAZE64;
    } else if (str == "sift128") {
        return FEAT_SIFT128;
    } else if (str == "aliked128") {
        return FEAT_ALIKED128;
    } else if (str == "superpoint256") {
        return FEAT_SUPERPOINT256;
    } else {
        std::terminate();
    }
}

#endif //AF_VSLAM_FEATURE_FACTORY_H
