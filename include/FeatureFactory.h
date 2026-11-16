#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "afvslam_log.hpp"
#include "Feature_orb32.h"
#include "Feature_kaze64.h"
// brisk / akaze come from standalone conda packages that not every platform has yet (CMakeLists.txt)
#ifdef ALLFEATURE_WITH_AKAZE
#include "Feature_akaze61.h"
#endif
#ifdef ALLFEATURE_WITH_BRISK
#include "Feature_brisk48.h"
#endif
// The learned features run on ONNX Runtime on every platform, SIFT on
// OpenCV's CPU implementation: no CMake option gates them any more. A feature compiled out (brisk / akaze without
// their conda package) keeps its FeatureType value (Types.h) but has no Feature instance, so requesting it in the
// settings is an error at startup (feature_compiled_out below) instead of a link error.
#include "Feature_sift128.h"
#include "Feature_aliked128.h"
#include "Feature_superpoint256.h"

// Returns a stateless singleton per feature type (every Feature virtual is const).
// Previously returned a fresh unique_ptr per call — a heap allocation inside every
// descriptor_distance/match dispatch, i.e. per descriptor comparison in the fuse and
// compute_distinctive_descriptors inner loops (issue #13 addendum, factory-singletons item).
[[noreturn]] inline void feature_compiled_out(const std::string& feature, const std::string& cmake_option) {
AF_ERROR("Feature '" << feature << "' is not available in this build: it needs " << cmake_option
         << ". Remove it from `features:` in the settings yaml or rebuild with it.");
std::terminate();
}

inline const AF_VSLAM::Feature& get_feature(const FeatureType& feature_type) {
static const AF_VSLAM::Orb32 orb32;
#ifdef ALLFEATURE_WITH_AKAZE
static const AF_VSLAM::Akaze61 akaze61;
#endif
#ifdef ALLFEATURE_WITH_BRISK
static const AF_VSLAM::Brisk48 brisk48;
#endif
static const AF_VSLAM::Kaze64 kaze64;
static const AF_VSLAM::Sift128 sift128;
static const AF_VSLAM::Aliked128 aliked128;
static const AF_VSLAM::Superpoint256 superpoint256;

switch (feature_type) {
    case FEAT_ORB32:
        return orb32;
    case FEAT_AKAZE61:
#ifdef ALLFEATURE_WITH_AKAZE
        return akaze61;
#else
        feature_compiled_out("akaze61", "the akaze conda package (find_package(akaze))");
#endif
    case FEAT_BRISK48:
#ifdef ALLFEATURE_WITH_BRISK
        return brisk48;
#else
        feature_compiled_out("brisk48", "the brisk conda package (find_package(brisk))");
#endif
    case FEAT_KAZE64:
        return kaze64;
    case FEAT_SIFT128:
        return sift128;
    case FEAT_ALIKED128:
        return aliked128;
    case FEAT_SUPERPOINT256:
        return superpoint256;
    default:
        // Not reachable from a settings yaml -- get_feature_type below rejects unknown names,
        // so every value arriving here came from the enum. It fires when a FeatureType is added
        // to Types.h without a case above.
        throw std::runtime_error("get_feature: no Feature for FeatureType value "
                                 + std::to_string(static_cast<int>(feature_type))
                                 + "; a value was added to the enum without a case in the factory");
}
}

// The names accepted in `features:` of the settings yaml, paired with their enum value. One
// table rather than an if-chain, so the names a bad entry is checked against and the names
// listed back to the user cannot drift apart.
inline const std::vector<std::pair<std::string, FeatureType>>& feature_name_table() {
static const std::vector<std::pair<std::string, FeatureType>> table{
    {"orb32",         FEAT_ORB32},
    {"akaze61",       FEAT_AKAZE61},
    {"brisk48",       FEAT_BRISK48},
    {"kaze64",        FEAT_KAZE64},
    {"sift128",       FEAT_SIFT128},
    {"aliked128",     FEAT_ALIKED128},
    {"superpoint256", FEAT_SUPERPOINT256},
};
return table;
}

// The FeatureType of a settings-yaml feature name.
// @throws std::runtime_error naming the unknown feature and listing the known ones.
inline FeatureType get_feature_type(const std::string& str) {
for (const auto& [name, feature_type] : feature_name_table())
    if (name == str)
        return feature_type;

std::string known;
for (const auto& [name, feature_type] : feature_name_table())
    known += (known.empty() ? "" : ", ") + name;

throw std::runtime_error("unknown feature '" + str + "' in `features:`; known features are "
                         + known);
}

