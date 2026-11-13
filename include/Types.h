//
// Created by fontan on 15/02/24.
//

#ifndef AF_VSLAM_TYPES_H
#define AF_VSLAM_TYPES_H

#include <Eigen/Dense>
#include <iostream>

namespace AF_VSLAM {
    using Seconds = double;

    // Chi-squared critical values. The 2-DOF 95% value was written out as a literal in SIX
    // places under six names (LocalMapping::CHI2_2DOF, FeatureMatcher::chi2_perc,
    // Initializer::chiSquare_th and ::chiSquare_thScore_fund, Tracking::RANSAC_TH2,
    // OptimizerParameters::chi2_2dof) and the 99% one inline in Sim3Solver, unnamed. The local
    // names stay -- each says what its own gate is for -- but the NUMBER is defined once.
    //
    // double, matching the literals they replace: the float members below initialize from these
    // and get the same float, while Sim3Solver's double arithmetic is unchanged.
    namespace ChiSquared {
        inline constexpr double dof2_95{5.991};
        inline constexpr double dof2_99{9.210};
        inline constexpr double dof3_95{7.815};
    }
}

enum FeatureType {
    FEAT_SUPERPOINT256 = 7,
    FEAT_ALIKED128 = 6,
    FEAT_SIFT128 = 5,
    FEAT_KAZE64 = 4,
    FEAT_BRISK48 = 2,
    FEAT_AKAZE61 = 1,
    FEAT_ORB32 = 0,
};

enum MatcherType {
    LIGHTGLUE_SUPERPOINT = 4,
    LIGHTGLUE_ALIKED = 3,
    LIGHTGLUE_SIFT = 2,
    BF_L2 = 1,
    BF_HAMMING = 0
};

//typedef int Descriptor_Distance_Type;
typedef float Descriptor_Distance_Type;

using KeyframeId = long unsigned int;
using FrameId = long unsigned int;
using PtId = long unsigned int;
using KeypointIndex = int;

typedef Eigen::Matrix<double,4,4> mat4;
typedef Eigen::Matrix<double,3,3> mat3;
typedef Eigen::Matrix<double,2,2> mat2;
typedef Eigen::Matrix<float,4,4> mat4f;
typedef Eigen::Matrix<float,3,3> mat3f;
typedef Eigen::Matrix<float,2,2> mat2f;

typedef Eigen::Matrix<double,4,1> vec4;
typedef Eigen::Matrix<double,3,1> vec3;
typedef Eigen::Matrix<float,4,1> vec4f;
typedef Eigen::Matrix<float,3,1> vec3f;

typedef Eigen::Matrix<float,2,1> vec2f;

typedef Eigen::Matrix<float,3,4> mat34f;


// Tracking pipeline state (owned by AF_VSLAM::Tracking; defined here so the
// viewer/drawer classes can hold it despite the Tracking<->FrameDrawer include cycle)
enum class TrackingState : int {
    SYSTEM_NOT_READY = -1,
    NO_IMAGES_YET = 0,
    NOT_INITIALIZED = 1,
    OK = 2,
    LOST = 3
};

enum VerbosityLevel{
    NONE = 0,
    LOW = 1,
    MEDIUM = 2,
    HIGH = 3,
    ABLATION = 4
};

#endif //AF_VSLAM_TYPES_H
