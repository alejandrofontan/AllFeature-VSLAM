//
// Created by fontan on 15/02/24.
//

#ifndef ANYFEATURE_VSLAM_TYPES_H
#define ANYFEATURE_VSLAM_TYPES_H

#include <Eigen/Dense>
#include <iostream>

enum FeatureType {
    FEAT_SUPERPOINT256 = 10,
    FEAT_ALIKED128 = 9,
    FEAT_ANYFEATNONBIN = 8,
    FEAT_ANYFEATBIN = 7,
    FEAT_R2D2 = 6,
    FEAT_SIFT128 = 5,
    FEAT_KAZE64 = 4,
    FEAT_SURF64 = 3,
    FEAT_BRISK = 2,
    FEAT_AKAZE61 = 1,
    FEAT_ORB = 0,
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


enum VerbosityLevel{
    NONE = 0,
    LOW = 1,
    MEDIUM = 2,
    HIGH = 3,
    ABLATION = 4
};

#endif //ANYFEATURE_VSLAM_TYPES_H
