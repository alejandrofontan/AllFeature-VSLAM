#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "Types.h"

namespace AF_VSLAM
{

// Everything measured per keypoint, for ONE feature type.
//
// Filled by three producers, which is why this type lives in a file of its own rather than
// beside any one of them: FeatureExtractor fills keypoints, descriptors and the scale arrays;
// Frame::undistort_keypoints replaces keypoints with their undistorted positions;
// Frame::get_depth and Frame::get_colors fill the rest.
//
// All vectors are indexed by KeypointIndex and have the same length, so the count is
// keypoints.size() -- there is no separate N to keep in step.
//
// sigma2, information and size are stored rather than derived from one another, so that a
// keypoint's size and its uncertainty can differ and so that the uncertainty can be genuinely
// anisotropic. Today's only producer is isotropic: size holds s, sigma2 holds s^2*I and
// information (1/s^2)*I.
struct KeypointData
{
    std::vector<cv::KeyPoint> keypoints;      // undistorted
    cv::Mat descriptors;                      // one row per keypoint
    std::vector<float> inv_depth;             // 0 where no valid depth
    std::vector<float> sigma2_inv_depth;      // 0 where no valid depth
    std::vector<cv::Vec3b> colors;            // BGR under each keypoint

    std::vector<mat2f> sigma2;
    std::vector<mat2f> information;
    std::vector<float> size;

    [[nodiscard]] std::size_t count() const { return keypoints.size(); }
};

// Every feature type's keypoint data, as Frame and KeyFrame hold it. Frame and KeyFrame used
// to declare five or six parallel maps each, plus their own copies of the accessors below --
// so a lookup for a feature type's keypoints and one for its descriptors were two separate
// tree descents.
struct FeatureData
{
    std::map<FeatureType, KeypointData> per_type;

    // trace/2 -- the isotropic summary of what may be an anisotropic covariance, not an
    // average of a number with itself (which is all it looks like while the one producer
    // fills both diagonal entries equally).
    [[nodiscard]] float get_sigma2(KeypointIndex keypoint_index, FeatureType feature_type) const
    {
        const mat2f& m = element(at(feature_type).sigma2, keypoint_index, feature_type);
        return 0.5f * (m(0,0) + m(1,1));
    }

    [[nodiscard]] mat2f get_information_2d(KeypointIndex keypoint_index, FeatureType feature_type) const
    {
        return element(at(feature_type).information, keypoint_index, feature_type);
    }

    // std::map::at and std::vector::at throw a std::out_of_range reading only "map::at" or
    // "vector::at" -- in a crash log that names neither which of the two lookups failed, nor
    // the feature type or the index. Nothing catches it, so that bare message is all the run
    // leaves behind. Same reason KeyFrame.cc wraps its map-point slots.
    [[nodiscard]] const KeypointData& at(FeatureType feature_type) const
    {
        const auto it = per_type.find(feature_type);
        if(it == per_type.end())
            throw std::out_of_range("FeatureData: no feature type "
                                    + std::to_string(static_cast<int>(feature_type)));
        return it->second;
    }

    [[nodiscard]] bool has(FeatureType feature_type) const
    {
        return per_type.find(feature_type) != per_type.end();
    }

    [[nodiscard]] static const mat2f& element(const std::vector<mat2f>& values,
                                              KeypointIndex keypoint_index, FeatureType feature_type)
    {
        if(keypoint_index < 0 || static_cast<std::size_t>(keypoint_index) >= values.size())
            throw std::out_of_range("FeatureData: feature type "
                                    + std::to_string(static_cast<int>(feature_type))
                                    + ", keypoint index " + std::to_string(keypoint_index)
                                    + " outside [0, " + std::to_string(values.size()) + ")");
        return values[static_cast<std::size_t>(keypoint_index)];
    }
};

}
