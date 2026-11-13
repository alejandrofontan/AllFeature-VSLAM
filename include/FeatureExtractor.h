#pragma once

#include <list>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "Types.h"
#include "Definitions.h"
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
    std::list<ExtractorNode>::iterator self_in_list;   // this node's own slot, so it can erase itself
    bool no_more;
};

// What every extractor reads from its feature's settings yaml: the keypoint budget. Anything else is specific to
// one extractor, which reads it itself through required_int / required_float - so a key in a feature's yaml is a
// key that feature uses, and a parameter added to one extractor touches no other yaml. (nOctaves, scaleFactor
// and detectTh used to be read here for all eight features; only ORB uses the first two, ORB and AKAZE the third.)
class FeatureExtractorSettings
{
public:

    // The keypoint budget, FeatureExtractor.maxNumFeatures. Every extractor honours it, but at
    // a different point in its pipeline, so it is not the same promise in each case:
    //   brisk48, kaze64, akaze61, aliked128  the detector runs unbounded and select_keypoints
    //                                        spreads the result down to this many -- a TARGET,
    //                                        which distribute_oct_tree may exceed by up to 3
    //   orb32                                split across pyramid levels as features_per_level,
    //                                        and applied per level, not to the frame total
    //   sift128                              cv::SIFT's nfeatures: a pre-filter by contrast,
    //                                        applied before SiftCpu's duplicate removal, so the
    //                                        count that survives is lower
    //   superpoint256                        top-K inside the graph's C++ tail, after the score
    //                                        threshold -- a hard cap
    // Tracking also multiplies it by scale_num_features for the denser initialization pass.
    int max_num_features;

    FeatureExtractorSettings(const FeatureType& feature_type,
                             const std::string &settings_yaml_file);

    // A key of the settings yaml this extractor cannot run without.
    // @throws std::runtime_error naming the file and the key when it is missing or not a number.
    [[nodiscard]] int required_int(const std::string& key) const;
    [[nodiscard]] float required_float(const std::string& key) const;

    std::string settings_yaml_file;

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

protected:

    void compute_size(KeypointData& data);
    void compute_sigma(KeypointData& data);

    [[nodiscard]] virtual float get_keypoint_size(const cv::KeyPoint& keypoint) const = 0;

    virtual void detect_and_compute(const Image&, std::vector<cv::KeyPoint>&, cv::Mat&){}
    std::vector<cv::KeyPoint> distribute_oct_tree(std::vector<cv::KeyPoint>& keys_to_distribute, const int &min_x,
        const int &max_x, const int &min_y, const int &max_y, const int &max_keypoints);
    void filter_keypoints_by_mask(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, const cv::Mat& mask) const;

    // Discard the detector's scale and pyramid level, flattening every keypoint to size 1 at
    // octave 0. Used by the extractors whose get_keypoint_size() returns a constant, which is
    // every multi-scale detector here except orb32 -- so the uncertainty compute_sigma derives
    // for them is the identity. Centralised so that decision has one place to be revisited.
    static void flatten_keypoint_scale(std::vector<cv::KeyPoint>& keypoints);

    // The tail shared by every extractor whose detector enforces no keypoint budget of its own:
    // mask out, keep everything if the detector stayed near the budget, and otherwise spread
    // the keypoints over the frame with the quadtree, gathering the descriptor rows to match.
    // `detected` and `detected_descriptors` are the detector's raw output and are consumed.
    void select_keypoints(const Image& img,
                          std::vector<cv::KeyPoint>& detected, cv::Mat& detected_descriptors,
                          std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
};

}
