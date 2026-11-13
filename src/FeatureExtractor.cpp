#include "FeatureExtractor.h"
#include "Utils.h"
#include "afvslam_log.hpp"

#include <algorithm>
#include <vector>

#include <opencv2/core/core.hpp>

#include <yaml-cpp/yaml.h>

using namespace cv;
using namespace std;

namespace
{
// Smallest sigma compute_sigma will invert. A tenth of a pixel: far below any real keypoint
// localisation accuracy, so it never clips a genuine measurement, and far above the point
// where 1/sigma^2 stops being representable.
constexpr float min_keypoint_sigma{0.1f};
}

AF_VSLAM::FeatureExtractorSettings::FeatureExtractorSettings(
        const FeatureType& feature_type,
        const std::string &settings_yaml_file):
        settings_yaml_file(settings_yaml_file), feature_type(feature_type){

    max_num_features = required_int("FeatureExtractor.maxNumFeatures");

    AF_INFO("Loading Feature Extractor Settings from : " + settings_yaml_file);
    AF_CONFIG_BEGIN("Feature Extractor Settings");
    AF_CONFIG_FIELD("maxNumFeatures:     ", max_num_features);
    AF_CONFIG_END();
}

namespace
{
// One key of a settings yaml as T. The file is loaded per call: this runs a handful of times at start-up, and it
// keeps yaml-cpp out of FeatureExtractor.h, which most of the library includes.
template <typename T>
T required_key(const std::string& settings_yaml, const std::string& key)
{
    YAML::Node settings;
    try
    {
        settings = YAML::LoadFile(settings_yaml);
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("cannot read " + settings_yaml + ": " + e.what());
    }
    if (!settings[key])
        throw std::runtime_error(settings_yaml + ": missing key '" + key + "'");
    try
    {
        return settings[key].as<T>();
    }
    catch (const std::exception&)
    {
        throw std::runtime_error(settings_yaml + ": key '" + key + "' is not a number");
    }
}
} // namespace

int AF_VSLAM::FeatureExtractorSettings::required_int(const std::string& key) const
{
    return required_key<int>(settings_yaml_file, key);
}

float AF_VSLAM::FeatureExtractorSettings::required_float(const std::string& key) const
{
    return required_key<float>(settings_yaml_file, key);
}

void AF_VSLAM::FeatureExtractor::operator()(const Image& img, KeypointData& data)
{
    setup_image(img);
    detect_and_compute(img, data.keypoints, data.descriptors);
    compute_size(data);
    compute_sigma(data);
}

void AF_VSLAM::FeatureExtractor::compute_size(KeypointData& data){
    const std::vector<cv::KeyPoint>& keypoints = data.keypoints;
    data.size.clear();
    data.size.reserve(keypoints.size());
    for(const auto& keypoint : keypoints)
        data.size.push_back(get_keypoint_size(keypoint));
}

void AF_VSLAM::FeatureExtractor::compute_sigma(KeypointData& data){
    std::vector<mat2f>& sigma2 = data.sigma2;
    std::vector<mat2f>& information = data.information;
    const std::vector<float>& sizes = data.size;
    sigma2.clear();
    information.clear();
    sigma2.reserve(sizes.size());
    information.reserve(sizes.size());
    for(size_t i = 0; i < sizes.size(); i++){
        const float sigma = std::max(sizes[i], min_keypoint_sigma);
        const float sigma_squared = sigma * sigma;
        sigma2.emplace_back(sigma_squared * mat2f::Identity());
        information.emplace_back((1.0f / sigma_squared) * mat2f::Identity());
    }
}

// Deterministic ordering for the node-expansion queue: the default pair
// comparison broke size ties by ExtractorNode POINTER value, which differs
// between runs (allocator/ASLR) and made the selected keypoints run-dependent
// (issue #16; same fix as ORB_SLAM2_Deterministic). Size first (the loop pops
// largest-first from the back), node x-coordinate on ties.
static bool node_comparison(const std::pair<int, AF_VSLAM::ExtractorNode*>& a,
                            const std::pair<int, AF_VSLAM::ExtractorNode*>& b)
{
    return (a.first != b.first) ? (a.first < b.first) : (a.second->upper_left.x < b.second->upper_left.x);
}

// Move one freshly divided child into the node list and, if it still holds more than one
// keypoint, register it as a candidate for the next round of splitting.
//
// Returns whether the child was registered, so the first of the two expansion loops below can
// count how many nodes the next round would split. That count was the only difference between
// the two copies of this block. Everything else has to stay exactly as it is: push_front order
// decides the final node order and so which keypoint survives each node, the address is only
// valid after the push, and self_in_list is set only for expandable children because it is how
// the second loop erases a parent it reached through a pointer rather than an iterator.
static bool push_child(const AF_VSLAM::ExtractorNode& child,
                       std::list<AF_VSLAM::ExtractorNode>& nodes,
                       std::vector<std::pair<int, AF_VSLAM::ExtractorNode*> >& expandable)
{
    if(child.keys.empty())
        return false;

    nodes.push_front(child);
    if(child.keys.size() <= 1)
        return false;

    expandable.push_back(std::make_pair(static_cast<int>(child.keys.size()), &nodes.front()));
    nodes.front().self_in_list = nodes.begin();
    return true;
}

// Spread keypoints evenly over the region by recursive quadtree splitting, then keep the
// highest-response keypoint in each surviving node. Returns the survivors, with class_id set to
// each one's index in keys_to_distribute so the caller can pick the matching descriptor rows.
//
// max_keypoints is a TARGET, not a hard cap, and the result can exceed it by up to three. The
// splitting works in rounds, and a round that splits one node replaces it with up to four, so
// crossing the target mid-round overshoots by that round's remainder. Two things keep the
// overshoot small: a full round only runs when the predicted result (nodes + 3 per expandable
// node) still fits, and the round that would not fit is replayed one parent at a time, stopping
// as soon as the count reaches the target. The one case that can overshoot further is a target
// smaller than four, where the very first round has nothing to predict from.
//
// This is deliberate, not a defect to fix. Every caller already tolerates it: the four
// classical extractors size their descriptor matrix from the returned count, and they only
// call this at all once the detector has produced more than oversize_keypoint_factor (1.05x)
// of the budget, so a handful over the target is well inside what they accept. Capping here
// would change which keypoints survive, and so every downstream trajectory.
std::vector<cv::KeyPoint> AF_VSLAM::FeatureExtractor::distribute_oct_tree(std::vector<cv::KeyPoint>& keys_to_distribute, const int &min_x,
                                       const int &max_x, const int &min_y, const int &max_y, const int &max_keypoints)
{
    // One root node per square tile across the region, so the tree starts roughly isotropic.
    //
    // Both clamps are load-bearing. round() returns 0 for any aspect ratio below 0.5 -- a
    // portrait or tall-crop region -- which gives an infinite node_width, an empty
    // initial_nodes, and a node_index of -1 from the clamp below; a zero region height divides
    // by zero before that. orb32 never reaches either: it hands in a landscape strip inset by
    // the ORB border. The extractors that pass the raw image bounds would, on a portrait frame.
    const int region_width = max_x - min_x;
    const int region_height = std::max(1, max_y - min_y);
    const int num_initial = std::max(1, static_cast<int>(
        std::round(static_cast<float>(region_width) / static_cast<float>(region_height))));

    const float node_width = static_cast<float>(region_width) / static_cast<float>(num_initial);

    std::list<ExtractorNode> nodes;

    std::vector<ExtractorNode*> initial_nodes;
    initial_nodes.resize(num_initial);

    for(int i=0; i<num_initial; i++)
    {
        ExtractorNode node;
        node.upper_left = cv::Point2i(node_width*static_cast<float>(i),0);
        node.upper_right = cv::Point2i(node_width*static_cast<float>(i+1),0);
        node.bottom_left = cv::Point2i(node.upper_left.x,max_y-min_y);
        node.bottom_right = cv::Point2i(node.upper_right.x,max_y-min_y);
        // The expected share, not the whole set: every root node used to reserve room for every
        // keypoint in the image, so the reservation cost scaled with num_initial squared.
        node.keys.reserve(keys_to_distribute.size() / num_initial + 1);

        nodes.push_back(node);
        initial_nodes[i] = &nodes.back();
    }

    // Associate points to childs. class_id carries the keypoint's index in keys_to_distribute
    // through to the caller, which uses it to pick the matching descriptor rows for the
    // survivors -- see the four classical extractors.
    for(size_t i=0;i<keys_to_distribute.size();i++)
    {
        cv::KeyPoint &keypoint = keys_to_distribute[i];
        keypoint.class_id = i;

        // Clamped, because pt.x == max_x lands one past the last node: node_width is
        // (max_x-min_x)/num_initial, so that keypoint divides to exactly num_initial. On a
        // 640x480 frame num_initial is 1 and node_width 639, so a keypoint in the last column
        // indexed initial_nodes[1] of a one-element vector -- a garbage ExtractorNode* that was
        // then written through. orb32 never hit it (it passes a border-inset rectangle); the
        // callers that pass the raw 0..cols-1 image bounds could.
        const int node_index = std::clamp(static_cast<int>(keypoint.pt.x / node_width),
                                          0, num_initial - 1);
        initial_nodes[node_index]->keys.push_back(keypoint);
    }

    std::list<ExtractorNode>::iterator node_it = nodes.begin();

    while(node_it!=nodes.end())
    {
        if(node_it->keys.size()==1)
        {
            node_it->no_more=true;
            node_it++;
        }
        else if(node_it->keys.empty())
            node_it = nodes.erase(node_it);
        else
            node_it++;
    }

    bool finished = false;

    // Both queues live out here so their capacity survives every round: the inner loop swaps
    // between them rather than copying, so after the first round neither allocates again.
    std::vector<std::pair<int,ExtractorNode*> > expandable;
    std::vector<std::pair<int,ExtractorNode*> > previous_expandable;
    expandable.reserve(nodes.size()*4);

    while(!finished)
    {
        int previous_size = nodes.size();

        node_it = nodes.begin();

        int num_to_expand = 0;

        expandable.clear();

        while(node_it!=nodes.end())
        {
            if(node_it->no_more)
            {
                // If node only contains one point do not subdivide and continue
                node_it++;
                continue;
            }
            else
            {
                // If more than one point, subdivide
                ExtractorNode top_left,top_right,lower_left,lower_right;
                node_it->divide_node(top_left,top_right,lower_left,lower_right);

                // Add childs if they contain points
                num_to_expand += push_child(top_left, nodes, expandable);
                num_to_expand += push_child(top_right, nodes, expandable);
                num_to_expand += push_child(lower_left, nodes, expandable);
                num_to_expand += push_child(lower_right, nodes, expandable);

                node_it=nodes.erase(node_it);
                continue;
            }
        }

        // Finish if there are more nodes than required features
        // or all nodes contain just one point
        if((int)nodes.size()>=max_keypoints || (int)nodes.size()==previous_size)
        {
            finished = true;
        }
        else if(((int)nodes.size()+num_to_expand*3)>max_keypoints)
        {

            while(!finished)
            {

                previous_size = nodes.size();

                previous_expandable.swap(expandable);
                expandable.clear();

                sort(previous_expandable.begin(),previous_expandable.end(), node_comparison);
                for(int j=previous_expandable.size()-1;j>=0;j--)
                {
                    ExtractorNode top_left,top_right,lower_left,lower_right;
                    previous_expandable[j].second->divide_node(top_left,top_right,lower_left,lower_right);

                    // Add childs if they contain points. The expansion count the first loop
                    // keeps is not needed here: this loop re-checks nodes.size() against the
                    // budget after every parent instead of predicting it a round ahead.
                    push_child(top_left, nodes, expandable);
                    push_child(top_right, nodes, expandable);
                    push_child(lower_left, nodes, expandable);
                    push_child(lower_right, nodes, expandable);

                    nodes.erase(previous_expandable[j].second->self_in_list);

                    if((int)nodes.size()>=max_keypoints)
                        break;
                }

                if((int)nodes.size()>=max_keypoints || (int)nodes.size()==previous_size)
                    finished = true;

            }
        }
    }

    // Retain the best point in each node. One survivor per node, so the count is already known
    // exactly -- it used to reserve settings->max_num_features, the whole-image budget, which
    // for orb32 is the sum over every pyramid level of what this one call can return.
    std::vector<cv::KeyPoint> result;
    result.reserve(nodes.size());
    for(std::list<ExtractorNode>::iterator node_it=nodes.begin(); node_it!=nodes.end(); node_it++)
    {
        std::vector<cv::KeyPoint> &node_keys = node_it->keys;
        cv::KeyPoint* best = &node_keys[0];
        float best_response = best->response;

        for(size_t k=1;k<node_keys.size();k++)
        {
            if(node_keys[k].response>best_response)
            {
                best = &node_keys[k];
                best_response = node_keys[k].response;
            }
        }

        result.push_back(*best);
    }

    return result;
}

void AF_VSLAM::FeatureExtractor::filter_keypoints_by_mask(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, const cv::Mat& mask) const
{
    if (mask.empty()) return; // no mask loaded for this frame/dataset -> keep everything

    std::vector<cv::KeyPoint> kept;
    std::vector<int> kept_rows;
    kept.reserve(keypoints.size());
    kept_rows.reserve(keypoints.size());
    for (size_t i = 0; i < keypoints.size(); ++i) {
        int x = cvRound(keypoints[i].pt.x), y = cvRound(keypoints[i].pt.y);
        if (x < 0 || y < 0 || x >= mask.cols || y >= mask.rows) continue;
        if (mask.at<uchar>(y, x) == 1) {
            cv::KeyPoint keypoint = keypoints[i];
            keypoint.class_id = static_cast<int>(kept.size()); // reindex to match the compacted descriptors row order
            kept.push_back(keypoint);
            kept_rows.push_back((int)i);
        }
    }

    cv::Mat kept_descriptors(static_cast<int>(kept_rows.size()), descriptors.cols, descriptors.type());
    for (size_t i = 0; i < kept_rows.size(); ++i)
        descriptors.row(kept_rows[i]).copyTo(kept_descriptors.row(static_cast<int>(i)));

    keypoints.swap(kept);
    descriptors = kept_descriptors;
}

void AF_VSLAM::ExtractorNode::divide_node(ExtractorNode &top_left, ExtractorNode &top_right, ExtractorNode &lower_left, ExtractorNode &lower_right)
{
    const int half_x = ceil(static_cast<float>(upper_right.x-upper_left.x)/2);
    const int half_y = ceil(static_cast<float>(bottom_right.y-upper_left.y)/2);

    //Define boundaries of childs
    top_left.upper_left = upper_left;
    top_left.upper_right = cv::Point2i(upper_left.x+half_x,upper_left.y);
    top_left.bottom_left = cv::Point2i(upper_left.x,upper_left.y+half_y);
    top_left.bottom_right = cv::Point2i(upper_left.x+half_x,upper_left.y+half_y);

    top_right.upper_left = top_left.upper_right;
    top_right.upper_right = upper_right;
    top_right.bottom_left = top_left.bottom_right;
    top_right.bottom_right = cv::Point2i(upper_right.x,upper_left.y+half_y);

    lower_left.upper_left = top_left.bottom_left;
    lower_left.upper_right = top_left.bottom_right;
    lower_left.bottom_left = bottom_left;
    lower_left.bottom_right = cv::Point2i(top_left.bottom_right.x,bottom_left.y);

    lower_right.upper_left = lower_left.upper_right;
    lower_right.upper_right = top_right.bottom_right;
    lower_right.bottom_left = lower_left.bottom_right;
    lower_right.bottom_right = bottom_right;

    // Associate points to childs. No reserve: each child used to reserve room for the parent's
    // whole key set, four times over, so a split allocated 4x what it could possibly need --
    // and allocated for children that end up empty, which deep in the tree is most of them.
    // push_back's geometric growth is cheaper here, since nodes shrink fast as the tree splits.
    for(size_t i=0;i<keys.size();i++)
    {
        const cv::KeyPoint &keypoint = keys[i];
        if(keypoint.pt.x<top_left.upper_right.x)
        {
            if(keypoint.pt.y<top_left.bottom_right.y)
                top_left.keys.push_back(keypoint);
            else
                lower_left.keys.push_back(keypoint);
        }
        else if(keypoint.pt.y<top_left.bottom_right.y)
            top_right.keys.push_back(keypoint);
        else
            lower_right.keys.push_back(keypoint);
    }

    if(top_left.keys.size()==1)
        top_left.no_more = true;
    if(top_right.keys.size()==1)
        top_right.no_more = true;
    if(lower_left.keys.size()==1)
        lower_left.no_more = true;
    if(lower_right.keys.size()==1)
        lower_right.no_more = true;

}