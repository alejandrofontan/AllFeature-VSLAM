#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <vector>

#include <opencv2/core.hpp>

#include "Types.h"
#include "Pose.h"
#include "Frame.h"
#include "PlaceRecognition.h"


namespace AF_VSLAM
{
using namespace std;

class Map;
class Frame;

class KeyFrame;
typedef std::shared_ptr<AF_VSLAM::KeyFrame> Keyframe;

class MapPoint;
typedef std::shared_ptr<AF_VSLAM::MapPoint> Pt;

struct KeyframeIdLess
{
    bool operator()(const Keyframe& a, const Keyframe& b) const;
};
using KeyframeIdSet = std::set<Keyframe, KeyframeIdLess>;

struct Connection
{
    Keyframe keyframe;
    int weight{};
};

struct PtIdLess
{
    bool operator()(const Pt& a, const Pt& b) const;
};


[[nodiscard]] bool contains_map_point(const std::vector<Pt>& sorted_points, const Pt& pt);

class KeyFrame : public std::enable_shared_from_this<KeyFrame>
{
public:

    // ========================================================================
    // Construction
    // ========================================================================

    KeyFrame(Frame& frame, std::shared_ptr<Map> map, std::shared_ptr<PlaceRecognition> place_recognition);

    std::shared_ptr<KeyFrame> this_keyframe() {
        return shared_from_this();
    }

    // ========================================================================
    // Pose
    // ========================================================================

    void set_pose(const mat4f& Tcw_);

    [[nodiscard]] Pose get_pose() const;

    [[nodiscard]] mat4f get_Tcw() const;
    [[nodiscard]] mat4f get_Twc() const;
    [[nodiscard]] mat3f get_Rcw() const;
    [[nodiscard]] vec3f get_tcw() const;
    [[nodiscard]] vec3f get_twc() const;

    // ========================================================================
    // Per-keyframe queries
    // ========================================================================

    void compute_global_descriptor();

    [[nodiscard]] std::optional<float> compute_scene_median_depth() const;

    // ========================================================================
    // Spatial index
    // ========================================================================

    [[nodiscard]] std::vector<size_t> get_features_in_area(float x, float y, float r, const FeatureType& feature_type) const;

    // ========================================================================
    // Map points
    // ========================================================================

    [[nodiscard]] bool add_map_point(const Pt& pt, KeypointIndex index);
    void erase_map_point_match(const Pt& pt, KeypointIndex index);
    void erase_map_point_match(const Pt& pt);
    bool replace_map_point_match(const Pt& pt, KeypointIndex index, const Pt& replacement);
    [[nodiscard]] std::vector<Pt> get_good_map_points(FeatureType feature_type) const;
    [[nodiscard]] std::vector<Pt> get_map_point_matches(FeatureType feature_type) const;
    [[nodiscard]] Pt get_map_point(KeypointIndex index, FeatureType feature_type) const;
    [[nodiscard]] int count_tracked_map_points(int min_observations) const;

    // ========================================================================
    // Covisibility graph
    // ========================================================================

    static inline int min_covisibility_weight{15};

    static void LoadParameters(const cv::FileStorage &fSettings);

    void update_connections();
    void add_connection(const Keyframe& pKF, int weight);
    void erase_connection(const Keyframe& pKF);
    [[nodiscard]] std::vector<Keyframe> get_best_covisibility_keyframes(int N) const;
    [[nodiscard]] std::vector<Keyframe> get_covisibles_by_weight(int w) const;
    [[nodiscard]] int get_weight(const Keyframe& pKF) const;
    [[nodiscard]] std::map<KeyframeId,Keyframe> get_connected_keyframes() const;
    [[nodiscard]] std::vector<Keyframe> get_covisible_keyframes() const;
    [[nodiscard]] std::vector<Connection> get_covisible_connections() const;

    // ========================================================================
    // Spanning tree and loop edges
    // ========================================================================

    void add_child(const Keyframe& keyframe);
    void erase_child(const Keyframe& keyframe);
    void set_parent(const Keyframe& keyframe);
    void add_loop_edge(const Keyframe& keyframe);
    [[nodiscard]] bool has_child(const Keyframe& keyframe) const;
    [[nodiscard]] std::vector<Keyframe> get_children() const;
    [[nodiscard]] Keyframe get_parent() const;
    [[nodiscard]] KeyframeIdSet get_loop_edges() const;

    // ========================================================================
    // Lifecycle
    // ========================================================================

    void pin();
    void unpin();
    void set_bad_flag();
    [[nodiscard]] bool is_bad() const;
    [[nodiscard]] bool is_origin() const { return keyId == 0; }
    static bool less_by_id(const Keyframe& pKF1, const Keyframe& pKF2){
        return pKF1->keyId < pKF2->keyId;
    }

    // ========================================================================
    // Data
    // ========================================================================

    std::vector<FeatureType> feature_types{};
    std::map<FrameId, std::map<FeatureType, std::vector<cv::DMatch>>> cache_matched_pairs_feat_type{};
    mutable std::mutex cache_mutex_;

    static inline KeyframeId next_id{0};
    static std::mutex set_bad_mutex_; // serialises set_bad_flag's spanning-tree update across threads (see there)
    KeyframeId keyId;
    const long unsigned int frame_id;

    const double timestamp;

    // Variables used by the tracking
    long unsigned int fuse_target_for_keyframe;

    // Variables used by the local mapping
    long unsigned int ba_local_for_keyframe;
    long unsigned int ba_fixed_for_keyframe;

    // Variables used by loop closing
    mat4f Tcw_gba{mat4f::Identity()}; // Eigen leaves matrices uninitialized: never read garbage poses
    mat4f Tcw_before_gba{mat4f::Identity()};
    long unsigned int ba_global_for_keyframe;

    std::shared_ptr<const Calibration> calibration;

    const FeatureData features;

    cv::Mat image;
    Eigen::VectorXf global_descriptor;

protected:

    // ========================================================================
    // Pose
    // ========================================================================

    Pose pose_;

    // ========================================================================
    // Data
    // ========================================================================

    std::map<FeatureType, std::vector<Pt>> map_points_;

    std::shared_ptr<PlaceRecognition> place_recognition_;

    const std::vector<Connection>& ordered_connections() const;
    void rebuild_ordered_connections() const;

    std::map<KeyframeId, Connection> connections_;
    mutable std::vector<Connection> ordered_connections_;
    mutable bool ordered_dirty_{true};

    bool first_connection_;
    Keyframe parent_;
    KeyframeIdSet children_;
    KeyframeIdSet loop_edges_;

    bool pinned_;          // held against culling; see pin()
    bool cull_requested_;   // set_bad_flag was called while pinned, and is owed
    bool bad_;

    std::shared_ptr<Map> map_;

    // ========================================================================
    // Spatial index
    // ========================================================================

    std::map<FeatureType, Frame::Grid> grid_;

    // ========================================================================
    // Locks
    // ========================================================================

    mutable std::mutex pose_mutex_;
    mutable std::mutex connections_mutex_;
    mutable std::mutex features_mutex_;
};

inline bool KeyframeIdLess::operator()(const Keyframe& a, const Keyframe& b) const
{
    return a->keyId < b->keyId;
}


}
