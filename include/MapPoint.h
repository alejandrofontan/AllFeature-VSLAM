#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "Types.h"
#include "KeyFrame.h"
#include "Observation.h"

namespace AF_VSLAM
{

class Map;

class MapPoint;
typedef std::shared_ptr<MapPoint> Pt;

class MapPoint: public std::enable_shared_from_this<MapPoint>
{
public:

    // ========================================================================
    // Construction
    // ========================================================================

    MapPoint(const vec3f& world_pos, Keyframe ref_keyframe, std::shared_ptr<Map> map,
             const FeatureType& featureType, const cv::Vec3b& color = cv::Vec3b(0, 0, 0));
    std::shared_ptr<MapPoint> this_point() {
        return shared_from_this();
    }

    static void LoadParameters(const cv::FileStorage &fSettings);

    // ========================================================================
    // Position
    // ========================================================================

    void set_world_pos(const vec3f& world_pos);
    vec3f get_world_pos() const;

    // ========================================================================
    // Observations
    // ========================================================================

    void add_observation(const Keyframe& projKeyframe, const KeypointIndex& projIndex);
    void add_observations(const std::vector<std::pair<Keyframe, KeypointIndex>>& new_observations);
    void erase_observation(const Keyframe& projKeyframe);

    std::map<KeyframeId,Obs> get_observations() const;

    [[nodiscard]] std::vector<std::pair<KeyframeId, Keyframe>> get_observers() const;

    Keyframe get_reference_keyframe() const;

    int number_of_observations() const;
    int num_observing_keyframes() const;

    int get_index_in_keyframe(const Keyframe& keyframe) const;
    bool is_in_keyframe(const Keyframe& keyframe) const;

    // ========================================================================
    // Descriptor
    // ========================================================================

    Pt compute_distinctive_descriptors();

    cv::Mat get_descriptor() const;

    // ========================================================================
    // Normal, scale and depth
    // ========================================================================

    void update_normal_and_depth();

    vec3f get_normal() const;

    float get_min_distance_invariance() const;
    float get_max_distance_invariance() const;

    float predict_size(const float &currentDist) const;

    // ========================================================================
    // Tracking statistics
    // ========================================================================

    void increase_visible(int n=1);
    void increase_found(int n=1);
    float get_found_ratio() const;

    // ========================================================================
    // Lifecycle
    // ========================================================================

    void set_bad_flag();
    bool is_bad() const;

    void replace(const Pt& replacement);
    Pt get_replaced() const;

    // ========================================================================
    // Data
    // ========================================================================

    static inline PtId next_id{0};

    PtId ptId;
    KeyframeId first_keyframe_id;

    FeatureType featureType;

    // Image color (OpenCV-native BGR) of the reference keypoint at creation.
    const cv::Vec3b color;

    // Variables used by the tracking
    FrameId last_frame_seen{0};

    // Variables used by local mapping
    long unsigned int ba_local_for_keyframe;
    long unsigned int fuse_candidate_for_keyframe;

    // Variables used by loop closing
    long unsigned int loop_point_for_keyframe;
    long unsigned int corrected_by_keyframe;
    long unsigned int corrected_reference;
    vec3f position_gba{vec3f::Zero()};
    long unsigned int ba_global_for_keyframe;

protected:

    // ========================================================================
    // Position
    // ========================================================================

    // Position in absolute coordinates
    vec3f position_;

    // ========================================================================
    // Observations
    // ========================================================================

    static inline int depth_observation_weight{2};
    static inline int min_observing_keyframes{2};

    [[nodiscard]] int observation_weight(const Keyframe& projKeyframe, const KeypointIndex& projIndex) const;

    void increase_observability(const Keyframe& projKeyframe, const KeypointIndex& projIndex);
    void decrease_observability(const Keyframe& projKeyframe, const KeypointIndex& projIndex);

    std::map<KeyframeId,Obs> observations_;

    int num_observations_;

    Keyframe ref_keyframe_;

    // ========================================================================
    // Descriptor
    // ========================================================================

    // Best descriptor to fast matching
    cv::Mat descriptor_;

    // ========================================================================
    // Normal, scale and depth
    // ========================================================================

    // Mean viewing direction
    vec3f normal_;

    float ref_distance_{0.0f};
    float ref_size_{0.0f};
    float min_distance_{0.0f};
    float max_distance_{0.0f};

    static constexpr float reference_keypoint_size{1.5f};
    static inline float depth_invariance_ratio{1.2f};

    // ========================================================================
    // Tracking statistics
    // ========================================================================

    // Tracking counters
    int num_visible_;
    int num_found_;

    // ========================================================================
    // Lifecycle
    // ========================================================================

    bool bad_;
    Pt replaced_;

    // ========================================================================
    // Data
    // ========================================================================

    std::shared_ptr<Map> map_;

    // ========================================================================
    // Locks
    // ========================================================================

    mutable std::mutex position_mutex_;
    mutable std::mutex features_mutex_;
};

}
