#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "Types.h"
#include "KeyFrame.h"

namespace AF_VSLAM
{

class Map;

struct Observation
{
    Keyframe keyframe{};
    KeypointIndex index{};
};

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
    ~MapPoint();

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

    void add_observation(const Keyframe& keyframe, const KeypointIndex& index);
    void add_observations(const std::vector<std::pair<Keyframe, KeypointIndex>>& new_observations);
    void erase_observation(const Keyframe& keyframe);

    std::map<KeyframeId, Observation> get_observations() const;

    [[nodiscard]] std::vector<std::pair<KeyframeId, Keyframe>> get_observers() const;

    Keyframe get_reference_keyframe() const;

    // WEIGHTED count: an observation with depth counts depth_observation_weight (2), one without counts 1,
    // as ORB-SLAM2 counts stereo observations twice. Compared against thresholds by map-point culling,
    // keyframe culling (as a pre-filter) and KeyFrame::count_tracked_map_points; "is it observed at all" is
    // number_of_observations() > 0. For the plain number of keyframes use num_observing_keyframes().
    int number_of_observations() const;
    int num_observing_keyframes() const;

    int get_index_in_keyframe(const Keyframe& keyframe) const;
    bool is_in_keyframe(const Keyframe& keyframe) const;

    // ========================================================================
    // Descriptor
    // ========================================================================

    Pt compute_distinctive_descriptors();

    // Shares the buffer with the point (no copy): read-only. The point never writes it in place, it

    // replaces it as a whole, so a returned Mat stays valid and unchanged. Empty before the first observation.

    cv::Mat get_descriptor() const;

    // ========================================================================
    // Normal, scale and depth
    // ========================================================================

    void update_normal_and_depth();

    vec3f get_normal() const;

    float get_min_distance_invariance() const;
    float get_max_distance_invariance() const;

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

    // Points currently alive, anywhere. Everything here is shared_ptr-owned, so a gap between
    // this and Map::map_points_in_map() is a point that was erased from the map but is still
    // held by something else -- retention, never a leak. Reported by System::Shutdown.
    static inline std::atomic<long> live_count{0};

    PtId ptId;
    KeyframeId first_keyframe_id;

    FeatureType featureType;

    // Image color (OpenCV-native BGR) of the reference keypoint at creation.
    const cv::Vec3b color;

    // ------------------------------------------------------------------------
    // Scratch marks of the algorithms. Each holds the id of the frame or keyframe an algorithm is
    // working for, so a point is handled once per pass ("already in this local BA").
    //
    // Contract: every mark has ONE owning thread, which is the only one that reads or writes it,
    // so none of them is locked. Code on another thread must not touch a mark it does not own.
    // ------------------------------------------------------------------------

    // Owned by the tracking thread (Tracking::search_local_points)
    FrameId last_frame_seen{0};

    // Owned by the local mapping thread (Optimizer's local BA; LocalMapping::search_in_neighbors)
    long unsigned int ba_local_for_keyframe;
    long unsigned int fuse_candidate_for_keyframe;

    // Owned by the loop closing thread (LoopClosing; Optimizer's essential-graph optimization)
    long unsigned int loop_point_for_keyframe;
    long unsigned int corrected_by_keyframe;
    long unsigned int corrected_reference;

    // Owned by whichever thread runs the global BA (Optimizer::global_bundle_adjustment, then
    // LoopClosing::apply_gba_correction). Two can exist: Loop Closing's BA thread and the final
    // System::GBA() on the main thread; nothing keeps them apart yet.
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
    // A point needs at least this many observing keyframes to survive losing one: erase_observation
    // removes it when fewer are left (ORB-SLAM2's `nObs <= 2`). MapPoint.MinObservingKeyframes.
    static inline int min_observing_keyframes{3};

    [[nodiscard]] int observation_weight(const Keyframe& keyframe, const KeypointIndex& index) const;

    void increase_observability(const Keyframe& keyframe, const KeypointIndex& index);
    void decrease_observability(const Keyframe& keyframe, const KeypointIndex& index);

    std::map<KeyframeId, Observation> observations_;

    int num_observations_;

    Keyframe ref_keyframe_;

    // ========================================================================
    // Descriptor
    // ========================================================================

    // The representative descriptor: the observation's with the least median distance to the others
    // (compute_distinctive_descriptors). Empty until the first observation.
    cv::Mat descriptor_;

    // ========================================================================
    // Normal, scale and depth
    // ========================================================================

    // Mean viewing direction
    vec3f normal_;

    // The distance range the point is expected to be seen from: its distance to the reference
    // keyframe divided / multiplied by depth_invariance_ratio (project_if_visible tests it)
    float min_distance_{0.0f};
    float max_distance_{0.0f};

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

    // position_mutex_ guards the geometry: position_, normal_, min_distance_, max_distance_.
    // features_mutex_ guards everything else that changes: observations_, num_observations_,
    //                 ref_keyframe_, descriptor_, bad_, replaced_, num_visible_, num_found_.
    // (KeyFrame has a features_mutex_ of its own: a different lock.)
    //
    // Rules:
    //  - A method takes one of the two. The one that needs both (update_normal_and_depth) takes them
    //    with a single std::scoped_lock; anywhere else that ever needs both must do the same, or take
    //    features_mutex_ first.
    //  - Neither is held across a call into a keyframe or the map that takes a lock: set_bad_flag and
    //    replace copy what they need, release, and only then touch the keyframes and the map.
    mutable std::mutex position_mutex_;
    mutable std::mutex features_mutex_;
};

}
