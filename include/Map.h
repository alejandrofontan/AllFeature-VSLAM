#pragma once

#include <memory>
#include <mutex>
#include <set>
#include <vector>

#include "KeyFrame.h"

namespace AF_VSLAM
{

class Map : public std::enable_shared_from_this<Map>
{
public:

    // ========================================================================
    // Construction
    // ========================================================================

    Map();

    std::shared_ptr<Map> this_map() {
        return shared_from_this();
    }

    // ========================================================================
    // Map points
    // ========================================================================

    Pt create_monocular_map_point(const vec3f& world_pos,
                                  const Keyframe& ref_keyframe, KeypointIndex ref_index,
                                  const Keyframe& proj_keyframe, KeypointIndex proj_index,
                                  FeatureType feature_type);
    Pt create_map_point(const vec3f& world_pos,
                        const Keyframe& ref_keyframe, KeypointIndex ref_index,
                        FeatureType feature_type);

    void add_map_point(Pt pt);
    void erase_map_point(Pt pt);

    std::vector<Pt> get_all_map_points();

    size_t map_points_in_map();

    // ========================================================================
    // Keyframes
    // ========================================================================

    void add_keyframe(Keyframe keyframe);
    void erase_keyframe(Keyframe keyframe);

    std::vector<Keyframe> get_all_keyframes();

    size_t keyframes_in_map();

    long unsigned int get_max_keyframe_id();

    // ========================================================================
    // Lifecycle
    // ========================================================================

    void clear();

    // ========================================================================
    // Data
    // ========================================================================

    std::vector<Keyframe> keyframe_origins_;

    // ========================================================================
    // Locks
    // ========================================================================

    std::mutex map_update_mutex_;

    // This avoid that two points are created simultaneously in separate threads (id conflict)
    std::mutex point_creation_mutex_;

protected:

    // ========================================================================
    // Map points
    // ========================================================================

    Pt make_map_point(const vec3f& world_pos, const Keyframe& ref_keyframe, KeypointIndex ref_index,
                      FeatureType feature_type,
                      const Keyframe& proj_keyframe, KeypointIndex proj_index);

    std::set<Pt, PtIdLess> map_points_;

    // ========================================================================
    // Keyframes
    // ========================================================================

    KeyframeIdSet keyframes_;

    // Highest keyId ever added, NOT the highest currently in the map: erase_keyframe deliberately
    // does not lower it. Optimizer::OptimizeEssentialGraph sizes three vectors as
    // get_max_keyframe_id()+1 and indexes them by keyId, so a value that shrank after a cull would
    // index past the end. Only clear() resets it, and only because every keyId restarts with it.
    long unsigned int max_keyframe_id_;

    // ========================================================================
    // Locks
    // ========================================================================

    std::mutex map_mutex_;
};

}
