#include "Map.h"
#include "afvslam_log.hpp"
#include "KeyFrame.h"
#include "MapPoint.h"

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace AF_VSLAM
{

// ============================================================================
// Construction
// ============================================================================

Map::Map():
    max_keyframe_id_(0)
{
}

// ============================================================================
// Map points
// ============================================================================

Pt Map::create_monocular_map_point(const vec3f& world_pos,
                                   const Keyframe& ref_keyframe, KeypointIndex ref_index,
                                   const Keyframe& proj_keyframe, KeypointIndex proj_index,
                                   FeatureType feature_type)
{
    return make_map_point(world_pos, ref_keyframe, ref_index, feature_type, proj_keyframe, proj_index);
}

Pt Map::create_map_point(const vec3f& world_pos,
                         const Keyframe& ref_keyframe, KeypointIndex ref_index,
                         FeatureType feature_type)
{
    return make_map_point(world_pos, ref_keyframe, ref_index, feature_type, nullptr, 0);
}

Pt Map::make_map_point(const vec3f& world_pos, const Keyframe& ref_keyframe, KeypointIndex ref_index,
                       FeatureType feature_type,
                       const Keyframe& proj_keyframe, KeypointIndex proj_index)
{
    // A "monocular" point is triangulated from two views, so the same keyframe twice has no
    // baseline and the position means nothing. Refuse it: building the point anyway would give it
    // a single observation, which min_observing_keyframes culls on its first erase_observation --
    // the caller's bug would show up later as a mysteriously short-lived point. Every caller
    // already handles nullptr.
    if(proj_keyframe && proj_keyframe == ref_keyframe)
    {
        AF_WARN("[Map] monocular map point asked for with the same keyframe as both views (keyId "
                << ref_keyframe->keyId << ", indices " << ref_index << " and " << proj_index
                << ") -- discarded");
        return nullptr;
    }

    // The colour is sampled ONCE, here, from the reference keyframe's keypoint, and is fixed for
    // the point's lifetime -- never re-sampled from a later observer and never averaged across
    // them. It is display data only (viewer "Point Color: rgb" and the PLY export), so it is read
    // without a lock. A point therefore shows the colour of the frame that created it, which on a
    // sequence with changing exposure is not the colour any other observer would have given it.
    auto pt = make_shared<MapPoint>(world_pos, ref_keyframe, this_map(), feature_type,
                                    ref_keyframe->features.at(feature_type).colors.at(ref_index));

    std::vector<std::pair<Keyframe, KeypointIndex>> observations{{ref_keyframe, ref_index}};
    if(proj_keyframe)   // the monocular form: two observations, two slots written
        observations.emplace_back(proj_keyframe, proj_index);

    pt->add_observations(observations);

    std::vector<std::pair<Keyframe, KeypointIndex>> published;
    published.reserve(observations.size());
    for(const auto& [keyframe, index] : observations)
    {
        if(!keyframe->add_map_point(pt, index))
        {
            for(const auto& [taken_keyframe, taken_index] : published)
                taken_keyframe->erase_map_point_match(pt, taken_index);
            return nullptr;
        }
        published.emplace_back(keyframe, index);
    }

    add_map_point(pt);
    return pt;
}

void Map::add_map_point(const Pt& pt)
{
    unique_lock<mutex> lock(map_mutex_);
    map_points_.insert(pt);
}

void Map::erase_map_point(const Pt& pt)
{
    unique_lock<mutex> lock(map_mutex_);
    map_points_.erase(pt);

    // This drops the map's reference; the point is destroyed when the last other owner lets go.
    // (Upstream ORB-SLAM2 held raw MapPoint*, where this really did leak and a delete was owed.)
}

vector<Pt> Map::get_all_map_points()
{
    unique_lock<mutex> lock(map_mutex_);
    return vector<Pt>(map_points_.begin(),map_points_.end());
}

size_t Map::map_points_in_map()
{
    unique_lock<mutex> lock(map_mutex_);
    return map_points_.size();
}

// ============================================================================
// Keyframes
// ============================================================================

void Map::add_keyframe(const Keyframe& keyframe)
{
    unique_lock<mutex> lock(map_mutex_);
    keyframes_.insert(keyframe);
    if(keyframe->keyId > max_keyframe_id_)
        max_keyframe_id_ = keyframe->keyId;
}

void Map::erase_keyframe(const Keyframe& keyframe)
{
    unique_lock<mutex> lock(map_mutex_);
    keyframes_.erase(keyframe);

    // As in erase_map_point: this drops the map's reference only. max_keyframe_id_ is deliberately
    // NOT lowered -- see its declaration. A culled keyframe also keeps its image, descriptors and
    // grid alive for as long as anything still holds it; KeyFrame::set_bad_flag releases none of
    // them.
}

vector<Keyframe> Map::get_all_keyframes()
{
    unique_lock<mutex> lock(map_mutex_);
    return vector<Keyframe>(keyframes_.begin(),keyframes_.end());
}

size_t Map::keyframes_in_map()
{
    unique_lock<mutex> lock(map_mutex_);
    return keyframes_.size();
}

long unsigned int Map::get_max_keyframe_id()
{
    unique_lock<mutex> lock(map_mutex_);
    return max_keyframe_id_;
}

// ============================================================================
// Lifecycle
// ============================================================================

void Map::clear()
{
    decltype(map_points_) released_points;
    decltype(keyframes_) released_keyframes;
    decltype(keyframe_origins_) released_origins;
    {
        unique_lock<mutex> lock(map_mutex_);
        released_points.swap(map_points_);
        released_keyframes.swap(keyframes_);
        released_origins.swap(keyframe_origins_);
        max_keyframe_id_ = 0;
    }
}

}
