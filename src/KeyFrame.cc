#include "KeyFrame.h"
#include "afvslam_log.hpp"
#include "Converter.h"
#include "Map.h"
#include "MapPoint.h"
#include "Observation.h"
#include <mutex>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

namespace AF_VSLAM
{

bool PtIdLess::operator()(const Pt& a, const Pt& b) const
{
    return a->ptId < b->ptId;
}

bool contains_map_point(const std::vector<Pt>& sorted_points, const Pt& pt)
{
    return std::binary_search(sorted_points.begin(), sorted_points.end(), pt, PtIdLess{});
}

namespace
{

std::string slot_context(KeyframeId key_id, FeatureType feature_type)
{
    return "KeyFrame " + std::to_string(key_id) + ", feature type "
           + std::to_string(static_cast<int>(feature_type));
}

template<class MapPoints>
auto& map_point_vector(MapPoints& map_points, FeatureType feature_type, KeyframeId key_id)
{
    const auto it = map_points.find(feature_type);
    if(it == map_points.end())
        throw std::out_of_range(slot_context(key_id, feature_type)
                                + ": this keyframe has no such feature type");
    return it->second;
}

template<class MapPoints>
auto& map_point_slot(MapPoints& map_points, FeatureType feature_type, KeypointIndex index,
                     KeyframeId key_id)
{
    auto& pts = map_point_vector(map_points, feature_type, key_id);
    if(index < 0 || static_cast<std::size_t>(index) >= pts.size())
        throw std::out_of_range(slot_context(key_id, feature_type) + ": keypoint index "
                                + std::to_string(index) + " outside [0, "
                                + std::to_string(pts.size()) + ")");
    return pts[static_cast<std::size_t>(index)];
}
}

void KeyFrame::LoadParameters(const cv::FileStorage &fSettings)
{
    const cv::FileNode node = fSettings["KeyFrame.MinCovisibilityWeight"];
    if(!node.empty())
        node >> min_covisibility_weight;
}

static bool stronger_first(const Connection& a, const Connection& b){
    return (a.weight != b.weight) ? (a.weight > b.weight) : (a.keyframe->keyId > b.keyframe->keyId);
}

static std::vector<Keyframe> keyframes_of(const std::vector<Connection>& connections,
                                          std::size_t count)
{
    std::vector<Keyframe> keyframes;
    keyframes.reserve(count);
    for(std::size_t i = 0; i < count; i++)
        keyframes.push_back(connections[i].keyframe);
    return keyframes;
}

// ============================================================================
// Construction
// ============================================================================

KeyFrame::KeyFrame(Frame& frame, std::shared_ptr<Map> map, std::shared_ptr<PlaceRecognition> place_recognition):
    feature_types(frame.feature_types), cache_matched_pairs_feat_type(frame.cache_matched_pairs_feat_type),
    frame_id(frame.frame_id), timestamp(frame.timestamp),
    fuse_target_for_keyframe(0), ba_local_for_keyframe(0), ba_fixed_for_keyframe(0), ba_global_for_keyframe(0),
    calibration(frame.calibration),
    features(frame.features),
    image(frame.image), global_descriptor(frame.global_descriptor),
    map_points_(frame.pts), place_recognition_(std::move(place_recognition)),
    first_connection_(true), parent_(nullptr), pinned_(false),
    cull_requested_(false), bad_(false), map_(std::move(map)),
    grid_((frame.build_grids(), std::move(frame.grid)))
{
    keyId = next_id++;
    set_pose(frame.get_Tcw());
    ++live_count;
}

KeyFrame::~KeyFrame()
{
    --live_count;
}

// ============================================================================
// Pose
// ============================================================================

void KeyFrame::set_pose(const mat4f& Tcw_)
{
    std::unique_lock<std::mutex> lock(pose_mutex_);
    if(!pose_.set(Tcw_))
        AF_WARN("[KeyFrame " << keyId << "] set_pose ignored a non-finite pose (a diverged"
                << " optimisation); keeping the previous one.");
}

Pose KeyFrame::get_pose() const
{
    std::unique_lock<std::mutex> lock(pose_mutex_);
    return pose_;
}

mat4f KeyFrame::get_Tcw() const
{
    std::unique_lock<std::mutex> lock(pose_mutex_);
    return pose_.Tcw;
}

mat4f KeyFrame::get_Twc() const
{
    std::unique_lock<std::mutex> lock(pose_mutex_);
    return pose_.Twc;
}

mat3f KeyFrame::get_Rcw() const
{
    std::unique_lock<std::mutex> lock(pose_mutex_);
    return pose_.Rcw;
}

vec3f KeyFrame::get_tcw() const
{
    std::unique_lock<std::mutex> lock(pose_mutex_);
    return pose_.tcw;
}

vec3f KeyFrame::get_twc() const
{
    std::unique_lock<std::mutex> lock(pose_mutex_);
    return pose_.twc;
}

// ============================================================================
// Per-keyframe queries
// ============================================================================

void KeyFrame::compute_global_descriptor()
{
    if(place_recognition_)
        place_recognition_->compute(*this);
}

std::optional<float> KeyFrame::compute_scene_median_depth() const
{
    std::vector<Pt> points;
    vec3f Rcw_row2;
    float tcw_z{};
    {
        std::unique_lock<std::mutex> lock(features_mutex_);
        std::unique_lock<std::mutex> lock2(pose_mutex_);
        std::size_t total = 0;
        for(const auto& [feature_type, pts] : map_points_)
            total += pts.size();
        points.reserve(total);
        for(const auto& [feature_type, pts] : map_points_)
            points.insert(points.end(), pts.begin(), pts.end());
        Rcw_row2 = pose_.Rcw.row(2).transpose();
        tcw_z = pose_.tcw(2);
    }

    std::vector<float> depths;
    depths.reserve(points.size());
    for(const Pt& pt : points)
        if(pt && !pt->is_bad())
            depths.push_back(Rcw_row2.dot(pt->get_world_pos()) + tcw_z);

    if(depths.empty())
        return std::nullopt;

    const size_t nth = (depths.size()-1)/2;
    std::nth_element(depths.begin(), depths.begin() + nth, depths.end());

    return depths[nth];
}

// ============================================================================
// Spatial index
// ============================================================================

std::vector<size_t> KeyFrame::get_features_in_area(const float x, const float y, const float r, const FeatureType& feature_type) const
{
    if(!features.has(feature_type))
        return {};

    return features_in_area(grid_.at(feature_type), features.at(feature_type).keypoints, *calibration, x, y, r);
}

// ============================================================================
// Map points
// ============================================================================

bool KeyFrame::add_map_point(const Pt& pt, KeypointIndex index)
{
    if(!pt)
        return false;

    std::unique_lock<std::mutex> lock(features_mutex_);
    Pt& slot = map_point_slot(map_points_, pt->featureType, index, keyId);
    if(slot)
        return false;
    slot = pt;
    return true;
}

void KeyFrame::erase_map_point_match(const Pt& pt, KeypointIndex index)
{
    replace_map_point_match(pt, index, nullptr);
}

void KeyFrame::erase_map_point_match(const Pt& pt)
{
    if(!pt)
        return;

    const KeypointIndex index = pt->get_index_in_keyframe(this_keyframe());
    if(index < 0)
        return;
    erase_map_point_match(pt, index);
}

bool KeyFrame::replace_map_point_match(const Pt& pt, KeypointIndex index, const Pt& replacement)
{
    if(!pt)
        return false;

    std::unique_lock<std::mutex> lock(features_mutex_);
    Pt& slot = map_point_slot(map_points_, pt->featureType, index, keyId);
    if(slot != pt)
        return false;
    slot = replacement;
    return true;
}

std::vector<Pt> KeyFrame::get_good_map_points(FeatureType feature_type) const
{
    std::vector<Pt> points;
    {
        std::unique_lock<std::mutex> lock(features_mutex_);
        points = map_point_vector(map_points_, feature_type, keyId);
    }

    points.erase(std::remove_if(points.begin(), points.end(),
                                [](const Pt& pt){ return !pt || pt->is_bad(); }),
                 points.end());
    std::sort(points.begin(), points.end(), PtIdLess{});
    points.erase(std::unique(points.begin(), points.end()), points.end());
    return points;
}

std::vector<Pt> KeyFrame::get_map_point_matches(FeatureType feature_type) const
{

    std::unique_lock<std::mutex> lock(features_mutex_);
    return map_point_vector(map_points_, feature_type, keyId);
}

Pt KeyFrame::get_map_point(KeypointIndex index, FeatureType feature_type) const
{
    std::unique_lock<std::mutex> lock(features_mutex_);
    return map_point_slot(map_points_, feature_type, index, keyId);
}

int KeyFrame::count_tracked_map_points(int min_observations) const
{
    std::map<FeatureType, std::vector<Pt>> snapshot;
    {
        std::unique_lock<std::mutex> lock(features_mutex_);
        snapshot = map_points_;
    }

    int num_points = 0;
    for(const auto& [feature_type, pts] : snapshot)
        for(const Pt& pt : pts)
        {
            if(!pt || pt->is_bad())
                continue;
            if(min_observations > 0 && pt->number_of_observations() < min_observations)
                continue;
            num_points++;
        }
    return num_points;
}

// ============================================================================
// Covisibility graph
// ============================================================================

void KeyFrame::update_connections()
{
    std::map<FeatureType, std::vector<Pt>> map_points;
    {
        std::unique_lock<std::mutex> lock(features_mutex_);
        map_points = map_points_;
    }

    std::map<KeyframeId, Connection> candidates;
    for(const auto& [feature_type, points] : map_points)
        for(const Pt& pt : points)
        {
            if(!pt || pt->is_bad())
                continue;

            for(const auto& [key_id, observer] : pt->get_observers())
            {
                if(key_id == keyId)
                    continue;
                Connection& candidate = candidates[key_id];
                if(!candidate.keyframe)       // once per neighbour, not once per shared point
                    candidate.keyframe = observer;
                candidate.weight++;
            }
        }

    if(candidates.empty())
        return;

    std::vector<Connection> edges;
    {
        std::unique_lock<std::mutex> lock(connections_mutex_);
        connections_ = std::move(candidates);
        ordered_dirty_ = true;
        edges = ordered_connections();
    }

    for(const Connection& edge : edges)
        edge.keyframe->add_connection(this_keyframe(), edge.weight);

    Keyframe first_parent;
    for(const Connection& edge : edges)
        if(!edge.keyframe->is_bad())
        {
            first_parent = edge.keyframe;
            break;
        }

    {
        std::unique_lock<std::mutex> lock(connections_mutex_);

        if(first_connection_ && !is_origin() && first_parent)
        {
            parent_ = first_parent;
            parent_->add_child(this_keyframe());
            first_connection_ = false;
        }
    }
}

void KeyFrame::add_connection(const Keyframe& keyframe, int weight)
{
    if(!keyframe)
        return;

    std::unique_lock<std::mutex> lock(connections_mutex_);

    if(bad_)
        return;

    const auto it = connections_.find(keyframe->keyId);
    if(it != connections_.end() && it->second.weight == weight)
        return;   // nothing changed, so nothing to re-sort

    connections_[keyframe->keyId] = Connection{keyframe, weight};
    ordered_dirty_ = true;
}

void KeyFrame::erase_connection(const Keyframe& keyframe)
{
    if(!keyframe)
        return;

    std::unique_lock<std::mutex> lock(connections_mutex_);

    if(connections_.erase(keyframe->keyId) > 0)
        ordered_dirty_ = true;
}

std::vector<Keyframe> KeyFrame::get_best_covisibility_keyframes(int N) const
{
    std::unique_lock<std::mutex> lock(connections_mutex_);

    const std::vector<Connection>& ordered = ordered_connections();
    const std::size_t count = std::min(ordered.size(), static_cast<std::size_t>(std::max(N, 0)));
    return keyframes_of(ordered, count);
}

std::vector<Keyframe> KeyFrame::get_covisibles_by_weight(int w) const
{
    std::unique_lock<std::mutex> lock(connections_mutex_);

    const std::vector<Connection>& ordered = ordered_connections();
    const auto it = std::partition_point(ordered.begin(), ordered.end(),
                                         [w](const Connection& c){ return c.weight >= w; });

    return keyframes_of(ordered, static_cast<std::size_t>(it - ordered.begin()));
}

int KeyFrame::get_weight(const Keyframe& keyframe) const
{

    if(!keyframe)
        return 0;

    std::unique_lock<std::mutex> lock(connections_mutex_);
    const auto it = connections_.find(keyframe->keyId);
    return it != connections_.end() ? it->second.weight : 0;
}

const std::vector<Connection>& KeyFrame::ordered_connections() const
{
    if(ordered_dirty_)
    {
        rebuild_ordered_connections();
        ordered_dirty_ = false;
    }
    return ordered_connections_;
}

void KeyFrame::rebuild_ordered_connections() const
{
    ordered_connections_.clear();
    ordered_connections_.reserve(connections_.size());

    const Connection* strongest = nullptr;
    for(const auto& [key_id, connection] : connections_)
    {
        if(connection.weight >= min_covisibility_weight)
            ordered_connections_.push_back(connection);
        if(!strongest || connection.weight > strongest->weight)
            strongest = &connection;
    }

    if(ordered_connections_.empty() && strongest)
        ordered_connections_.push_back(*strongest);

    std::sort(ordered_connections_.begin(), ordered_connections_.end(), stronger_first);
}

std::map<KeyframeId,Keyframe> KeyFrame::get_connected_keyframes() const
{
    std::unique_lock<std::mutex> lock(connections_mutex_);
    std::map<KeyframeId,Keyframe> keyframes;
    for(const auto& [id, connection] : connections_)
        keyframes.emplace_hint(keyframes.end(), id, connection.keyframe);   // already ordered
    return keyframes;
}

std::vector<Connection> KeyFrame::get_covisible_connections() const
{
    std::unique_lock<std::mutex> lock(connections_mutex_);
    return ordered_connections();
}

std::vector<Keyframe> KeyFrame::get_covisible_keyframes() const
{
    std::unique_lock<std::mutex> lock(connections_mutex_);
    const std::vector<Connection>& ordered = ordered_connections();
    return keyframes_of(ordered, ordered.size());
}

// ============================================================================
// Spanning tree and loop edges
// ============================================================================

void KeyFrame::add_child(const Keyframe& keyframe)
{
    if(!keyframe)
        return;

    std::unique_lock<std::mutex> lock(connections_mutex_);

    if(bad_)
        return;

    children_.insert(keyframe);
}

void KeyFrame::erase_child(const Keyframe& keyframe)
{
    if(!keyframe)
        return;

    std::unique_lock<std::mutex> lock(connections_mutex_);
    children_.erase(keyframe);
}

void KeyFrame::set_parent(const Keyframe& keyframe)
{
    if(!keyframe)
        return;

    std::unique_lock<std::mutex> lock(connections_mutex_);
    parent_ = keyframe;
    keyframe->add_child(this_keyframe());
}

void KeyFrame::add_loop_edge(const Keyframe& keyframe)
{
    if(!keyframe)
        return;

    std::unique_lock<std::mutex> lock(connections_mutex_);
    pinned_ = true;   // see unpin(): a loop edge pins the keyframe permanently
    loop_edges_.insert(keyframe);
}

bool KeyFrame::has_child(const Keyframe& keyframe) const
{
    if(!keyframe)
        return false;

    std::unique_lock<std::mutex> lock(connections_mutex_);
    return children_.count(keyframe) != 0;
}

std::vector<Keyframe> KeyFrame::get_children() const
{
    std::unique_lock<std::mutex> lock(connections_mutex_);
    return {children_.begin(), children_.end()};
}

Keyframe KeyFrame::get_parent() const
{
    std::unique_lock<std::mutex> lock(connections_mutex_);
    return parent_;
}

KeyframeIdSet KeyFrame::get_loop_edges() const
{
    std::unique_lock<std::mutex> lock(connections_mutex_);
    return loop_edges_;
}

// ============================================================================
// Lifecycle
// ============================================================================

void KeyFrame::pin()
{
    std::unique_lock<std::mutex> lock(connections_mutex_);
    pinned_ = true;
}

void KeyFrame::unpin()
{
    bool cull_now = false;
    {
        std::unique_lock<std::mutex> lock(connections_mutex_);

        if(loop_edges_.empty())
            pinned_ = false;

        cull_now = cull_requested_ && !pinned_;
    }

    if(cull_now)
        set_bad_flag();
}

std::mutex KeyFrame::set_bad_mutex_;

void KeyFrame::set_bad_flag()
{
    {
        std::unique_lock<std::mutex> lock(connections_mutex_);
        if(is_origin() || bad_)   // bad_ once set is final: culling twice must not re-run the erasures below
            return;
        else if(pinned_)
        {
            cull_requested_ = true;   // unpin() will come back to this
            return;
        }
    }

    std::lock_guard<std::mutex> serialise(set_bad_mutex_);
    {
        std::unique_lock<std::mutex> lock(connections_mutex_);
        if(bad_)
            return;
    }

    std::map<KeyframeId, Connection> connected;
    std::map<FeatureType, std::vector<Pt>> map_points;
    {
        std::unique_lock<std::mutex> lock(connections_mutex_);
        connected = connections_;
    }
    {
        std::unique_lock<std::mutex> lock(features_mutex_);
        map_points = map_points_;
    }

    for(const auto& [id, connection] : connected)
        connection.keyframe->erase_connection(this_keyframe());

    for(const auto& [ft, points] : map_points)
        for(const Pt& point : points)
            if(point)
                point->erase_observation(this_keyframe());

    KeyframeIdSet children;
    Keyframe parent;
    {
        std::unique_lock<std::mutex> lock(connections_mutex_);
        children = children_;
        parent = parent_;
    }

    struct ChildLinks
    {
        Keyframe child;
        std::vector<Connection> covisibles;
    };
    std::vector<ChildLinks> links;
    links.reserve(children.size());
    for(const Keyframe& child : children)
        if(!child->is_bad())
            links.push_back({child, child->get_covisible_connections()});

    KeyframeIdSet parent_candidates;
    if(parent)
        parent_candidates.insert(parent);

    std::vector<std::pair<Keyframe, Keyframe>> rehung;   // child, its new parent
    KeyframeIdSet rehung_children;
    while(true)
    {
        int best_weight = -1;
        Keyframe best_child;
        Keyframe best_parent;

        for(const ChildLinks& link : links)
        {
            if(rehung_children.count(link.child))
                continue;

            for(const Connection& covisible : link.covisibles)
            {
                if(!parent_candidates.count(covisible.keyframe))
                    continue;
                if(covisible.weight > best_weight)
                {
                    best_weight = covisible.weight;
                    best_child = link.child;
                    best_parent = covisible.keyframe;
                }
            }
        }

        if(!best_child)
            break;

        rehung.emplace_back(best_child, best_parent);
        rehung_children.insert(best_child);
        parent_candidates.insert(best_child);
    }

    for(const auto& [child, new_parent] : rehung)
        child->set_parent(new_parent);

    if(parent)
    {
        for(const Keyframe& child : children)
            if(!rehung_children.count(child))
                child->set_parent(parent);
        parent->erase_child(this_keyframe());
    }

    {
        std::unique_lock<std::mutex> lock(connections_mutex_);

        connections_.clear();
        ordered_connections_.clear();   // released now, not on the next read
        ordered_dirty_ = false;

        for(const Keyframe& child : rehung_children)
            children_.erase(child);

        bad_ = true;
    }

    map_->erase_keyframe(this_keyframe());
    place_recognition_->erase(this_keyframe());
}

bool KeyFrame::is_bad() const
{
    std::unique_lock<std::mutex> lock(connections_mutex_);
    return bad_;
}

}
