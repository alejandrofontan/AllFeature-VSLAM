#pragma once

#include <atomic>
#include <chrono>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>

#include "KeyFrame.h"
#include "Types.h"

namespace AF_VSLAM
{

class FeatureMatcher;
class KeyframeInformation;
class LoopClosing;
class Map;
class Viewer;

struct LocalMappingParameters
{

    std::string information_kernel{"megaloc"};
    static constexpr float keyframe_culling_max_unexplained_covisibility{0.7f};   // tau default for the covisibility kernel (starting point, untuned)

    // cull_map_points()
    float map_point_culling_min_found_ratio{0.25f}; // cull when found/visible drops below this
    int map_point_culling_min_observations{3};      // observations a point needs by observation_test_age to survive (depth-verified ones count more)
    int map_point_culling_observation_test_age{2};  // keyframes after creation at which the observation test applies
    int map_point_culling_probation_age{3};         // keyframes after creation at which a surviving point graduates out of the probation list

    // create_new_map_points()
    int create_new_map_points_keyframes{5};                      // covisible keyframes matched and triangulated against
    float create_new_map_points_min_baseline_depth_ratio{0.01f}; // skip neighbors with baseline / median scene depth below this
    float create_new_map_points_max_parallax_cos{0.9998f};       // two-view triangulation needs cos(parallax) below this (parallax > ~1.15 deg)

    // fuse_with_neighbors()
    int fuse_with_neighbors_keyframes{20};       // covisible keyframes fused with the new keyframe
    int fuse_with_neighbors_second_keyframes{5}; // second-degree neighbors added per covisible keyframe
    float fuse_descriptor_quantile{0.5f};
    float fuse_viewing_cos_limit{0.5f};

    std::atomic<float> keyframe_culling_max_unexplained{0.3f};  // a keyframe may be culled only if every keyframe ever inserted stays at least (1 - this) explained by the alive ones
    int keyframe_culling_min_age{5};               // keyframes younger than this many keyframes (by keyId) are never culled
    int keyframe_culling_min_keyframes{5};         // never cull below this many alive keyframes
    std::string keyframe_culling_scope{"map"};     // explainers + candidates: "map" = all alive keyframes, "local" = current keyframe + its covisible keyframes
    int keyframe_culling_max_per_call{5};          // at most this many culls per cull_keyframes() call (0 = unlimited); spreads the response to a raised threshold over several keyframes
    std::string keyframe_culling_objective{"unique"}; // gram-greedy ranks the feasible candidates by: "unique" = own unique information v_i, "minimax" = worst unexplained view after the cull, "total-loss" = v_i plus the rise of every other view
    bool keyframe_culling_centred{true};           // double-centre the kernel (Pearson correlation of mean-centred descriptors) instead of raw cosine

};

class LocalMapping
{
public:
    LocalMapping(std::shared_ptr<Map> map, const std::vector<FeatureType>& feature_types);

    static LocalMappingParameters params;
    static void LoadParameters(const cv::FileStorage &fSettings);

    void set_loop_closer(std::shared_ptr<LoopClosing> loop_closer) { loop_closer_ = std::move(loop_closer); }
    void set_viewer(std::shared_ptr<Viewer> viewer) { viewer_ = std::move(viewer); }
    void set_keyframe_information(std::shared_ptr<KeyframeInformation> keyframe_information) { keyframe_information_ = std::move(keyframe_information); }

    void run();

    void insert_keyframe(const Keyframe& keyframe);
    bool has_new_keyframes() const;

    bool accepts_keyframes() const;

    void request_stop();
    bool is_stopped() const;
    bool is_stop_requested() const;
    void release();
    bool set_insertion_lock(bool locked);

    void request_reset();

    void request_finish();
    bool is_finished() const;

    void release_map_references();

private:

    static constexpr std::chrono::milliseconds IDLE_POLL_PERIOD{3};

    static constexpr const char* KEYFRAME_CULLING_METHOD{"gram-greedy"};

    void process_keyframe();
    void register_keyframe();
    void cull_map_points();
    void create_new_map_points();

    [[nodiscard]] std::vector<Keyframe> select_triangulation_neighbors() const;

    void create_depth_seeded_points();
    void fuse_with_neighbors();

    void estimate_fuse_max_descriptor_distance(FeatureType feature_type);
    struct FuseCounts { int observed{0}; int merged{0}; };   // observations added, points merged
    FuseCounts fuse_into_keyframe(const Keyframe& keyframe, const std::vector<Pt>& map_points, FeatureType feature_type);
    void local_bundle_adjustment();
    void cull_keyframes();

    void set_accept_keyframes(bool accept);
    bool stop_if_requested();
    void reset_if_requested();
    bool is_reset_requested() const;
    bool is_finish_requested() const;
    void set_finished();

    mutable std::mutex finish_mutex_;          // guards finish_requested_ and finished_
    mutable std::mutex reset_mutex_;           // guards reset_requested_
    mutable std::mutex new_keyframes_mutex_;   // guards new_keyframes_
    mutable std::mutex stop_mutex_;            // guards stopped_, stop_requested_ and insertion_locked_
    mutable std::mutex accept_mutex_;          // guards accept_keyframes_

    std::shared_ptr<Map> map_;
    std::shared_ptr<LoopClosing> loop_closer_;
    std::shared_ptr<FeatureMatcher> matcher_;
    std::shared_ptr<Viewer> viewer_;

    std::list<Keyframe> new_keyframes_;
    Keyframe current_keyframe_;

    std::shared_ptr<KeyframeInformation> keyframe_information_{};
    std::list<Pt> recent_map_points_;

    bool stopped_{false};
    bool stop_requested_{false};
    bool insertion_locked_{false};
    bool accept_keyframes_{true};
    bool reset_requested_{false};
    bool finish_requested_{false};
    bool finished_{true};

    bool warned_no_kernel_{false};
    int last_over_budget_{-1};

    std::map<int, int> local_mapping_times_{};
};

}
