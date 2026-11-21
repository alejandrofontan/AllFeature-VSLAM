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

// Held by pointer only: their headers are included where they are used (LocalMapping.cc)
class FeatureMatcher;
class KeyframeInformation;
class LoopClosing;
class Map;
class Viewer;

// Tunable local-mapping heuristics, loaded from the settings YAML at System startup.
// Same pattern as TrackingParameters/OptimizerParameters: compiled-in defaults,
// overridden only by keys present in the file, so settings YAMLs without a
// LocalMapping.* block keep working unchanged.
struct LocalMappingParameters
{
    // The keyframe information kernel (KeyframeInformation.h) that the information
    // culler AND Tracking's insertion bands read: "megaloc" (the retrieval store's
    // MegaLoc cosine, centred; no kernel at all in a build without MegaLoc) or "covisibility"
    // (cosine of the keyframes' shared map points, raw). The two
    // kernels live on different scales, so KeyframeCullingMaxUnexplained and
    // KeyframeCullingCentred take a per-kernel default when their keys are absent.
    std::string information_kernel{"megaloc"};
    static constexpr float keyframe_culling_max_unexplained_covisibility{0.7f};   // tau default for the covisibility kernel (starting point, untuned)

    // cull_map_points()
    float map_point_culling_min_found_ratio{0.25f}; // cull when found/visible drops below this
    int map_point_culling_min_observations{2};      // cull when, observation_test_age keyframes after creation, a point has at most this many observations (depth-verified ones count twice)
    int map_point_culling_observation_test_age{2};  // keyframes after creation at which the observation test applies
    int map_point_culling_probation_age{3};         // keyframes after creation at which a surviving point graduates out of the probation list

    // create_new_map_points()
    int create_new_map_points_keyframes{5};                      // covisible keyframes matched and triangulated against
    float create_new_map_points_min_baseline_depth_ratio{0.01f}; // skip neighbors with baseline / median scene depth below this
    float create_new_map_points_max_parallax_cos{0.9998f};       // two-view triangulation needs cos(parallax) below this (parallax > ~1.15 deg)

    // search_in_neighbors()
    int search_in_neighbors_keyframes{20};       // covisible keyframes fused with the new keyframe
    int search_in_neighbors_second_keyframes{5}; // second-degree neighbors added per covisible keyframe
    // fusion accepts a match only if its descriptor distance is at most this quantile of the distances between map
    // points' descriptors and their own observations (estimated at every keyframe; also used by Loop Closing)
    float fuse_descriptor_quantile{0.5f};
    // a point is projected into a keyframe for fusion only if the keyframe views it within this cosine of its
    // mean viewing direction (60 deg). Key Tracking.ViewingCosLimit: one value for tracking and both fusions.
    float fuse_viewing_cos_limit{0.5f};

    // cull_keyframes()
    // Atomic: the Viewer's "KF Max Unexplained" slider writes it while LocalMapping reads it.
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

    // Tunable parameters, loaded from the settings YAML at System startup
    static LocalMappingParameters params;
    static void LoadParameters(const cv::FileStorage &fSettings);

    void set_loop_closer(std::shared_ptr<LoopClosing> loop_closer) { loop_closer_ = std::move(loop_closer); }
    void set_viewer(std::shared_ptr<Viewer> viewer) { viewer_ = std::move(viewer); }
    void set_keyframe_information(std::shared_ptr<KeyframeInformation> keyframe_information) { keyframe_information_ = std::move(keyframe_information); }

    // Main function: runs on the Local Mapping thread. One process_keyframe() per
    // queued keyframe; the stop/reset/finish protocols are honored between iterations.
    void run();

    // Keyframe queue, fed by Tracking and drained by run() (process_keyframe). The
    // refinement stages re-check it and yield to a waiting keyframe, so insertion
    // latency stays bounded.
    void insert_keyframe(const Keyframe& keyframe);
    bool has_new_keyframes() const;

    // Busy flag published by run(): false while a keyframe is being processed. Tracking
    // reads it to decide between a normal and an emergency keyframe insertion.
    bool accepts_keyframes() const;

    // Stop protocol, used by Loop Closing while it corrects the map: request_stop()
    // asks the thread to pause at its next safe point; run() pauses there via
    // stop_if_requested() and idles until release(). Tracking wraps each keyframe
    // insertion in set_insertion_lock(true/false), which defers a pending stop until
    // the insertion is done; it returns false if the thread is already stopped, in
    // which case the caller must not insert.
    void request_stop();
    bool is_stopped() const;
    bool is_stop_requested() const;
    void release();
    bool set_insertion_lock(bool locked);

    // Reset protocol, used by Tracking::reset: request_reset() blocks the caller until
    // the Local Mapping thread has dropped its keyframe queue and recent map points
    // (reset_if_requested, at the end of each run() iteration).
    void request_reset();

    // Finish protocol, used by System::Shutdown: request_finish() makes run() return at
    // its next check; is_finished() turns true once it has (set_finished also marks the
    // thread stopped, so waiters on is_stopped() are released).
    void request_finish();
    bool is_finished() const;

    // Drops what the thread still refers to once it has finished: the keyframe it processed last and the points on
    // probation, some erased since. Called by System::Shutdown before the lifetime summary.
    void release_map_references();

private:

    // create_new_map_points(): the numerical zero for the homogeneous scale of a triangulated point
    // (its reprojection gate is ErrorModel::params.chi2_mono)
    static constexpr float HOMOGENEOUS_W_EPSILON{1e-12f};

    // run() and request_reset(): how long the thread sleeps when idle or paused, and how often a waiter polls
    static constexpr std::chrono::milliseconds IDLE_POLL_PERIOD{3};

    // cull_keyframes(): placecell's culling method
    static constexpr const char* KEYFRAME_CULLING_METHOD{"gram-greedy"};

    // One full mapping iteration for the keyframe at the head of the queue (see run())
    void process_keyframe();
    void process_new_keyframe();
    void cull_map_points();
    void create_new_map_points();
    // create_new_map_points() helper: the create_new_map_points_keyframes most covisible keyframes that have
    // enough baseline to triangulate against
    [[nodiscard]] std::vector<Keyframe> select_triangulation_neighbors() const;
    // create_new_map_points() helper: back-project unmatched keypoints with depth
    void create_depth_seeded_points();
    void search_in_neighbors();
    // search_in_neighbors() helper: the fuse_descriptor_quantile quantile of the distances between the current
    // keyframe's map points' descriptors and their other observations; stored in the map for both fusions
    void estimate_fuse_max_descriptor_distance(FeatureType feature_type);
    // search_in_neighbors() helper: fuses `map_points` into `keyframe`'s keypoints of `feature_type`
    void fuse_into_keyframe(const Keyframe& keyframe, const std::vector<Pt>& map_points, FeatureType feature_type);
    // Optimizer::local_bundle_adjustment on the current keyframe (skipped on a too-small map)
    void local_bundle_adjustment();
    // Keyframe culling: joint-information culling on the keyframe information kernel; see the definition for
    // the exact rule. Without a kernel nothing is culled.
    void cull_keyframes();
    // Called by run(): publish the busy flag / pause at its safe point (true once stopped) / perform a pending
    // reset / exit when asked / publish the exit
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

    // Keyframe information kernel (KeyframeInformation.h, LocalMapping.InformationKernel):
    // its placecell store holds one row per keyframe ever processed (rows are never
    // removed when a keyframe is culled -- culled keyframes stay as the culling
    // "history"), keyed by frame_id. process_new_keyframe registers each keyframe,
    // cull_keyframes refreshes the alive ones and culls on it. Null when the
    // system has no information kernel (megaloc kernel in a build without MegaLoc).
    std::shared_ptr<KeyframeInformation> keyframe_information_{};
    std::list<Pt> recent_map_points_;

    bool stopped_{false};
    bool stop_requested_{false};
    bool insertion_locked_{false};
    bool accept_keyframes_{true};
    bool reset_requested_{false};
    bool finish_requested_{false};
    bool finished_{true};

    // cull_keyframes(): the missing-kernel warning is printed once, and the over-budget history once per change
    bool warned_no_kernel_{false};
    int last_over_budget_{-1};

    // Time of each mapping iteration, as a histogram in ms buckets (LocalMappingProfiler): the Viewer shows its
    // median
    std::map<int, int> local_mapping_times_{};
};

}
