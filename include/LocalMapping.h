#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include "KeyFrame.h"
#include "Map.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"
#include "FeatureMatcher.h"
#include "Viewer.h"

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>


namespace AF_VSLAM
{

class Tracking;
class LoopClosing;
class Map;
class Viewer;

// Tunable local-mapping heuristics, loaded from the settings YAML at System startup.
// Same pattern as TrackingParameters/OptimizerParameters: compiled-in defaults,
// overridden only by keys present in the file, so settings YAMLs without a
// LocalMapping.* block keep working unchanged. The remaining LocalMapping
// constants below migrate here as their functions get cleaned (issue #15 N2).
struct LocalMappingParameters
{
    // cull_keyframes(): "heuristic" (ORB-SLAM2-style point redundancy, default) or
    // "information" (joint information on the online MegaLoc similarity kernel; needs
    // vpr: megaloc — falls back to heuristic with a one-time warning otherwise)
    std::string keyframe_culling_method{"heuristic"};

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
    float search_in_neighbors_radius{5.0f};      // projection search radius (pixels) for fusion

    // cull_keyframes_heuristic()
    float keyframe_culling_redundancy_ratio{0.9f}; // cull when more than this fraction of a keyframe's points is redundant
    int keyframe_culling_min_observations{3};      // a point is redundant when this many OTHER keyframes observe it

    // cull_keyframes_information()
    // Atomic: the Viewer's "Cull Max Unexplained" slider writes it while LocalMapping reads it.
    std::atomic<float> keyframe_culling_max_unexplained{0.3f};  // a keyframe may be culled only if every keyframe ever inserted stays at least (1 - this) explained by the alive ones
    int keyframe_culling_min_age{5};               // keyframes younger than this many keyframes (by keyId) are never culled
    int keyframe_culling_min_keyframes{5};         // never cull below this many alive keyframes
    std::string keyframe_culling_scope{"map"};     // explainers + candidates: "map" = all alive keyframes, "local" = current keyframe + its covisible keyframes
    int keyframe_culling_max_per_call{5};          // at most this many culls per cull_keyframes() call (0 = unlimited); spreads the response to a raised threshold over several keyframes
    bool keyframe_culling_centred{true};           // double-centre the kernel (Pearson correlation of mean-centred descriptors) instead of raw cosine

    LocalMappingParameters() = default;
    LocalMappingParameters(const LocalMappingParameters& o) { *this = o; }
    LocalMappingParameters& operator=(const LocalMappingParameters& o)
    {
        keyframe_culling_method = o.keyframe_culling_method;
        map_point_culling_min_found_ratio = o.map_point_culling_min_found_ratio;
        map_point_culling_min_observations = o.map_point_culling_min_observations;
        map_point_culling_observation_test_age = o.map_point_culling_observation_test_age;
        map_point_culling_probation_age = o.map_point_culling_probation_age;
        create_new_map_points_keyframes = o.create_new_map_points_keyframes;
        create_new_map_points_min_baseline_depth_ratio = o.create_new_map_points_min_baseline_depth_ratio;
        create_new_map_points_max_parallax_cos = o.create_new_map_points_max_parallax_cos;
        search_in_neighbors_keyframes = o.search_in_neighbors_keyframes;
        search_in_neighbors_second_keyframes = o.search_in_neighbors_second_keyframes;
        search_in_neighbors_radius = o.search_in_neighbors_radius;
        keyframe_culling_redundancy_ratio = o.keyframe_culling_redundancy_ratio;
        keyframe_culling_min_observations = o.keyframe_culling_min_observations;
        keyframe_culling_max_unexplained.store(o.keyframe_culling_max_unexplained.load());
        keyframe_culling_min_age = o.keyframe_culling_min_age;
        keyframe_culling_min_keyframes = o.keyframe_culling_min_keyframes;
        keyframe_culling_scope = o.keyframe_culling_scope;
        keyframe_culling_max_per_call = o.keyframe_culling_max_per_call;
        keyframe_culling_centred = o.keyframe_culling_centred;
        return *this;
    }
};

class LocalMapping
{
public:
    LocalMapping(std::shared_ptr<Map> map, const std::vector<FeatureType>& feature_types,
                 int image_width, int image_height);

    // Tunable parameters, loaded from the settings YAML at System startup
    static LocalMappingParameters params;
    static void LoadParameters(const cv::FileStorage &fSettings);

    void SetLoopCloser(std::shared_ptr<LoopClosing> loop_closer){loop_closer_ = std::move(loop_closer);}
    void SetViewer(std::shared_ptr<Viewer> viewer){viewer_ = std::move(viewer);}

    // Online keyframe x keyframe VPR similarity matrix (cosine of the keyframes' MegaLoc global
    // descriptors), grown in process_new_keyframe: row/col k = k-th keyframe with a descriptor
    // processed by LocalMapping (insertion order, see keyframe_vpr_order()); rows/cols are never
    // removed when a keyframe is culled (culled keyframes stay as culling "history"). Empty
    // when the VPR backend does not produce global descriptors (vpr: bow / none). Both return
    // copies taken under the matrix mutex.
    Eigen::MatrixXf keyframe_vpr_matrix() const;
    std::vector<Keyframe> keyframe_vpr_order() const;
    bool has_keyframe_vpr_matrix() const;

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
    void set_accept_keyframes(bool accept);

    // Stop protocol, used by Loop Closing while it corrects the map: request_stop()
    // asks the thread to pause at its next safe point; run() pauses there via
    // stop_if_requested() and idles until release(),
    // which also drops the keyframes queued meanwhile. Tracking wraps each keyframe
    // insertion in set_insertion_lock(true/false), which defers a pending stop until
    // the insertion is done; it returns false if the thread is already stopped, in
    // which case the caller must not insert.
    void request_stop();
    bool is_stopped() const;
    bool is_stop_requested() const;
    void release();
    bool set_insertion_lock(bool locked);

    // Reset protocol, used by Tracking::reset: request_reset() blocks the caller until
    // the Local Mapping thread has dropped its keyframe queue, recent map points and
    // VPR matrix (reset_if_requested, at the end of each run() iteration).
    void request_reset();

    // Finish protocol, used by System::Shutdown: request_finish() makes run() return at
    // its next check; is_finished() turns true once it has (set_finished also marks the
    // thread stopped, so waiters on is_stopped() are released).
    void request_finish();
    bool is_finished() const;


protected:

    // run(): local BA needs more keyframes than this in the map
    static constexpr int LOCAL_BA_MIN_KEYFRAMES{2};

    // create_new_map_points(): 2-DoF chi-square 95% quantile (reprojection gate),
    // and the numerical zero for the homogeneous scale of a triangulated point
    static constexpr float CHI2_2DOF{5.991f};
    static constexpr float HOMOGENEOUS_W_EPSILON{1e-12f};

    // One full mapping iteration for the keyframe at the head of the queue (see run())
    void process_keyframe();
    void process_new_keyframe();
    void cull_map_points();
    void create_new_map_points();
    // create_new_map_points() helpers: brute-force match the new keyframe against its
    // neighbors (cached, OMP-parallel), and back-project unmatched keypoints with depth
    void cache_neighbor_matches(const std::vector<Keyframe>& neighbors);
    void create_depth_seeded_points();
    void search_in_neighbors();
    // Keyframe culling: dispatches on params.keyframe_culling_method.
    void cull_keyframes();
    // "heuristic": mark as bad the covisible keyframes whose map points are (mostly)
    // already observed by enough other keyframes; see the definition for the exact rule.
    void cull_keyframes_heuristic();
    // "information": joint-information culling on the online keyframe VPR matrix; see
    // the definition for the exact rule.
    void cull_keyframes_information();
    // Called by run(): pause at its safe point (true once stopped) / perform a pending
    // reset / exit when asked / publish the exit
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

    // Online keyframe VPR matrix (see keyframe_vpr_matrix())
    mutable std::mutex vpr_mutex_;
    std::vector<Keyframe> vpr_keyframes_{};       // k -> keyframe
    Eigen::MatrixXf keyframe_vpr_matrix_{};        // k x k, raw cosine
    void grow_keyframe_vpr_matrix(const Keyframe& keyframe);  // called once per processed keyframe
    void print_keyframe_vpr_matrix() const;
    std::list<Pt> recent_map_points_;

    bool stopped_{false};
    bool stop_requested_{false};
    bool insertion_locked_{false};
    bool accept_keyframes_{true};
    bool reset_requested_{false};
    bool finish_requested_{false};
    bool finished_{true};

    // Per-iteration timing histograms (ms buckets). local_mapping_times_ is always
    // recorded -- it feeds the viewer's median-time display; the per-stage histograms
    // are PROFILING_EXHAUSTIVE-only (LocalMapping_aux.h: LocalMappingProfiler/StageTimer).
    std::map<int, int> local_mapping_times_{};
    std::map<int, int> create_new_map_points_times_{};
    std::map<int, int> search_in_neighbors_times_{};
    std::map<int, int> local_ba_times_{};
    void log_profile();

};

} //namespace ORB_SLAM

#endif // LOCALMAPPING_H
