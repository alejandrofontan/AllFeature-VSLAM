#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/persistence.hpp>

#include "Types.h"

namespace AF_VSLAM
{

class FeatureMatcher;
class KeyFrame;
class KeyframeInformation;
class LoopClosing;
class Map;
class MapPoint;
class Viewer;

typedef std::shared_ptr<AF_VSLAM::KeyFrame> Keyframe;
typedef std::shared_ptr<AF_VSLAM::MapPoint> Pt;

struct LocalMappingParameters
{
    // Keyframe information kernel: "megaloc" or "covisibility"; tau's default when its key is absent, covisibility
    std::string information_kernel{"megaloc"};
    static constexpr float keyframe_culling_max_unexplained_covisibility{0.7f};

    // cull_map_points()
    float map_point_culling_min_found_ratio{0.25f};
    int map_point_culling_min_observations{3};
    int map_point_culling_observation_test_age{2};
    int map_point_culling_probation_age{3};

    // create_new_map_points()
    int create_new_map_points_keyframes{5};
    float create_new_map_points_min_baseline_depth_ratio{0.01f};
    float create_new_map_points_min_parallax_cos{0.0f};
    float create_new_map_points_max_parallax_cos{0.9998f};

    // fuse_with_neighbors()
    int fuse_with_neighbors_keyframes{20};
    int fuse_with_neighbors_second_keyframes{5};
    float fuse_descriptor_quantile{0.5f};
    int fuse_estimate_min_samples{50};
    float fuse_viewing_cos_limit{0.5f};

    // cull_keyframes()
    std::atomic<float> keyframe_culling_max_unexplained{0.3f};
    int keyframe_culling_min_age{5};
    int keyframe_culling_min_keyframes{5};
    std::string keyframe_culling_scope{"map"};
    int keyframe_culling_max_per_call{5};
    std::string keyframe_culling_objective{"unique"};
    bool keyframe_culling_centred{true};
};

class LocalMapping
{
public:

    // ========================================================================
    // Construction
    // ========================================================================

    // The thread is started by System after set_loop_closer (required) and the optional setters below
    LocalMapping(std::shared_ptr<Map> map, const std::vector<FeatureType>& feature_types);
    ~LocalMapping();

    static LocalMappingParameters params;
    static void LoadParameters(const cv::FileStorage &fSettings);

    void set_loop_closer(std::shared_ptr<LoopClosing> loop_closer) { loop_closer_ = std::move(loop_closer); }
    void set_viewer(std::shared_ptr<Viewer> viewer) { viewer_ = std::move(viewer); }
    void set_keyframe_information(std::shared_ptr<KeyframeInformation> keyframe_information)
    {
        keyframe_information_ = std::move(keyframe_information);
    }

    // ========================================================================
    // Thread
    // ========================================================================

    void run();

    // ========================================================================
    // Keyframe queue
    // ========================================================================

    void insert_keyframe(const Keyframe& keyframe);
    bool has_new_keyframes() const;
    bool accepts_keyframes() const;

    // ========================================================================
    // Stop protocol
    // ========================================================================

    void request_stop();
    bool is_stopped() const;
    bool is_stop_requested() const;
    void release();
    bool set_insertion_lock(bool locked);

    // ========================================================================
    // Reset and finish
    // ========================================================================

    void request_reset();
    void request_finish();
    bool is_finished() const;
    void release_map_references();

private:

    // What a fusion into one keyframe did: offered points it cannot see, observations added, points merged
    struct FuseCounts { int not_visible{0}; int observed{0}; int merged{0}; };

    static constexpr std::chrono::milliseconds IDLE_POLL_PERIOD{3};

    // The thread's name in the log and in the report
    static constexpr const char* THREAD_NAME{"local_mapping"};

    static constexpr const char* KEYFRAME_CULLING_METHOD{"gram-greedy"};

    // ========================================================================
    // Stages
    // ========================================================================

    void process_keyframe();
    void register_keyframe();
    void cull_map_points();
    void create_new_map_points();
    [[nodiscard]] std::vector<Keyframe> select_triangulation_neighbors() const;
    void create_depth_seeded_points();
    void fuse_with_neighbors();
    void estimate_fuse_max_descriptor_distance(FeatureType feature_type);
    [[nodiscard]] FuseCounts fuse_into_keyframe(const Keyframe& keyframe, const std::vector<Pt>& map_points,
                                                FeatureType feature_type);
    void local_bundle_adjustment();
    void cull_keyframes();

    // ========================================================================
    // Thread synchronization
    // ========================================================================

    void set_accept_keyframes(bool accept);
    bool stop_if_requested();
    void reset_if_requested();
    bool is_reset_requested() const;
    bool is_finish_requested() const;
    void set_finished();

    mutable std::mutex finish_mutex_;
    mutable std::mutex reset_mutex_;
    mutable std::mutex new_keyframes_mutex_;
    mutable std::mutex stop_mutex_;
    mutable std::mutex accept_mutex_;

    bool stopped_{false};
    bool stop_requested_{false};
    bool insertion_locked_{false};
    bool accept_keyframes_{true};
    bool reset_requested_{false};
    bool finish_requested_{false};
    bool finished_{true};

    // ========================================================================
    // State
    // ========================================================================

    std::shared_ptr<Map> map_;
    std::shared_ptr<LoopClosing> loop_closer_;
    std::unique_ptr<FeatureMatcher> matcher_;
    std::shared_ptr<Viewer> viewer_;
    std::shared_ptr<KeyframeInformation> keyframe_information_{};

    std::list<Keyframe> new_keyframes_;
    Keyframe current_keyframe_;
    std::list<Pt> recent_map_points_;

    bool warned_no_kernel_{false};
    int last_over_budget_{-1};

    std::map<int, int> local_mapping_times_{};
};

}
