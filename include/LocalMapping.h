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
    LocalMapping(shared_ptr<Map> pMap, const float bMonocular, const vector<FeatureType>& featureTypes, const int& image_width, const int& image_height);

    // Tunable parameters, loaded from the settings YAML at System startup
    static LocalMappingParameters params;
    static void LoadParameters(const cv::FileStorage &fSettings);

    void SetLoopCloser(std::shared_ptr<LoopClosing>  loopCloser_){loopCloser = loopCloser_;};
    void SetTracker(std::shared_ptr<Tracking> tracker_){tracker = tracker_;};
    void SetViewer(std::shared_ptr<Viewer> viewer_){viewer = viewer_;};

    // Online keyframe x keyframe VPR similarity matrix (cosine of the keyframes' MegaLoc global
    // descriptors), grown in ProcessNewKeyFrame: row/col k = k-th keyframe with a descriptor
    // processed by LocalMapping (insertion order, see keyframe_vpr_order()); rows/cols are never
    // removed when a keyframe is culled (culled keyframes stay as culling "history"). Empty
    // when the VPR backend does not produce global descriptors (vpr: bow / none). Both return
    // copies taken under the matrix mutex.
    Eigen::MatrixXf keyframe_vpr_matrix() const;
    std::vector<Keyframe> keyframe_vpr_order() const;
    bool has_keyframe_vpr_matrix() const;

    // Main function
    void Run();

    void insert_keyframe(Keyframe pKF);

    // Thread Synch
    void request_stop();
    void request_reset();
    bool Stop();
    void release();
    bool is_stopped();
    bool is_stop_requested();
    bool accepts_keyframes();
    void SetAcceptKeyFrames(bool flag);
    bool set_insertion_lock(bool flag);
    void InterruptBA();
    void RequestFinish();
    bool isFinished();
    int KeyframesInQueue(){
        unique_lock<std::mutex> lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }

    std::map<int, int> localMapping_times{};
    std::map<int, int> createNewMapPoints_times{};
    std::map<int, int> localbundleadjustment_times{};
    std::map<int, int> searchInNeighbors_times{};

    // void medianLocalMappingTime(){
    //     if(!localMappingTime.empty()){
    //         std::vector<double> tmp = localMappingTime;
    //         std::sort(tmp.begin(), tmp.end());
    //         double median;
    //         size_t n = tmp.size();
    //         if(n % 2 == 1) median = tmp[n/2];
    //         else median = 0.5*(tmp[n/2 - 1] + tmp[n/2]);

    //         const double sum = std::accumulate(localMappingTime.begin(), localMappingTime.end(), 0.0);
    //         double stddev = 0.0;
    //         if (n >= 2) {
    //             double sq_sum = 0.0;
    //             for (double x : localMappingTime) {
    //                 const double d = x - median;
    //                 sq_sum += d * d;
    //             }
    //             stddev = std::sqrt(sq_sum / static_cast<double>(n - 1));
    //         }
    //         std::cout << std::fixed << std::setprecision(2) << "Local  median / std: " << " / " << median*1000 << " / " << stddev*1000 << " / " << " ms" << std::endl;
    //     }
    // }
protected:

    vector<FeatureType> featureTypes{};

    // Parameters for local mapping
    const float CHI2_2DOF{5.991f};

    // CreateNewMapPoints()
    const int CREATE_NEW_MAP_POINTS_BEST_COVISIBILITY_KEYFRAMES{5};
    const float CREATE_NEW_MAP_POINTS_RATIO_BASELINE_DEPTH{0.01f};
    const float CREATE_NEW_MAP_POINTS_MIN_COS{0.9998f};

    // MapPointCulling()
    const int MAP_POINT_CULLING_MIN_NUM_OBSERVATIONS{2};

    // SearchInNeighbors()
    const int SEARCH_IN_NEIGHBORS_NUM_KEYFRAMES{20};
    const int SEARCH_IN_NEIGHBORS_NUM_KEYFRAMES_SECOND{5};
    const float SEARCH_IN_NEIGHBORS_RADIUS_TH{5.f};

    bool CheckNewKeyFrames();
    void ProcessNewKeyFrame();
    void CreateNewMapPoints();
    mat3f ComputeF12(Keyframe &pKF1, Keyframe &pKF2);
    mat3f SkewSymmetricMatrix(const vec3f &v);
    void MapPointCulling();
    void SearchInNeighbors(const FeatureType& featureType);
    // Keyframe culling: dispatches on params.keyframe_culling_method.
    void cull_keyframes();
    // "heuristic": mark as bad the covisible keyframes whose map points are (mostly)
    // already observed by enough other keyframes; see the definition for the exact rule.
    void cull_keyframes_heuristic();
    // "information": joint-information culling on the online keyframe VPR matrix; see
    // the definition for the exact rule.
    void cull_keyframes_information();
    void ResetIfRequested();
    bool CheckFinish();
    void SetFinish();

    std::mutex mMutexFinish;
    std::mutex mMutexReset;
    std::mutex mMutexNewKFs;
    std::mutex mMutexStop;
    std::mutex mMutexAccept;

    std::shared_ptr<Map> mpMap;
    std::shared_ptr<LoopClosing> loopCloser;
    std::shared_ptr<Tracking> tracker;
    std::shared_ptr<FeatureMatcher> matcher;
    std::shared_ptr<Viewer> viewer;

    std::list<Keyframe> mlNewKeyFrames;
    Keyframe current_keyframe_;

    // Online keyframe VPR matrix (see keyframe_vpr_matrix())
    mutable std::mutex vpr_mutex_;
    std::vector<Keyframe> vpr_keyframes_{};       // k -> keyframe
    Eigen::MatrixXf keyframe_vpr_matrix_{};        // k x k, raw cosine
    void grow_keyframe_vpr_matrix(const Keyframe& keyframe);  // called once per processed keyframe
    void print_keyframe_vpr_matrix() const;
    std::list<Pt> mlpRecentAddedMapPoints;

    bool mbAbortBA;
    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    bool mbAcceptKeyFrames;
    bool mbMonocular;
    bool mbResetRequested;
    bool mbFinishRequested;
    bool mbFinished;

    const int image_width;
    const int image_height;

};

} //namespace ORB_SLAM

#endif // LOCALMAPPING_H
