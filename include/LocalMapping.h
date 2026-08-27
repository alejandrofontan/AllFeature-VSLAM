#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include "KeyFrame.h"
#include "Map.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"
#include "FeatureMatcher.h"
#include "Viewer.h"


#include <mutex>

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
    // cull_keyframes()
    float keyframe_culling_redundancy_ratio{0.9f}; // cull when more than this fraction of a keyframe's points is redundant
    int keyframe_culling_min_observations{3};      // a point is redundant when this many OTHER keyframes observe it
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
    // Mark as bad the covisible keyframes whose map points are (mostly) already
    // observed by enough other keyframes; see the definition for the exact rule.
    void cull_keyframes();
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
