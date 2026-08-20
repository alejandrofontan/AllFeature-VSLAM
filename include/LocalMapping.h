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


namespace AF_VSLAM
{

class Tracking;
class LoopClosing;
class Map;
class Viewer;

class LocalMapping
{
public:
    LocalMapping(shared_ptr<Map> pMap, const float bMonocular, const vector<FeatureType>& featureTypes, const int& image_width, const int& image_height);

    void SetLoopCloser(std::shared_ptr<LoopClosing>  loopCloser_){loopCloser = loopCloser_;};
    void SetTracker(std::shared_ptr<Tracking> tracker_){tracker = tracker_;};
    void SetViewer(std::shared_ptr<Viewer> viewer_){viewer = viewer_;};

    // Main function
    void Run();

    void insert_keyframe(Keyframe pKF);

    // Thread Synch
    void RequestStop();
    void RequestReset();
    bool Stop();
    void Release();
    bool isStopped();
    bool stopRequested();
    bool AcceptKeyFrames();
    void SetAcceptKeyFrames(bool flag);
    bool SetNotStop(bool flag);
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
    int featureProcessNewKeyframe{0};

    // Parameters for local mapping
    const float CHI2_2DOF{5.991f};

    // KeyFrameCulling()
    const float KEYFRAME_CULLING_COVISIBILITY_THRESHOLD{0.9f}; // 0.9f
    const int KEYFRAME_CULLING_MIN_NUM_OBSERVATIONS{3}; // 3

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
    void KeyFrameCulling();
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
    Keyframe mpCurrentKeyFrame;
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
