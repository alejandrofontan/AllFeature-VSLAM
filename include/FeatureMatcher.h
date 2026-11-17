#pragma once

#include <mutex>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>

#include "Frame.h"
#include "KeyFrame.h"
#include "MapPoint.h"

// Learned matching: the three LightGlue variants (superpoint / aliked / sift) run on ONNX Runtime through the
// process-wide sessions of onnx/FeatureModels.h; classical descriptors
// go through the brute-force matcher (BruteForceMatcher.h). No CMake backend option gates the dispatch any more.

namespace AF_VSLAM
{

class FeatureMatcher
{
public:

    // Loads (once per process) and warms up the LightGlue variant of every learned feature in `feature_types`.
    FeatureMatcher(const int& image_width, const int& image_height,
        const std::vector<FeatureType>& feature_types,
        std::string name = "Extractor", float nnratio=0.6, bool checkOri=true);

    // Computes the Hamming distance between two ORB descriptors
    static Descriptor_Distance_Type descriptor_distance(const cv::Mat &a, const cv::Mat &b, const FeatureType& featureType_);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::vector<cv::DMatch> match_descriptors(const cv::Mat& desc1, const cv::Mat& desc2,
        const std::vector<cv::KeyPoint>& kps1, const std::vector<cv::KeyPoint>& kps2, const FeatureType& feat_type);
    std::vector<cv::DMatch> match_descriptors_only(const cv::Mat& desc1, const cv::Mat& desc2,  const FeatureType& feat_type);

    // Robust geometric match filtering via PoseLib LO-RANSAC: a fundamental matrix, a homography if that finds
    // no consensus, and the matches unfiltered if neither does. It replaced OpenCV's findFundamentalMat/USAC,
    // which threw on degenerate inputs and ground for seconds per call at zero-baseline frames.
    // Every match takes part and the result keeps the input order; there is no cap on the count.
    // Every queryIdx / trainIdx must be a valid index into kps1 / kps2.
    static constexpr double filter_inlier_threshold_px = 3.0;  // Sampson (F) / transfer (H) error, pixels
    static constexpr size_t filter_min_inliers = 8;            // fewer: the model is rejected
    static constexpr size_t filter_min_iterations = 100;       // RANSAC rounds, lower and upper bound
    static constexpr size_t filter_max_iterations = 1000;
    static std::vector<cv::DMatch> filter_matches_geometric(const std::vector<cv::DMatch>& matches,
        const std::vector<cv::KeyPoint>& kps1, const std::vector<cv::KeyPoint>& kps2);

    // Takes the two FeatureData directly: splitting them into per-type descriptor and keypoint
    // maps at the call site copied every keypoint vector on every call.
    std::map<FeatureType, std::vector<cv::DMatch>> match_descriptors_parallel(
        const std::vector<FeatureType>& feature_types,
        const FeatureData& features1, const FeatureData& features2);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Matches features from a keyframe to a frame across all requested feature types.
    // Matches descriptors per type, filters outliers jointly with PoseLib LO-RANSAC, and associates
    // inlier matches to map points. Writes the frame's match cache; the keyframe is not modified.
    // Returns the number of valid map point matches per feature type.
    std::map<FeatureType, int> match_keyframe_to_frame(const Keyframe& keyframe, Frame& frame,
        std::map<FeatureType, std::vector<Pt>>& map_pts_matches,
        const std::vector<FeatureType>& feat_types);

    // Gives a map point to the frame keypoints that have none yet: descriptor matching of the unmatched
    // keypoints against the candidates, then a check that the candidate projects near the keypoint.
    // `map_pts` are the candidates Tracking found visible and `projections` their projections into the frame,
    // parallel arrays. Accepted matches are written to frame.pts.
    // Used in: Tracking::search_local_points
    // The gate: a match is accepted when the squared offset between keypoint and projection is below this many
    // times the keypoint's own position variance (KeypointData::sigma2 / information), the model pose
    // optimization tests outliers with -- so a coarser keypoint gets a wider gate. 64 is (8 sigmas)^2: on
    // ETH table_3 the correct orb32 matches lie within 8 px at sigma 1, and beyond that it is mostly wrong
    // pairings. It is well above the optimizer's chi2 of 5.99 because the gate also has to absorb the pose's
    // and the map point's error. Was a fixed 20 px (400 in these units at sigma 1).
    static constexpr float projection_match_scale_sigma2 = 64.0f;
    void match_map_points_to_frame(Frame& frame, const std::vector<Pt>& map_pts,
                                   const std::vector<cv::Point2f>& projections);

    // Makes sure the matches between `keyframe` and each of `others` are in `keyframe`'s match cache: pairs
    // already cached are skipped; the rest are matched (all pairs and feature types in parallel), filtered
    // geometrically per pair, jointly over the feature types, and stored in `keyframe` only. The only writer of
    // the cache for keyframe pairs; tracking writes it for a frame against its reference keyframe, filtered too.
    // The cache is a hand-off: read the pairs back, then release them with clear_keyframe_matches.
    // Used in: LocalMapping::create_new_map_points, LoopClosing::compute_sim3, LoopClosing::search_loop_map_points.
    // The pairs are read back with get_keyframe_matches.
    void match_keyframe_to_keyframes(const Keyframe& keyframe, const std::vector<Keyframe>& others,
        const std::vector<FeatureType>& feat_types);

    // The matched keypoint pairs (index in keyframe1, index in keyframe2) that keyframe1's match cache holds for
    // keyframe2, per feature type. A read only: it never matches. A pair of keyframes that was not cached first
    // (match_keyframe_to_keyframes) gives no pairs.
    // Used in: LocalMapping::create_new_map_points, LoopClosing::compute_sim3
    static std::map<FeatureType, std::vector<std::pair<size_t, size_t>>> get_keyframe_matches(
        const Keyframe& keyframe1, const Keyframe& keyframe2);

    // Releases every match list `keyframe` holds; the cache is otherwise never pruned. Called by the thread that
    // filled them once it has read them back.
    // Used in: LocalMapping::process_keyframe, LoopClosing::run
    static void clear_keyframe_matches(const Keyframe& keyframe);

    // Matches keypoints between two frames across all feature types for monocular initialization.
    // Runs parallel descriptor matching, filters outliers jointly with the geometric filter
    // (filter_matches_geometric), and populates matched_pairs per feature type.
    // Used in: Tracking::monocular_initialization
    map<FeatureType, vector<pair<size_t, size_t>>> match_frames_for_initialization(const Frame& F1, const Frame& F2,
        const vector<FeatureType>& feat_types);

    // Projects map points into a keyframe and fuses them with existing keypoints.
    // Applies visibility checks (depth, image bounds, scale, viewing angle), then matches by descriptor distance within radius_th.
    // Replaces the weaker map point if one already exists, or adds a new observation.
    // Used in: LocalMapping::SearchInNeighbors
    const float cos_viewing_angle_th = 0.5f;
    const float chi2_perc = ChiSquared::dof2_95;
    int fuse_map_points_to_keyframe(const Keyframe& keyframe, const vector<Pt>& map_pts, const float& radius_th, const FeatureType& feat_type);

    /// Projects candidate map points into a keyframe using a Sim3 and validates projections
    /// against cached matches. Writes accepted matches into @p pts_matched.
    /// @return Number of new map point assignments. Used in: LoopClosing::search_loop_map_points
    const float radiusTh_factor = 12.0f;
    int search_by_projection_for_compute_sim3(const Keyframe& keyframe, const mat4f& Scw,
        const vector<Pt>& pts, vector<Pt>& pts_matched,
        const map<PtId, Keyframe>& pt_to_keyframe_id);

    int fuse_map_points_to_keyframe(Keyframe& keyframe, const mat4f& Scw,
        const std::vector<Pt> &map_pts, const float& radius_th, vector<Pt> &replace_pts, const FeatureType& feat_type);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Project MapPoints seen in KeyFrame into the Frame and search matches.
    // Used in relocalisation (Tracking)
    int search_by_projection(Frame &CurrentFrame, Keyframe pKF, const std::set<Pt> &sAlreadyFound, const float& radiusTh, const bool& useHighMatchingThreshold, const FeatureType& featureType);
    static void setDescriptorDistanceThresholds(const string &feature_settings_yaml_file, const FeatureType& featureType);

public:

    static VerbosityLevel verbosity;
    static std::map<FeatureType, Descriptor_Distance_Type> TH_LOW;
    static std::map<FeatureType, Descriptor_Distance_Type> descDistTh_high_reloc;
    static std::map<FeatureType, Descriptor_Distance_Type> descDistTh_low_reloc;

    static constexpr int HISTO_LENGTH = 30;
    static float radiusScale;

protected:

    // filter_matches_geometric on the matches of several feature types at once: one model is fitted to all of
    // them, since every type sees the same two views. `matches_by_type` holds per-type keypoint indices into
    // features1 (queryIdx) and features2 (trainIdx); so does the result, which keeps the surviving matches.
    static std::map<FeatureType, std::vector<cv::DMatch>> filter_matches_geometric_jointly(
        const std::map<FeatureType, std::vector<cv::DMatch>>& matches_by_type,
        const FeatureData& features1, const FeatureData& features2);

    string name{};
    // Returns a reused thread-local buffer, cleared per call — do not hold across calls.
    static vector<vector<int>>& initRotationHistogram(float& rotFactor, const int& histLength);
    static void updateRotationHistogram(vector<vector<int>>& rotHist,
                                        const KeypointIndex& idx,
                                        const cv::KeyPoint& keyPt, const cv::KeyPoint& refKeyPt,
                                        const float& rotFactor, const int& histLength);
    static void computeThreeMaxima(vector<vector<int>>& rotHist, int &ind1, int &ind2, int &ind3);
    static void filterMatchesWithOrientation(vector<vector<int>>& rotHist, vector<Pt>& points, int& nMatches);
    static void filterMatchesWithOrientation(vector<vector<int>>& rotHist, vector<int>& matches, int& nMatches);

    float mfNNratio;
    bool mbCheckOrientation;

    const Descriptor_Distance_Type highest_possible_distance{std::numeric_limits<Descriptor_Distance_Type>::max()};

    // Brute-force matching now goes through BruteForceMatcher.h (issue #13 P2), which
    // replicates the cv::BFMatcher(<norm>, crossCheck=true) semantics these members had.
    int image_width;
    int image_height;


    // Matching options
    std::vector<FeatureType> feature_types;
};

}
