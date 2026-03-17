#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "include/FeatureMatcher.h"
#include "Thirdparty/SuperPoint-SuperGlue-TensorRT/include/super_glue.h"

std::vector<cv::DMatch> ANYFEATURE_VSLAM::FeatureMatcher::superglueMatching(
    const std::vector<cv::KeyPoint>& kps1, const cv::Mat& desc1,
    const std::vector<cv::KeyPoint>& kps2, const cv::Mat& desc2,
    float min_score)
{
    if (kps1.empty() || kps2.empty())
        return {};

    auto make_fp = [](const std::vector<cv::KeyPoint>& kps, const cv::Mat& desc)
        -> Eigen::Matrix<double, 259, Eigen::Dynamic>
    {
        const int N = (int)kps.size();
        Eigen::Matrix<double, 259, Eigen::Dynamic> fp(259, N);
        for (int i = 0; i < N; ++i) {
            fp(0, i) = kps[i].response;
            fp(1, i) = kps[i].pt.x;
            fp(2, i) = kps[i].pt.y;
            for (int d = 0; d < 256; ++d)
                fp(3 + d, i) = desc.at<float>(i, d);
        }
        return fp;
    };

    Eigen::Matrix<double, 259, Eigen::Dynamic> fp0 = make_fp(kps1, desc1);
    Eigen::Matrix<double, 259, Eigen::Dynamic> fp1 = make_fp(kps2, desc2);

    // Serialize all SuperGlue calls — IExecutionContext is not thread-safe
    std::lock_guard<std::mutex> lock(superglue_mutex_);

    std::vector<cv::DMatch> matches;
    matcher_superglue->matching_points(fp0, fp1, matches, false, imageWidth, imageHeight);

    if (min_score > 0.0f) {
        matches.erase(
            std::remove_if(matches.begin(), matches.end(),
                [min_score](const cv::DMatch& m) {
                    return (1.0f - m.distance) < min_score;
                }),
            matches.end());
    }

    return matches;
}