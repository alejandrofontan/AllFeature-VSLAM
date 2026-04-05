#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "include/FeatureMatcher.h"

#ifdef CHECK
#undef CHECK
#endif

#include "Thirdparty/SuperPoint-LightGlue-TensorRT/include/light_glue.h"
#include "Thirdparty/SuperPoint-LightGlue-TensorRT/include/super_point.h"

std::vector<cv::DMatch> AF_VSLAM::FeatureMatcher::matcherLightglueSuperpoint(
    const std::vector<cv::KeyPoint>& kps1, const cv::Mat& desc1,
    const std::vector<cv::KeyPoint>& kps2, const cv::Mat& desc2,
    float min_score)
{

    // Build 258-row feature matrices: rows 0-1 = x/y, rows 2-257 = descriptor
    auto make_fp = [](const std::vector<cv::KeyPoint>& kps, const cv::Mat& desc)
        -> Eigen::Matrix<double, 258, Eigen::Dynamic>
    {
        const int N = (int)kps.size();
        Eigen::Matrix<double, 258, Eigen::Dynamic> fp(258, N);
        for (int i = 0; i < N; ++i) {
            fp(0, i) = kps[i].pt.x;
            fp(1, i) = kps[i].pt.y;
            for (int d = 0; d < 256; ++d)
                fp(2 + d, i) = desc.at<float>(i, d);
        }
        return fp;
    };

    Eigen::Matrix<double, 258, Eigen::Dynamic> fp0_ = make_fp(kps1, desc1);
    Eigen::Matrix<double, 258, Eigen::Dynamic> fp1_ = make_fp(kps2, desc2);
    std::vector<cv::DMatch> matches;

    std::lock_guard<std::mutex> lock(lightglue_superpoint_mutex_);
    matcher_lightglue_superpoint->matching_points(fp0_, fp1_, matches);

    return matches;
}
