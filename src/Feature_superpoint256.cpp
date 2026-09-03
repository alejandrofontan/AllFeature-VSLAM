#include "Feature_superpoint256.h"
#include "afvslam_log.hpp"

#include <opencv2/opencv.hpp>

AF_VSLAM::FeatureExtractor_superpoint256::FeatureExtractor_superpoint256(std::shared_ptr<FeatureExtractorSettings> &settings_):
        FeatureExtractor(settings_){

        AF_INFO("Initializing SuperPoint256 feature extractor...");
        AF_INFO("Building SuperPoint inference engine...");

        const std::string& config_path = Superpoint256::getConfigYaml();
        const std::string& model_dir = Superpoint256::getModelsDir();
        Configs configs(config_path, model_dir);
        AF_INFO("SuperPoint256 config loaded.");
        extractor = std::make_shared<SuperPoint>(configs.superpoint_config);
        if (!extractor->build()) {
            throw std::runtime_error(
                "SuperPoint: failed to build/load TensorRT engine.\n"
                "  config: " + config_path + "\n"
                "  weights: " + model_dir);
        }
        AF_INFO("SuperPoint engine loaded from: " + model_dir);
        AF_INFO("Engine ready.");
}


void AF_VSLAM::FeatureExtractor_superpoint256::detectAndCompute(
    const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{

    Eigen::Matrix<double, 258, Eigen::Dynamic> features;
    Eigen::Matrix<double, 1, Eigen::Dynamic> features_score;
    extractor->infer(img.grayImg, features, features_score);

    int N = (int)features.cols();

    if (N == 0) {
        keypoints.clear();
        descriptors = cv::Mat();
        return;
    }

    // --- Build keypoints and descriptors ---
    // New API: features row 0=x, row 1=y, rows 2..257=descriptor; separate features_score row 0=score
    std::vector<cv::KeyPoint> keypoints_;
    keypoints_.reserve(N);

    cv::Mat descriptors_(N, 256, CV_32F);
    for (int i = 0; i < N; ++i) {
        cv::KeyPoint kp;
        kp.pt.x     = (float)features(0, i);
        kp.pt.y     = (float)features(1, i);
        kp.response = (float)features_score(0, i);
        kp.size     = 1.f;
        kp.angle    = 0.f;
        kp.octave   = 0;
        kp.class_id = i;
        keypoints_.push_back(kp);
        for (int d = 0; d < 256; ++d)
            descriptors_.at<float>(i, d) = (float)features(2 + d, i);
    }

    // Discard keypoints outside the mask (if one was loaded for this frame)
    FilterKeypointsByMask(keypoints_, descriptors_, img.mask);

    keypoints = keypoints_;
    descriptors.create((int)keypoints.size(), 256, CV_32F);
    for (int i = 0; i < (int)keypoints.size(); ++i)
        descriptors_.row(keypoints[i].class_id).copyTo(descriptors.row(i));
}

int AF_VSLAM::FeatureExtractor_superpoint256::GetKeypointOctave(const cv::KeyPoint&) const{
    return 0;
}

float AF_VSLAM::FeatureExtractor_superpoint256::GetKeypointSize(const cv::KeyPoint&) const{
    return 1.0f;
}

float AF_VSLAM::DescriptorDistance_superpoint256(const cv::Mat &a, const cv::Mat &b){
    return (Descriptor_Distance_Type) cv::norm(a, b, cv::NORM_L2);
}