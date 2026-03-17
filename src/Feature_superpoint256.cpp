#include "Feature_superpoint256.h"

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

ANYFEATURE_VSLAM::FeatureExtractor_superpoint256::FeatureExtractor_superpoint256(std::shared_ptr<FeatureExtractorSettings> &settings_):
        FeatureExtractor(settings_){


        std::cout << "Building inference engines...\n";
        std::string config_path = "/home/alejandro/VSLAM-LAB/Baselines/AllFeature-VSLAM-DEV/Thirdparty/SuperPoint-SuperGlue-TensorRT/config/config.yaml";
        std::string model_dir = "/home/alejandro/VSLAM-LAB/Baselines/AllFeature-VSLAM-DEV/Thirdparty/SuperPoint-SuperGlue-TensorRT/weights/";
        Configs configs(config_path, model_dir);
        std::cout << "SuperPoint256 config loaded.\n";
        extractor = std::make_shared<SuperPoint>(configs.superpoint_config);
        if (!extractor->build()) {
            throw std::runtime_error(
                "SuperPoint: failed to build/load TensorRT engine.\n"
                "  config: " + config_path + "\n"
                "  weights: " + model_dir);
        }
        std::cout << "SuperPoint engine loaded from: " << model_dir << "\n";
                std::cout << "Engines ready.\n\n";
}


void ANYFEATURE_VSLAM::FeatureExtractor_superpoint256::detectAndCompute(
    const Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    Eigen::Matrix<double, 259, Eigen::Dynamic> fp0;
    extractor->infer(img.grayImg, fp0);
    int N = (int)fp0.cols();
    if (N == 0) {
        keypoints.clear();
        descriptors = cv::Mat();
        return;
    }

    // --- Build keypoints and descriptors ---
    std::vector<cv::KeyPoint> keypoints_;
    keypoints_.reserve(N);

    // Descriptors: N rows x 256 cols, float32
    cv::Mat descriptors_(N, 256, CV_32F);
    for (int i = 0; i < N; ++i) {
        cv::KeyPoint kp;
        kp.pt.x    = (float)fp0(1, i);
        kp.pt.y    = (float)fp0(2, i);
        kp.response = (float)fp0(0, i);
        kp.size    = 1.f;
        kp.angle   = 0.f;
        kp.octave  = 0;
        kp.class_id = i;
        keypoints_.push_back(kp);

        // Copy descriptor row (rows 3..258)
        for (int d = 0; d < 256; ++d)
            descriptors_.at<float>(i, d) = (float)fp0(3 + d, i);
    }

    // --- Retain keypoints (with octree distribution if over budget) ---
    //if ((int)keypoints_.size() <= (int)(settings->OVERSIZE_KEYPOINT_FACTOR * settings->maxNumFeatures)) {
        keypoints    = keypoints_;
        //descriptors  = descriptors_;
        //return;
    //}

    // Too many — distribute via octree, then gather matching descriptor rows
    // keypoints = DistributeOctTree(
    //     keypoints_, 0, img.grayImg.cols - 1, 0, img.grayImg.rows - 1,
    //     settings->maxNumFeatures, 0);

    descriptors.create((int)keypoints.size(), 256, CV_32F);
    for (int i = 0; i < (int)keypoints.size(); ++i)
        descriptors_.row(keypoints[i].class_id).copyTo(descriptors.row(i));

    std::cout << "SuperPoint256: " << keypoints.size() << " keypoints retained after octree distribution." << std::endl;
}

int ANYFEATURE_VSLAM::FeatureExtractor_superpoint256::GetKeypointOctave(const cv::KeyPoint& keypoint) const{
    return 0;
}

float ANYFEATURE_VSLAM::FeatureExtractor_superpoint256::GetKeypointSize(const cv::KeyPoint& keypoint) const{
    return 1.0f;
}

float ANYFEATURE_VSLAM::DescriptorDistance_superpoint256(const cv::Mat &a, const cv::Mat &b){
    return (Descriptor_Distance_Type) cv::norm(a, b, cv::NORM_L2);
}