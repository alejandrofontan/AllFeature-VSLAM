//
// Created by fontan on 30/03/24.
//

#ifndef AF_VSLAM_IMAGE_H
#define AF_VSLAM_IMAGE_H

#include <opencv2/core/core.hpp>
#include <string>

namespace AF_VSLAM {

    class Image {
    public:
        explicit Image(const std::string& imagePath);
        explicit Image(const cv::Mat& image);
        cv::Mat img{};
        cv::Mat grayImg{};
        cv::Mat mask{};
        cv::Mat depthImg{};

        std::string imageFile{};
        std::string imageName{};
        std::string depthFile{};

        // depthFactor converts raw depth-image units to meters: metric_depth = raw_value / depthFactor
        // (e.g. 5000 for TUM/ETH-style datasets storing depth at 1/5000 m resolution). Defaults to 1
        // (no scaling, i.e. the depth image is assumed to already be metric).
        void LoadDepth(const std::string& depthPath, const float& depthFactor = 1.0f);
        void GetGrayImage(const bool& rgb);
        void FixImageSize(const int& new_width, const int& new_height);
    };

}

#endif //AF_VSLAM_IMAGE_H
