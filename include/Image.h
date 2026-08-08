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
        std::string maskFile{};

        void LoadMask(const std::string& maskPath);
        void LoadDepth(const std::string& depthPath);
        void GetGrayImage(const bool& rgb);
        void FixImageSize(const int& new_width, const int& new_height);
    };

}

#endif //AF_VSLAM_IMAGE_H
