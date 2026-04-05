//
// Created by fontan on 30/03/24.
//

#ifndef AF_VSLAM_IMAGE_H
#define AF_VSLAM_IMAGE_H

#include<opencv2/core/core.hpp>

namespace AF_VSLAM {

    class Image {
    public:
        explicit Image(const std::string& imagePath);
        explicit Image(const cv::Mat& image);
        cv::Mat img{};
        cv::Mat grayImg{};
        cv::Mat mask{};

        std::string imageFile{};
        std::string imageName{};
        std::string keypointBinFile{};
        std::string scoresBinFile{};
        std::string descriptorsBinFile{};

        void LoadMask(const std::string& maskPath);
        void GetGrayImage(const bool& rgb);
        void FixImageSize(const int& new_width, const int& new_height);
    };

}

#endif //AF_VSLAM_IMAGE_H
