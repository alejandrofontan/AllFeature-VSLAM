//
// Created by fontan on 30/03/24.
//

#include "Image.h"

#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>

#include "afvslam_log.hpp"

namespace
{

// The colour image is 8-bit everywhere downstream (the extractors, the viewer, the point colours). A file
// read as stored can be deeper: 16-bit is scaled from its full range, floating point from [0, 1].
void to_8bit(cv::Mat& image)
{
    if (image.empty() || image.depth() == CV_8U)
        return;

    double scale = 0.0;
    switch (image.depth()) {
        case CV_16U: scale = 255.0 / 65535.0; break;
        case CV_32F:
        case CV_64F: scale = 255.0; break;
        default:
            throw std::runtime_error("Image: unsupported pixel depth (OpenCV depth code " + std::to_string(image.depth())
                                     + "); 8-bit, 16-bit unsigned and floating-point images are accepted");
    }

    static std::once_flag warned;
    std::call_once(warned, [&image] {
        AF_WARN("Image: " << (image.depth() == CV_16U ? "16-bit" : "floating-point") << " images are converted to 8-bit, "
                << (image.depth() == CV_16U ? "taking 0..65535 as the full range" : "taking 0..1 as the full range")
                << " (reported once)");
    });
    image.convertTo(image, CV_8U, scale);
}

}

AF_VSLAM::Image::Image(const std::string &imagePath): imageFile{imagePath} {

    img = cv::imread(imageFile,cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << imageFile << std::endl;
    }
    to_8bit(img);

    std::filesystem::path p(imagePath);
    imageName = p.filename().string();
}

AF_VSLAM::Image::Image(const cv::Mat& image){
    img = image.clone();
    to_8bit(img);
}

void AF_VSLAM::Image::LoadDepth(const std::string &depthPath, const float& depthFactor) {
    depthFile = depthPath;
    depthImg = cv::imread(depthFile, cv::IMREAD_UNCHANGED);
    if (depthImg.empty()) {
        std::cerr << "Failed to load depth image: " << depthFile << std::endl;
        return;
    }

    // A raw value sitting exactly at the format's max representable value is a sensor
    // saturation/out-of-range sentinel, not a real depth reading -- map it to 0, matching
    // the same "0 = invalid" convention already relied on downstream (Frame::get_depth).
    double maxRawValue = -1.0;
    switch (depthImg.type()) {
        case CV_8U:  maxRawValue = 255.0;   break;
        case CV_16U: maxRawValue = 65535.0; break;
        case CV_16S: maxRawValue = 32767.0; break;
    }
    if (maxRawValue > 0.0) {
        depthImg.setTo(0, depthImg == maxRawValue);
    }

    // Convert to metric CV_32F. Guard against a near-zero factor (missing/invalid
    // calibration) blowing up to a huge scale — treat it as "already metric" instead.
    const float scale = (std::fabs(depthFactor) < 1e-5f) ? 1.0f : (1.0f / depthFactor);
    if (scale != 1.0f || depthImg.type() != CV_32F) {
        depthImg.convertTo(depthImg, CV_32F, scale);
    }
}

void AF_VSLAM::Image::get_gray_image() {
    if (img.channels() == 3) {
        cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
    } else if (img.channels() == 4) {
        cvtColor(img, grayImg, cv::COLOR_BGRA2GRAY);
    } else {
        grayImg = img;
    }
}

void AF_VSLAM::Image::fix_image_size(const int& new_width, const int& new_height){
    if (grayImg.empty()) {
        throw std::runtime_error("fix_image_size: grayImg is empty — call get_gray_image() first");
    }

    cv::Size new_size(new_width, new_height);
    // Shrinking: every output pixel is the average of the source pixels it covers (no aliasing). Linear
    // interpolation reads only the 2x2 pixels around each position and skips the rest. Enlarging: linear.
    const bool shrinks = new_width <= img.cols && new_height <= img.rows && (new_width < img.cols || new_height < img.rows);
    const int interpolation = shrinks ? cv::INTER_AREA : cv::INTER_LINEAR;
    cv::resize(grayImg, grayImg, new_size, 0, 0, interpolation);
    cv::resize(img, img, new_size, 0, 0, interpolation);

    // Floor to a multiple of SIZE_MULTIPLE (top-left anchored: the principal point is unchanged)
    const int cropWidth = (img.cols / SIZE_MULTIPLE) * SIZE_MULTIPLE;
    const int cropHeight = (img.rows / SIZE_MULTIPLE) * SIZE_MULTIPLE;

    cv::Rect cropRect(0, 0, cropWidth, cropHeight);
    img = img(cropRect).clone();
    grayImg = grayImg(cropRect).clone();

    if (!depthImg.empty()) {
        cv::resize(depthImg, depthImg, new_size, 0, 0, cv::INTER_NEAREST);
        depthImg = depthImg(cropRect).clone();
    }

    if (!mask.empty()) {
        cv::resize(mask, mask, new_size, 0, 0, cv::INTER_NEAREST);
        mask = mask(cropRect).clone();
    }
}