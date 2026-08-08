//
// Created by fontan on 30/03/24.
//

#include "Image.h"

#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <filesystem>

AF_VSLAM::Image::Image(const std::string &imagePath): imageFile{imagePath} {

    img = cv::imread(imageFile,cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << imageFile << std::endl;
    }

    std::filesystem::path p(imagePath);
    imageName = p.filename().string();
}

AF_VSLAM::Image::Image(const cv::Mat& image){
    img = image.clone();
}

void AF_VSLAM::Image::LoadMask(const std::string &maskPath) {
    maskFile = maskPath;
    mask = cv::imread(maskFile, cv::IMREAD_UNCHANGED);
    if (mask.empty()) {
        std::cerr << "Failed to load mask image: " << maskFile << std::endl;
    }
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
    // the same "0 = invalid" convention already relied on downstream (Frame::GetDepth).
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

void AF_VSLAM::Image::GetGrayImage(const bool& rgb) {
    if (img.channels() == 3) {
        cvtColor(img, grayImg, rgb ? cv::COLOR_RGB2GRAY : cv::COLOR_BGR2GRAY);
    } else if (img.channels() == 4) {
        cvtColor(img, grayImg, rgb ? cv::COLOR_RGBA2GRAY : cv::COLOR_BGRA2GRAY);
    } else {
        grayImg = img;
    }
}

void AF_VSLAM::Image::FixImageSize(const int& new_width, const int& new_height){
    if (grayImg.empty()) {
        throw std::runtime_error("FixImageSize: grayImg is empty — call GetGrayImage() first");
    }

    cv::Size new_size(new_width, new_height);
    cv::resize(grayImg, grayImg, new_size);
    cv::resize(img, img, new_size);

    // Floor to a multiple of 32: the learned feature extractors (SuperPoint/ALIKED)
    // downsample internally by a power-of-2 stride and expect input dimensions
    // divisible by it.
    int cropWidth = (img.cols / 32) * 32;
    int cropHeight = (img.rows / 32) * 32;

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