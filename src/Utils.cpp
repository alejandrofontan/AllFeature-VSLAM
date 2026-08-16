//
// Created by fontan on 20/02/24.
//

#include <fstream>
#include "Utils.h"
#include <utility>
#include "Feature_superpoint256.h"

std::random_device rd;
//std::mt19937 AF_VSLAM::RandomIntegerGenerator::randomIntGenerator{std::mt19937(rd())};
std::mt19937 AF_VSLAM::RandomIntegerGenerator::randomIntGenerator{123456u};

void AF_VSLAM::printInfo(const std::string& function, const std::string& message,
                          const VerbosityLevel& verbosityLevel, const VerbosityLevel& verbosityLevelRequired,
                          std::string color){
    if(verbosityLevel >= verbosityLevelRequired)
        std::cout << color << "[" << function << "] : " <<  message << "\x1b[0m"<< std::endl;
}

void AF_VSLAM::printError(const std::string& function, const std::string& message){
    std::cout << "\x1b[91m" << "[" << function << "] : " <<  message << "\x1b[0m"<< std::endl;
}

std::string AF_VSLAM::featureName(const FeatureType& featureType){
    const AF_VSLAM::Feature& ft = get_feature(featureType);
    return ft.getFeatureName();
}

cv::Scalar AF_VSLAM::getFeatureColor(const FeatureType& featureType, const int& format, const bool& normalize){

    Eigen::Matrix<float,3,1> color{0,0,0};
    const AF_VSLAM::Feature& ft = get_feature(featureType);
    color = ft.getColor();

    if (normalize){
        color /= 255.0f;
    }
    switch(format) {
        case 0:
            return {color(0),color(1),color(2)};
        default:
            return {color(2),color(1),color(0)};
    }
}

std::string AF_VSLAM::matType(const int& matTypeIndex){
    switch(matTypeIndex) {
        case 0:
            return "CV_8U";
        case 1:
            return "CV_8S";
        case 2:
            return "CV_16U";
        case 3:
            return "CV_16S";
        case 4:
            return "CV_32S";
        case 5:
            return "CV_32F";
        case 6:
            return "CV_64F ";
        default:
            return "UNKNOWN";
    }
}

std::vector<std::vector<float>> AF_VSLAM::loadBinFile(const std::string& filename, const int& numFloats ){

    std::vector<std::vector<float>> floats{};
    std::vector<double> floatsRow(numFloats);

    std::ifstream binFile(filename, std::ios::binary);
    while (binFile.read(reinterpret_cast<char*>(floatsRow.data()), numFloats * sizeof(double))) {
        std::vector<float> floats_;
        for (double f : floatsRow)
            floats_.push_back(float(f));
        floats.push_back(floats_);
    }
    binFile.close();
    return floats;
}

std::string AF_VSLAM::replaceAllOccurrences(std::string str, const std::string& from, const std::string& to) {
    size_t startPos = 0;
    while ((startPos = str.find(from, startPos)) != std::string::npos) {
        str.replace(startPos, from.length(), to);
        startPos += to.length(); // Handles case when 'to' is a substring of 'from'
    }
    return str;
}

int AF_VSLAM::RandomIntegerGenerator::GetRandomInteger(const int& minNumber, const int& maxNumber){
    std::uniform_int_distribution<> distrib(minNumber, maxNumber);
    return distrib(randomIntGenerator);
}

void AF_VSLAM::median_tracking_time(std::map<int, int> &timeMap, const std::string& stage, const bool& activate){
    if (!activate)
        return;
    if (timeMap.empty()) {
        return;
    }

    double median = AF_VSLAM::map_median(timeMap);
    std::cout << stage + "median / std  / max time: " << " / " << median << " ms" << std::endl;
}

double AF_VSLAM::map_median(std::map<int, int>& map_){
    // Calculate total count of all elements
    int totalCount = 0;
    for (const auto& pair : map_) {
        totalCount += pair.second;
    }

    int midLow = (totalCount - 1) / 2;  // Lower middle index
    int midHigh = totalCount / 2;        // Upper middle index (same as midLow if odd)

    int cumulative = 0;
    int lowVal = -1, highVal = -1;

    for (const auto& pair : map_) {
        cumulative += pair.second;

        if (lowVal == -1 && cumulative > midLow) {
            lowVal = pair.first;
        }
        if (highVal == -1 && cumulative > midHigh) {
            highVal = pair.first;
        }
        if (lowVal != -1 && highVal != -1) {
            break;
        }
    }

    double median = (lowVal + highVal) / 2.0;
    return median;
}