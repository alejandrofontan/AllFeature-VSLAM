//
// Created by fontan on 20/02/24.
//

#include <fstream>
#include "Utils.h"
#include <utility>
#include "Feature_superpoint256.h"

std::random_device rd;
//std::mt19937 ANYFEATURE_VSLAM::RandomIntegerGenerator::randomIntGenerator{std::mt19937(rd())};
std::mt19937 ANYFEATURE_VSLAM::RandomIntegerGenerator::randomIntGenerator{123456u};

void ANYFEATURE_VSLAM::printInfo(const std::string& function, const std::string& message,
                          const VerbosityLevel& verbosityLevel, const VerbosityLevel& verbosityLevelRequired,
                          std::string color){
    if(verbosityLevel >= verbosityLevelRequired)
        std::cout << color << "[" << function << "] : " <<  message << "\x1b[0m"<< std::endl;
}

void ANYFEATURE_VSLAM::printError(const std::string& function, const std::string& message){
    std::cout << "\x1b[91m" << "[" << function << "] : " <<  message << "\x1b[0m"<< std::endl;
}

std::string ANYFEATURE_VSLAM::featureName(const FeatureType& featureType){
    std::unique_ptr<ANYFEATURE_VSLAM::Feature> ft = get_feature(featureType);
    return ft->getFeatureName();
}

cv::Scalar ANYFEATURE_VSLAM::getFeatureColor(const FeatureType& featureType, const int& format, const bool& normalize){

    Eigen::Matrix<float,3,1> color{0,0,0};
    std::unique_ptr<ANYFEATURE_VSLAM::Feature> ft = get_feature(featureType);
    color = ft->getColor();

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

std::string ANYFEATURE_VSLAM::matType(const int& matTypeIndex){
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
    }
}

std::vector<std::vector<float>> ANYFEATURE_VSLAM::loadBinFile(const std::string& filename, const int& numFloats ){

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

std::string ANYFEATURE_VSLAM::replaceAllOccurrences(std::string str, const std::string& from, const std::string& to) {
    size_t startPos = 0;
    while ((startPos = str.find(from, startPos)) != std::string::npos) {
        str.replace(startPos, from.length(), to);
        startPos += to.length(); // Handles case when 'to' is a substring of 'from'
    }
    return str;
}

int ANYFEATURE_VSLAM::RandomIntegerGenerator::GetRandomInteger(const int& minNumber, const int& maxNumber){
    std::uniform_int_distribution<> distrib(minNumber, maxNumber);
    return distrib(randomIntGenerator);
}

void ANYFEATURE_VSLAM::medianTrackingTime(std::vector<double> &timeVector, const std::string& stage, const bool& activate){
    if(!activate)
        return;
    if(timeVector.empty()){
        return;
    }
    std::vector<double> tmp = timeVector;
    std::sort(tmp.begin(), tmp.end());
    double median;
    size_t n = tmp.size();
    if(n % 2 == 1) median = tmp[n/2];
    else median = 0.5*(tmp[n/2 - 1] + tmp[n/2]);
    const double sum = std::accumulate(timeVector.begin(), timeVector.end(), 0.0);
    double stddev = 0.0;
    if (n >= 2) {
        double sq_sum = 0.0;
        for (double x : timeVector) {
            const double d = x - median;
            sq_sum += d * d;
        }
        stddev = std::sqrt(sq_sum / static_cast<double>(n - 1));
    }
    std::cout << stage + "median / std  / max time: " << " / " << 1000*median << " / " << 1000*stddev << " / " << 1000*tmp.back() << " ms" << std::endl;
}

double ANYFEATURE_VSLAM::vector_median(std::vector<double>& vector_){
    std::vector<double> tmp = vector_;
    std::sort(tmp.begin(), tmp.end());
    size_t n = tmp.size();
    if(n == 0)
        return 0.0;
    if(n % 2 == 1) return tmp[n/2];
    else return 0.5*(tmp[n/2 - 1] + tmp[n/2]);
}