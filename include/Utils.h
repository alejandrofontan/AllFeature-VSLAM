//
// Created by fontan on 20/02/24.
//

#ifndef AF_VSLAM_UTILS_H
#define AF_VSLAM_UTILS_H

#include "FeatureFactory.h"

#include<opencv2/core/core.hpp>

#include <vector>
#include <random>

namespace AF_VSLAM{
    void printInfo(const std::string& function, const std::string& message,
                   const VerbosityLevel& verbosityLevel, const VerbosityLevel& verbosityLevelRequired,
                   std::string color = "\x1b[0m");
    void printError(const std::string& function, const std::string& message);

    std::string featureName(const FeatureType& featureType);

    std::string matType(const int& matTypeIndex);
    cv::Scalar getFeatureColor(const FeatureType& featureType, const int& format, const bool& normalize = false);
    std::vector<std::vector<float>> loadBinFile(const std::string& filename, const int& numFloats);
    std::string replaceAllOccurrences(std::string str, const std::string& from, const std::string& to);

    void medianTrackingTime(std::vector<double> &timeVector, const std::string& stage, const bool& activate);
    double vector_median(std::vector<double>& vector_);

    class RandomIntegerGenerator
    {
    private:

        static std::mt19937 randomIntGenerator;

    public:
        static int GetRandomInteger(const int& minNumber, const int& maxNumber);
    };
}



#endif //AF_VSLAM_UTILS_H
