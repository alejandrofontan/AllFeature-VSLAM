#include <opencv2/opencv.hpp>
#include <iostream>
#include<FeatureFactory.h>
#include <yaml-cpp/yaml.h>
#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<opencv2/core/core.hpp>
#include "sys/sysinfo.h"

#include<System.h>
#include<FrameDrawer.h>

using namespace std;
namespace ANYFEATURE_VSLAM{
    using Seconds = double;
}

static bool isMostlyFlatOrClipped(const cv::Mat& bgrOrGray) {
    cv::Mat gray;
    if (bgrOrGray.channels() == 3) cv::cvtColor(bgrOrGray, gray, cv::COLOR_BGR2GRAY);
    else gray = bgrOrGray;

    // Downsample for speed
    cv::Mat small;
    cv::resize(gray, small, cv::Size(), 0.25, 0.25, cv::INTER_AREA);

    // Count near-black and near-white pixels
    const int N = small.rows * small.cols;
    int nearBlack = 0, nearWhite = 0;
    for (int r = 0; r < small.rows; ++r) {
        const uchar* p = small.ptr<uchar>(r);
        for (int c = 0; c < small.cols; ++c) {
            uchar v = p[c];
            nearBlack += (v <= 2);
            nearWhite += (v >= 253);
        }
    }

    double fracBlack = (double)nearBlack / N;
    double fracWhite = (double)nearWhite / N;

    // Tune thresholds to your stream
    return (fracBlack > 0.98) || (fracWhite > 0.98);
}

static bool isTooBlurryOrTextureless(const cv::Mat& bgrOrGray) {
    cv::Mat gray;
    if (bgrOrGray.channels() == 3) cv::cvtColor(bgrOrGray, gray, cv::COLOR_BGR2GRAY);
    else gray = bgrOrGray;

    cv::Mat small;
    cv::resize(gray, small, cv::Size(), 0.5, 0.5, cv::INTER_AREA);

    cv::Mat lap;
    cv::Laplacian(small, lap, CV_64F);

    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);
    double var = stddev[0] * stddev[0];

    // Tune: higher threshold = stricter. Start around 30–100 depending on resolution.
    return var < 50.0;
}

static bool isFrozenFrame(const cv::Mat& current, const cv::Mat& prevGood) {
    if (prevGood.empty()) return false;

    cv::Mat g1, g2;
    if (current.channels() == 3) cv::cvtColor(current, g1, cv::COLOR_BGR2GRAY);
    else g1 = current;
    if (prevGood.channels() == 3) cv::cvtColor(prevGood, g2, cv::COLOR_BGR2GRAY);
    else g2 = prevGood;

    cv::Mat s1, s2;
    cv::resize(g1, s1, cv::Size(160, 120));
    cv::resize(g2, s2, cv::Size(160, 120));

    cv::Mat diff;
    cv::absdiff(s1, s2, diff);

    double meanDiff = cv::mean(diff)[0];
    // Tune: if your scene is static, make this lower. Start ~0.5–2.0
    return meanDiff < 1.0;
}

int main() {

    // ALLFEATURE_VSLAM inputs
    std::string calibration_yaml{"/home/alejandro/Desktop/calibration.yaml"};
    std::string settings_yaml{"/home/alejandro/VSLAM-LAB/Baselines/AllFeature-VSLAM-DEV/vslamlab_allfeature-dev_settings.yaml"};
    bool verbose{true};

    std::string path_to_vocabulary_folder("/home/alejandro/VSLAM-LAB/Baselines/AllFeature-VSLAM-DEV/allfeature_vocabulary");
    bool fixImageSize = false;
    YAML::Node settings = YAML::LoadFile(settings_yaml);
    const vector<std::string> features = settings["features"].as<vector<std::string>>();
    bool debug = (bool)settings["debug"].as<bool>();
    std::cout << "[vslamlab_anyfeature_mono.cpp] Debug mode = " << debug << std::endl;

    vector<FeatureType> featureTypes{};
    for(const auto& feat : features) {
        auto featureType = get_feature_type(feat);
        featureTypes.push_back(featureType);
        std::cout << "[vslamlab_anyfeature_mono.cpp] Loaded feature from settings YAML: " << feat << std::endl;
    }

    std::map<FeatureType, std::string> feature_settings_yaml_file;
    feature_settings_yaml_file[FEAT_ORB] = "settings/orb32_settings.yaml";
    feature_settings_yaml_file[FEAT_AKAZE61] = "settings/akaze61_settings.yaml";
    feature_settings_yaml_file[FEAT_BRISK] = "settings/brisk48_settings.yaml";
    feature_settings_yaml_file[FEAT_SURF64] = "settings/surf64_settings.yaml";
    feature_settings_yaml_file[FEAT_KAZE64] = "settings/kaze64_settings.yaml";
    feature_settings_yaml_file[FEAT_SIFT128] = "settings/sift128_settings.yaml";
    feature_settings_yaml_file[FEAT_ALIKED128] = "settings/aliked128_settings.yaml";
    feature_settings_yaml_file[FEAT_SUPERPOINT256] = "settings/superpoint256_settings.yaml";

    // Setting AllFeature-VSLAM
    ANYFEATURE_VSLAM::System SLAM(path_to_vocabulary_folder,
                                  calibration_yaml, settings_yaml, feature_settings_yaml_file,
                                  ANYFEATURE_VSLAM::System::MONOCULAR,
                                  verbose,
                                  featureTypes,
                                  fixImageSize);

    // If OpenCV was built with FFmpeg, this usually works:
    //std::string url = "udp://127.0.0.1:5000";
    const std::string url = "rtmp://10.68.61.71:1935/live/stream";

    // Some builds prefer explicitly selecting FFmpeg backend:
    cv::VideoCapture cap(url, cv::CAP_FFMPEG);

    if (!cap.isOpened()) {
        std::cerr << "Failed to open stream: " << url << std::endl;
        return 1;
    }

    //cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    // cv::Mat frame;//, lastGood;
    // int idx{0};
    // while (true) {

    //     // Block until at least one frame is available
    //     if (!cap.grab()) {
    //         cv::waitKey(30);
    //         continue;
    //     }

    //     // Drain any extra queued frames; keep the newest
    //     int drained = 0;
    //     while (cap.grab()) {               // grabs next if already available
    //         drained++;
    //         // stop draining after some limit to avoid infinite loop if stream is extremely fast
    //         if (drained > 5) break;
    //     }

    //     cap.retrieve(frame);
    //     if (frame.empty()) continue;

    //     // if(isMostlyFlatOrClipped(frame)) {
    //     //     std::cout << "Skipping frame due to mostly flat or clipped image." << std::endl;
    //     //     continue;
    //     // }
    //     // if (isTooBlurryOrTextureless(frame)) {
    //     //     std::cout << "Skipping frame due to blur or lack of texture." << std::endl;
    //     //     continue;
    //     // }

    //     // if (isFrozenFrame(frame, lastGood)) {
    //     //     std::cout << "Skipping frame due to frozen frame detected." << std::endl;
    //     //     continue;
    //     // }

    //     //lastGood = frame.clone();

    //     cv::imshow("UDP Stream", frame);

    //     // ANYFEATURE_VSLAM::Image im(frame);
    //     // ANYFEATURE_VSLAM::Seconds tframe = idx * (1.0/30.0); // or use a real timestamp
    //     // SLAM.TrackMonocular(im, tframe);
    //     idx++;

    //     int key = cv::waitKey(30);
    //     if (key == 'q' || key == 27) break;
    // }

    cv::Mat frame;
    int idx{0};
    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            // stream might not have started yet; avoid busy spin
            cv::waitKey(30);
            continue;
        }

        ANYFEATURE_VSLAM::Image im(frame);
        ANYFEATURE_VSLAM::Seconds tframe = idx * (1.f/30.f);

        SLAM.TrackMonocular(im,tframe);
        idx++;

        // cv::imshow("UDP Stream", frame);
        int key = cv::waitKey(30);
        if (key == 'q' || key == 27) { // q or ESC
            break;
        }
    }

    cap.release();
    return 0;
}
