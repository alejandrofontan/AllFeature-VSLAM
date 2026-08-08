#include <opencv2/opencv.hpp>
#include <iostream>
#include<FeatureFactory.h>
#include <yaml-cpp/yaml.h>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<opencv2/core/core.hpp>

#include<System.h>
#include<FrameDrawer.h>

static bool hasPrefix(const std::string& str, const std::string& prefix) {
    return str.rfind(prefix, 0) == 0;
}

static constexpr float kAssumedFps = 30.f;
static constexpr int kWaitKeyMs = 30;

int main(int argc, char** argv) {

    // ALLFEATURE_VSLAM inputs (defaults are for local testing; override via CLI: key:value)
    std::string calibration_yaml{"/home/alejandro/Desktop/calibration.yaml"};
    std::string settings_yaml{"/home/alejandro/VSLAM-LAB/Baselines/AllFeature-VSLAM-DEV/vslamlab_allfeature-dev_settings.yaml"};
    std::string path_to_vocabulary_folder("/home/alejandro/VSLAM-LAB/Baselines/AllFeature-VSLAM-DEV/allfeature_vocabulary");
    std::string stream_url{"rtmp://10.68.61.71:1935/live/stream"};
    bool verbose{true};
    bool fixImageSize = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (hasPrefix(arg, "calibration_yaml:")) { calibration_yaml = arg.substr(std::string("calibration_yaml:").size()); continue; }
        if (hasPrefix(arg, "settings_yaml:")) { settings_yaml = arg.substr(std::string("settings_yaml:").size()); continue; }
        if (hasPrefix(arg, "vocabulary_folder:")) { path_to_vocabulary_folder = arg.substr(std::string("vocabulary_folder:").size()); continue; }
        if (hasPrefix(arg, "stream_url:")) { stream_url = arg.substr(std::string("stream_url:").size()); continue; }
    }

    YAML::Node settings;
    try {
        settings = YAML::LoadFile(settings_yaml);
    } catch (const YAML::Exception& e) {
        std::cerr << "Failed to load settings_yaml '" << settings_yaml << "': " << e.what() << std::endl;
        return 1;
    }
    const std::vector<std::string> features = settings["features"].as<std::vector<std::string>>();
    bool debug = (bool)settings["debug"].as<bool>();
    std::cout << "[vslamlab_allfeature_mono_stream.cpp] Debug mode = " << debug << std::endl;

    std::vector<FeatureType> featureTypes{};
    for(const auto& feat : features) {
        auto featureType = get_feature_type(feat);
        featureTypes.push_back(featureType);
        std::cout << "[vslamlab_allfeature_mono_stream.cpp] Loaded feature from settings YAML: " << feat << std::endl;
    }

    // Setting AllFeature-VSLAM
    AF_VSLAM::System SLAM(path_to_vocabulary_folder,
                                  calibration_yaml, settings_yaml,
                                  AF_VSLAM::System::MONOCULAR,
                                  verbose,
                                  featureTypes,
                                  fixImageSize);

    // Some builds prefer explicitly selecting FFmpeg backend:
    const std::string& url = stream_url;
    cv::VideoCapture cap(url, cv::CAP_FFMPEG);

    if (!cap.isOpened()) {
        std::cerr << "Failed to open stream: " << url << std::endl;
        return 1;
    }

    cv::Mat frame;
    int idx{0};
    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            // stream might not have started yet; avoid busy spin
            cv::waitKey(kWaitKeyMs);
            continue;
        }

        AF_VSLAM::Image im(frame);
        AF_VSLAM::Seconds tframe = idx * (1.f/kAssumedFps);

        SLAM.TrackMonocular(im,tframe);
        idx++;

        // cv::imshow("UDP Stream", frame);
        int key = cv::waitKey(kWaitKeyMs);
        if (key == 'q' || key == 27) { // q or ESC
            break;
        }
    }

    cap.release();
    return 0;
}
