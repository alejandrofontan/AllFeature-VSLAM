#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <opencv2/core/core.hpp>

#include <System.h>
#include <FeatureFactory.h>
#include <yaml-cpp/yaml.h>
#include <FrameDrawer.h>
#include "afvslam_log.hpp"

#include "DebugKeyStepper.h"
#include "StringUtils.h"

std::string AF_VSLAM::FrameDrawer::exp_folder{};

void LoadImages(const std::string &pathToSequence, const std::string &rgb_csv,
                std::vector<std::string> &imageFilenames, std::vector<std::string> &depthFilenames,
                std::vector<std::string> &maskFilenames, std::vector<AF_VSLAM::Seconds> &timestamps,
                const std::string cam_name = "rgb_0", const std::string depth_cam_name = "depth_0",
                const std::string mask_cam_name = "mask_0");
std::string paddingZeros(const std::string& number, const size_t numberOfZeros = 5);

void removeSubstring(std::string& str, const std::string& substring) {
    size_t pos;
    while ((pos = str.find(substring)) != std::string::npos) {
        str.erase(pos, substring.length());
    }
}

bool hasPrefix(const std::string& str, const std::string& prefix) {
    return str.rfind(prefix, 0) == 0;
}


int main(int argc, char **argv)
{
    af_print_banner();

    // AF_VSLAM inputs
    std::string sequence_path;
    std::string calibration_yaml;
    std::string rgb_csv;
    std::string exp_folder;
    std::string exp_id{"0"};
    std::string settings_yaml{"vslamlab_allfeature-dev_settings.yaml"};
    bool verbose{true};

    std::string feature{"orb32"};
    std::string path_to_vocabulary_folder("allfeature_vocabulary");
    bool fixImageSize = true;

    std::cout << std::endl;

    AF_CONFIG_BEGIN("Arguments:");
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (hasPrefix(arg, "sequence_path:")) {
            removeSubstring(arg, "sequence_path:");
            sequence_path =  arg;
            AF_CONFIG_FIELD("Path to sequence: ", sequence_path);
            continue;
        }
        if (hasPrefix(arg, "calibration_yaml:")) {
            removeSubstring(arg, "calibration_yaml:");
            calibration_yaml =  arg;
            AF_CONFIG_FIELD("Path to calibration.yaml: ", calibration_yaml);
            continue;
        }
        if (hasPrefix(arg, "rgb_csv:")) {
            removeSubstring(arg, "rgb_csv:");
            rgb_csv =  arg;
            AF_CONFIG_FIELD("Path to rgb_csv: ", rgb_csv);
            continue;
        }
        if (hasPrefix(arg, "exp_folder:")) {
            removeSubstring(arg, "exp_folder:");
            exp_folder =  arg;
            AF_CONFIG_FIELD("Path to exp_folder: ", exp_folder);
            continue;
        }
        if (hasPrefix(arg, "exp_id:")) {
            removeSubstring(arg, "exp_id:");
            exp_id =  arg;
            AF_CONFIG_FIELD("Experiment id: ", exp_id);
            continue;
        }
        if (hasPrefix(arg, "settings_yaml:")) {
            removeSubstring(arg, "settings_yaml:");
            settings_yaml =  arg;
            AF_CONFIG_FIELD("Path to settings_yaml: ", settings_yaml);
            continue;
        }
        if (hasPrefix(arg, "verbose:")) {
            removeSubstring(arg, "verbose:");
            verbose = bool(std::stoi(arg));
            AF_CONFIG_FIELD("Verbose mode: ", verbose);
            continue;
        }
        if (hasPrefix(arg, "vocabulary_folder:")) {
            removeSubstring(arg, "vocabulary_folder:");
            path_to_vocabulary_folder = arg;
            AF_CONFIG_FIELD("Path to vocabulary folder: ", path_to_vocabulary_folder);
            continue;
        }
    }

    // AnyFeature-VSLAM inputs
    YAML::Node settings;
    try {
        settings = YAML::LoadFile(settings_yaml);
    } catch (const YAML::Exception& e) {
        std::cerr << "Failed to load settings_yaml: '" << settings_yaml << "': " << e.what() << std::endl;
        return 1;
    }
    const std::vector<std::string> features = settings["features"].as<std::vector<std::string>>();
    bool debug = settings["debug"].as<bool>();
    AF_CONFIG_FIELD("Debug mode: ", debug);

    AF_VSLAM::FrameDrawer::exp_folder = exp_folder;

    std::vector<FeatureType> featureTypes{};
    for(const auto& feat : features) {
        auto featureType = get_feature_type(feat);
        featureTypes.push_back(featureType);
        AF_CONFIG_FIELD("Feature added: ", feat);
    }
    AF_CONFIG_END();

    // Retrieve paths to images
    std::vector<std::string> imageFilenames{};
    std::vector<std::string> depthFilenames{};
    std::vector<std::string> maskFilenames{};
    std::vector<AF_VSLAM::Seconds> timestamps{};
    LoadImages(sequence_path, rgb_csv, imageFilenames, depthFilenames, maskFilenames, timestamps);

    size_t nImages = imageFilenames.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.

    AF_VSLAM::System SLAM(path_to_vocabulary_folder,
                                  calibration_yaml, settings_yaml,
                                  AF_VSLAM::System::RGBD,
                                  verbose,
                                  featureTypes,
                                  fixImageSize);

    // Vector for tracking time statistics
    std::vector<AF_VSLAM::Seconds> vTimesTrack;
    vTimesTrack.resize(nImages);

    std::cout << std::endl << "-------------------------------" << std::endl;
    AF_INFO("Start processing sequence ...");
    AF_INFO("Images in the sequence: " << nImages);

    // Main loop
    std::atomic<int> allowance{0};
    std::atomic<bool> quit{false};
    std::thread keyThread{};
    if (debug){
        keyThread = startKeyAllowanceThread(allowance, quit);
    }

    for (size_t ni = 0; ni < nImages; /* ni++ happens when a frame passes */) {
        // Wait until we have at least 1 "allowed" frame, or quit
        if (debug){
            while (!quit.load(std::memory_order_relaxed) && allowance.load(std::memory_order_relaxed) == 0){
                usleep(1000); // 1 ms
            }
            if (quit.load(std::memory_order_relaxed)) break;
            allowance.store(0, std::memory_order_relaxed);
        }

        // Read image from file
        AF_VSLAM::Image im(imageFilenames[ni]);
        if (im.img.empty()) {
            std::cerr << "Failed to load image: '" << imageFilenames[ni] << "'" << std::endl;
            ++ni;
            continue;
        }

        im.LoadDepth(depthFilenames[ni]);
        if (im.depthImg.empty()) {
            std::cerr << "Failed to load depth image: '" << depthFilenames[ni] << "'" << std::endl;
            ++ni;
            continue;
        }

        if (!maskFilenames.empty()) {
            im.LoadMask(maskFilenames[ni]);
        }

        AF_VSLAM::Seconds tframe = timestamps[ni];

        // Pass the image to the SLAM system
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        SLAM.Track(im,tframe);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        AF_VSLAM::Seconds ttrack = std::chrono::duration_cast<std::chrono::duration<AF_VSLAM::Seconds> >(t2 - t1).count();
        vTimesTrack[ni] = ttrack;

        // Wait to load the next frame
        AF_VSLAM::Seconds T = 0.0;
        if(ni < nImages-1)
            T = timestamps[ni+1] - tframe;
        else if(ni > 0)
            T = tframe - timestamps[ni-1];

        if(ttrack < T)
            usleep(1.0 * (T-ttrack)  * 1e6);

        // Advance to next image only after processing this one
        ++ni;
    }

    if (debug){
        quit.store(true, std::memory_order_relaxed);
        if (keyThread.joinable()) keyThread.join();
    }

    std::string resultsPath_expId = exp_folder + "/" + paddingZeros(exp_id);
    SLAM.SaveKeyFrameTrajectoryVSLAMLAB(resultsPath_expId + "_" + "KeyFrameTrajectory_beforeGBA.csv");

    // Perform Global Bundle Adjustment
    SLAM.GBA();

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    std::sort(vTimesTrack.begin(),vTimesTrack.end());
    AF_VSLAM::Seconds totaltime = 0.0;
    for(size_t ni = 0; ni < nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    std::cout << "-------" << std::endl << std::endl;
    std::cout << "median tracking time: " << vTimesTrack[nImages/2] << std::endl;
    std::cout << "mean tracking time: " << totaltime/nImages << std::endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryVSLAMLAB(resultsPath_expId + "_" + "KeyFrameTrajectory.csv");
    SLAM.SavePointCloudVSLAMLAB(resultsPath_expId + "_" + "PointCloud.ply", imageFilenames);

    return 0;
}

void LoadImages(const std::string &pathToSequence, const std::string &rgb_csv,
                std::vector<std::string> &imageFilenames, std::vector<std::string> &depthFilenames,
                std::vector<std::string> &maskFilenames, std::vector<AF_VSLAM::Seconds> &timestamps,
                const std::string cam_name, const std::string depth_cam_name,
                const std::string mask_cam_name)
{

    imageFilenames.clear();
    depthFilenames.clear();
    maskFilenames.clear();
    timestamps.clear();

    std::ifstream in(rgb_csv);
    if (!in.is_open()) {
        throw std::runtime_error("LoadImages: failed to open rgb_csv file: " + rgb_csv);
    }
    std::string line;

    // Read and map the header row to find indices
    if (!std::getline(in, line)) return;
    if (!line.empty() && line.back() == '\r') line.pop_back();

    std::vector<std::string> headers = split(line, ',');
    std::map<std::string, int> col_map;
    for (size_t i = 0; i < headers.size(); ++i) {
        col_map[headers[i]] = i;
    }

    // Required headers
    const std::string header_ts = "ts_" + cam_name + " (ns)";
    const std::string header_rgb0 = "path_" + cam_name;
    const std::string header_depth = "path_" + depth_cam_name;
    const std::string header_mask = "path_" + mask_cam_name;

    // Safely get indices
    auto get_index = [&](const std::string& key) -> int {
        auto it = col_map.find(key);
        if (it == col_map.end()) {
            throw std::runtime_error("LoadImages: required column '" + key + "' not found in " + rgb_csv);
        }
        return it->second;
    };
    auto get_index_optional = [&](const std::string& key) -> int {
        auto it = col_map.find(key);
        return (it == col_map.end()) ? -1 : it->second;
    };

    int ts_idx = get_index(header_ts);
    int rgb0_idx = get_index(header_rgb0);
    int depth_idx = get_index(header_depth);
    int mask_idx = get_index_optional(header_mask);

    // Read and process data lines using fixed indices
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (!line.empty() && line.back() == '\r') line.pop_back();

        std::vector<std::string> tokens = split(line, ',');
        int max_idx = std::max({ts_idx, rgb0_idx, depth_idx});
        if (mask_idx >= 0) max_idx = std::max(max_idx, mask_idx);
        if (tokens.size() <= static_cast<size_t>(max_idx)) {
            throw std::runtime_error("LoadImages: malformed row (too few columns) in " + rgb_csv + ": " + line);
        }

        // Assign variables using indices, regardless of column order
        std::string t_str = tokens[ts_idx];
        std::string rel_rgb0_path = tokens[rgb0_idx];
        std::string rel_depth_path = tokens[depth_idx];

        AF_VSLAM::Seconds t = static_cast<double>(std::stoll(t_str)) * 1e-9;

        timestamps.push_back(t);
        imageFilenames.push_back(pathToSequence + "/" + rel_rgb0_path);
        depthFilenames.push_back(pathToSequence + "/" + rel_depth_path);
        if (mask_idx >= 0) {
            maskFilenames.push_back(pathToSequence + "/" + tokens[mask_idx]);
        }
    }
}

std::string paddingZeros(const std::string& number, const size_t numberOfZeros){
    if (number.size() >= numberOfZeros)
        return number;
    return std::string(numberOfZeros - number.size(), '0') + number;
}