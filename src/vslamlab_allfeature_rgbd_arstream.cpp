/**
 * vslamlab_allfeature_rgbd_arstream.cpp - online RGB-D AllFeature-VSLAM on an ARStream relay
 *
 * Author: Alejandro Fontan
 * Assisted by: Claude (Fable 5.1)
 * Version: 1.0
 * Created: 2026-10-15
 *
 * Frames arrive over ZeroMQ from `arstream publish` (github.com/VSLAM-LAB/arstream, conda package libarstream)
 * with ARKit's LiDAR depth: 16-bit millimetres at its native 256x192, upsampled with nearest neighbour to the image
 * and masked below min_confidence (ARKit's 0 low / 1 medium / 2 high), exactly the depth_0/ of VSLAM-LAB's
 * iphone-lidar dataset; the calibration's depth_factor turns it into the metric CV_32F map Image::depthImg
 * carries. A stream without depth stops with a message. Stops at the relay's end-of-stream message or on Ctrl-C,
 * then saves the same outputs as vslamlab_allfeature_rgbd. VSLAM-LAB runs it with
 * `pixi run stream-iphone --run allfeature-dev --mode rgbd`.
 *
 *   vslamlab_allfeature_rgbd_arstream stream_url:tcp://localhost:5556 calibration_yaml:<file> exp_folder:<dir>
 *                                     [exp_id:<n>] [settings_yaml:<file>] [verbose:<0|1>] [min_confidence:<0|1|2>]
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <execinfo.h>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <arstream/arstream.hpp>
#include <opencv2/core/core.hpp>
#include <yaml-cpp/yaml.h>

#include <System.h>
#include <FeatureFactory.h>
#include <FrameDrawer.h>
#include "afvslam_log.hpp"
#include "SegmentationSettings.h"

// An uncaught exception on any thread ends the process; name it and print the throwing thread's stack first
// (symbolised by the dynamic linker: `atos -o <binary> <addr>` resolves what it leaves as offsets)
static void on_terminate() {
    if (auto current = std::current_exception()) {
        try { std::rethrow_exception(current); }
        catch (const std::exception& e) { std::cerr << "terminate: uncaught " << e.what() << std::endl; }
        catch (...) { std::cerr << "terminate: uncaught non-standard exception" << std::endl; }
    }
    void* frames[64];
    const int depth = backtrace(frames, 64);
    backtrace_symbols_fd(frames, depth, 2);
    std::abort();
}

static arstream::Subscriber* g_subscriber = nullptr;
static AF_VSLAM::System* g_system = nullptr;
// Stop the subscriber and lift a Viewer pause, so a paused run still reaches the end of its loop and saves
static void on_signal(int) {
    if (g_subscriber) g_subscriber->stop();
    if (g_system) g_system->set_paused(false);
}

static bool arg_has_prefix(const std::string& str, const std::string& prefix) {
    return str.rfind(prefix, 0) == 0;
}

static std::string padding_zeros(const std::string& number, const size_t numberOfZeros = 5) {
    std::string zeros{};
    for (size_t iZero{}; iZero < numberOfZeros - number.size(); ++iZero)
        zeros += "0";
    return zeros + number;
}

#ifdef ARSTREAM_HAS_STATUS
// Reports the SLAM state to the relay, which shows it as the phone app's banner (Viewer.StreamStatus). Its own
// thread sends 10 times a second, so a pause (the tracking loop blocked in System::Track) still reports PAUSED, and
// silence (the app turns the banner grey) means this process is gone or stuck.
class StatusReporter {
public:
    explicit StatusReporter(AF_VSLAM::System& slam) : slam_(slam), thread_([this] { run(); }) {}
    ~StatusReporter() {
        stop_ = true;
        thread_.join();
    }

private:
    void run() {
        arstream::StatusPublisher publisher;   // never blocks: dropped while the relay is not there
        while (!stop_) {
            publisher.send(state_name(), slam_.GetFramesProcessedCount());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    std::string state_name() {
        if (slam_.is_paused())
            return "PAUSED";
        switch (static_cast<TrackingState>(slam_.GetTrackingState())) {
            case TrackingState::SYSTEM_NOT_READY: return "NOT READY";
            case TrackingState::NO_IMAGES_YET:    return "WAITING";
            case TrackingState::NOT_INITIALIZED:  return "INITIALIZING";
            case TrackingState::OK:               return "OK";
            case TrackingState::LOST:             return "LOST";
        }
        return "?";
    }

    AF_VSLAM::System& slam_;
    std::atomic<bool> stop_{false};
    std::thread thread_;   // last member: started once the others exist
};
#endif

int main(int argc, char** argv) {
    std::set_terminate(on_terminate);
    af_print_banner();

    // AF_VSLAM inputs
    std::string stream_url{arstream::kDefaultUrl};
    std::string calibration_yaml;
    std::string exp_folder;
    std::string exp_id{"0"};
    std::string settings_yaml{"vslamlab_allfeature-dev_settings.yaml"};
    bool verbose{true};
    bool fixImageSize = true;
    int min_confidence{1};

    if (argc > 1 && std::string(argv[1]) == "--help") {
        std::cout << "Usage: vslamlab_allfeature_rgbd_arstream stream_url:<tcp://host:port> calibration_yaml:<file> exp_folder:<dir>"
                  << " [exp_id:<n>] [settings_yaml:<file>] [verbose:<0|1>] [min_confidence:<0|1|2>]" << std::endl;
        return 0;
    }

    std::cout << std::endl;
    AF_CONFIG_BEGIN("Arguments:");
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg_has_prefix(arg, "stream_url:")) { stream_url = arg.substr(11); AF_CONFIG_FIELD("Stream url: ", stream_url); continue; }
        if (arg_has_prefix(arg, "calibration_yaml:")) { calibration_yaml = arg.substr(17); AF_CONFIG_FIELD("Path to calibration.yaml: ", calibration_yaml); continue; }
        if (arg_has_prefix(arg, "exp_folder:")) { exp_folder = arg.substr(11); AF_CONFIG_FIELD("Path to exp_folder: ", exp_folder); continue; }
        if (arg_has_prefix(arg, "exp_id:")) { exp_id = arg.substr(7); AF_CONFIG_FIELD("Experiment id: ", exp_id); continue; }
        if (arg_has_prefix(arg, "settings_yaml:")) { settings_yaml = arg.substr(14); AF_CONFIG_FIELD("Path to settings_yaml: ", settings_yaml); continue; }
        if (arg_has_prefix(arg, "verbose:")) { verbose = bool(std::stoi(arg.substr(8))); AF_CONFIG_FIELD("Verbose mode: ", verbose); continue; }
        if (arg_has_prefix(arg, "min_confidence:")) { min_confidence = std::stoi(arg.substr(15)); AF_CONFIG_FIELD("Min depth confidence: ", min_confidence); continue; }
    }
    if (calibration_yaml.empty() || exp_folder.empty()) {
        std::cerr << "calibration_yaml: and exp_folder: are required (--help)" << std::endl;
        return 1;
    }

    // AllFeature-VSLAM inputs
    YAML::Node settings;
    try {
        settings = YAML::LoadFile(settings_yaml);
    } catch (const YAML::Exception& e) {
        std::cerr << "Failed to load settings_yaml: '" << settings_yaml << "': " << e.what() << std::endl;
        return 1;
    }
    const std::vector<std::string> features = settings["features"].as<std::vector<std::string>>();
    const bool debug = settings["debug"].as<bool>();
    AF_CONFIG_FIELD("Debug mode: ", debug);
    // Viewer.StartPaused: track one black frame (loads every model session, starts no map), then pause until the
    // Viewer's Pause is unticked, so the phone can wait on the table while Core ML warms up. Needs the viewer.
    const bool start_paused = verbose && settings["Viewer.StartPaused"] && settings["Viewer.StartPaused"].as<int>() != 0;
    AF_CONFIG_FIELD("Start paused: ", start_paused);
    // Viewer.StreamStatus: report the SLAM state to the relay, which shows it as the phone app's banner
    // (libarstream >= 0.3.0 and an app that shows it; off by default)
    const bool stream_status = settings["Viewer.StreamStatus"] && settings["Viewer.StreamStatus"].as<int>() != 0;
#ifdef ARSTREAM_HAS_STATUS
    AF_CONFIG_FIELD("Stream status: ", stream_status);
#else
    if (stream_status)
        AF_WARN("Viewer.StreamStatus needs libarstream >= 0.3.0 (status messages): ignored");
#endif

    // Depth scale factor: metric_depth = raw_depth / depth_factor, from the calibration.yaml camera entry matching
    // settings_yaml's cam_mono (the relay writes 1000: millimetres). Defaults to 1 if the calibration declares none.
    float depth_factor = 1.0f;
    try {
        YAML::Node calibration = YAML::LoadFile(calibration_yaml);
        const std::string cam_name = settings["cam_mono"].as<std::string>();
        for (const auto& cam : calibration["cameras"]) {
            if (cam["cam_name"].as<std::string>() == cam_name) {
                if (cam["depth_factor"])
                    depth_factor = cam["depth_factor"].as<float>();
                break;
            }
        }
    } catch (const YAML::Exception& e) {
        std::cerr << "Failed to load calibration_yaml: '" << calibration_yaml << "': " << e.what() << std::endl;
        return 1;
    }
    AF_CONFIG_FIELD("Depth factor: ", depth_factor);

    AF_VSLAM::FrameDrawer::exp_folder = exp_folder;

    std::vector<FeatureType> featureTypes{};
    for (const auto& feat : features) {
        featureTypes.push_back(get_feature_type(feat));  // an unknown name terminates the process (FeatureFactory.h)
        AF_CONFIG_FIELD("Feature added: ", feat);
    }
    segmentation::SegmentationSettings segmentationSettings{};
    try {
        segmentationSettings.load(settings);
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    AF_CONFIG_END();

    // Online segmentation: build/load the engine before the SLAM threads exist (policy and announcements in
    // SegmentationSettings.h).
    std::unique_ptr<segmentation::Segmenter> segmenter{};
    try {
        segmenter = segmentationSettings.makeSegmenter();
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    // Subscribe before the SLAM system starts so no frame published meanwhile is missed (conflate keeps the newest)
    arstream::Subscriber subscriber(stream_url);
    g_subscriber = &subscriber;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    AF_VSLAM::System SLAM(calibration_yaml, settings_yaml, AF_VSLAM::System::RGBD, verbose, featureTypes, fixImageSize);

    g_system = &SLAM;

    // Stream length is unknown; 0 keeps the frame counter unset while still reporting whether masks are in use.
    SLAM.SetSequenceInfo(0, segmenter != nullptr);
#ifdef ARSTREAM_HAS_STATUS
    // Declared after SLAM, so it stops before SLAM is destroyed
    std::unique_ptr<StatusReporter> statusReporter;
    if (stream_status)
        statusReporter = std::make_unique<StatusReporter>(SLAM);
#endif

    bool noDepth = false;
    // The whole run (tracking loop, GBA, Shutdown, saves) as one unit: on macOS with the viewer on, System::Run
    // executes it in a worker thread while the Pangolin viewer owns the main thread; elsewhere it runs inline.
    auto processStream = [&]() {
        std::vector<AF_VSLAM::Seconds> vTimesTrack;

        std::cout << std::endl << "-------------------------------" << std::endl;
        AF_INFO("Subscribed to " << stream_url << ", waiting for RGB-D frames (Ctrl-C or the relay's end of stream stops) ...");
        std::cout.flush();

        arstream::Frame frame;
        size_t received = 0;
        while (subscriber.next(frame)) {
            if (!frame.hasDepth()) {
                AF_WARN("The stream carries no depth (not a LiDAR phone, or Depth off in the app) - use the mono entry point");
                noDepth = true;
                break;
            }
            ++received;
            if (received == 1 && start_paused) {
                // Warm-up: a black frame (no depth) of the stream's size runs every extractor (the fixed-shape exports
                // load on the first frame) and has no keypoints, so no map starts; the real frame is dropped
                const cv::Mat black = cv::Mat::zeros(frame.image.size(), frame.image.type());
                AF_VSLAM::Image warm(black);
                warm.depthImg = cv::Mat::zeros(frame.image.size(), CV_32F);
                if (segmenter)
                    warm.mask = segmenter->inferMask(warm.img);
                SLAM.Track(warm, frame.timestamp);
                SLAM.set_paused(true);
                AF_INFO("Warm-up frame done: paused until the Viewer's Pause is unticked (Viewer.StartPaused)");
                std::cout.flush();
                continue;
            }
            AF_VSLAM::Image im(frame.image);
            frame.depthResized(min_confidence).convertTo(im.depthImg, CV_32F, 1.0 / depth_factor);  // metres, 0 = none
            if (segmenter)
                im.mask = segmenter->inferMask(im.img);

            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            SLAM.Track(im, frame.timestamp);
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
            vTimesTrack.push_back(std::chrono::duration_cast<std::chrono::duration<AF_VSLAM::Seconds>>(t2 - t1).count());
        }
        AF_INFO((subscriber.ended() ? "End of stream" : "Stopped") << " after " << received << " frames");

        std::string resultsPath_expId = exp_folder + "/" + padding_zeros(exp_id);
        SLAM.SaveKeyFrameTrajectoryVSLAMLAB(resultsPath_expId + "_" + "KeyFrameTrajectory_beforeGBA.csv");

        // Perform Global Bundle Adjustment
        SLAM.GBA();

        // Stop all threads
        SLAM.Shutdown();

        // Tracking time statistics
        if (!vTimesTrack.empty()) {
            std::sort(vTimesTrack.begin(), vTimesTrack.end());
            AF_VSLAM::Seconds totaltime = 0.0;
            for (const auto t : vTimesTrack)
                totaltime += t;
            std::cout << "-------" << std::endl << std::endl;
            std::cout << "median tracking time: " << vTimesTrack[vTimesTrack.size() / 2] << std::endl;
            std::cout << "mean tracking time: " << totaltime / vTimesTrack.size() << std::endl;
        }

        // Save camera trajectory
        SLAM.SaveKeyFrameTrajectoryVSLAMLAB(resultsPath_expId + "_" + "KeyFrameTrajectory.csv");
        SLAM.SavePointCloudVSLAMLAB(resultsPath_expId + "_" + "PointCloud.ply");

        // placecell kernel / history / profile / plots (PlaceCell.Dump; no-op otherwise)
        SLAM.SavePlaceCellDiagnostics(resultsPath_expId + "_placecell");
    };

    SLAM.Run(processStream);

    return noDepth ? 2 : 0;
}
