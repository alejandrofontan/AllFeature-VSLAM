#include "System.h"
#include "Converter.h"
#include "FeatureMatcher.h"
#include "Optimizer.h"
#include "PlaceRecognitionMegaLoc.h"

#include <placecell/viz.h>

#include <algorithm>
#include <fstream>

#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>
#include <yaml-cpp/yaml.h>
#include "afvslam_log.hpp"

namespace AF_VSLAM
{

System::System(const string &strCalibrationFile, const string &strSettingsFile,
               const eSensor sensor,
               const bool activateVisualization,
               const vector<FeatureType>& featureTypes,
               const bool& fixImageSize):
               featureTypes(featureTypes), mSensor(sensor), viewer(static_cast<shared_ptr<Viewer>>(nullptr)), mbReset(false)
{
    // Output welcome message

    if(mSensor==MONOCULAR)
        AF_INFO("Monocular sensor selected");
    else if(mSensor==STEREO)
        AF_INFO("Stereo sensor selected");
    else if(mSensor==RGBD)
        AF_INFO("RGB-D sensor selected");

    //Check settings file
    cv::FileStorage fsCalibration(strCalibrationFile.c_str(), cv::FileStorage::READ);
    if(!fsCalibration.isOpened())
    {
       cerr << "Failed to open calibration file at: " << strCalibrationFile << endl;
       exit(-1);
    }

    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "Failed to open settings file at: " << strSettingsFile << endl;
        exit(-1);
    }

    Optimizer::LoadParameters(fsSettings);
    Tracking::LoadParameters(fsSettings);
    LocalMapping::LoadParameters(fsSettings);
    placecell_settings = PlaceCellSettings::Load(fsSettings);

    ////////////////////////////////////////////////////////////////////////////////
    // Visual place recognition (VPR) backend selection (PlaceRecognition.h). Settings
    // keys (all optional):
    //   vpr:               "megaloc" (default) | "none"
    //   feature_vpr:       local feature that geometrically verifies retrieved candidates
    //                      (reloc PnP, loop Sim3); must be listed in features:;
    //                      defaults to the first feature.
    //   megaloc_onnx:      MegaLoc model (default PlaceRecognitionMegaLoc::kDefaultOnnx)
    //   megaloc_precision: "fp16" (default) | "fp32"
    //   PlaceRecognition.MegaLocMinSimilarity, PlaceRecognition.MaxCandidates:
    //                      see PlaceRecognitionMegaLocParameters
    //   PlaceCell.*:       placecell's print manager / profiler / recorder / visualizer
    //                      (PlaceCellSettings.h)
    // A request that cannot be satisfied (missing model, unknown backend) is a hard error.
    std::string vpr_method{"megaloc"};
    std::string feature_vpr_name{};
    std::string megaloc_onnx{PlaceRecognitionMegaLoc::kDefaultOnnx};
    std::string megaloc_precision{"fp16"};
    PlaceRecognitionMegaLocParameters megaloc_params{};
    try {
        const YAML::Node settingsNode = YAML::LoadFile(strSettingsFile);
        if(settingsNode["vpr"]) vpr_method = settingsNode["vpr"].as<std::string>();
        if(settingsNode["feature_vpr"]) feature_vpr_name = settingsNode["feature_vpr"].as<std::string>();
        if(settingsNode["megaloc_onnx"]) megaloc_onnx = settingsNode["megaloc_onnx"].as<std::string>();
        if(settingsNode["megaloc_precision"]) megaloc_precision = settingsNode["megaloc_precision"].as<std::string>();
        if(settingsNode["PlaceRecognition.MegaLocMinSimilarity"])
            megaloc_params.min_similarity = settingsNode["PlaceRecognition.MegaLocMinSimilarity"].as<float>();
        if(settingsNode["PlaceRecognition.MaxCandidates"])
            megaloc_params.max_candidates = settingsNode["PlaceRecognition.MaxCandidates"].as<int>();
    } catch (const std::exception& e) {
        AF_ERROR("[System] Failed to parse settings file '" + strSettingsFile + "': " + std::string(e.what()));
        exit(-1);
    }

    std::string feature_list{};
    for (const auto& ft : featureTypes)
        feature_list += (feature_list.empty() ? "" : ", ") + feature_name(ft);

    // Explicit feature_vpr must name a configured feature — hard error otherwise
    FeatureType vpr_feature{featureTypes[0]};
    if(!feature_vpr_name.empty()){
        bool found{false};
        for (const auto& ft : featureTypes)
            if(feature_name(ft) == feature_vpr_name){ vpr_feature = ft; found = true; break; }
        if(!found){
            AF_ERROR("[System] feature_vpr '" + feature_vpr_name + "' is not in features: [" + feature_list + "]");
            exit(-1);
        }
    }

    if(vpr_method == "none"){
        AF_INFO("[System] VPR: none (disabled by settings) — no loop closing, no relocalization; a tracking loss is permanent");
        place_recognition = make_shared<PlaceRecognitionNone>(vpr_feature);
    }
    else if(vpr_method == "megaloc"){
        if(!std::ifstream(megaloc_onnx).good()){
            AF_ERROR("[System] vpr: megaloc requested but the model is missing ('" + megaloc_onnx
                     + "') — let the VSLAM-LAB wrapper download it, generate it with "
                       "Thirdparty/placecell/tools/export_megaloc.py, "
                       "or set megaloc_onnx: to an existing export");
            exit(-1);
        }
        // stdout is fully buffered when redirected to a log file: flush so the
        // announcement is visible during a 1-2 min engine build.
        const bool engineCached = std::ifstream(megaloc_onnx + "." + megaloc_precision + ".engine").good();
        AF_INFO("[System] VPR: megaloc | feature_vpr: " + feature_name(vpr_feature)
                + (feature_vpr_name.empty() ? " (defaulted: first feature)" : "")
                + " | " + (engineCached ? "loading cached TensorRT engine for " : "building TensorRT engine (1-2 min) for ")
                + megaloc_onnx + " ...");
        std::cout.flush();

        // placecell's diagnostic managers (PlaceCell.* keys). The Logger is process-wide:
        // its level is applied through Options (PLACECELL_VERBOSITY in the environment
        // wins, by placecell's contract); the Profiler and the Recorder belong to the store.
        placecell::PlaceCell::Options placecell_options{};
        placecell_options.name = "placecell";
        placecell_options.profile = placecell_settings.profile;
        placecell_options.record = placecell_settings.record;
        placecell_options.report_on_destruction = false;   // printed explicitly in Shutdown() (PlaceCell.PrintProfile)
        if(const auto level = placecell::Logger::parse(placecell_settings.verbosity))
            placecell_options.verbosity = *level;
        else
            AF_WARN("[System] PlaceCell.Verbosity '" << placecell_settings.verbosity
                    << "' is not a placecell log level (off|error|warn|info|debug|trace or 0-5) — keeping placecell's default");
        placecell::Logger::instance().set_show_elapsed(placecell_settings.log_elapsed);

        try {
            // Engine build/load + CUDA warmup happen in the embedder constructor
            place_cell = make_shared<placecell::MegaLocPlaceCell>(megaloc_onnx, megaloc_precision, placecell_options);
        } catch (const std::exception& e) {
            AF_ERROR("[System] MegaLoc backend setup failed: " + std::string(e.what()));
            exit(-1);
        }
        place_recognition = make_shared<PlaceRecognitionMegaLoc>(place_cell, vpr_feature, megaloc_params);
        AF_INFO("[System] VPR: megaloc ready (engine " << (place_cell->embedder().loaded_from_cache() ? "cached" : "built")
                << ": " << place_cell->embedder().engine_path() << ", " << place_cell->embedder().descriptor_dim()
                << "-d, min similarity " << megaloc_params.min_similarity
                << ", max candidates " << megaloc_params.max_candidates << ")");
        AF_INFO("[System] placecell diagnostics: verbosity "
                << placecell::Logger::name(placecell::Logger::instance().level())
                << (placecell::Logger::instance().level_from_environment() ? " (PLACECELL_VERBOSITY)" : "")
                << " | profile " << (placecell_settings.profile ? "on" : "off")
                << (placecell_settings.print_profile ? " (table at shutdown)" : "")
                << " | record " << (placecell_settings.record ? "on" : "off")
                << " | dump " << (placecell_settings.dump ? "on" : "off")
                << " | visualize " << (placecell_settings.visualize ? "on" : "off")
                << (placecell_settings.visualize && !activateVisualization ? " (no Viewer: verbose:0)" : ""));
        std::cout.flush();
    }
    else{
        AF_ERROR("[System] Unknown vpr backend '" + vpr_method + "' (options: megaloc, none)");
        exit(-1);
    }

    //Create the Map
    mpMap = make_shared<Map>();

    //Create Drawers. These are used by the Viewer
    frameDrawer = make_shared<FrameDrawer>(mpMap,featureTypes);
    mapDrawer = make_shared<MapDrawer>(mpMap, strSettingsFile,featureTypes);

    // Feature settings yaml file
    std::map<FeatureType, string> feature_settings_yaml_file;
    for (const auto& featureType : featureTypes){
        const AF_VSLAM::Feature& ft = get_feature(featureType);
        feature_settings_yaml_file[featureType] = ft.getSettingsYamlFile();
    }

    // Initialize matching thresholds
    for (const auto& featureType : featureTypes){
         FeatureMatcher::setDescriptorDistanceThresholds(feature_settings_yaml_file.at(featureType), featureType);
    }

    //Initialize the Tracking thread
    //(it will live in the main thread of execution, the one that called this constructor)
    tracker = make_shared<Tracking>(place_recognition, frameDrawer, mapDrawer,
                             mpMap,
                             strCalibrationFile, strSettingsFile,
                             feature_settings_yaml_file,
                             featureTypes, fixImageSize);

    //Initialize the Local Mapping thread and launch
    localMapper = make_shared<LocalMapping>(mpMap, featureTypes, tracker->get_image_width(), tracker->get_image_height());
    mptLocalMapping = make_shared<thread>(&AF_VSLAM::LocalMapping::run, localMapper);

    //Initialize the Loop Closing thread and launch. Without an active VPR backend the object
    //is still constructed (other threads hold pointers to it and enqueue keyframes) but its
    //thread never starts: LoopClosing starts with finished_ = true, so Shutdown()'s
    //is_finished()/isRunningGBA() waits pass immediately.
    loopCloser =  make_shared<LoopClosing>(mpMap, place_recognition, localMapper, mapDrawer,
        mSensor!=MONOCULAR, featureTypes,
        tracker->get_image_width(), tracker->get_image_height());
    if(place_recognition->is_active())
        mptLoopClosing = make_shared<thread>(&AF_VSLAM::LoopClosing::Run, loopCloser);

    //Initialize the Viewer thread and launch
    if(activateVisualization)
    {
        viewer = make_shared<Viewer>(this, frameDrawer,mapDrawer,tracker,
                                    strCalibrationFile, strSettingsFile,
                                    featureTypes);
        mptViewer = make_shared<thread>(&Viewer::Run, viewer);
        tracker->set_viewer(viewer);
        localMapper->set_viewer(viewer);
    }

    //Set pointers between threads
    tracker->set_local_mapper(localMapper);
    tracker->set_loop_closing(loopCloser);

    localMapper->set_loop_closer(loopCloser);
    if(place_cell)
        localMapper->set_placecell(place_cell);   // keyframe descriptors for the online VPR matrix
}

mat4f System::TrackStereo(const cv::Mat &, const cv::Mat &, const double &)
{
    std::cout << "This function (System::TrackStereo) has not been modified yet to work with AnyFeature-VSLAM"<< endl;
    std::terminate();

    // if(mSensor!=STEREO)
    // {
    //     cerr << "ERROR: you called TrackStereo but input sensor was not set to STEREO." << endl;
    //     exit(-1);
    // }

    // // Check mode change
    // {
    //     unique_lock<mutex> lock(mMutexMode);
    //     if(mbActivateLocalizationMode)
    //     {
    //        localMapper->request_stop();

    //         // Wait until Local Mapping has effectively stopped
    //         while(!localMapper->is_stopped())
    //         {
    //             usleep(1000);
    //         }

    //         tracker->InformOnlyTracking(true);
    //         mbActivateLocalizationMode = false;
    //     }
    //     if(mbDeactivateLocalizationMode)
    //     {
    //         tracker->InformOnlyTracking(false);
    //         localMapper->release();
    //         mbDeactivateLocalizationMode = false;
    //     }
    // }

    // // Check reset
    // {
    // unique_lock<mutex> lock(mMutexReset);
    // if(mbReset)
    // {
    //     tracker->reset();
    //     mbReset = false;
    // }
    // }

    // mat4f Tcw = tracker->GrabImageStereo(imLeft,imRight,timestamp);

    // unique_lock<mutex> lock2(mMutexState);
    // mTrackingState = tracker->state_;
    // mTrackedMapPoints = tracker->currentFrame.pts;
    // mTrackedKeyPointsUn = tracker->currentFrame.keypoints;
    // return Tcw;
}

mat4f System::TrackRGBD(const cv::Mat &, const cv::Mat &, const double &)
{
    std::cout << "This function (System::TrackRGBD) has not been modified yet to work with AnyFeature-VSLAM"<< endl;
    std::terminate();
    // if(mSensor!=RGBD)
    // {
    //     cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
    //     exit(-1);
    // }

    // // Check mode change
    // {
    //     unique_lock<mutex> lock(mMutexMode);
    //     if(mbActivateLocalizationMode)
    //     {
    //         localMapper->request_stop();

    //         // Wait until Local Mapping has effectively stopped
    //         while(!localMapper->is_stopped())
    //         {
    //             usleep(1000);
    //         }

    //         tracker->InformOnlyTracking(true);
    //         mbActivateLocalizationMode = false;
    //     }
    //     if(mbDeactivateLocalizationMode)
    //     {
    //         tracker->InformOnlyTracking(false);
    //         localMapper->release();
    //         mbDeactivateLocalizationMode = false;
    //     }
    // }

    // // Check reset
    // {
    // unique_lock<mutex> lock(mMutexReset);
    // if(mbReset)
    // {
    //     tracker->reset();
    //     mbReset = false;
    // }
    // }

    // mat4f Tcw = tracker->GrabImageRGBD(im,depthmap,timestamp);

    // unique_lock<mutex> lock2(mMutexState);
    // mTrackingState = tracker->state_;
    // mTrackedMapPoints = tracker->currentFrame.pts;
    // mTrackedKeyPointsUn = tracker->currentFrame.keypoints;
    // return Tcw;
}

mat4f System::Track(Image &im, const double &timestamp)
{
    std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();

    if(mSensor!=MONOCULAR && mSensor!=RGBD)
    {
        cerr << "ERROR: you called Track but input sensor was not set to mono or RGBD." << endl;
        exit(-1);
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        tracker->reset();
        mbReset = false;
    }
    }

    mat4f Tcw = tracker->grab_image(im, timestamp);

    mnFramesProcessed.fetch_add(1);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = static_cast<int>(tracker->state_);
    mTrackedMapPoints = tracker->current_frame_.pts;
    mTrackedKeyPointsUn = tracker->current_frame_.keypoints;

    std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
    double t_duration = std::chrono::duration_cast<std::chrono::duration<double> >(t_end - t_start).count();
    trackingTime.push_back(t_duration);

    return Tcw;
}

bool System::MapChanged()
{
    static int n=0;
    int curn = mpMap->GetLastBigChangeIdx();
    if(n<curn)
    {
        n=curn;
        return true;
    }
    else
        return false;
}

void System::reset()
{
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

void System::Shutdown()
{
    localMapper->request_finish();
    loopCloser->request_finish();
    if(viewer)
    {
        viewer->RequestFinish();
        while(!viewer->isFinished())
            usleep(5000);
    }

    // Wait until all thread have effectively stopped
    while(!localMapper->is_finished() || !loopCloser->is_finished() || loopCloser->isRunningGBA())
    {
        usleep(5000);
    }

    // The threads have returned; join them so their std::thread objects are not destroyed
    // joinable with System (that calls std::terminate at exit, issue #12).
    for(const auto& thread : {mptLocalMapping, mptLoopClosing, mptViewer})
        if(thread && thread->joinable())
            thread->join();

    if(viewer)
        pangolin::BindToContext(viewer->GetWindowTitle());

    // Every placecell caller has stopped: the profile table is complete
    if(place_cell && placecell_settings.print_profile)
    {
        std::cout.flush();   // keep the (stdout) AF_ lines before the (stderr) table when redirected
        place_cell->print_profile();
    }
}

void System::SavePlaceCellDiagnostics(const std::string& directory)
{
    if(!place_cell || !placecell_settings.dump)
        return;

    AF_INFO("Saving placecell diagnostics to " << directory << " ...");
    try {
        // Kernel (.npy, raw + centred), views.csv, recorder CSVs, profile.csv
        place_cell->dump(directory);

        // The three plots at their full-size defaults (the Viewer panels are smaller);
        // windows stay off (renders to cv::Mat and cv::imwrite only, headless-safe)
        placecell::viz::Visualizer::Options viz_options{};
        viz_options.windows = false;
        viz_options.max_hz = 0.0;
        viz_options.kernel.centred = placecell_settings.visualize_centred;
        viz_options.history.last_n = static_cast<size_t>(std::max(0, placecell_settings.visualize_history_last_n));
        placecell::viz::Visualizer visualizer(*place_cell, viz_options);
        visualizer.update(true);
        visualizer.save(directory);
    } catch (const std::exception& e) {
        AF_WARN("[System] placecell diagnostics dump failed: " << e.what());
        return;
    }
    AF_INFO("placecell diagnostics saved!");
}

const placecell::PlaceCell* System::GetPlaceCell() const
{
    return place_cell.get();
}

void System::SetSequenceInfo(const size_t nImages, const bool useMasks)
{
    mnSequenceImages.store(nImages);
    mbUseMasks.store(useMasks);
}

std::string System::GetModalityDescription() const
{
    std::string modality;
    switch(mSensor)
    {
        case MONOCULAR: modality = "Monocular"; break;
        case STEREO:    modality = "Stereo";    break;
        case RGBD:      modality = "RGB-D";     break;
    }
    if(mbUseMasks.load())
        modality += " + masks";
    return modality;
}


void System::SaveKeyFrameTrajectoryVSLAMLAB(const string &filename)
{
    AF_INFO("Saving keyframe trajectory to " << filename << " ...");

    vector<Keyframe> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    //cv::Mat Two = vpKFs[0]->get_pose_inverse();

    std::ofstream f(filename.c_str());
    f.imbue(std::locale::classic());

    // CSV header
    f << "ts (ns),tx (m),ty (m),tz (m),qx,qy,qz,qw\n";

    for(size_t i=0; i<vpKFs.size(); i++)
    {

        Keyframe pKF = vpKFs[i];

       // pKF->set_pose(pKF->get_pose()*Two);

        if(pKF->is_bad())
            continue;

        mat3f R = pKF->get_rotation().transpose();
        Eigen::Quaternionf q(R);
        vec3f t = pKF->get_camera_center();

        long long ts_ns = static_cast<long long>(std::round(pKF->timestamp * 1e9));
        f << std::fixed << std::setprecision(9) << ts_ns << ','
          << std::scientific << std::setprecision(7)
          << static_cast<double>(t(0)) << ','
          << static_cast<double>(t(1)) << ','
          << static_cast<double>(t(2)) << ','
          << static_cast<double>(q.x()) << ','
          << static_cast<double>(q.y()) << ','
          << static_cast<double>(q.z()) << ','
          << static_cast<double>(q.w()) << '\n';
    }

    f.close();
    AF_INFO("Keyframe trajectory saved!");
}

struct PointRGB {
    float x, y, z;
    uint8_t r, g, b;
};

bool write_ply_binary(const std::string& path, const std::vector<PointRGB>& pts) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;

    out << "ply\n";
    out << "format binary_little_endian 1.0\n";
    out << "element vertex " << pts.size() << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "property uchar red\n";
    out << "property uchar green\n";
    out << "property uchar blue\n";
    out << "end_header\n";

    for (const auto& p : pts) {
        out.write(reinterpret_cast<const char*>(&p.x), sizeof(float));
        out.write(reinterpret_cast<const char*>(&p.y), sizeof(float));
        out.write(reinterpret_cast<const char*>(&p.z), sizeof(float));
        out.write(reinterpret_cast<const char*>(&p.r), sizeof(uint8_t));
        out.write(reinterpret_cast<const char*>(&p.g), sizeof(uint8_t));
        out.write(reinterpret_cast<const char*>(&p.b), sizeof(uint8_t));
    }

    return static_cast<bool>(out);
}

auto to_u8 = [](double v) {
    v = std::clamp(v, 0.0, 1.0);
    return static_cast<uint8_t>(std::lround(v * 255.0));
};

void System::SavePointCloudVSLAMLAB(const string &filename, const vector<string>& imageFilenames)
{
    AF_INFO("Saving point cloud to " << filename << " ...");

    std::vector<PointRGB> pts;
    auto mapPoints = mpMap->get_all_map_points();
    int numPoints = 0;
    std::map<int, cv::Mat> imageCache;
    float alpha = 0.0f;
    for (auto& mp: mapPoints) {
        if (mp->is_bad()) continue;
        numPoints++;
        PointRGB p;
        vec3f pos = mp->get_world_pos();
        p.x = pos(0);
        p.y = pos(1);
        p.z = pos(2);

        FeatureType ft = mp->featureType;
        cv::Scalar color = AF_VSLAM::getFeatureColor(ft, 0, true);
        // p.r = 255;
        // p.g = 255;
        // p.b = 255;

        auto keyfram = mp->GetCurrentRefKeyframe();
        int idx = mp->GetIndexInKeyFrame(keyfram);
        int imgIdx = keyfram->frame_id;

        if (imageCache.find(imgIdx) == imageCache.end()) {
            cv::Mat cvimg = cv::imread(imageFilenames[imgIdx],cv::IMREAD_UNCHANGED);
            imageCache[imgIdx] = cvimg;
        }
        cv::Mat cvimg = imageCache[imgIdx];
        int u = static_cast<int>(keyfram->mvKeys.at(mp->featureType)[idx].pt.x);
        int v = static_cast<int>(keyfram->mvKeys.at(mp->featureType)[idx].pt.y);
        if (cvimg.channels() == 1) {
            uint8_t intensity = cvimg.at<uint8_t>(v, u);
            p.r = intensity;
            p.g = intensity;
            p.b = intensity;
        } else if (cvimg.channels() == 3) {
            cv::Vec3b bgr = cvimg.at<cv::Vec3b>(v, u);
            p.r = to_u8(alpha * (float(bgr[2]) / 255.0f) + (1.0f - alpha) * (float(color[0])));
            p.g = to_u8(alpha * (float(bgr[1]) / 255.0f) + (1.0f - alpha) * (float(color[1])));
            p.b = to_u8(alpha * (float(bgr[0]) / 255.0f) + (1.0f - alpha) * (float(color[2])));
        }
        pts.push_back(p);
    }
    AF_INFO("Point cloud saved! Number of points: " << numPoints);
    write_ply_binary(filename, pts);
}



int System::GetTrackingState()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackingState;
}

std::map<FeatureType,std::vector<Pt>> System::GetTrackedMapPoints()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedMapPoints;
}

std::map<FeatureType, std::vector<cv::KeyPoint>> System::GetTrackedKeyPointsUn()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedKeyPointsUn;
}

void System::SaveStatistics(const std::string &filename){
    // auto keyframes = mpMap->GetAllKeyFrames();
    // auto pts = mpMap->get_all_map_points();

    // size_t numKeyframes{0};
    // for (auto& keyframe: keyframes) {
    //     if(keyframe->is_bad())
    //         continue;
    //     numKeyframes++;
    // }

    // size_t numPts{0};
    // size_t numObservations{0};
    // for (auto& pt: pts) {
    //     if(pt->is_bad())
    //         continue;
    //     numPts++;
    //     numObservations += pt->number_of_observations();
    // }
    // float numObservationsPerPt = float(numObservations)/float(numPts);

    // double median_tracking_time{};
    // AF_VSLAM::vectorMedian(median_tracking_time,trackingTime);

    // double medianLocalMapppingTime{};
    // AF_VSLAM::vectorMedian(medianLocalMapppingTime,localMapper->localMappingTime);

    // double medianLoopClosingTime{};
    // AF_VSLAM::vectorMedian(medianLoopClosingTime,loopCloser->loopClosingTime);

    // long long finalVirtualMemUsed{virtualMemUsed.back()};
    // long long firstVirtualMemUsed{virtualMemUsed.front()};
    // long long maxVirtualMemUsed{0};
    // AF_VSLAM::vectorMax(maxVirtualMemUsed,virtualMemUsed);

    // ofstream f;
    // string statisticsFile = filename + ".txt";
    // f.open(statisticsFile.c_str());
    // f << fixed;
    // f << setprecision(0) << numKeyframes << " " << numPts << " " << numObservations << " " << setprecision(3) << numObservationsPerPt  <<
    // " " << setprecision(9) << median_tracking_time << " " << medianLocalMapppingTime << " " << medianLoopClosingTime <<
    // " " << tracker->num_tracked_frames_ <<" " << loopCloser->numOfLoopClosures << setprecision(0) <<
    // " " << firstVirtualMemUsed <<" " << maxVirtualMemUsed << " " << finalVirtualMemUsed <<
    // endl;

    // // Create statistics yaml file
    string statisticsFile_yaml = filename + ".yaml";
    // YAML::Node node;

    // node["graph"]["numKeyframes"] = numKeyframes;
    // node["graph"]["numPts"] = numPts;
    // node["graph"]["numObservations"] = numObservations;
    // node["graph"]["numObservationsPerPt"] = numObservationsPerPt;

    // node["profiling"]["median_tracking_time"] = median_tracking_time;
    // node["profiling"]["medianLocalMapppingTime"] = medianLocalMapppingTime;
    // node["profiling"]["medianLoopClosingTime"] = medianLoopClosingTime;

    // node["recall"]["num_tracked_frames_"] = tracker->num_tracked_frames_;
    // node["recall"]["loopCloser->numOfLoopClosures"] = loopCloser->numOfLoopClosures;

    // node["memory"]["firstVirtualMemUsed"] = firstVirtualMemUsed;
    // node["memory"]["maxVirtualMemUsed"] = maxVirtualMemUsed;
    // node["memory"]["finalVirtualMemUsed"] = finalVirtualMemUsed;

    // std::ofstream fout(statisticsFile_yaml);

    // fout << node;
    // fout.close();
    AF_INFO(statisticsFile_yaml + " file written successfully!");

}

void System::setImageSize(const int width, const int height){
    image_width = width;
    image_height = height;
}

void System::GBA(){
    AF_INFO("Starting Global Bundle Adjustment...");
    Optimizer::global_bundle_adjustment(mpMap, 100);
    AF_INFO("Global Bundle Adjustment finished.");
}

}
//namespace ORB_SLAM
