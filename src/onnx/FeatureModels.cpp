/**
 * Module: AllFeature-VSLAM - onnx/FeatureModels.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.1)
 * - Version: 1.1
 * - Created: 2026-10-16
 * - Updated: 2026-10-16 (static-shape exports: features_static_shapes)
 * - License: GPLv3 License
 */

#include "onnx/FeatureModels.h"

#include <algorithm>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "afvslam_log.hpp"

namespace AF_VSLAM::onnx
{

namespace
{

/// The padded exports of one variant present in models_dir: the L of every 'lightglue_<features>_L<L>.onnx'
/// (the '_fp16' twins have a non-numeric tail and are skipped), ascending.
std::vector<int> padded_lengths(const std::string& features)
{
    const std::string prefix = "lightglue_" + features + "_L", ext = ".onnx";
    std::vector<int> lengths;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(feature_models_config().models_dir, ec))
    {
        const std::string name = entry.path().filename().string();
        if (name.size() <= prefix.size() + ext.size() || name.compare(0, prefix.size(), prefix) != 0 ||
            name.compare(name.size() - ext.size(), ext.size(), ext) != 0)
            continue;
        const std::string digits = name.substr(prefix.size(), name.size() - prefix.size() - ext.size());
        if (digits.find_first_not_of("0123456789") != std::string::npos)
            continue;
        lengths.push_back(std::stoi(digits));
    }
    std::sort(lengths.begin(), lengths.end());
    return lengths;
}

/// Create and warm up one matcher session: the first Run of a session pays allocation and (CUDA / Core ML) kernel
/// selection; keep it out of the first frame. Sixteen points per side on a 640x480 frame.
std::unique_ptr<LightGlueOnnx> load_matcher(const std::string& file, const std::string& features)
{
    auto matcher = std::make_unique<LightGlueOnnx>(feature_model_path(file), feature_models_config().ort);
    std::vector<cv::KeyPoint> kps;
    for (int i = 0; i < 16; ++i)
        kps.emplace_back(40.0f * i + 20.0f, 30.0f * (i % 4) + 15.0f, 3.0f, 10.0f * i);
    cv::Mat desc(16, matcher->input_dim(), CV_32F);
    cv::randn(desc, 0.0, 1.0);
    for (int i = 0; i < desc.rows; ++i)
        cv::normalize(desc.row(i), desc.row(i));
    matcher->match(kps, desc, kps, desc, 640, 480);
    AF_INFO("LightGlue (" << features << ") warmed up: " << matcher->last_graph_ms() << " ms for 16x16 points on "
                          << file);
    return matcher;
}

/// One variant's sessions: the dynamic file and the padded exports, each loaded on first use.
struct Variant
{
    std::unique_ptr<LightGlueOnnx> dynamic;
    bool scanned{false};
    std::vector<int> lengths;                                 // padded exports present, ascending
    std::map<int, std::unique_ptr<LightGlueOnnx>> padded;     // by L
    bool warned_overflow{false};
};

} // namespace

FeatureModelsConfig& feature_models_config()
{
    static FeatureModelsConfig config;
    return config;
}

std::string feature_model_path(const std::string& file)
{
    return feature_models_config().models_dir + "/" + file;
}

std::string static_model_path(const std::string& file, int width, int height, const std::string& export_key)
{
    const std::string size = std::to_string(width) + "x" + std::to_string(height);
    const std::string stem = file.size() > 5 && file.compare(file.size() - 5, 5, ".onnx") == 0
                                 ? file.substr(0, file.size() - 5) : file;
    const std::string path = feature_model_path(stem + "_" + size + ".onnx");
    if (!std::filesystem::exists(path))
        throw std::runtime_error("features_static_shapes is on but " + path + " does not exist; export it with: "
                                 "pixi run -e export-features export-features --only " + export_key + " --static " + size);
    return path;
}

namespace
{

std::mutex& variants_mutex()
{
    static std::mutex mutex;
    return mutex;
}

std::map<std::string, Variant>& variants()
{
    static std::map<std::string, Variant> map;
    return map;
}

/// The padded exports of `features`, scanned once (caller holds variants_mutex()).
void scan_padded(Variant& v, const std::string& features)
{
    if (v.scanned)
        return;
    v.lengths = padded_lengths(features);
    v.scanned = true;
    if (v.lengths.empty())
        throw std::runtime_error("features_static_shapes is on but " + feature_models_config().models_dir +
                                 " has no lightglue_" + features + "_L<L>.onnx; export one with: pixi run -e "
                                 "export-features export-features --only lightglue_" + features + " --static-length 1024");
}

LightGlueOnnx& padded_session(Variant& v, const std::string& features, int length)
{
    auto& slot = v.padded[length];
    if (!slot)
        slot = load_matcher("lightglue_" + features + "_L" + std::to_string(length) + ".onnx", features);
    return *slot;
}

} // namespace

std::vector<int> preload_lightglue(const std::string& features)
{
    std::lock_guard<std::mutex> lock(variants_mutex());
    Variant& v = variants()[features];
    if (!v.dynamic)
        v.dynamic = load_matcher("lightglue_" + features + ".onnx", features);
    if (!feature_models_config().static_shapes)
        return {};
    scan_padded(v, features);
    for (const int length : v.lengths)
        padded_session(v, features, length);
    return v.lengths;
}

LightGlueOnnx& shared_lightglue(const std::string& features, int n_max)
{
    std::lock_guard<std::mutex> lock(variants_mutex());
    Variant& v = variants()[features];
    const FeatureModelsConfig& config = feature_models_config();
    if (config.static_shapes && n_max > 0)
    {
        scan_padded(v, features);
        const auto it = std::lower_bound(v.lengths.begin(), v.lengths.end(), n_max);   // smallest L >= n_max
        if (it != v.lengths.end())
            return padded_session(v, features, *it);
        if (!v.warned_overflow)
        {
            AF_WARN("LightGlue (" << features << "): " << n_max << " keypoints on a side exceed the largest padded export (L="
                                  << v.lengths.back() << "); such pairs run on the dynamic file (reported once)");
            v.warned_overflow = true;
        }
    }
    if (!v.dynamic)
        v.dynamic = load_matcher("lightglue_" + features + ".onnx", features);
    return *v.dynamic;
}

} // namespace AF_VSLAM::onnx
