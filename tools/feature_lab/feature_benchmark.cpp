/**
 * Module: AllFeature-VSLAM - tools/feature_lab/feature_benchmark.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5.2)
 * - Version: 0.14 (SuperPoint threshold / NMS variant; geometric error of every match; brute-force matcher variant; descriptor distance of every match; feature variants; MegaLoc similarity matrix over the visited images, random pairs above a similarity threshold,
 *   every feature extracted, matched and geometrically checked on each pair, one csv per feature)
 * - Created: 2026-11-12
 * - License: GPLv3 License
 *
 * feature_benchmark: the entry point of the feature lab. Takes the images to work on and the features to run on
 * them, and measures every feature on the same image pairs:
 *
 *   1. visit 1 of every `step` images of the list;
 *   2. embed each with MegaLoc (placecell's embedder, the SLAM's `vpr: megaloc` model) and build the similarity
 *      matrix of the visited images (cosine of the descriptors), saved as <output-dir>/vpr_megaloc.npy with its row
 *      order in vpr_megaloc_images.csv;
 *   3. the candidate pairs are the pairs of visited images whose similarity is above --vpr-threshold: pairs that
 *      plausibly see the same place, whatever their distance in the sequence;
 *   4. draw --pairs of them at random without repetition (fewer if there are fewer candidates; --seed makes the
 *      draw repeatable), and for each drawn pair run every feature: extract the keypoints of both images, match
 *      them, check the matches geometrically.
 * Per feature it uses the keypoint extractor and the feature matcher the SLAM uses (same settings yaml, same model
 * files). Matches are the matcher's raw output (LightGlue, or cross-checked brute force).
 *
 * Output: <output-dir>/<feature>.csv, one row per image pair, written as the pair is processed:
 *   image_0, image_1   the pair, as the csv names the images (image_0 is the earlier one)
 *   index_0, index_1   their row in the image list (0-based), so index_1 - index_0 is the pair's gap in frames
 *   vpr_similarity     MegaLoc cosine similarity of the pair (the value that made it a candidate)
 *   keypoints_0, _1    keypoints extracted on each image
 *   matches            matched keypoints
 *   matched_percent    100 * matches / min(keypoints_0, keypoints_1): the share of the matches the pair could have
 *                      at most (a keypoint matches at most once); 0 when a side has no keypoints
 *   correct_matches    matches that pass the geometric check below
 *   correct_percent    100 * correct_matches / matches; 0 when the pair has no matches
 *   geometry           the model that judged the pair: fundamental | homography | none (see below)
 *   extract_ms_0, _1   wall time of the keypoint extraction on each image (the extractor call alone: no image
 *                      decoding, no grey conversion). An image is extracted once per feature, the first time a
 *                      drawn pair needs it, so its time repeats on every row the image appears in
 *   match_ms           wall time of the matcher call on the pair
 *   geometry_ms        wall time of the geometric check on the pair
 *
 * Geometric check (check_geometry): the SLAM's own filter, FeatureMatcher::filter_matches_by_fundamental, with one
 * difference. A fundamental matrix is fitted to the pair's matches with PoseLib LO-RANSAC (inlier threshold 3 px,
 * 100-1000 iterations) and its inliers are the correct matches; if it finds fewer than 8 inliers, a homography is
 * tried the same way (planar scenes, pure rotation). The difference: when neither model finds 8 inliers the SLAM
 * passes every match through unfiltered, and here the pair has 0 correct matches (geometry: none).
 * What "correct" means, then: consistent with one two-view geometry estimated from the matches themselves. It needs
 * no ground truth, but it is an upper bound on the truly correct matches - a wrong match that happens to fall
 * within 3 px of the epipolar line passes, and RANSAC finds a small consensus even among unrelated images. It runs
 * on the keypoints as extracted (no undistortion), so it is exact for pinhole sequences only.
 * The first image of a learned feature can be slower than the rest: the models are warmed up on another image size.
 *
 * Every feature's csv has the same pairs in the same order, so the files join row by row.
 *
 * Output: <output-dir>/<feature>_distances.csv, one row per MATCH of every pair (hundreds of thousands of rows):
 *   distance           descriptor distance between the two matched keypoints, in the feature's own metric - the one
 *                      the SLAM's descriptor thresholds (FeatureMatcher.TH_LOW, ...) are expressed in: Hamming for
 *                      the binary descriptors (orb32, akaze61, brisk48), L2 for the others. For a LightGlue feature
 *                      this is NOT what the matcher decided on (it scores keypoints with their positions too)
 *   correct            1 if the match passed the geometric check, 0 if not
 *   error_px           the match's error against the pair's two-view model, in pixels: the Sampson distance to the
 *                      fundamental matrix (the first-order distance of the two keypoints to the nearest pair that
 *                      satisfies the epipolar constraint exactly), or the transfer error |H x0 - x1| for a
 *                      homography; -1 when the pair has no model. Correct matches are below 3 by construction. It
 *                      is an epipolar error, not a 3D reprojection error: no points are triangulated here
 *
 * Cost: one MegaLoc embedding per visited image (the embedder is freed before the features run); then, per
 * feature, each image is decoded and extracted at most once - the first time a drawn pair needs it - and its
 * keypoints stay cached for the later pairs (about 0.1-0.7 MB per image and feature). All the requested features'
 * models are loaded at the same time, since the features share the inner loop.
 *
 * Grey conversion is BGR2GRAY, the order cv::imread gives. The SLAM currently applies RGB2GRAY to the same data
 * (the open colour-order item), so its keypoints differ slightly from these.
 *
 * Run it from the repository root: the per-feature settings (settings/<feature>_settings.yaml) and the learned
 * models (feature_models/) are resolved against the working directory, as in the SLAM executables. The learned
 * models run with the compiled-in defaults of onnx::feature_models_config() (provider auto, dynamic shapes).
 * The matcher is built for the size of the first image as it is on disk: no resize to the nominal area yet.
 *
 *   ./bin/feature_benchmark --rgb-csv <rgb.csv> --features orb32 aliked128 [--step 10] [--pairs 1000]
 *                           [--vpr-threshold 0.55] [--seed 0] [--megaloc-onnx <file>] [--output-dir <dir>]
 *                           [--sequence-path <dir>] [--cam rgb_0]
 *                           [--variant <label> <feature> [model=<file>] [max_keypoints=<n>]
 *                                      [matcher=bruteforce]] ...
 *
 *   --rgb-csv        a VSLAM-LAB image list: a header row, then one row per image, with the columns
 *                    'ts_<cam> (ns)' and 'path_<cam>' (any order, other columns ignored)
 *   --features       one or more feature names, as in the settings yaml's `features:` (orb32, sift128, ...)
 *   --variant        one more run, of <feature> with something changed, reported under <label> (its csv is
 *                    <label>.csv); repeatable, and it runs on the same pairs as everything else. Changes:
 *                      max_keypoints=<n>  the keypoint budget, overriding the settings yaml's max_num_features
 *                      model=<file>       another model file for the extractor (aliked128 only); a bare file
 *                                         name is looked up in feature_models/
 *                      threshold=<x>      superpoint256 only: the detection threshold of the C++ tail (the sidecar's
 *                                         detection_threshold, 0 by default); 0 keeps the best max_keypoints maxima
 *                                         whatever their score
 *                      nms_radius=<n>     superpoint256 only: the NMS radius of the C++ tail (the sidecar's, 4 px)
 *                      matcher=bruteforce the cross-checked brute-force matcher (mutual nearest neighbour on the
 *                                         descriptor distance: L2 for float descriptors, Hamming for binary - the
 *                                         one orb32 uses) instead of the feature's own, i.e. instead of LightGlue
 *                                         for sift128 / aliked128 / superpoint256; no change for the others
 *                    e.g. --variant aliked128_threshold aliked128 model=aliked_n16_threshold.onnx
 *                         --variant aliked128_2048 aliked128 max_keypoints=2048
 *                    --features may be left out when at least one --variant is given.
 *   --step           visit 1 of every <step> images, starting at the first (default: 1, every image)
 *   --pairs          how many random pairs to draw from the candidates, without repetition (default: 1000)
 *   --vpr-threshold  a pair of visited images is a candidate when its MegaLoc similarity is above this (default:
 *                    0.55, the SLAM's PlaceRecognition.MegaLocMinSimilarity: unrelated places ~0.35-0.4)
 *   --seed           seed of the random draw (default: 0)
 *   --megaloc-onnx   the MegaLoc model (default: megaloc_models/megaloc_322x322.onnx, relative to the working
 *                    directory)
 *   --output-dir     where the per-feature results go, created if needed (default: tools/feature_lab/experiments,
 *                    relative to the working directory like the settings and the models)
 *   --sequence-path  the folder the csv's image paths are relative to (default: the csv's own folder)
 *   --cam            the camera whose columns are read (default: rgb_0)
 *
 * Exit code 0 = inputs are usable, 1 = they are not (missing file, unknown feature, no images), 2 = usage.
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <algorithm>
#include <memory>
#include <optional>
#include <random>
#include <regex>
#include <sstream>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/imgcodecs.hpp>

#include "onnx/AlikedOnnx.h"
#include "onnx/FeatureModels.h"
#include "onnx/SuperPointOnnx.h"
#include "placecell/kernel_io.h"
#include "placecell/megaloc_embedder.h"

#include "FeatureExtractor.h"
#include "FeatureFactory.h"
#include "FeatureMatcher.h"
#include "Image.h"

// PoseLib's headers trip this project's -Wextra (unused parameters): silence it for the include only
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <PoseLib/robust.h>
#pragma GCC diagnostic pop
#include "KeypointData.h"
#include "StringUtils.h"

namespace
{

struct ImageEntry
{
    double timestamp{0.0};   // seconds
    std::string name;        // the path as the csv gives it, relative to the sequence
    std::string path;        // sequence_path / name
};

/// One run of the benchmark: a feature as the settings yaml configures it (--features), or with a change (--variant).
struct RunSpec
{
    std::string label;       // names the run in the report and its csv
    std::string feature;     // settings-yaml feature name
    std::string model;       // extractor model file ("" = the feature's own)
    int max_keypoints{0};    // keypoint budget (0 = the settings yaml's)
    bool bruteforce{false};  // match with the cross-checked brute force instead of the feature's own matcher
    float threshold{-1.0f};  // superpoint256: detection threshold of the tail (< 0 = the sidecar's)
    int nms_radius{-1};      // superpoint256: NMS radius of the tail (< 0 = the sidecar's)
};

struct Arguments
{
    std::vector<RunSpec> variants;
    std::string rgb_csv;
    std::string sequence_path;
    std::string cam{"rgb_0"};
    std::string output_dir{"tools/feature_lab/experiments"};
    std::vector<std::string> features;
    size_t step{1};   // visit 1 of every `step` images
    size_t pairs{1000};             // random pairs drawn from the candidates
    float vpr_threshold{0.55f};     // a pair is a candidate above this MegaLoc similarity
    unsigned seed{0};               // of the random draw
    std::string megaloc_onnx{"megaloc_models/megaloc_322x322.onnx"};
};

void print_usage()
{
    std::fprintf(stderr, "usage: feature_benchmark --rgb-csv <rgb.csv> --features <name> [<name> ...] "
                         "[--step 1] [--pairs 1000] [--vpr-threshold 0.55] [--seed 0] [--megaloc-onnx <file>] "
                         "[--output-dir <dir>] [--sequence-path <dir>] [--cam rgb_0] "
                         "[--variant <label> <feature> [model=<file>] [max_keypoints=<n>] [matcher=bruteforce]] ...\n");
}

/// Flags take one value, except --features, which takes every token up to the next flag.
Arguments parse_arguments(const std::vector<std::string>& args)
{
    Arguments parsed;
    for (size_t i = 0; i < args.size(); ++i)
    {
        const std::string& key = args[i];
        if (key == "--features")
        {
            while (i + 1 < args.size() && args[i + 1].rfind("--", 0) != 0)
                parsed.features.push_back(args[++i]);
            continue;
        }
        if (key == "--variant")
        {
            std::vector<std::string> tokens;
            while (i + 1 < args.size() && args[i + 1].rfind("--", 0) != 0)
                tokens.push_back(args[++i]);
            if (tokens.size() < 2)
                throw std::invalid_argument("--variant needs <label> <feature> [model=<file>] [max_keypoints=<n>] [matcher=bruteforce]");
            RunSpec variant{tokens[0], tokens[1], "", 0, false, -1.0f, -1};
            for (size_t t = 2; t < tokens.size(); ++t)
            {
                const size_t equals = tokens[t].find('=');
                const std::string name = tokens[t].substr(0, equals);
                const std::string value = equals == std::string::npos ? "" : tokens[t].substr(equals + 1);
                if (name == "model" && !value.empty())
                    variant.model = value;
                else if (name == "max_keypoints")
                {
                    try { variant.max_keypoints = std::stoi(value); } catch (const std::exception&) {}
                    if (variant.max_keypoints < 1)
                        throw std::invalid_argument("--variant " + variant.label + ": max_keypoints must be an integer >= 1");
                }
                else if (name == "matcher" && value == "bruteforce")
                    variant.bruteforce = true;
                else if (name == "threshold")
                {
                    try { variant.threshold = std::stof(value); } catch (const std::exception&) {}
                    if (variant.threshold < 0.0f)
                        throw std::invalid_argument("--variant " + variant.label + ": threshold must be a number >= 0");
                }
                else if (name == "nms_radius")
                {
                    try { variant.nms_radius = std::stoi(value); } catch (const std::exception&) {}
                    if (variant.nms_radius < 0)
                        throw std::invalid_argument("--variant " + variant.label + ": nms_radius must be an integer >= 0");
                }
                else
                    throw std::invalid_argument("--variant " + variant.label + ": unknown change '" + tokens[t] +
                                                "' (options: model=<file>, max_keypoints=<n>, matcher=bruteforce, threshold=<x>, nms_radius=<n>)");
            }
            parsed.variants.push_back(variant);
            continue;
        }
        if (key != "--rgb-csv" && key != "--sequence-path" && key != "--cam" && key != "--step" &&
            key != "--output-dir" && key != "--pairs" && key != "--vpr-threshold" && key != "--seed" &&
            key != "--megaloc-onnx")
            throw std::invalid_argument("unknown argument '" + key + "'");
        if (i + 1 >= args.size())
            throw std::invalid_argument("'" + key + "' needs a value");
        const std::string& value = args[++i];
        if (key == "--rgb-csv") parsed.rgb_csv = value;
        else if (key == "--sequence-path") parsed.sequence_path = value;
        else if (key == "--output-dir") parsed.output_dir = value;
        else if (key == "--megaloc-onnx") parsed.megaloc_onnx = value;
        else if (key == "--pairs" || key == "--seed")
        {
            long number = -1;
            try { number = std::stol(value); } catch (const std::exception&) {}
            if (number < (key == "--pairs" ? 1 : 0))
                throw std::invalid_argument(key + " must be an integer >= " + (key == "--pairs" ? "1" : "0") + ", got '" + value + "'");
            if (key == "--pairs") parsed.pairs = static_cast<size_t>(number);
            else parsed.seed = static_cast<unsigned>(number);
        }
        else if (key == "--vpr-threshold")
        {
            try { parsed.vpr_threshold = std::stof(value); }
            catch (const std::exception&) { throw std::invalid_argument("--vpr-threshold must be a number, got '" + value + "'"); }
        }
        else if (key == "--step")
        {
            int step = 0;
            try { step = std::stoi(value); } catch (const std::exception&) {}
            if (step < 1)
                throw std::invalid_argument("--step must be an integer >= 1, got '" + value + "'");
            parsed.step = static_cast<size_t>(step);
        }
        else parsed.cam = value;
    }
    if (parsed.rgb_csv.empty())
        throw std::invalid_argument("--rgb-csv is required");
    if (parsed.features.empty() && parsed.variants.empty())
        throw std::invalid_argument("--features needs at least one feature name (or give a --variant)");
    if (parsed.sequence_path.empty())
        parsed.sequence_path = std::filesystem::path(parsed.rgb_csv).parent_path().string();
    return parsed;
}

/// The feature type of a settings-yaml feature name. get_feature_type() ends the process on a name it does not
/// know, so the name is checked here first and reported instead.
FeatureType to_feature_type(const std::string& name)
{
    static const std::vector<std::string> known{"orb32", "akaze61", "brisk48", "kaze64",
                                                "sift128", "aliked128", "superpoint256"};
    for (const std::string& candidate : known)
        if (candidate == name)
            return get_feature_type(name);
    std::string list;
    for (const std::string& candidate : known)
        list += " " + candidate;
    throw std::runtime_error("unknown feature '" + name + "' (options:" + list + ")");
}

/// The image list of a VSLAM-LAB rgb.csv, same column contract as the SLAM entry points' LoadImages.
std::vector<ImageEntry> load_images(const std::string& rgb_csv, const std::string& sequence_path,
                                    const std::string& cam)
{
    std::ifstream file(rgb_csv);
    if (!file)
        throw std::runtime_error("cannot open " + rgb_csv);

    std::string line;
    if (!std::getline(file, line))
        throw std::runtime_error(rgb_csv + " is empty");
    const std::vector<std::string> headers = split(line, ',');
    std::map<std::string, size_t> column;
    for (size_t i = 0; i < headers.size(); ++i)
        column[headers[i]] = i;

    const std::string timestamp_header = "ts_" + cam + " (ns)", path_header = "path_" + cam;
    for (const std::string& header : {timestamp_header, path_header})
        if (column.find(header) == column.end())
            throw std::runtime_error("column '" + header + "' not found in " + rgb_csv);
    const size_t timestamp_column = column.at(timestamp_header), path_column = column.at(path_header);

    std::vector<ImageEntry> images;
    while (std::getline(file, line))
    {
        if (line.empty() || line == "\r")
            continue;
        const std::vector<std::string> tokens = split(line, ',');
        if (tokens.size() <= std::max(timestamp_column, path_column))
            throw std::runtime_error("malformed row in " + rgb_csv + ": " + line);
        images.push_back({static_cast<double>(std::stoll(tokens[timestamp_column])) * 1e-9, tokens[path_column],
                          (std::filesystem::path(sequence_path) / tokens[path_column]).string()});
    }
    return images;
}

/// ALIKED on a model file of the caller's choice. The same steps as Aliked128Extractor::detect_and_compute -
/// the graph, the mask, the octree distribution down to the budget when the graph returns more than it - on another
/// file: the SLAM's extractor loads the fixed name aliked_n16.onnx.
class AlikedFileExtractor : public AF_VSLAM::FeatureExtractor
{
  public:
    AlikedFileExtractor(std::shared_ptr<AF_VSLAM::FeatureExtractorSettings>& settings, const std::string& onnx_path)
        : FeatureExtractor(settings), aliked_(onnx_path, AF_VSLAM::onnx::feature_models_config().ort)
    {
    }

  protected:
    void detect_and_compute(const AF_VSLAM::Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override
    {
        std::vector<cv::KeyPoint> all;
        cv::Mat all_descriptors;
        aliked_.detect(img.img, all, all_descriptors);
        filter_keypoints_by_mask(all, all_descriptors, img.mask);
        if (static_cast<float>(all.size()) < settings->oversize_keypoint_factor * static_cast<float>(settings->max_num_features))
        {
            keypoints = all;
            descriptors = all_descriptors;
            return;
        }
        keypoints = distribute_oct_tree(all, 0, img.grayImg.cols - 1, 0, img.grayImg.rows - 1, settings->max_num_features);
        descriptors.create(static_cast<int>(keypoints.size()), all_descriptors.cols, all_descriptors.type());
        for (int i = 0; i < static_cast<int>(keypoints.size()); ++i)
            all_descriptors.row(keypoints[i].class_id).copyTo(descriptors.row(i));
    }
    [[nodiscard]] float get_keypoint_size(const cv::KeyPoint&) const override { return 1.0f; }

  private:
    AF_VSLAM::onnx::AlikedOnnx aliked_;
};

/// SuperPoint with a sidecar of the caller's choice. The same steps as Superpoint256Extractor::detect_and_compute
/// (the dense graph, the C++ tail with the keypoint budget, the mask); the sidecar is where the tail's detection
/// threshold and NMS radius live, and the SLAM's extractor always reads the model's own.
class SuperPointSidecarExtractor : public AF_VSLAM::FeatureExtractor
{
  public:
    SuperPointSidecarExtractor(std::shared_ptr<AF_VSLAM::FeatureExtractorSettings>& settings, const std::string& onnx_path,
                               const std::string& sidecar_path)
        : FeatureExtractor(settings), superpoint_(onnx_path, AF_VSLAM::onnx::feature_models_config().ort, sidecar_path)
    {
    }

  protected:
    void detect_and_compute(const AF_VSLAM::Image& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override
    {
        superpoint_.detect(img.grayImg, settings->max_num_features, keypoints, descriptors);
        filter_keypoints_by_mask(keypoints, descriptors, img.mask);
    }
    [[nodiscard]] float get_keypoint_size(const cv::KeyPoint&) const override { return 1.0f; }

  private:
    AF_VSLAM::onnx::SuperPointOnnx superpoint_;
};

/// A copy of SuperPoint's sidecar with the tail's detection threshold and / or NMS radius replaced, written to
/// `output_dir` as '<label>_superpoint_v1.onnx.yaml'. Returns its path.
std::string write_superpoint_sidecar(const std::string& onnx_path, const RunSpec& spec, const std::string& output_dir)
{
    std::ifstream in(onnx_path + ".yaml");
    if (!in)
        throw std::runtime_error("cannot read " + onnx_path + ".yaml");
    std::stringstream buffer;
    buffer << in.rdbuf();
    std::string text = buffer.str();
    const auto replace_key = [&](const std::string& key, const std::string& value) {
        const std::regex line("(^|\\n)" + key + ":[^\\n]*");
        if (!std::regex_search(text, line))
            throw std::runtime_error(onnx_path + ".yaml has no '" + key + "' key");
        text = std::regex_replace(text, line, "$1" + key + ": " + value + "   # overridden by --variant " + spec.label);
    };
    if (spec.threshold >= 0.0f)
        replace_key("detection_threshold", std::to_string(spec.threshold));
    if (spec.nms_radius >= 0)
        replace_key("nms_radius", std::to_string(spec.nms_radius));
    const std::string path = (std::filesystem::path(output_dir) / (spec.label + "_superpoint_v1.onnx.yaml")).string();
    std::ofstream out(path);
    out << text;
    if (!out)
        throw std::runtime_error("cannot write " + path);
    return path;
}

/// The keypoint extractor of one feature type, built as Tracking builds it: the feature's settings yaml, then the
/// feature's own factory.
/// A variant's changes are applied on top: the budget replaces the yaml's, a model file goes through AlikedFileExtractor.
std::shared_ptr<AF_VSLAM::FeatureExtractor> build_extractor(const FeatureType feature_type, const RunSpec& spec,
                                                            const std::string& output_dir)
{
    const AF_VSLAM::Feature& feature = get_feature(feature_type);
    const std::string& settings_yaml = feature.get_settings_yaml_file();
    if (!std::filesystem::exists(settings_yaml))
        throw std::runtime_error(settings_yaml + " not found: run from the repository root");
    auto settings = std::make_shared<AF_VSLAM::FeatureExtractorSettings>(feature_type, settings_yaml);
    if (spec.max_keypoints > 0)
        settings->max_num_features = spec.max_keypoints;
    if (spec.threshold >= 0.0f || spec.nms_radius >= 0)
    {
        if (feature_type != FEAT_SUPERPOINT256 || !spec.model.empty())
            throw std::runtime_error("--variant " + spec.label + ": threshold= / nms_radius= are supported for superpoint256 "
                                     "only, on its own model file");
        const std::string onnx_path = AF_VSLAM::onnx::feature_model_path(AF_VSLAM::Superpoint256::s_model_file);
        return std::make_shared<SuperPointSidecarExtractor>(settings, onnx_path,
                                                            write_superpoint_sidecar(onnx_path, spec, output_dir));
    }
    if (spec.model.empty())
        return feature.create_extractor(settings);
    if (feature_type != FEAT_ALIKED128)
        throw std::runtime_error("--variant " + spec.label + ": model= is supported for aliked128 only");
    const std::string path = spec.model.find('/') == std::string::npos ? AF_VSLAM::onnx::feature_model_path(spec.model)
                                                                       : spec.model;
    return std::make_shared<AlikedFileExtractor>(settings, path);
}

/// The feature matcher of one feature type, built as System + Tracking build it: the descriptor-distance thresholds
/// from the feature's settings yaml, then a FeatureMatcher for this type alone (it loads the LightGlue session of a
/// learned feature).
std::shared_ptr<AF_VSLAM::FeatureMatcher> build_matcher(const FeatureType feature_type, const int image_width,
                                                        const int image_height)
{
    const AF_VSLAM::Feature& feature = get_feature(feature_type);
    AF_VSLAM::FeatureMatcher::setDescriptorDistanceThresholds(feature.get_settings_yaml_file(), feature_type);
    return std::make_shared<AF_VSLAM::FeatureMatcher>(image_width, image_height, std::vector<FeatureType>{feature_type},
                                                      "feature_benchmark/" + feature.get_feature_name());
}

using Clock = std::chrono::steady_clock;

double ms_since(const Clock::time_point& t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

/// Running totals of one feature's pass over the images.
struct FeatureStats
{
    size_t images{0}, pairs{0};
    size_t keypoints{0}, matches{0}, correct{0};
    double extract_ms{0.0}, match_ms{0.0}, geometry_ms{0.0};
};

/// Decode one image and extract its keypoints into `data` (overwritten). The decoded image lives only inside this
/// call. Returns false, leaving `data` untouched, if the file cannot be read.
bool extract(const std::string& path, AF_VSLAM::FeatureExtractor& extractor, AF_VSLAM::KeypointData& data,
             double& extract_ms)
{
    AF_VSLAM::Image image(path);
    if (image.img.empty())
        return false;
    image.get_gray_image(false);   // cv::imread order: BGR
    data = AF_VSLAM::KeypointData{};
    const auto t0 = Clock::now();
    extractor(image, data);
    extract_ms = ms_since(t0);
    return true;
}

/// The outcome of the geometric check of one pair's matches.
struct GeometryCheck
{
    size_t correct{0};           // matches consistent with the model
    const char* model{"none"};   // "fundamental" | "homography" | "none"
    std::vector<char> inliers;   // per match, 1 = consistent with the model; all 0 when no model was found
    std::vector<float> errors;   // per match, error against the model in px (Sampson / transfer); -1 without a model
};

/// Sampson distance of one correspondence to a fundamental matrix (x1^T F x0 = 0), in pixels.
double sampson_error(const Eigen::Matrix3d& F, const poselib::Point2D& x0, const poselib::Point2D& x1)
{
    const Eigen::Vector3d h0(x0.x(), x0.y(), 1.0), h1(x1.x(), x1.y(), 1.0);
    const Eigen::Vector3d Fx0 = F * h0, Ftx1 = F.transpose() * h1;
    const double denominator = Fx0.head<2>().squaredNorm() + Ftx1.head<2>().squaredNorm();
    return denominator > 0.0 ? std::abs(h1.dot(Fx0)) / std::sqrt(denominator) : 0.0;
}

/// Transfer error of one correspondence under a homography (x1 ~ H x0), in pixels.
double transfer_error(const Eigen::Matrix3d& H, const poselib::Point2D& x0, const poselib::Point2D& x1)
{
    const Eigen::Vector3d projected = H * Eigen::Vector3d(x0.x(), x0.y(), 1.0);
    if (std::abs(projected.z()) < 1e-12)
        return 1e6;
    return (projected.hnormalized() - x1).norm();
}

/// The SLAM's match filter (FeatureMatcher::filter_matches_by_fundamental: PoseLib LO-RANSAC, fundamental matrix
/// first, homography when it finds no consensus), same threshold and iteration budget - except that a pair where
/// neither model finds a consensus has no correct matches, instead of all its matches passed through.
GeometryCheck check_geometry(const std::vector<cv::DMatch>& matches, const std::vector<cv::KeyPoint>& keypoints0,
                             const std::vector<cv::KeyPoint>& keypoints1)
{
    constexpr double inlier_threshold_px = 3.0;   // Sampson (F) / transfer (H) error
    constexpr size_t min_inliers = 8;             // below this a model is not a consensus
    constexpr size_t max_iterations = 1000, min_iterations = 100;

    GeometryCheck none;
    none.inliers.assign(matches.size(), 0);
    none.errors.assign(matches.size(), -1.0f);
    if (matches.size() < min_inliers)
        return none;
    std::vector<poselib::Point2D> points0, points1;
    points0.reserve(matches.size());
    points1.reserve(matches.size());
    for (const cv::DMatch& m : matches)
    {
        const cv::Point2f& p0 = keypoints0[m.queryIdx].pt;
        const cv::Point2f& p1 = keypoints1[m.trainIdx].pt;
        points0.emplace_back(p0.x, p0.y);
        points1.emplace_back(p1.x, p1.y);
    }

    std::vector<char> inliers;
    poselib::RelativePoseOptions fundamental_options;
    fundamental_options.max_error = inlier_threshold_px;
    fundamental_options.ransac.max_iterations = max_iterations;
    fundamental_options.ransac.min_iterations = min_iterations;
    Eigen::Matrix3d F;
    poselib::RansacStats stats = poselib::estimate_fundamental(points0, points1, fundamental_options, &F, &inliers);
    if (stats.num_inliers >= min_inliers)
    {
        std::vector<float> errors(matches.size());
        for (size_t m = 0; m < matches.size(); ++m)
            errors[m] = static_cast<float>(sampson_error(F, points0[m], points1[m]));
        return {stats.num_inliers, "fundamental", std::move(inliers), std::move(errors)};
    }

    poselib::HomographyOptions homography_options;
    homography_options.max_error = inlier_threshold_px;
    homography_options.ransac.max_iterations = max_iterations;
    homography_options.ransac.min_iterations = min_iterations;
    Eigen::Matrix3d H;
    inliers.clear();
    stats = poselib::estimate_homography(points0, points1, homography_options, &H, &inliers);
    if (stats.num_inliers >= min_inliers)
    {
        std::vector<float> errors(matches.size());
        for (size_t m = 0; m < matches.size(); ++m)
            errors[m] = static_cast<float>(transfer_error(H, points0[m], points1[m]));
        return {stats.num_inliers, "homography", std::move(inliers), std::move(errors)};
    }
    return none;
}

} // namespace

int main(int argc, char** argv)
{
    Arguments arguments;
    try
    {
        arguments = parse_arguments(std::vector<std::string>(argv + 1, argv + argc));
    }
    catch (const std::invalid_argument& e)
    {
        std::fprintf(stderr, "feature_benchmark: %s\n", e.what());
        print_usage();
        return 2;
    }

    try
    {
        // The runs: every --features name as the settings configure it, then every --variant
        std::vector<RunSpec> specs;
        for (const std::string& name : arguments.features)
            specs.push_back({name, name, "", 0, false, -1.0f, -1});
        specs.insert(specs.end(), arguments.variants.begin(), arguments.variants.end());
        std::vector<FeatureType> feature_types;
        for (size_t f = 0; f < specs.size(); ++f)
        {
            feature_types.push_back(to_feature_type(specs[f].feature));
            for (size_t g = 0; g < f; ++g)
                if (specs[g].label == specs[f].label)
                    throw std::runtime_error("two runs are called '" + specs[f].label + "': give each --variant its own label");
        }

        const std::vector<ImageEntry> images = load_images(arguments.rgb_csv, arguments.sequence_path, arguments.cam);
        if (images.empty())
            throw std::runtime_error(arguments.rgb_csv + " lists no images");
        size_t missing = 0;
        for (const ImageEntry& image : images)
            if (!std::filesystem::exists(image.path))
                ++missing;

        std::printf("feature_benchmark\n");
        std::printf("  rgb csv:       %s (camera %s)\n", arguments.rgb_csv.c_str(), arguments.cam.c_str());
        std::printf("  sequence path: %s\n", arguments.sequence_path.c_str());
        std::printf("  images:        %zu, %.3f s from the first to the last\n", images.size(),
                    images.back().timestamp - images.front().timestamp);
        std::printf("    first: %s\n", images.front().path.c_str());
        std::printf("    last:  %s\n", images.back().path.c_str());
        std::printf("  features:      %zu\n", specs.size());
        for (size_t f = 0; f < specs.size(); ++f)
        {
            const AF_VSLAM::Feature& feature = get_feature(feature_types[f]);   // ends the process if compiled out
            std::printf("    %-22s settings %s", specs[f].label.c_str(), feature.get_settings_yaml_file().c_str());
            if (!specs[f].model.empty())
                std::printf(", model %s", specs[f].model.c_str());
            if (specs[f].max_keypoints > 0)
                std::printf(", max keypoints %d", specs[f].max_keypoints);
            if (specs[f].bruteforce)
                std::printf(", brute-force matcher");
            if (specs[f].threshold >= 0.0f)
                std::printf(", threshold %g", static_cast<double>(specs[f].threshold));
            if (specs[f].nms_radius >= 0)
                std::printf(", NMS radius %d", specs[f].nms_radius);
            std::printf("\n");
        }
        if (missing > 0)
            throw std::runtime_error(std::to_string(missing) + " of " + std::to_string(images.size()) +
                                     " image files do not exist under " + arguments.sequence_path);

        std::filesystem::create_directories(arguments.output_dir);
        std::printf("  output dir:    %s\n", arguments.output_dir.c_str());

        // 1-2. The visited images (1 of every `step`) and their MegaLoc similarity matrix. An unreadable image is
        //      left out here and so takes part in no pair. The embedder lives only inside this block.
        std::vector<size_t> visited;        // indices into `images`, ascending
        Eigen::MatrixXf similarity;         // [visited, visited] cosine of the MegaLoc descriptors
        int image_width = 0, image_height = 0;   // of the first visited image, as it is on disk (the matchers' size)
        {
            if (!std::filesystem::exists(arguments.megaloc_onnx))
                throw std::runtime_error(arguments.megaloc_onnx + " not found: run from the repository root, or pass --megaloc-onnx");
            placecell::MegaLocEmbedder embedder(arguments.megaloc_onnx);
            const size_t expected = (images.size() + arguments.step - 1) / arguments.step;
            Eigen::MatrixXf descriptors(static_cast<Eigen::Index>(expected), embedder.descriptor_dim());   // one row per image
            size_t unreadable = 0;
            const auto t0 = Clock::now();
            for (size_t i = 0; i < images.size(); i += arguments.step)
            {
                const cv::Mat bgr = cv::imread(images[i].path, cv::IMREAD_COLOR);
                if (bgr.empty())
                {
                    ++unreadable;
                    continue;
                }
                if (visited.empty())
                {
                    image_width = bgr.cols;
                    image_height = bgr.rows;
                }
                descriptors.row(static_cast<Eigen::Index>(visited.size())) = embedder.embed(bgr).transpose();
                visited.push_back(i);
            }
            if (visited.size() < 2)
                throw std::runtime_error("fewer than 2 readable images to visit");
            descriptors.conservativeResize(static_cast<Eigen::Index>(visited.size()), Eigen::NoChange);
            similarity = descriptors * descriptors.transpose();   // the descriptors are L2-normalised
            std::printf("  vpr:           MegaLoc (%s, onnxruntime/%s), 1 of every %zu images: %zu images in %.1f s",
                        arguments.megaloc_onnx.c_str(), embedder.provider().c_str(), arguments.step, visited.size(),
                        ms_since(t0) / 1000.0);
            if (unreadable > 0)
                std::printf(", %zu unreadable images skipped", unreadable);
            std::printf("\n");
        }
        const std::filesystem::path output_dir(arguments.output_dir);
        placecell::save_npy((output_dir / "vpr_megaloc.npy").string(), similarity);
        {
            std::ofstream rows(output_dir / "vpr_megaloc_images.csv");
            rows << "row,index,image\n";
            for (size_t r = 0; r < visited.size(); ++r)
                rows << r << ',' << visited[r] << ',' << images[visited[r]].name << '\n';
        }

        // 3-4. The candidate pairs (similarity above the threshold) and the random draw, without repetition:
        //      shuffle the candidates and keep the first `pairs`.
        struct Pair { size_t a, b; };   // rows of `visited`, a < b
        std::vector<Pair> pairs;
        for (size_t a = 0; a < visited.size(); ++a)
            for (size_t b = a + 1; b < visited.size(); ++b)
                if (similarity(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b)) > arguments.vpr_threshold)
                    pairs.push_back({a, b});
        const size_t all_pairs = visited.size() * (visited.size() - 1) / 2, candidates = pairs.size();
        if (candidates == 0)
            throw std::runtime_error("no pair of visited images has a MegaLoc similarity above the threshold");
        std::mt19937 generator(arguments.seed);
        std::shuffle(pairs.begin(), pairs.end(), generator);
        if (pairs.size() > arguments.pairs)
            pairs.resize(arguments.pairs);
        std::printf("  pairs:         %zu of the %zu possible are above similarity %.2f; %zu drawn at random (seed %u)%s\n",
                    candidates, all_pairs, static_cast<double>(arguments.vpr_threshold), pairs.size(), arguments.seed,
                    pairs.size() < arguments.pairs ? " - every candidate, fewer than --pairs" : "");
        std::printf("  image size:    %dx%d\n", image_width, image_height);
        std::fflush(stdout);

        // Everything one feature needs for the whole run: its extractor and matcher, its csv, its totals, and the
        // keypoints of the images extracted so far (by row of `visited`; filled the first time a pair needs the image)
        struct Extracted
        {
            double extract_ms;
            AF_VSLAM::KeypointData data;
        };
        struct FeatureRun
        {
            FeatureType type;
            std::string name, csv_path;
            std::shared_ptr<AF_VSLAM::FeatureExtractor> extractor;
            std::shared_ptr<AF_VSLAM::FeatureMatcher> matcher;
            std::ofstream csv;
            std::ofstream distances;   // <label>_distances.csv: one row per match
            bool bruteforce{false};    // match with the cross-checked brute force (a variant's matcher=bruteforce)
            std::vector<std::optional<Extracted>> cache;
            FeatureStats stats;
        };
        std::vector<FeatureRun> runs(feature_types.size());
        for (size_t f = 0; f < feature_types.size(); ++f)
        {
            FeatureRun& run = runs[f];
            run.type = feature_types[f];
            run.name = specs[f].label;
            run.bruteforce = specs[f].bruteforce;
            run.extractor = build_extractor(run.type, specs[f], arguments.output_dir);
            run.matcher = build_matcher(run.type, image_width, image_height);
            run.cache.resize(visited.size());
            run.csv_path = (output_dir / (run.name + ".csv")).string();
            run.csv.open(run.csv_path);
            if (!run.csv)
                throw std::runtime_error("cannot write " + run.csv_path);
            const std::string distances_path = (output_dir / (run.name + "_distances.csv")).string();
            run.distances.open(distances_path);
            if (!run.distances)
                throw std::runtime_error("cannot write " + distances_path);
            run.distances << "distance,correct,error_px\n";
            run.csv << "image_0,image_1,index_0,index_1,vpr_similarity,keypoints_0,keypoints_1,matches,matched_percent,"
                       "correct_matches,correct_percent,geometry,extract_ms_0,extract_ms_1,match_ms,geometry_ms\n";
        }

        // The benchmark loop: every drawn pair, every feature on it
        for (const Pair& pair : pairs)
        {
            const ImageEntry& image0 = images[visited[pair.a]];
            const ImageEntry& image1 = images[visited[pair.b]];
            const float pair_similarity = similarity(static_cast<Eigen::Index>(pair.a), static_cast<Eigen::Index>(pair.b));
            for (FeatureRun& run : runs)
            {
                // The pair's two images, extracted now if no earlier pair needed them
                const auto extracted = [&](const size_t row, const ImageEntry& image) -> const Extracted&
                {
                    std::optional<Extracted>& slot = run.cache[row];
                    if (!slot)
                    {
                        slot.emplace();
                        if (!extract(image.path, *run.extractor, slot->data, slot->extract_ms))
                            throw std::runtime_error("cannot read " + image.path);
                        ++run.stats.images;
                        run.stats.keypoints += slot->data.count();
                        run.stats.extract_ms += slot->extract_ms;
                    }
                    return *slot;
                };
                const Extracted& first = extracted(pair.a, image0);
                const Extracted& second = extracted(pair.b, image1);

                const auto t0 = Clock::now();
                const std::vector<cv::DMatch> matches = run.bruteforce
                    ? run.matcher->match_descriptors_only(first.data.descriptors, second.data.descriptors, run.type)
                    : run.matcher->match_descriptors(first.data.descriptors, second.data.descriptors,
                                                     first.data.keypoints, second.data.keypoints, run.type);
                const double match_ms = ms_since(t0);

                const auto t1 = Clock::now();
                const GeometryCheck geometry = check_geometry(matches, first.data.keypoints, second.data.keypoints);
                const double geometry_ms = ms_since(t1);

                // The descriptor distance of every match, in the feature's own metric, with the check's verdict
                // (outside the timed stages)
                const AF_VSLAM::Feature& feature = get_feature(run.type);
                for (size_t m = 0; m < matches.size(); ++m)
                {
                    const float distance = feature.descriptor_distance(first.data.descriptors.row(matches[m].queryIdx),
                                                                       second.data.descriptors.row(matches[m].trainIdx));
                    char row[64];
                    std::snprintf(row, sizeof(row), "%.4f,%d,%.3f\n", static_cast<double>(distance),
                                  geometry.inliers[m] ? 1 : 0, static_cast<double>(geometry.errors[m]));
                    run.distances << row;
                }

                run.stats.match_ms += match_ms;
                run.stats.geometry_ms += geometry_ms;
                run.stats.matches += matches.size();
                run.stats.correct += geometry.correct;
                ++run.stats.pairs;

                const size_t max_matches = std::min(first.data.count(), second.data.count());
                const double matched_percent =
                    max_matches > 0 ? 100.0 * static_cast<double>(matches.size()) / static_cast<double>(max_matches) : 0.0;
                const double correct_percent =
                    matches.empty() ? 0.0 : 100.0 * static_cast<double>(geometry.correct) / static_cast<double>(matches.size());
                char similarity_text[32], percents[64], times[96];
                std::snprintf(similarity_text, sizeof(similarity_text), "%.4f", static_cast<double>(pair_similarity));
                std::snprintf(percents, sizeof(percents), "%.2f,%zu,%.2f", matched_percent, geometry.correct, correct_percent);
                std::snprintf(times, sizeof(times), "%.3f,%.3f,%.3f,%.3f", first.extract_ms, second.extract_ms, match_ms,
                              geometry_ms);
                run.csv << image0.name << ',' << image1.name << ',' << visited[pair.a] << ',' << visited[pair.b] << ','
                        << similarity_text << ',' << first.data.count() << ',' << second.data.count() << ','
                        << matches.size() << ',' << percents << ',' << geometry.model << ',' << times << '\n';
            }
        }

        for (FeatureRun& run : runs)
        {
            run.csv.close();
            run.distances.close();
            if (!run.csv || !run.distances)
                throw std::runtime_error("writing " + run.csv_path + " or its _distances.csv failed");
            const FeatureStats& stats = run.stats;
            const double n_images = static_cast<double>(std::max<size_t>(stats.images, 1));
            const double n_pairs = static_cast<double>(std::max<size_t>(stats.pairs, 1));
            std::printf("    %-22s %zu images, %zu pairs | mean keypoints %.0f, mean matches %.0f, mean correct %.0f | "
                        "extract %.1f ms/image, match %.1f ms/pair, geometry %.1f ms/pair | %s\n",
                        run.name.c_str(), stats.images, stats.pairs, static_cast<double>(stats.keypoints) / n_images,
                        static_cast<double>(stats.matches) / n_pairs, static_cast<double>(stats.correct) / n_pairs,
                        stats.extract_ms / n_images, stats.match_ms / n_pairs, stats.geometry_ms / n_pairs,
                        run.csv_path.c_str());
        }
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "feature_benchmark: %s\n", e.what());
        return 1;
    }
    return 0;
}
