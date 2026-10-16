/**
 * test_feature_parity - parity + timing harness for the learned feature models on ONNX Runtime
 * (docs/plans/learned_features_onnxruntime.md, test T2).
 *
 * Runs one exported model on one frame (or pair) and compares the C++ output against the upstream PyTorch
 * reference that tools/export_features.py dumped into feature_models/reference/: keypoints matched to the
 * reference by nearest neighbour (fraction within 0.5 px, max position difference), descriptor cosine over the
 * matched pairs, score difference, and the wall time of the graph and of graph + tail over --iters calls.
 * Exit code 0 = the T2 thresholds hold (--min-fraction, --min-cos; the Core ML provider is measured with
 * looser ones), 1 = they do not, 2 = usage / unsupported model.
 *
 *   detector:  ./bin/test_feature_parity --model feature_models/superpoint_v1.onnx \
 *                  --image <frame.png> --reference feature_models/reference/<frame>.superpoint.txt
 *              (same with feature_models/aliked_n16.onnx and <frame>.aliked.txt)
 *   sift:      ./bin/test_feature_parity --model sift --image <frame.png> --reference <frame>.sift.txt
 *              (cv::SIFT on the CPU, no ONNX file; the reference is cv2.SIFT through lightglue/sift.py)
 *   matcher:   ./bin/test_feature_parity --model feature_models/lightglue_superpoint.onnx \
 *                  --reference0 <frame0>.superpoint.txt --reference1 <frame1>.superpoint.txt \
 *                  --matches <frame0>__<frame1>.lightglue_superpoint.txt
 *   common:    [--provider auto|cuda|coreml|cpu] [--threads 4] [--coreml-units ALL|CPUAndGPU|CPUAndNeuralEngine|CPUOnly]
 *              [--iters 10] [--max-keypoints 1000] [--min-fraction 0.99] [--min-cos 0.999]
 *
 * The matcher mode feeds the matcher with the reference detector outputs (the extractor's parity is measured
 * separately), so a matcher failure is the matcher's alone.
 */

#include "SiftCpu.h"
#include "onnx/AlikedOnnx.h"
#include "onnx/LightGlueOnnx.h"
#include "onnx/SuperPointOnnx.h"

#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace
{

struct Reference
{
    int n{0};
    int width{0}, height{0};   // 'size=WxH' of the header
    std::vector<std::string> columns;
    std::vector<std::vector<float>> rows;   // one per keypoint, in column order

    [[nodiscard]] int col(const std::string& name) const
    {
        const auto it = std::find(columns.begin(), columns.end(), name);
        return it == columns.end() ? -1 : static_cast<int>(it - columns.begin());
    }
};

/// The text dump of tools/export_features.py write_references: a '# ... columns: x y score ...' header, then rows.
Reference load_reference(const std::string& path)
{
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("cannot open reference " + path);
    Reference ref;
    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty())
            continue;
        if (line[0] == '#')
        {
            const auto size_pos = line.find("size=");
            if (size_pos != std::string::npos)
                std::sscanf(line.c_str() + size_pos, "size=%dx%d", &ref.width, &ref.height);
            const auto pos = line.find("columns:");
            if (pos != std::string::npos)
            {
                std::istringstream cols(line.substr(pos + 8));
                std::string c;
                while (cols >> c)
                    ref.columns.push_back(c);
            }
            continue;
        }
        std::istringstream values(line);
        std::vector<float> row;
        float v;
        while (values >> v)
            row.push_back(v);
        ref.rows.push_back(std::move(row));
    }
    ref.n = static_cast<int>(ref.rows.size());
    if (ref.columns.empty() || ref.n == 0)
        throw std::runtime_error("reference " + path + " has no columns header or no rows");
    return ref;
}

/// A detector reference as the matcher's input: cv::KeyPoints (scale -> size, ori -> angle in degrees when the
/// columns exist) and a [N, D] CV_32F descriptor matrix.
void reference_to_features(const Reference& ref, std::vector<cv::KeyPoint>& kps, cv::Mat& desc)
{
    const int cx = ref.col("x"), cy = ref.col("y"), cs = ref.col("score"), cd = ref.col("d0");
    const int cscale = ref.col("scale"), cori = ref.col("ori");
    const int dim = static_cast<int>(ref.columns.size()) - cd;
    kps.clear();
    kps.reserve(ref.n);
    desc.create(ref.n, dim, CV_32F);
    for (int i = 0; i < ref.n; ++i)
    {
        cv::KeyPoint kp(ref.rows[i][cx], ref.rows[i][cy], 1.0f);
        kp.response = ref.rows[i][cs];
        if (cscale >= 0)
            kp.size = ref.rows[i][cscale];
        if (cori >= 0)
            kp.angle = ref.rows[i][cori] * static_cast<float>(180.0 / CV_PI);
        kp.class_id = i;
        kps.push_back(kp);
        std::copy_n(ref.rows[i].begin() + cd, dim, desc.ptr<float>(i));
    }
}

/// The match list dump: rows 'i0 i1 score'.
struct MatchReference
{
    std::vector<std::array<int, 2>> pairs;
    std::vector<float> scores;
};

MatchReference load_matches(const std::string& path)
{
    const Reference raw = load_reference(path);
    MatchReference m;
    const int c0 = raw.col("i0"), c1 = raw.col("i1"), cs = raw.col("score");
    for (const auto& row : raw.rows)
    {
        m.pairs.push_back({static_cast<int>(row[c0]), static_cast<int>(row[c1])});
        m.scores.push_back(row[cs]);
    }
    return m;
}

std::string arg(const std::vector<std::string>& args, const std::string& key, const std::string& fallback)
{
    for (size_t i = 0; i + 1 < args.size(); ++i)
        if (args[i] == key)
            return args[i + 1];
    return fallback;
}

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

/// Keypoints + descriptors of the C++ side against the reference rows: nearest neighbour in the reference for
/// every detected keypoint; a hit is within `radius` px.
struct DetectorParity
{
    int n_ref{0}, n_got{0}, hits{0};
    double max_dxy{0.0}, max_dscore{0.0}, min_cos{1.0};
};

DetectorParity compare_detector(const Reference& ref, const std::vector<cv::KeyPoint>& kps, const cv::Mat& desc,
                                double radius)
{
    DetectorParity p;
    p.n_ref = ref.n;
    p.n_got = static_cast<int>(kps.size());
    const int cx = ref.col("x"), cy = ref.col("y"), cs = ref.col("score"), cd = ref.col("d0");
    const int dim = desc.cols;
    for (int i = 0; i < p.n_got; ++i)
    {
        int best = -1;
        double best_d2 = std::numeric_limits<double>::max();
        for (int j = 0; j < ref.n; ++j)
        {
            const double dx = ref.rows[j][cx] - kps[i].pt.x, dy = ref.rows[j][cy] - kps[i].pt.y;
            const double d2 = dx * dx + dy * dy;
            if (d2 < best_d2)
            {
                best_d2 = d2;
                best = j;
            }
        }
        const double d = std::sqrt(best_d2);
        if (best < 0 || d >= radius)
            continue;
        p.hits++;
        p.max_dxy = std::max(p.max_dxy, d);
        p.max_dscore = std::max(p.max_dscore, std::abs(static_cast<double>(ref.rows[best][cs]) - kps[i].response));
        double dot = 0.0, na = 0.0, nb = 0.0;
        const float* row = desc.ptr<float>(i);
        for (int c = 0; c < dim; ++c)
        {
            const double a = ref.rows[best][cd + c], b = row[c];
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        p.min_cos = std::min(p.min_cos, dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12));
    }
    return p;
}

} // namespace

int main(int argc, char** argv)
{
    const std::vector<std::string> args(argv + 1, argv + argc);
    const std::string model = arg(args, "--model", "");
    const std::string image_path = arg(args, "--image", "");
    const std::string reference_path = arg(args, "--reference", "");
    if (model.empty())
    {
        std::fprintf(stderr, "usage: test_feature_parity --model <onnx> (--image <png> --reference <txt> | "
                             "--reference0 <txt> --reference1 <txt> --matches <txt>) [--provider auto] [--threads 4] "
                             "[--iters 10] [--max-keypoints 1000] [--min-fraction 0.99] [--min-cos 0.999]\n");
        return 2;
    }
    AF_VSLAM::onnx::OrtSettings settings;
    settings.provider = arg(args, "--provider", "auto");
    settings.threads = std::stoi(arg(args, "--threads", "4"));
    settings.coreml_compute_units = arg(args, "--coreml-units", "ALL");   // ALL | CPUAndGPU | CPUAndNeuralEngine | CPUOnly
    const int iters = std::stoi(arg(args, "--iters", "10"));
    const int max_keypoints = std::stoi(arg(args, "--max-keypoints", "1000"));
    const double min_fraction = std::stod(arg(args, "--min-fraction", "0.99"));
    const double min_cos = std::stod(arg(args, "--min-cos", "0.999"));

    const std::string kind = model == "sift" ? "sift" : YAML::LoadFile(model + ".yaml")["model"].as<std::string>();

    if (kind == "lightglue")
    {
        const std::string ref0_path = arg(args, "--reference0", ""), ref1_path = arg(args, "--reference1", "");
        const std::string matches_path = arg(args, "--matches", "");
        if (ref0_path.empty() || ref1_path.empty() || matches_path.empty())
        {
            std::fprintf(stderr, "matcher mode needs --reference0 --reference1 --matches\n");
            return 2;
        }
        const Reference ref0 = load_reference(ref0_path), ref1 = load_reference(ref1_path);
        const MatchReference ref_matches = load_matches(matches_path);
        std::vector<cv::KeyPoint> kps0, kps1;
        cv::Mat desc0, desc1;
        reference_to_features(ref0, kps0, desc0);
        reference_to_features(ref1, kps1, desc1);

        AF_VSLAM::onnx::LightGlueOnnx lg(model, settings);
        std::vector<cv::DMatch> got = lg.match(kps0, desc0, kps1, desc1, ref0.width, ref0.height);
        std::vector<double> graph_ms, total_ms;
        for (int i = 0; i < iters; ++i)
        {
            lg.match(kps0, desc0, kps1, desc1, ref0.width, ref0.height);
            graph_ms.push_back(lg.last_graph_ms());
            total_ms.push_back(lg.last_total_ms());
        }
        // identical assignments: the (i0, i1) pairs present in both lists; score difference over those
        std::map<std::array<int, 2>, float> ref_map;
        for (size_t k = 0; k < ref_matches.pairs.size(); ++k)
            ref_map[ref_matches.pairs[k]] = ref_matches.scores[k];
        int identical = 0;
        double max_dscore = 0.0;
        for (const cv::DMatch& m : got)
        {
            const auto it = ref_map.find({m.queryIdx, m.trainIdx});
            if (it == ref_map.end())
                continue;
            identical++;
            max_dscore = std::max(max_dscore, std::abs(static_cast<double>(it->second) - (1.0 - m.distance)));
        }
        const int n_ref = static_cast<int>(ref_matches.pairs.size()), n_got = static_cast<int>(got.size());
        const double fraction = n_ref ? static_cast<double>(identical) / n_ref : 0.0;
        const bool pass = fraction >= min_fraction && std::abs(n_got - n_ref) <= std::max(1, n_ref / 100);
        std::printf("lightglue_%s %s (%s, %s weights) N0=%d N1=%d %dx%d\n", lg.features().c_str(), model.c_str(),
                    lg.provider().c_str(), lg.precision().c_str(), ref0.n, ref1.n, ref0.width, ref0.height);
        std::printf("  matches=%d (reference %d)  identical pairs=%d/%d (%.4f)  max|dscore|=%.2e\n", n_got, n_ref,
                    identical, n_ref, fraction, max_dscore);
        std::printf("  time over %d calls: graph median %.1f ms, graph+extraction median %.1f ms\n", iters,
                    median(graph_ms), median(total_ms));
        std::printf("%s\n", pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    }

    if (image_path.empty() || reference_path.empty())
    {
        std::fprintf(stderr, "detector mode needs --image and --reference\n");
        return 2;
    }
    const cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::fprintf(stderr, "cannot read %s\n", image_path.c_str());
        return 2;
    }
    const Reference ref = load_reference(reference_path);

    if (kind == "superpoint")
    {
        AF_VSLAM::onnx::SuperPointOnnx sp(model, settings);
        std::vector<cv::KeyPoint> kps;
        cv::Mat desc;
        sp.detect(image, max_keypoints, kps, desc);   // warm-up + the compared result
        std::vector<double> graph_ms, total_ms;
        for (int i = 0; i < iters; ++i)
        {
            std::vector<cv::KeyPoint> k;
            cv::Mat d;
            sp.detect(image, max_keypoints, k, d);
            graph_ms.push_back(sp.last_graph_ms());
            total_ms.push_back(sp.last_total_ms());
        }
        const DetectorParity p = compare_detector(ref, kps, desc, 0.5);
        const double fraction = p.n_got ? static_cast<double>(p.hits) / p.n_got : 0.0;
        const bool count_ok = std::abs(p.n_got - p.n_ref) <= std::max(1, p.n_ref / 100);
        const bool pass = count_ok && fraction >= min_fraction && p.min_cos >= min_cos;
        std::printf("superpoint %s (%s, %s weights) %dx%d\n", model.c_str(), sp.provider().c_str(),
                    sp.precision().c_str(), image.cols, image.rows);
        std::printf("  N=%d (reference %d)  within 0.5px: %d/%d (%.4f)  max|dxy| matched=%.2e px  "
                    "max|dscore|=%.2e  min cos=%.7f\n",
                    p.n_got, p.n_ref, p.hits, p.n_got, fraction, p.max_dxy, p.max_dscore, p.min_cos);
        std::printf("  time over %d calls: graph median %.1f ms, graph+tail median %.1f ms\n", iters,
                    median(graph_ms), median(total_ms));
        std::printf("%s\n", pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    }
    if (kind == "aliked")
    {
        AF_VSLAM::onnx::AlikedOnnx aliked(model, settings);
        std::vector<cv::KeyPoint> kps;
        cv::Mat desc;
        aliked.detect(image, kps, desc);
        std::vector<double> graph_ms, total_ms;
        for (int i = 0; i < iters; ++i)
        {
            std::vector<cv::KeyPoint> k;
            cv::Mat d;
            aliked.detect(image, k, d);
            graph_ms.push_back(aliked.last_graph_ms());
            total_ms.push_back(aliked.last_total_ms());
        }
        const DetectorParity p = compare_detector(ref, kps, desc, 0.5);
        const double fraction = p.n_got ? static_cast<double>(p.hits) / p.n_got : 0.0;
        const bool count_ok = std::abs(p.n_got - p.n_ref) <= std::max(1, p.n_ref / 100);
        const bool pass = count_ok && fraction >= min_fraction && p.min_cos >= min_cos;
        std::printf("aliked %s (%s, %s weights) %dx%d\n", model.c_str(), aliked.provider().c_str(),
                    aliked.precision().c_str(), image.cols, image.rows);
        std::printf("  N=%d (reference %d)  within 0.5px: %d/%d (%.4f)  max|dxy| matched=%.2e px  "
                    "max|dscore|=%.2e  min cos=%.7f\n",
                    p.n_got, p.n_ref, p.hits, p.n_got, fraction, p.max_dxy, p.max_dscore, p.min_cos);
        std::printf("  time over %d calls: graph median %.1f ms, graph+conversion median %.1f ms\n", iters,
                    median(graph_ms), median(total_ms));
        std::printf("%s\n", pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    }
    if (kind == "sift")
    {
        AF_VSLAM::SiftCpu sift(max_keypoints);
        std::vector<cv::KeyPoint> kps;
        cv::Mat desc;
        sift.detect(image, kps, desc);
        std::vector<double> total_ms;
        for (int i = 0; i < iters; ++i)
        {
            std::vector<cv::KeyPoint> k;
            cv::Mat d;
            sift.detect(image, k, d);
            total_ms.push_back(sift.last_total_ms());
        }
        const DetectorParity p = compare_detector(ref, kps, desc, 0.5);
        // scale and orientation parity over the matched pairs (the LightGlue-SIFT inputs)
        const int cx = ref.col("x"), cy = ref.col("y"), cscale = ref.col("scale"), cori = ref.col("ori");
        double max_dscale = 0.0, max_dori = 0.0;
        for (const cv::KeyPoint& k : kps)
            for (int j = 0; j < ref.n; ++j)
                if (std::abs(ref.rows[j][cx] - k.pt.x) < 1e-3 && std::abs(ref.rows[j][cy] - k.pt.y) < 1e-3)
                {
                    max_dscale = std::max(max_dscale, std::abs(static_cast<double>(ref.rows[j][cscale]) - k.size));
                    max_dori = std::max(max_dori, std::abs(static_cast<double>(ref.rows[j][cori]) - k.angle * CV_PI / 180.0));
                }
        const double fraction = p.n_got ? static_cast<double>(p.hits) / p.n_got : 0.0;
        const bool count_ok = std::abs(p.n_got - p.n_ref) <= std::max(1, p.n_ref / 100);
        const bool pass = count_ok && fraction >= min_fraction && p.min_cos >= min_cos;
        std::printf("sift cv::SIFT (cpu, nfeatures %d) %dx%d\n", max_keypoints, image.cols, image.rows);
        std::printf("  N=%d (reference %d)  within 0.5px: %d/%d (%.4f)  max|dxy| matched=%.2e px  "
                    "max|dscore|=%.2e  min cos=%.7f  max|dscale|=%.2e  max|dori|=%.2e rad\n",
                    p.n_got, p.n_ref, p.hits, p.n_got, fraction, p.max_dxy, p.max_dscore, p.min_cos, max_dscale,
                    max_dori);
        std::printf("  time over %d calls: median %.1f ms\n", iters, median(total_ms));
        std::printf("%s\n", pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    }
    std::fprintf(stderr, "model kind '%s' not supported by this harness yet\n", kind.c_str());
    return 2;
}
