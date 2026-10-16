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
 *   ./bin/test_feature_parity --model feature_models/superpoint_v1.onnx \
 *       --image <frame.png> --reference feature_models/reference/<frame>.superpoint.txt \
 *       [--provider auto|cuda|coreml|cpu] [--threads 4] [--iters 10] [--max-keypoints 1000]
 */

#include "onnx/SuperPointOnnx.h"

#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
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
    if (model.empty() || image_path.empty() || reference_path.empty())
    {
        std::fprintf(stderr, "usage: test_feature_parity --model <onnx> --image <png> --reference <txt> "
                             "[--provider auto] [--threads 4] [--iters 10] [--max-keypoints 1000] "
                             "[--min-fraction 0.99] [--min-cos 0.999]\n");
        return 2;
    }
    AF_VSLAM::onnx::OrtSettings settings;
    settings.provider = arg(args, "--provider", "auto");
    settings.threads = std::stoi(arg(args, "--threads", "4"));
    const int iters = std::stoi(arg(args, "--iters", "10"));
    const int max_keypoints = std::stoi(arg(args, "--max-keypoints", "1000"));
    const double min_fraction = std::stod(arg(args, "--min-fraction", "0.99"));
    const double min_cos = std::stod(arg(args, "--min-cos", "0.999"));

    const std::string kind = YAML::LoadFile(model + ".yaml")["model"].as<std::string>();
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
    std::fprintf(stderr, "model kind '%s' not supported by this harness yet\n", kind.c_str());
    return 2;
}
