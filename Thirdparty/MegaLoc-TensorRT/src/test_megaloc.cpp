/**
 * Module: AllFeature-VSLAM - MegaLoc-TensorRT - test_megaloc.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.0
 * - Created: 2026-08-28
 * - License: GPLv3 License
 *
 * Standalone harness for the MegaLoc TensorRT module: builds/loads the engine, embeds
 * every image given, times the steady state, prints the pairwise cosine-similarity
 * matrix and - when the export script's self-check output is supplied - the cosine
 * between each TensorRT descriptor and the PyTorch reference descriptor of the same
 * image (README preprocessing). That last number is the end-to-end fidelity check of
 * the ONNX export + fp16 engine + C++ preprocessing chain.
 *
 * Usage:
 *   test_megaloc <model.onnx> <image> [<image> ...] [--precision fp16|fp32]
 *                [--reference <model>_reference.txt] [--iters N]
 */

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "tensorrt_megaloc.hpp"

namespace
{
double msSince(const std::chrono::steady_clock::time_point& t0)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
        .count();
}

/** '<path> v0 v1 ...' per line, '#' comments (written by export_megaloc.py). */
std::map<std::string, std::vector<float>> readReference(const std::string& path)
{
    std::map<std::string, std::vector<float>> ref;
    std::ifstream in(path);
    if (!in)
    {
        std::printf("cannot read reference file %s\n", path.c_str());
        return ref;
    }
    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream iss(line);
        std::string imagePath;
        iss >> imagePath;
        std::vector<float> d;
        float v;
        while (iss >> v)
            d.push_back(v);
        ref[imagePath] = std::move(d);
    }
    return ref;
}

std::string baseName(const std::string& path)
{
    const auto pos = path.find_last_of('/');
    return pos == std::string::npos ? path : path.substr(pos + 1);
}
} // namespace

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s <model.onnx> <image> [<image> ...] [--precision fp16|fp32] "
                    "[--reference <file>] [--iters N]\n",
                    argv[0]);
        return 1;
    }
    const std::string onnxPath = argv[1];
    std::string precision = "fp16";
    std::string referencePath;
    int iters = 50;
    std::vector<std::string> images;
    for (int i = 2; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--precision") == 0 && i + 1 < argc)
            precision = argv[++i];
        else if (std::strcmp(argv[i], "--reference") == 0 && i + 1 < argc)
            referencePath = argv[++i];
        else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc)
            iters = std::atoi(argv[++i]);
        else
            images.emplace_back(argv[i]);
    }

    auto t0 = std::chrono::steady_clock::now();
    megaloc::TensorRTMegaLoc net(onnxPath, precision);
    std::printf("engine %s in %.0f ms: %s\nIO tensors:\n%s", net.loadedFromCache() ? "loaded" : "built",
                msSince(t0), net.enginePath().c_str(), net.describeIOTensors().c_str());

    std::vector<std::vector<float>> descs;
    std::vector<std::string> names;
    for (const auto& path : images)
    {
        const cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
        if (img.empty())
        {
            std::printf("cannot read image %s\n", path.c_str());
            return 1;
        }
        t0 = std::chrono::steady_clock::now();
        descs.push_back(net.infer(img));
        std::printf("%s: %dx%dx%d -> %zu-d in %.1f ms%s\n", baseName(path).c_str(), img.cols,
                    img.rows, img.channels(), descs.back().size(), msSince(t0),
                    descs.size() == 1 ? " (first call includes CUDA/TRT warmup)" : "");
        names.push_back(path);
    }

    if (iters > 0)
    {
        const cv::Mat img = cv::imread(images.front(), cv::IMREAD_UNCHANGED);
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i)
            (void)net.infer(img);
        std::printf("steady state: %.2f ms/image over %d iters\n", msSince(t0) / iters, iters);
    }

    if (descs.size() > 1)
    {
        std::printf("pairwise cosine similarity (TensorRT %s):\n", precision.c_str());
        for (size_t i = 0; i < descs.size(); ++i)
        {
            std::printf("  ");
            for (size_t j = 0; j < descs.size(); ++j)
                std::printf(" %6.3f", megaloc::TensorRTMegaLoc::cosine(descs[i], descs[j]));
            std::printf("   %s\n", baseName(names[i]).c_str());
        }
    }

    if (!referencePath.empty())
    {
        const auto ref = readReference(referencePath);
        std::printf("cosine(TensorRT, PyTorch reference) per image:\n");
        float worst = 1.0f;
        for (size_t i = 0; i < descs.size(); ++i)
        {
            // match by basename so the reference can have been written from another cwd
            const std::vector<float>* r = nullptr;
            for (const auto& [p, d] : ref)
                if (baseName(p) == baseName(names[i]))
                    r = &d;
            if (r == nullptr)
            {
                std::printf("  %s: not in reference file\n", baseName(names[i]).c_str());
                continue;
            }
            const float c = megaloc::TensorRTMegaLoc::cosine(descs[i], *r);
            worst = std::min(worst, c);
            std::printf("  %s: %.5f\n", baseName(names[i]).c_str(), c);
        }
        std::printf("worst: %.5f\n", worst);
    }
    return 0;
}
