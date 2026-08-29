/**
 * Module: AllFeature-VSLAM - MegaLoc-TensorRT - tensorrt_megaloc.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.0
 * - Created: 2026-08-28
 * - License: GPLv3 License (MegaLoc itself is MIT)
 */

#include "tensorrt_megaloc.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <NvOnnxParser.h>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

namespace megaloc
{

namespace
{
void cudaCheck(cudaError_t status, const char* what)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("TensorRTMegaLoc: ") + what + " failed: " +
                                 cudaGetErrorString(status));
}
} // namespace

void TensorRTMegaLoc::Logger::log(Severity severity, const char* msg) noexcept
{
    if (severity <= Severity::kWARNING)
        std::printf("[TensorRT] %s\n", msg);
}

TensorRTMegaLoc::TensorRTMegaLoc(const std::string& onnxPath, const std::string& precision,
                                 const std::string& sidecarYamlPath)
    : precision_(precision), enginePath_(onnxPath + "." + precision + ".engine")
{
    if (precision_ != "fp16" && precision_ != "fp32")
        throw std::runtime_error("TensorRTMegaLoc: precision must be fp16 or fp32, got " +
                                 precision_);

    loadSidecarYaml(sidecarYamlPath.empty() ? onnxPath + ".yaml" : sidecarYamlPath);

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_)
        throw std::runtime_error("TensorRTMegaLoc: createInferRuntime failed");

    if (tryLoadEngine(onnxPath))
    {
        loadedFromCache_ = true;
    }
    else
    {
        std::printf("[TensorRTMegaLoc] no usable engine cache, building from %s (takes 1-2 "
                    "minutes)...\n",
                    onnxPath.c_str());
        std::fflush(stdout);
        buildEngine(onnxPath);
        if (!tryLoadEngine(onnxPath))
            throw std::runtime_error("TensorRTMegaLoc: freshly built engine failed to load: " +
                                     enginePath_);
    }
    setupInference();
}

TensorRTMegaLoc::~TensorRTMegaLoc()
{
    if (dInput_ != nullptr)
        cudaFree(dInput_);
    if (dOutput_ != nullptr)
        cudaFree(dOutput_);
    if (stream_ != nullptr)
        cudaStreamDestroy(stream_);
}

void TensorRTMegaLoc::loadSidecarYaml(const std::string& yamlPath)
{
    YAML::Node root;
    try
    {
        root = YAML::LoadFile(yamlPath);
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("TensorRTMegaLoc: cannot read sidecar " + yamlPath + ": " +
                                 e.what());
    }
    for (int c = 0; c < 3; ++c)
    {
        mean_[c] = root["mean"][c].as<float>();
        std_[c] = root["std"][c].as<float>();
    }
    inputName_ = root["input_tensor"].as<std::string>();
    outputName_ = root["output_tensor"].as<std::string>();
    inputH_ = root["input_height"].as<int>();
    inputW_ = root["input_width"].as<int>();
    descriptorDim_ = root["descriptor_dim"].as<int>();
}

void TensorRTMegaLoc::buildEngine(const std::string& onnxPath)
{
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
    if (!builder)
        throw std::runtime_error("TensorRTMegaLoc: createInferBuilder failed");

    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));

    // The parser owns the weight memory it read; keep it alive until the build is done.
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger_));
    if (!parser->parseFromFile(onnxPath.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        std::ostringstream oss;
        oss << "TensorRTMegaLoc: ONNX parse failed for " << onnxPath;
        for (int i = 0; i < parser->getNbErrors(); ++i)
            oss << "\n  " << parser->getError(i)->desc();
        throw std::runtime_error(oss.str());
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    // Scratch memory TRT may use while TRYING kernel variants during the build; a ViT-B
    // with 8448-d output wants more headroom than the segmentation model.
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 2ULL << 30);
    if (precision_ == "fp16")
    {
        if (!builder->platformHasFastFp16())
            std::printf("[TensorRTMegaLoc] warning: no fast fp16 on this GPU, building anyway\n");
        config->setFlag(nvinfer1::BuilderFlag::kFP16);

        // Mixed precision: fp16 is allowed in the ViT backbone (the bulk of the compute),
        // everything after it - Sinkhorn optimal transport (exp/logsumexp over scores up
        // to ~20), the cluster aggregation and the 16640->8448 linear - stays fp32. A
        // uniformly-fp16 engine loses ~2-7% cosine against the fp32 reference on real
        // frames; pinning the aggregator recovers it at negligible cost. Layer names come
        // from the ONNX node names, which the export wrapper scopes as '/backbone/...'.
        config->setFlag(nvinfer1::BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
        int pinned = 0, backbone = 0;
        for (int i = 0; i < network->getNbLayers(); ++i)
        {
            nvinfer1::ILayer* layer = network->getLayer(i);
            const std::string name = layer->getName() ? layer->getName() : "";
            if (name.find("/backbone/") != std::string::npos)
            {
                ++backbone;
                continue;
            }
            if (layer->getNbOutputs() == 0 ||
                layer->getOutput(0)->getType() != nvinfer1::DataType::kFLOAT)
                continue; // shape/int layers carry no float precision
            if (layer->getType() == nvinfer1::LayerType::kCONSTANT ||
                layer->getType() == nvinfer1::LayerType::kSHAPE)
                continue;
            layer->setPrecision(nvinfer1::DataType::kFLOAT);
            ++pinned;
        }
        std::printf("[TensorRTMegaLoc] mixed precision: %d backbone layers fp16-eligible, "
                    "%d aggregator layers pinned to fp32 (of %d)\n",
                    backbone, pinned, network->getNbLayers());
    }

    auto blob = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!blob)
        throw std::runtime_error(
            "TensorRTMegaLoc: buildSerializedNetwork failed (see [TensorRT] log)");

    std::ofstream out(enginePath_, std::ios::binary);
    out.write(static_cast<const char*>(blob->data()), static_cast<std::streamsize>(blob->size()));
    if (!out)
        throw std::runtime_error("TensorRTMegaLoc: cannot write engine cache " + enginePath_);
    std::printf("[TensorRTMegaLoc] engine cached: %s (%.1f MB)\n", enginePath_.c_str(),
                static_cast<double>(blob->size()) / (1024.0 * 1024.0));
}

bool TensorRTMegaLoc::tryLoadEngine(const std::string& onnxPath)
{
    std::ifstream in(enginePath_, std::ios::binary | std::ios::ate);
    if (!in)
        return false; // no cache yet - normal on first run
    // A re-exported ONNX must not be served by the engine built from its predecessor
    // (the cache file name only encodes path + precision).
    std::error_code ec;
    const auto onnxTime = std::filesystem::last_write_time(onnxPath, ec);
    const auto engineTime = std::filesystem::last_write_time(enginePath_, ec);
    if (!ec && onnxTime > engineTime)
    {
        std::printf("[TensorRTMegaLoc] engine cache %s is older than the ONNX - rebuilding\n",
                    enginePath_.c_str());
        return false;
    }
    const auto size = static_cast<size_t>(in.tellg());
    std::vector<char> blob(size);
    in.seekg(0);
    in.read(blob.data(), static_cast<std::streamsize>(size));
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), size));
    return engine_ != nullptr;
}

void TensorRTMegaLoc::setupInference()
{
    context_.reset(engine_->createExecutionContext());
    if (!context_)
        throw std::runtime_error("TensorRTMegaLoc: createExecutionContext failed");

    const nvinfer1::Dims inDims = engine_->getTensorShape(inputName_.c_str());
    const nvinfer1::Dims outDims = engine_->getTensorShape(outputName_.c_str());
    if (inDims.nbDims != 4 || outDims.nbDims != 2)
        throw std::runtime_error("TensorRTMegaLoc: unexpected tensor ranks - wrong ONNX? " +
                                 describeIOTensors());
    if (inDims.d[2] != inputH_ || inDims.d[3] != inputW_)
        throw std::runtime_error("TensorRTMegaLoc: sidecar input size disagrees with engine: " +
                                 describeIOTensors());
    if (outDims.d[1] != descriptorDim_)
        throw std::runtime_error(
            "TensorRTMegaLoc: sidecar descriptor_dim disagrees with engine: " +
            describeIOTensors());
    if (engine_->getTensorDataType(outputName_.c_str()) != nvinfer1::DataType::kFLOAT)
        throw std::runtime_error("TensorRTMegaLoc: output is not float32 - wrong ONNX export?");

    const size_t nIn = 3ULL * inputH_ * inputW_;
    hostInput_.resize(nIn);
    hostOutput_.resize(static_cast<size_t>(descriptorDim_));
    cudaCheck(cudaStreamCreate(&stream_), "cudaStreamCreate");
    cudaCheck(cudaMalloc(&dInput_, nIn * sizeof(float)), "cudaMalloc(input)");
    cudaCheck(cudaMalloc(&dOutput_, hostOutput_.size() * sizeof(float)), "cudaMalloc(output)");

    if (!context_->setTensorAddress(inputName_.c_str(), dInput_) ||
        !context_->setTensorAddress(outputName_.c_str(), dOutput_))
        throw std::runtime_error("TensorRTMegaLoc: setTensorAddress failed (name mismatch?)");
}

std::vector<float> TensorRTMegaLoc::infer(const cv::Mat& imageBGR)
{
    // --- preprocess: must match preprocess_cv2() in export_megaloc.py ---
    // (3-channel BGR -> area resize -> RGB -> /255 -> (x-mean)/std -> CHW planar)
    cv::Mat bgr = imageBGR;
    if (imageBGR.channels() == 1)
        cv::cvtColor(imageBGR, bgr, cv::COLOR_GRAY2BGR);
    else if (imageBGR.channels() == 4)
        cv::cvtColor(imageBGR, bgr, cv::COLOR_BGRA2BGR);
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(inputW_, inputH_), 0, 0, cv::INTER_AREA);

    const int hw = inputH_ * inputW_;
    float* rPlane = hostInput_.data(); // CHW: R plane first (BGR->RGB swap)
    float* gPlane = hostInput_.data() + hw;
    float* bPlane = hostInput_.data() + 2 * hw;
    for (int y = 0; y < inputH_; ++y)
    {
        const cv::Vec3b* row = resized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < inputW_; ++x)
        {
            const int i = y * inputW_ + x;
            bPlane[i] = (row[x][0] / 255.0f - mean_[2]) / std_[2];
            gPlane[i] = (row[x][1] / 255.0f - mean_[1]) / std_[1];
            rPlane[i] = (row[x][2] / 255.0f - mean_[0]) / std_[0];
        }
    }

    // --- inference: H2D, enqueue, D2H, all ordered on one stream ---
    cudaCheck(cudaMemcpyAsync(dInput_, hostInput_.data(), hostInput_.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              "H2D copy");
    if (!context_->enqueueV3(stream_))
        throw std::runtime_error("TensorRTMegaLoc: enqueueV3 failed");
    cudaCheck(cudaMemcpyAsync(hostOutput_.data(), dOutput_, hostOutput_.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_),
              "D2H copy");
    cudaCheck(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");

    // --- postprocess: host-side re-normalisation (fp16 leaves |d| slightly off 1) ---
    std::vector<float> descriptor = hostOutput_;
    double sq = 0.0;
    for (const float v : descriptor)
        sq += static_cast<double>(v) * v;
    const float inv = sq > 0.0 ? static_cast<float>(1.0 / std::sqrt(sq)) : 0.0f;
    for (float& v : descriptor)
        v *= inv;
    return descriptor;
}

float TensorRTMegaLoc::cosine(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.empty() || a.size() != b.size())
        return 0.0f;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    const double den = std::sqrt(na) * std::sqrt(nb);
    return den > 0.0 ? static_cast<float>(dot / den) : 0.0f;
}

std::string TensorRTMegaLoc::describeIOTensors() const
{
    std::ostringstream oss;
    for (int i = 0; i < engine_->getNbIOTensors(); ++i)
    {
        const char* name = engine_->getIOTensorName(i);
        const nvinfer1::Dims dims = engine_->getTensorShape(name);
        oss << (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT ? "  input  "
                                                                                 : "  output ")
            << name << " [";
        for (int d = 0; d < dims.nbDims; ++d)
            oss << (d ? "," : "") << dims.d[d];
        oss << "] dtype=" << static_cast<int>(engine_->getTensorDataType(name)) << "\n";
    }
    return oss.str();
}

} // namespace megaloc
