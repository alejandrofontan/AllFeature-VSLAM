/**
 * Module: AllFeature-VSLAM - Segmentation-TensorRT - tensorrt_seg.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 0.3 (full inference: preprocess -> enqueueV3 -> binary mask)
 * - Created: 2026-08-17
 * - License: GPLv3 License
 */

#include "tensorrt_seg.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <NvOnnxParser.h>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

namespace segmentation
{

namespace
{
/** Every CUDA runtime call returns a status; unchecked failures surface later
 *  as inscrutable downstream errors, so fail loudly at the call site. */
void cudaCheck(cudaError_t status, const char* what)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("TensorRTSeg: ") + what + " failed: " +
                                 cudaGetErrorString(status));
}
} // namespace

void TensorRTSeg::Logger::log(Severity severity, const char* msg) noexcept
{
    if (severity <= Severity::kWARNING)
        std::printf("[TensorRT] %s\n", msg);
}

TensorRTSeg::TensorRTSeg(const std::string& onnxPath, const std::string& precision,
                         const std::string& classesYamlPath)
    : precision_(precision), enginePath_(onnxPath + "." + precision + ".engine")
{
    if (precision_ != "fp16" && precision_ != "fp32")
        throw std::runtime_error("TensorRTSeg: precision must be fp16 or fp32, got " + precision_);

    loadClassesYaml(classesYamlPath.empty() ? onnxPath + ".classes.yaml" : classesYamlPath);

    // The runtime is the deserialization factory; it must outlive the engine.
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_)
        throw std::runtime_error("TensorRTSeg: createInferRuntime failed");

    if (tryLoadEngine())
    {
        loadedFromCache_ = true;
    }
    else
    {
        std::printf("[TensorRTSeg] no usable engine cache, building from %s (takes ~a minute)...\n",
                    onnxPath.c_str());
        buildEngine(onnxPath);
        if (!tryLoadEngine())
            throw std::runtime_error("TensorRTSeg: freshly built engine failed to load: " +
                                     enginePath_);
    }
    setupInference();
}

TensorRTSeg::~TensorRTSeg()
{
    // CUDA resources are not RAII-wrapped; release them before the context /
    // engine / runtime unique_ptrs run their (reverse-declaration-order) dtors.
    if (dInput_ != nullptr)
        cudaFree(dInput_);
    if (dOutput_ != nullptr)
        cudaFree(dOutput_);
    if (stream_ != nullptr)
        cudaStreamDestroy(stream_);
}

void TensorRTSeg::loadClassesYaml(const std::string& yamlPath)
{
    YAML::Node root;
    try
    {
        root = YAML::LoadFile(yamlPath);
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("TensorRTSeg: cannot read sidecar " + yamlPath + ": " + e.what());
    }

    for (int c = 0; c < 3; ++c)
    {
        mean_[c] = root["mean"][c].as<float>();
        std_[c] = root["std"][c].as<float>();
    }
    inputName_ = root["input_tensor"].as<std::string>();
    outputName_ = root["output_tensor"].as<std::string>();

    // LUT over class ids: default static (1), listed ids dynamic (0). Sized
    // generously so an out-of-range id (a bug upstream) maps to static rather
    // than out-of-bounds.
    staticLut_.assign(256, 1);
    for (const auto& id : root["dynamic_class_ids"])
    {
        const int i = id.as<int>();
        if (i < 0 || i >= static_cast<int>(staticLut_.size()))
            throw std::runtime_error("TensorRTSeg: dynamic class id out of range in " + yamlPath);
        staticLut_[i] = 0;
    }
    // YAML input size is cross-checked against the engine in setupInference().
    inputH_ = root["input_height"].as<int>();
    inputW_ = root["input_width"].as<int>();
}

void TensorRTSeg::buildEngine(const std::string& onnxPath)
{
    // 1) Builder: the factory for everything below.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
    if (!builder)
        throw std::runtime_error("TensorRTSeg: createInferBuilder failed");

    // 2) Empty network graph. TRT 10 only supports explicit-batch networks,
    //    so the legacy flags argument is simply 0.
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));

    // 3) ONNX parser populates the network from the file. It also OWNS the
    //    weight memory it read, so it must stay alive until the build is done.
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger_));
    if (!parser->parseFromFile(onnxPath.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        std::ostringstream oss;
        oss << "TensorRTSeg: ONNX parse failed for " << onnxPath;
        for (int i = 0; i < parser->getNbErrors(); ++i)
            oss << "\n  " << parser->getError(i)->desc();
        throw std::runtime_error(oss.str());
    }

    // 4) Build configuration. kFP16 means "TRT may run layers in fp16 where
    //    profitable" - network I/O keeps the exported dtypes (float32/int32).
    //    The workspace limit caps scratch GPU memory used to TRY kernel
    //    variants during the build; 1 GiB is comfortable for this model.
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
    if (precision_ == "fp16")
    {
        if (!builder->platformHasFastFp16())
            std::printf("[TensorRTSeg] warning: no fast fp16 on this GPU, building anyway\n");
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // 5) The expensive step: per-layer kernel benchmarking on THIS GPU,
    //    producing a serialized engine blob.
    auto blob = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!blob)
        throw std::runtime_error("TensorRTSeg: buildSerializedNetwork failed (see [TensorRT] log)");

    std::ofstream out(enginePath_, std::ios::binary);
    out.write(static_cast<const char*>(blob->data()), static_cast<std::streamsize>(blob->size()));
    if (!out)
        throw std::runtime_error("TensorRTSeg: cannot write engine cache " + enginePath_);
    std::printf("[TensorRTSeg] engine cached: %s (%.1f MB)\n", enginePath_.c_str(),
                static_cast<double>(blob->size()) / (1024.0 * 1024.0));
}

bool TensorRTSeg::tryLoadEngine()
{
    std::ifstream in(enginePath_, std::ios::binary | std::ios::ate);
    if (!in)
        return false; // no cache yet - normal on first run
    const auto size = static_cast<size_t>(in.tellg());
    std::vector<char> blob(size);
    in.seekg(0);
    in.read(blob.data(), static_cast<std::streamsize>(size));

    // Fails (returns null, logs why) on GPU/TRT-version mismatch or a
    // truncated file -> caller falls back to a rebuild that overwrites it.
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), size));
    return engine_ != nullptr;
}

void TensorRTSeg::setupInference()
{
    // The engine is immutable; the context is its mutable "running instance"
    // (activation memory, tensor addresses). One context <-> one in-flight
    // inference; we run synchronously, so one is all we need.
    context_.reset(engine_->createExecutionContext());
    if (!context_)
        throw std::runtime_error("TensorRTSeg: createExecutionContext failed");

    // The engine's own shapes are the ground truth for buffer sizes; the
    // sidecar's input size is only cross-checked, never trusted over them.
    const nvinfer1::Dims inDims = engine_->getTensorShape(inputName_.c_str());
    const nvinfer1::Dims outDims = engine_->getTensorShape(outputName_.c_str());
    if (inDims.nbDims != 4 || outDims.nbDims != 3)
        throw std::runtime_error("TensorRTSeg: unexpected tensor ranks - wrong ONNX? " +
                                 describeIOTensors());
    if (inDims.d[2] != inputH_ || inDims.d[3] != inputW_)
        throw std::runtime_error("TensorRTSeg: sidecar input size disagrees with engine: " +
                                 describeIOTensors());
    if (engine_->getTensorDataType(outputName_.c_str()) != nvinfer1::DataType::kINT32)
        throw std::runtime_error("TensorRTSeg: output is not int32 - wrong ONNX export?");

    const size_t nIn = 3ULL * inputH_ * inputW_;
    const size_t nOut = 1ULL * inputH_ * inputW_;
    hostInput_.resize(nIn);
    hostOutput_.resize(nOut);
    cudaCheck(cudaStreamCreate(&stream_), "cudaStreamCreate");
    cudaCheck(cudaMalloc(&dInput_, nIn * sizeof(float)), "cudaMalloc(input)");
    cudaCheck(cudaMalloc(&dOutput_, nOut * sizeof(int32_t)), "cudaMalloc(output)");

    // Addresses are fixed for the object's lifetime, so bind them once here
    // instead of before every enqueue.
    if (!context_->setTensorAddress(inputName_.c_str(), dInput_) ||
        !context_->setTensorAddress(outputName_.c_str(), dOutput_))
        throw std::runtime_error("TensorRTSeg: setTensorAddress failed (name mismatch?)");
}

cv::Mat TensorRTSeg::inferMask(const cv::Mat& frame)
{
    // --- preprocess: must match preprocess() in export_efficientvit_seg.py ---
    // (resize -> RGB -> /255 -> (x-mean)/std -> CHW). Grayscale is replicated
    // to 3 channels first, exactly like cv2.imread does for the Python check.
    cv::Mat bgr = frame;
    if (frame.channels() == 1)
        cv::cvtColor(frame, bgr, cv::COLOR_GRAY2BGR);
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(inputW_, inputH_));

    const int hw = inputH_ * inputW_;
    float* rPlane = hostInput_.data();          // CHW: R plane first (BGR->RGB swap)
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

    // --- inference: H2D copy, enqueue, D2H copy, all ordered on one stream ---
    cudaCheck(cudaMemcpyAsync(dInput_, hostInput_.data(), hostInput_.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              "H2D copy");
    if (!context_->enqueueV3(stream_))
        throw std::runtime_error("TensorRTSeg: enqueueV3 failed");
    cudaCheck(cudaMemcpyAsync(hostOutput_.data(), dOutput_, hostOutput_.size() * sizeof(int32_t),
                              cudaMemcpyDeviceToHost, stream_),
              "D2H copy");
    // The three operations above only ENTERED the stream's queue; this is the
    // point where the CPU actually waits for the GPU to finish them in order.
    cudaCheck(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");

    // --- postprocess: class ids -> binary mask -> dilate dynamic -> frame size ---
    cv::Mat mask(inputH_, inputW_, CV_8UC1);
    uchar* out = mask.ptr<uchar>(0);
    for (int i = 0; i < hw; ++i)
        out[i] = staticLut_[static_cast<uint8_t>(hostOutput_[i])];

    // Keypoints concentrate on object boundaries, so grow the dynamic (0)
    // region a little: erosion of the static mask = dilation of the dynamic
    // region. Done at network resolution, before upsampling.
    if (dilateRadiusPx_ > 0)
    {
        const int k = 2 * dilateRadiusPx_ + 1;
        cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k)));
    }

    // Nearest neighbour: a binary mask must never be interpolated.
    cv::Mat maskFull;
    cv::resize(mask, maskFull, frame.size(), 0, 0, cv::INTER_NEAREST);
    return maskFull;
}

std::string TensorRTSeg::describeIOTensors() const
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

} // namespace segmentation
