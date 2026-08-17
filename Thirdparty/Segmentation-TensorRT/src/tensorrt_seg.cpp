/**
 * Module: AllFeature-VSLAM - Segmentation-TensorRT - tensorrt_seg.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 0.2 (engine build/load lifecycle)
 * - Created: 2026-08-17
 * - License: GPLv3 License
 */

#include "tensorrt_seg.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <NvOnnxParser.h>

namespace segmentation
{

void TensorRTSeg::Logger::log(Severity severity, const char* msg) noexcept
{
    if (severity <= Severity::kWARNING)
        std::printf("[TensorRT] %s\n", msg);
}

TensorRTSeg::TensorRTSeg(const std::string& onnxPath, const std::string& precision)
    : precision_(precision), enginePath_(onnxPath + "." + precision + ".engine")
{
    if (precision_ != "fp16" && precision_ != "fp32")
        throw std::runtime_error("TensorRTSeg: precision must be fp16 or fp32, got " + precision_);

    // The runtime is the deserialization factory; it must outlive the engine.
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_)
        throw std::runtime_error("TensorRTSeg: createInferRuntime failed");

    if (tryLoadEngine())
    {
        loadedFromCache_ = true;
        return;
    }
    std::printf("[TensorRTSeg] no usable engine cache, building from %s (takes ~a minute)...\n",
                onnxPath.c_str());
    buildEngine(onnxPath);
    if (!tryLoadEngine())
        throw std::runtime_error("TensorRTSeg: freshly built engine failed to load: " + enginePath_);
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
