# MegaLoc-TensorRT: MegaLoc global descriptors for the online visual-place-recognition
# backend of AllFeature-VSLAM (`vpr: megaloc`, see README.md). Pulled in via
# add_subdirectory from the main CMakeLists; the SLAM library itself links the
# tensorrt_megaloc target (PlaceRecognitionMegaLoc owns the engine).
#
# Builds only inside the pixi env (CONDA_PREFIX provides CUDA, TensorRT, OpenCV).
cmake_minimum_required(VERSION 3.16)
project(MegaLoc-TensorRT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)   # CUDA::cudart (cudaMalloc, streams, memcpy)
find_package(OpenCV REQUIRED)        # image I/O + preprocessing
find_package(yaml-cpp REQUIRED)      # sidecar <onnx>.yaml parsing

# TensorRT ships no CMake config - locate headers/libs in the pixi env manually.
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS $ENV{CONDA_PREFIX}/include)
find_library(NVINFER_LIB nvinfer HINTS $ENV{CONDA_PREFIX}/lib)
find_library(NVONNXPARSER_LIB nvonnxparser HINTS $ENV{CONDA_PREFIX}/lib)
if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER_LIB OR NOT NVONNXPARSER_LIB)
    message(FATAL_ERROR "TensorRT not found - run the build inside the pixi env "
                        "(CONDA_PREFIX=$ENV{CONDA_PREFIX})")
endif()
message(STATUS "TensorRT (MegaLoc): ${NVINFER_LIB}")

# PIC: this static library is linked into the SHARED libAllFeature-VSLAM.so.
add_library(tensorrt_megaloc STATIC src/tensorrt_megaloc.cpp)
set_target_properties(tensorrt_megaloc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(tensorrt_megaloc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(tensorrt_megaloc PUBLIC
    ${NVINFER_LIB}
    ${NVONNXPARSER_LIB}
    ${OpenCV_LIBS}
    CUDA::cudart
    yaml-cpp
)
target_compile_options(tensorrt_megaloc PRIVATE -Wall -Wextra)

add_executable(test_megaloc src/test_megaloc.cpp)
target_link_libraries(test_megaloc PRIVATE tensorrt_megaloc)
target_compile_options(test_megaloc PRIVATE -Wall -Wextra)
