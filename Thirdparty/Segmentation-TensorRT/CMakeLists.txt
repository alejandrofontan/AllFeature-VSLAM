# Segmentation-TensorRT: online static/dynamic masking for AllFeature-VSLAM
# (see segmentation.md). Standalone project for now; added via add_subdirectory
# from the main CMakeLists in Phase 2.
#
# Builds only inside the pixi env (CONDA_PREFIX provides CUDA, TensorRT, OpenCV).
cmake_minimum_required(VERSION 3.16)
project(Segmentation-TensorRT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)   # CUDA::cudart (cudaMalloc, streams, memcpy)
find_package(OpenCV REQUIRED)        # image I/O + preprocessing

# TensorRT ships no CMake config - locate headers/libs in the pixi env manually.
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS $ENV{CONDA_PREFIX}/include)
find_library(NVINFER_LIB nvinfer HINTS $ENV{CONDA_PREFIX}/lib)
find_library(NVONNXPARSER_LIB nvonnxparser HINTS $ENV{CONDA_PREFIX}/lib)
if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER_LIB OR NOT NVONNXPARSER_LIB)
    message(FATAL_ERROR "TensorRT not found - run the build inside the pixi env "
                        "(CONDA_PREFIX=$ENV{CONDA_PREFIX})")
endif()
message(STATUS "TensorRT: ${NVINFER_LIB}")

add_executable(test_segmentation src/test_segmentation.cpp)
target_include_directories(test_segmentation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(test_segmentation PRIVATE
    ${NVINFER_LIB}
    ${NVONNXPARSER_LIB}
    ${OpenCV_LIBS}
    CUDA::cudart
)
target_compile_options(test_segmentation PRIVATE -Wall -Wextra)
