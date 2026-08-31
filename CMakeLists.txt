cmake_minimum_required(VERSION 3.16)
project(AllFeature-VSLAM)

# Compiler settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# Compiler flags
add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native)
# -Wno-overloaded-virtual: this project's own translation units transitively include vendored
# DBoW2 headers (FOrb.h, FBrief.h, ...) whose FXxx classes intentionally hide the private, never
# dynamically-dispatched FClass::meanValue with their own differently-typed static meanValue --
# see the matching note in Thirdparty/DBoW2/CMakeLists.txt for the full rationale. Scoped to CXX
# only (via generator expression) since this directory's compile options are inherited by
# add_subdirectory()'d targets that include plain C sources (e.g. vendored g2o's os_specific.c),
# and -Woverloaded-virtual is a C++-only concept -- passing it unconditionally produced its own
# "valid for C++/ObjC++ but not for C" warning from the C frontend.
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-overloaded-virtual>)

set(SRCS_ALLFEATURE
        src/System.cc
        src/Tracking.cc
        src/Tracking_aux.cc
        src/LocalMapping.cc
        src/LocalMapping_aux.cc
        src/LoopClosing.cc
        src/FeatureMatcher.cpp
        src/BruteForceMatcher.cpp
        src/FrameDrawer.cc
        src/Converter.cc
        src/MapPoint.cc
        src/KeyFrame.cc
        src/Map.cc
        src/MapDrawer.cc
        src/Optimizer.cc
        src/PnPsolver.cc
        src/Frame.cc
        src/KeyFrameDatabase.cc
        src/Sim3Solver.cc
        src/Initializer.cc
        src/Viewer.cc
        src/Observation.cpp
        src/FeatureExtractor.cpp
        src/Utils.cpp
        src/Vocabulary.cpp
        src/PlaceRecognitionBoW.cc
        src/PlaceRecognitionMegaLoc.cc
        src/Image.cpp
        src/Feature.cpp
        src/Feature_orb32.cpp
        src/Feature_brisk48.cpp
        src/Feature_akaze61.cpp
        src/Feature_surf64.cpp
        src/Feature_kaze64.cpp
        src/Feature_sift128.cpp
        src/Feature_aliked128.cpp
        src/Feature_superpoint256.cpp
        src/FeatureMatcher_lightglue.cpp
        src/FeatureMatcher_superglue.cpp
)

# ---- CUDA (help Torch/Caffe2 legacy FindCUDA) ----
# Prefer the pixi/conda CUDA layout when present
if(DEFINED ENV{CONDA_PREFIX})
  set(_CUDA_PREFIX "$ENV{CONDA_PREFIX}/targets/x86_64-linux")
  if(EXISTS "${_CUDA_PREFIX}/include/cuda.h")
    set(CUDA_TOOLKIT_ROOT_DIR "${_CUDA_PREFIX}" CACHE PATH "CUDA root" FORCE)
    set(CUDAToolkit_ROOT      "${_CUDA_PREFIX}" CACHE PATH "CUDA root" FORCE)
    set(CUDA_INCLUDE_DIRS     "${_CUDA_PREFIX}/include" CACHE PATH "CUDA include" FORCE)
    # Common lib locations in conda/pixi CUDA packages:
    if(EXISTS "${_CUDA_PREFIX}/lib")
      list(APPEND CMAKE_LIBRARY_PATH "${_CUDA_PREFIX}/lib")
    endif()
    if(EXISTS "${_CUDA_PREFIX}/lib64")
      list(APPEND CMAKE_LIBRARY_PATH "${_CUDA_PREFIX}/lib64")
    endif()
  endif()
endif()

# Optional but often helps some configs that look at CMAKE_PREFIX_PATH
if(DEFINED CUDA_TOOLKIT_ROOT_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${CUDA_TOOLKIT_ROOT_DIR}")
endif()

# ---- SuperPoint-LightGlue ----
set(SUPERPOINT_LIGHTGLUE_DIR "${PROJECT_SOURCE_DIR}/Thirdparty/SuperPoint-LightGlue-TensorRT"
    CACHE PATH "Path to SuperPoint-LightGlue source")
set(SUPERPOINT_LIGHTGLUE_BUILD_EXECUTABLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${SUPERPOINT_LIGHTGLUE_DIR} ${CMAKE_BINARY_DIR}/superpoint_lightglue)

# ---- g2o ----
add_subdirectory(${PROJECT_SOURCE_DIR}/Thirdparty/g2o ${CMAKE_BINARY_DIR}/g2o)

# ---- PoseLib (robust estimation: replaces OpenCV USAC in FeatureMatcher) ----
set(MARCH_NATIVE ON CACHE BOOL "" FORCE)  # match this project's -march=native flags
add_subdirectory(${PROJECT_SOURCE_DIR}/Thirdparty/PoseLib ${CMAKE_BINARY_DIR}/poselib)
# Vendored code compiles under this project's global -Wall -Wextra -Werror flags;
# suppress diagnostics for the third-party target only so -Werror can't break it.
target_compile_options(PoseLib PRIVATE -w)

# ---- Segmentation-TensorRT (online static/dynamic masking, segmentation.md) ----
# First-party module (not vendored), so it keeps this project's warning flags.
# Only the CLI entry points consume it; the SLAM library itself stays unaware
# of how Image::mask gets filled.
add_subdirectory(${PROJECT_SOURCE_DIR}/Thirdparty/Segmentation-TensorRT ${CMAKE_BINARY_DIR}/segmentation_tensorrt)

# ---- MegaLoc-TensorRT (visual place recognition backend `vpr: megaloc`) ----
# First-party module (not vendored). Unlike segmentation, the SLAM library itself
# consumes it: PlaceRecognitionMegaLoc owns the engine, so the library links it.
add_subdirectory(${PROJECT_SOURCE_DIR}/Thirdparty/MegaLoc-TensorRT ${CMAKE_BINARY_DIR}/megaloc_tensorrt)

# ---- mimalloc (drop-in allocator, issue #13 addendum M1) ----
# Must be the SHARED library, linked FIRST into each executable: libmimalloc.so then appears
# before libc in the process's symbol lookup order, so its malloc/free interpose for every
# shared library in the process (OpenCV, torch, this project). A static mimalloc linked into a
# non-exporting executable would only cover the executable's own allocations.
set(MI_BUILD_STATIC OFF CACHE BOOL "" FORCE)
set(MI_BUILD_OBJECT OFF CACHE BOOL "" FORCE)
set(MI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${PROJECT_SOURCE_DIR}/Thirdparty/mimalloc ${CMAKE_BINARY_DIR}/mimalloc)
target_compile_options(mimalloc PRIVATE -w)

# Dependencies
find_package(OpenCV REQUIRED )
find_package(Eigen3  REQUIRED NO_MODULE)
find_package(Pangolin REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter)
find_package(brisk REQUIRED)
find_package(akaze REQUIRED)
find_package(SiftGPU CONFIG REQUIRED)
# find_package(LightGlue) transitively re-triggers libtorch's own Caffe2Config.cmake (via
# LightGlueConfig.cmake's find_dependency(Torch)), which mis-calls
# find_package_handle_standard_args() with a package-name mismatch (nvtx3 vs Caffe2) -- a dev/
# author warning from Torch's installed CMake config, not from any CMakeLists.txt in this
# project. Scoped around just this call (not a global -Wno-dev) so any real future dev warning
# of our own still surfaces.
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON)
find_package(LightGlue REQUIRED HINTS "${PROJECT_SOURCE_DIR}/Thirdparty/Light_Glue_CPP/build")
unset(CMAKE_SUPPRESS_DEVELOPER_WARNINGS)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
else()
  message(WARNING "OpenMP NOT found")
endif()

set(DBoW2_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/include")
set(DBoW2_LIBRARIES "${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so")

add_library(${PROJECT_NAME} SHARED ${SRCS_ALLFEATURE})

# Include directories
target_include_directories(${PROJECT_NAME} PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${DBoW2_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBRARIES}
        Eigen3::Eigen
        ${YAML_CPP_LIBRARIES}
        g2o
        ${DBoW2_LIBRARIES}
        ${brisk_LIBRARIES}
        ${akaze_LIBRARIES}
        ${Pangolin_LIBRARIES}
        OpenMP::OpenMP_CXX
        LightGlue::lib
        superpoint_lightglue_lib
        SiftGPU::sift_gpu
        PoseLib::PoseLib
        tensorrt_megaloc
)


# Example executables
# mimalloc must stay FIRST in each executable's link line (see the mimalloc section above).
add_executable(vslamlab_allfeature_mono src/vslamlab_allfeature_mono.cpp)
target_link_libraries(vslamlab_allfeature_mono mimalloc ${PROJECT_NAME} tensorrt_seg)

add_executable(vslamlab_allfeature_mono_stream src/vslamlab_allfeature_mono_stream.cpp)
target_link_libraries(vslamlab_allfeature_mono_stream mimalloc ${PROJECT_NAME} tensorrt_seg)

add_executable(vslamlab_allfeature_rgbd src/vslamlab_allfeature_rgbd.cpp)
target_link_libraries(vslamlab_allfeature_rgbd mimalloc ${PROJECT_NAME} tensorrt_seg)

# Parity/benchmark harness for the brute-force matcher (issue #13 P2) — not installed.
add_executable(test_bfmatcher_parity src/test_bfmatcher_parity.cpp)
target_link_libraries(test_bfmatcher_parity mimalloc ${PROJECT_NAME})

# Install targets and headers
install(TARGETS ${PROJECT_NAME} vslamlab_allfeature_mono vslamlab_allfeature_mono_stream vslamlab_allfeature_rgbd
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY include/ DESTINATION include)