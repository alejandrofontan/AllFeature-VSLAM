cmake_minimum_required(VERSION 3.16)
project(AllFeature-VSLAM)

# Compiler settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)

# Vendored libraries without an explicit type (placecell, PoseLib) are linked statically into libAllFeature-VSLAM.so,
# also when a package build passes -DBUILD_SHARED_LIBS=ON (they have no install rules of their own).
set(BUILD_SHARED_LIBS OFF)

# Output directories: bin/ and lib/ in the source tree, where build.sh and the pixi tasks expect them. A package
# build (pixi-build compiles this checkout in place) turns it off and keeps everything in its build directory.
option(ALLFEATURE_OUTPUT_IN_SOURCE "Write executables and libraries to bin/ and lib/ of the source tree" ON)
if(ALLFEATURE_OUTPUT_IN_SOURCE)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
endif()

# Compiler flags. The architecture flag applies to every C/C++ target, vendored subprojects included (g2o,
# Light_Glue_CPP and PoseLib follow it): Eigen code compiled for different targets in one process breaks alignment.
# ALLFEATURE_NATIVE=ON tunes for this machine (build.sh; -march=native also works with clang on Apple Silicon); OFF
# targets x86-64-v3 (AVX2) on x86-64, as the conda package does, and the compiler's default on every other
# architecture (arm64: the armv8-a baseline with NEON; Apple clang rejects x86 -march values).
option(ALLFEATURE_NATIVE "Optimise for the build machine's CPU (-march=native) instead of the portable baseline" OFF)
set(ALLFEATURE_MARCH_FLAG)
if(ALLFEATURE_NATIVE)
  set(ALLFEATURE_MARCH_FLAG -march=native)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  set(ALLFEATURE_MARCH_FLAG -march=x86-64-v3)
endif()
message(STATUS "AllFeature architecture flag: '${ALLFEATURE_MARCH_FLAG}' (${CMAKE_SYSTEM_PROCESSOR})")
# C/C++ only: these flags must not reach nvcc (Light_Glue_CPP's CUDA kernels)
add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-Wall;-Wextra;-Wpedantic;-O3;${ALLFEATURE_MARCH_FLAG}>")
# GPU architectures of Light_Glue_CPP's CUDA kernels: this machine's GPU with ALLFEATURE_NATIVE, otherwise Turing to
# Hopper plus consumer Blackwell ("90" also embeds PTX that newer GPUs JIT-compile)
if(ALLFEATURE_NATIVE)
  set(ALLFEATURE_CUDA_ARCHITECTURES_DEFAULT native)
else()
  set(ALLFEATURE_CUDA_ARCHITECTURES_DEFAULT "75-real;80-real;86-real;89-real;90;120-real")
endif()
set(ALLFEATURE_CUDA_ARCHITECTURES "${ALLFEATURE_CUDA_ARCHITECTURES_DEFAULT}" CACHE STRING
    "CUDA architectures of the Light_Glue_CPP kernels")

# GPU backends (docs/plans/mac_port.md, phase 1). Each option gates a third-party project, its find_package
# calls, the source files that use it and a compile definition of the same name; a feature or backend
# requested in the settings YAML but compiled out is a clear error at startup (FeatureFactory.h, System.cc,
# SegmentationSettings.h). All three default to ON on Linux (today's build) and OFF on macOS, where none
# of the libraries exists. With every option OFF the project needs OpenCV, Eigen, Pangolin, yaml-cpp,
# OpenMP, brisk and akaze only.
if(APPLE)
  set(ALLFEATURE_GPU_DEFAULT OFF)
else()
  set(ALLFEATURE_GPU_DEFAULT ON)
endif()
option(ALLFEATURE_WITH_SIFTGPU  "SiftGPU (CUDA): the sift128 feature and its SiftMatchGPU matcher" ${ALLFEATURE_GPU_DEFAULT})
option(ALLFEATURE_WITH_TORCH    "libtorch + Light_Glue_CPP (CUDA kernels): the aliked128 feature and the LightGlue matcher" ${ALLFEATURE_GPU_DEFAULT})
option(ALLFEATURE_WITH_TENSORRT "TensorRT: the superpoint256 feature + SuperPoint-LightGlue matcher, online segmentation and the MegaLoc VPR backend (placecell::megaloc)" ${ALLFEATURE_GPU_DEFAULT})
foreach(_opt ALLFEATURE_WITH_SIFTGPU ALLFEATURE_WITH_TORCH ALLFEATURE_WITH_TENSORRT)
  if(${_opt})
    add_compile_definitions(${_opt})
  endif()
endforeach()
message(STATUS "AllFeature backends: SiftGPU=${ALLFEATURE_WITH_SIFTGPU} torch/LightGlue=${ALLFEATURE_WITH_TORCH} TensorRT=${ALLFEATURE_WITH_TENSORRT}")

# brisk / akaze (standalone feature libraries from prefix.dev/vslamlab, VSLAM-LAB/conda-recipes): optional, so a
# platform without the packages yet (osx-arm64 until their recipes are built there, docs/plans/mac_port.md
# decision 2) still builds with the other features; brisk48 / akaze61 are then refused at startup like the
# GPU-backed features (FeatureFactory.h).
find_package(brisk QUIET)
find_package(akaze QUIET)
if(brisk_FOUND)
  add_compile_definitions(ALLFEATURE_WITH_BRISK)
else()
  message(WARNING "brisk not found: the brisk48 feature is compiled out")
endif()
if(akaze_FOUND)
  add_compile_definitions(ALLFEATURE_WITH_AKAZE)
else()
  message(WARNING "akaze not found: the akaze61 feature is compiled out")
endif()

set(SRCS_ALLFEATURE
        src/System.cc
        src/Tracking.cc
        src/Tracking_aux.cc
        src/LocalMapping.cc
        src/LocalMapping_aux.cc
        src/LoopClosing.cc
        src/LoopClosing_aux.cc
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
        src/Sim3Solver.cc
        src/Initializer.cc
        src/Viewer.cc
        src/Observation.cpp
        src/FeatureExtractor.cpp
        src/Utils.cpp
        src/KeyframeInformation.cc
        src/PlaceCellSettings.cc
        src/Image.cpp
        src/Feature.cpp
        src/Feature_orb32.cpp
        src/Feature_surf64.cpp
        src/Feature_kaze64.cpp
)
if(ALLFEATURE_WITH_SIFTGPU)
  list(APPEND SRCS_ALLFEATURE src/Feature_sift128.cpp)
endif()
if(brisk_FOUND)
  list(APPEND SRCS_ALLFEATURE src/Feature_brisk48.cpp)
endif()
if(akaze_FOUND)
  list(APPEND SRCS_ALLFEATURE src/Feature_akaze61.cpp)
endif()
if(ALLFEATURE_WITH_TORCH)
  list(APPEND SRCS_ALLFEATURE src/Feature_aliked128.cpp src/FeatureMatcher_lightglue.cpp)
endif()
if(ALLFEATURE_WITH_TENSORRT)
  list(APPEND SRCS_ALLFEATURE src/Feature_superpoint256.cpp src/FeatureMatcher_superglue.cpp src/PlaceRecognitionMegaLoc.cc)
endif()

# ---- CUDA (help Torch/Caffe2 legacy FindCUDA) ----
# Prefer the pixi/conda CUDA layout when present
if((ALLFEATURE_WITH_TORCH OR ALLFEATURE_WITH_TENSORRT OR ALLFEATURE_WITH_SIFTGPU) AND DEFINED ENV{CONDA_PREFIX})
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

# ---- SuperPoint-LightGlue (TensorRT) ----
if(ALLFEATURE_WITH_TENSORRT)
  set(SUPERPOINT_LIGHTGLUE_DIR "${PROJECT_SOURCE_DIR}/Thirdparty/SuperPoint-LightGlue-TensorRT"
      CACHE PATH "Path to SuperPoint-LightGlue source")
  set(SUPERPOINT_LIGHTGLUE_BUILD_EXECUTABLES OFF CACHE BOOL "" FORCE)
  add_subdirectory(${SUPERPOINT_LIGHTGLUE_DIR} ${CMAKE_BINARY_DIR}/superpoint_lightglue)
endif()

# ---- g2o ----
add_subdirectory(${PROJECT_SOURCE_DIR}/Thirdparty/g2o ${CMAKE_BINARY_DIR}/g2o)

# ---- PoseLib (robust estimation: replaces OpenCV USAC in FeatureMatcher) ----
set(MARCH_NATIVE ${ALLFEATURE_NATIVE} CACHE BOOL "" FORCE)  # PoseLib's own -march=native switch follows ours
# PoseLib's install rules are unconditional, so an install also gets its static archive, headers and CMake config
# (unused at runtime: it is linked into the library).
add_subdirectory(${PROJECT_SOURCE_DIR}/Thirdparty/PoseLib ${CMAKE_BINARY_DIR}/poselib)
# Vendored code compiles under this project's global -Wall -Wextra -Werror flags;
# suppress diagnostics for the third-party target only so -Werror can't break it.
target_compile_options(PoseLib PRIVATE -w)

# ---- Segmentation-TensorRT (online static/dynamic masking, segmentation.md) ----
# First-party module (not vendored), so it keeps this project's warning flags.
# Only the CLI entry points consume it; the SLAM library itself stays unaware
# of how Image::mask gets filled.
if(ALLFEATURE_WITH_TENSORRT)
  add_subdirectory(${PROJECT_SOURCE_DIR}/Thirdparty/Segmentation-TensorRT ${CMAKE_BINARY_DIR}/segmentation_tensorrt)
endif()

# ---- placecell (MegaLoc descriptors for the `vpr: megaloc` backend) ----
# Standalone first-party library (github.com/alejandrofontan/placecell), pulled in as a
# git submodule. System owns the placecell::MegaLocPlaceCell and injects it into
# PlaceRecognitionMegaLoc; the SLAM library links placecell::megaloc (the Eigen-only
# core placecell::placecell comes in transitively) and placecell::viz, whose kernel /
# information / alive renders the Viewer shows as Pangolin panels and
# System::SavePlaceCellDiagnostics writes as PNGs (PlaceCell.* settings keys). viz only
# needs OpenCV core/imgproc/imgcodecs here: its cv::imshow windows are never used
# (the pixi env ships the headless OpenCV build, where imshow throws).
set(PLACECELL_WITH_MEGALOC ${ALLFEATURE_WITH_TENSORRT})
set(PLACECELL_WITH_VIZ ON)
set(PLACECELL_BUILD_EXAMPLES OFF)
set(PLACECELL_BUILD_PYTHON OFF)
add_subdirectory(${PROJECT_SOURCE_DIR}/Thirdparty/placecell ${CMAKE_BINARY_DIR}/placecell)

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
if(ALLFEATURE_WITH_SIFTGPU)
  find_package(SiftGPU CONFIG REQUIRED)
endif()
# ---- Light_Glue_CPP (ALIKED extractor + LightGlue matcher, libtorch + CUDA kernels) ----
# Built in the same tree (static LightGlue::lib); it compiles LIGHTGLUE_MODELS_DIR (default: the bare
# `lightglue_models`, resolved against the cwd at runtime) into the library and finds Torch itself, with libtorch's
# Caffe2Config dev warning already scoped there.
if(ALLFEATURE_WITH_TORCH)
  set(LIGHTGLUE_CUDA_ARCHITECTURES "${ALLFEATURE_CUDA_ARCHITECTURES}")
  add_subdirectory(${PROJECT_SOURCE_DIR}/Thirdparty/Light_Glue_CPP ${CMAKE_BINARY_DIR}/light_glue_cpp)
  target_compile_options(LightGlue_lib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-w>)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
else()
  message(WARNING "OpenMP NOT found")
endif()


add_library(${PROJECT_NAME} SHARED ${SRCS_ALLFEATURE})

# Include directories
target_include_directories(${PROJECT_NAME} PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBRARIES}
        Eigen3::Eigen
        ${YAML_CPP_LIBRARIES}
        g2o
        ${Pangolin_LIBRARIES}
        OpenMP::OpenMP_CXX
        PoseLib::PoseLib
        placecell::placecell
        placecell::viz
)
if(ALLFEATURE_WITH_TORCH)
  target_link_libraries(${PROJECT_NAME} LightGlue::lib)
endif()
if(ALLFEATURE_WITH_TENSORRT)
  target_link_libraries(${PROJECT_NAME} superpoint_lightglue_lib placecell::megaloc)
endif()
if(ALLFEATURE_WITH_SIFTGPU)
  target_link_libraries(${PROJECT_NAME} SiftGPU::sift_gpu)
endif()
if(brisk_FOUND)
  target_link_libraries(${PROJECT_NAME} ${brisk_LIBRARIES})
endif()
if(akaze_FOUND)
  target_link_libraries(${PROJECT_NAME} ${akaze_LIBRARIES})
endif()


# Example executables
# mimalloc must stay FIRST in each executable's link line (see the mimalloc section above).
# Online segmentation (SegmentationSettings.h) is consumed by the entry points only, so the SLAM library stays
# unaware of how Image::mask gets filled; without TensorRT the header provides a stub and nothing to link.
if(ALLFEATURE_WITH_TENSORRT)
  set(ALLFEATURE_SEGMENTATION_LIB tensorrt_seg)
else()
  set(ALLFEATURE_SEGMENTATION_LIB)
endif()
add_executable(vslamlab_allfeature_mono src/vslamlab_allfeature_mono.cpp)
target_link_libraries(vslamlab_allfeature_mono mimalloc ${PROJECT_NAME} ${ALLFEATURE_SEGMENTATION_LIB})

add_executable(vslamlab_allfeature_mono_stream src/vslamlab_allfeature_mono_stream.cpp)
target_link_libraries(vslamlab_allfeature_mono_stream mimalloc ${PROJECT_NAME} ${ALLFEATURE_SEGMENTATION_LIB})

add_executable(vslamlab_allfeature_rgbd src/vslamlab_allfeature_rgbd.cpp)
target_link_libraries(vslamlab_allfeature_rgbd mimalloc ${PROJECT_NAME} ${ALLFEATURE_SEGMENTATION_LIB})

# Parity/benchmark harness for the brute-force matcher (issue #13 P2) — not installed.
add_executable(test_bfmatcher_parity src/test_bfmatcher_parity.cpp)
target_link_libraries(test_bfmatcher_parity mimalloc ${PROJECT_NAME})

# Install targets and headers
install(TARGETS ${PROJECT_NAME} vslamlab_allfeature_mono vslamlab_allfeature_mono_stream vslamlab_allfeature_rgbd
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

# Headers under include/AllFeature-VSLAM/: they have generic names (Frame.h, System.h, Types.h, ...)
install(DIRECTORY include/ DESTINATION include/AllFeature-VSLAM)

# SuperPoint-LightGlue-TensorRT has no install rules of its own; libAllFeature-VSLAM.so links it
if(ALLFEATURE_WITH_TENSORRT)
  install(TARGETS superpoint_lightglue_lib LIBRARY DESTINATION lib)
endif()

# Reference copies of the settings (VSLAM-LAB's allfeature environment copies them next to the models)
install(DIRECTORY settings/ DESTINATION share/allfeature-vslamlab/settings FILES_MATCHING PATTERN "*.yaml")
install(FILES vslamlab_allfeature-dev_settings.yaml DESTINATION share/allfeature-vslamlab)