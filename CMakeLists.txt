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

set(SRCS_ALLFEATURE
        src/System.cc
        src/Tracking.cc
        src/LocalMapping.cc
        src/LoopClosing.cc
        src/FeatureMatcher.cpp
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
        src/Image.cpp
        src/Feature.cpp
        src/Feature_orb32.cpp
        src/Feature_brisk48.cpp
        src/Feature_akaze61.cpp
        src/Feature_surf64.cpp
        src/Feature_kaze64.cpp
        src/Feature_sift128.cpp
        src/Feature_r2d2_128.cpp
        src/Feature_anyFeatBin.cpp
        src/Feature_anyFeatNonBin.cpp
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

# Dependencies
find_package(OpenCV REQUIRED )
find_package(Eigen3  REQUIRED NO_MODULE)
find_package(Pangolin REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(g2o REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter)
find_package(brisk REQUIRED)
find_package(akaze REQUIRED)
find_package(SiftGPU CONFIG REQUIRED)
find_package(LightGlue REQUIRED HINTS "${PROJECT_SOURCE_DIR}/Thirdparty/Light_Glue_CPP/build")

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
        ${g2o_LIBRARIES}
        ${DBoW2_LIBRARIES}
        ${brisk_LIBRARIES}
        ${akaze_LIBRARIES}
        ${Pangolin_LIBRARIES}
        OpenMP::OpenMP_CXX
        LightGlue::lib
        superpoint_lightglue_lib
        SiftGPU::sift_gpu
)


# Example executables
add_executable(vslamlab_allfeature_mono src/vslamlab_allfeature_mono.cpp)
target_link_libraries(vslamlab_allfeature_mono ${PROJECT_NAME})

add_executable(vslamlab_allfeature_mono_stream src/vslamlab_allfeature_mono_stream.cpp)
target_link_libraries(vslamlab_allfeature_mono_stream ${PROJECT_NAME})

# Install targets and headers
install(TARGETS ${PROJECT_NAME} vslamlab_allfeature_mono vslamlab_allfeature_mono_stream
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY include/ DESTINATION include)