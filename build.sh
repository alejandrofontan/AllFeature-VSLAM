#!/bin/bash

# The pixi/conda cuda-nvcc package's activation script unconditionally injects `-ccbin=${CXX}`
# via NVCC_PREPEND_FLAGS, for the benefit of nvcc invocations outside CMake's control. Every CUDA
# target built here goes through CMake's own native `LANGUAGES CUDA` support instead (see
# Thirdparty/Light_Glue_CPP/CMakeLists.txt, Thirdparty/SuperPoint-LightGlue-TensorRT), which
# independently supplies its own `-ccbin` derived from CMAKE_CXX_COMPILER on every nvcc call --
# same compiler, so it's a harmless but noisy duplicate ("incompatible redefinition for option
# 'compiler-bindir'"). Clearing it here (scoped to this script's process only, not the conda
# activation script) removes the duplicate without affecting anything CMake itself controls.
unset NVCC_PREPEND_FLAGS

delete_if_exists() {
  local folder=$1
  build_folder="${folder}/build"
  bin_folder="${folder}/bin"
  lib_folder="${folder}/lib"
  if [ -d "$build_folder" ]; then
    rm -rf "$build_folder"
  fi
  if [ -d "$bin_folder" ]; then
    rm -rf "$bin_folder"
  fi
  if [ -d "$lib_folder" ]; then
    rm -rf "$lib_folder"
  fi
}

build_library() {
  library_name="$1"
  source_folder="$2"
  verbose="$3"
  force_build="$4"

  build_folder="$source_folder/build"
  bin_folder="$source_folder/bin"
  lib_folder="$source_folder/lib"

  if [ "$force_build" = true ]; then
  	delete_if_exists ${source_folder}
  fi

  if [ "$verbose" = true ]; then
    echo "[${library_name}][build.sh] Compile ${library_name} ... "
  	cmake -G Ninja -B $build_folder -S $source_folder -DCMAKE_PREFIX_PATH=$source_folder -DCMAKE_INSTALL_PREFIX=$source_folder
  	cmake --build $build_folder --config Release --parallel 4
    #ninja -C build 2>&1 | grep "error:" | head -30
  else
    echo "[${library_name}][build.sh] Compile ${library_name} (output disabled) ... "
  	cmake -G Ninja -B $build_folder -S $source_folder -DCMAKE_PREFIX_PATH=$source_folder -DCMAKE_INSTALL_PREFIX=$source_folder > /dev/null 2>&1
  	cmake --build $build_folder --config Release --parallel 4 > /dev/null 2>&1
  fi
}

# Check inputs
force_build=false
verbose=false
for input in "$@"
do
    if [ "$input" = "-f" ]; then
  	force_build=true
    fi
    if [ "$input" = "-v" ]; then
  	verbose=true
    fi
    if [ "$input" = "-fv" ] || [ "$input" = "-vf" ]; then
  	verbose=true
    force_build=true
    fi
done

# Baseline Dir
LIBRARY_PATH=$(realpath "$0")
LIBRARY_DIR=$(dirname "$LIBRARY_PATH")

# Build Light_Glue_CPP
library_name="Light_Glue_CPP"
source_folder="${LIBRARY_DIR}/Thirdparty/${library_name}"
build_library ${library_name} ${source_folder} ${verbose} ${force_build}

# Build AllFeature-VSLAM
library_name="AllFeature-VSLAM"
source_folder="${LIBRARY_DIR}"
build_library ${library_name} ${source_folder} ${verbose} ${force_build}
