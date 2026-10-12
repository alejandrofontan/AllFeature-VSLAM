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
  	rm -rf "$source_folder/build-cpu"
  fi

  # Local builds are tuned for this machine (-march=native, CUDA kernels for this GPU); the conda package
  # builds the same CMake project with ALLFEATURE_NATIVE off (x86-64-v3) and an explicit CUDA architecture list.
  local extra_flags=(-DALLFEATURE_NATIVE=ON)
  # -c: GPU-free build (docs/plans/mac_port.md phase 1) in its own build directory, same bin/ and lib/ outputs
  if [ "$cpu_only" = true ]; then
    build_folder="$source_folder/build-cpu"
    extra_flags+=(-DALLFEATURE_WITH_SIFTGPU=OFF -DALLFEATURE_WITH_TORCH=OFF -DALLFEATURE_WITH_TENSORRT=OFF)
  fi

  if [ "$verbose" = true ]; then
    echo "[${library_name}][build.sh] Compile ${library_name} ... "
  	cmake -G Ninja -B $build_folder -S $source_folder -DCMAKE_PREFIX_PATH=$source_folder -DCMAKE_INSTALL_PREFIX=$source_folder "${extra_flags[@]}"
  	cmake --build $build_folder --config Release --parallel $parallel
    #ninja -C build 2>&1 | grep "error:" | head -30
  else
    echo "[${library_name}][build.sh] Compile ${library_name} (output disabled) ... "
  	cmake -G Ninja -B $build_folder -S $source_folder -DCMAKE_PREFIX_PATH=$source_folder -DCMAKE_INSTALL_PREFIX=$source_folder "${extra_flags[@]}" > /dev/null 2>&1
  	cmake --build $build_folder --config Release --parallel $parallel > /dev/null 2>&1
  fi
}

# Parallel compile jobs: 4 on Linux; 3 on macOS (16 GB unified memory, 1 GB swap on the target laptop).
# Override with ALLFEATURE_BUILD_JOBS.
if [ -n "${ALLFEATURE_BUILD_JOBS:-}" ]; then
  parallel=$ALLFEATURE_BUILD_JOBS
elif [ "$(uname -s)" = "Darwin" ]; then
  parallel=3
else
  parallel=4
fi

# Check inputs: -f force (delete build/bin/lib first), -v verbose, -c CPU-only (no CUDA/TensorRT/libtorch/SiftGPU);
# flags may be combined (-fv, -cv, -fcv, ...)
force_build=false
verbose=false
cpu_only=false
for input in "$@"
do
    case "$input" in
      --cpu) cpu_only=true ;;
      -*)
        flags="${input#-}"
        [[ "$flags" == *f* ]] && force_build=true
        [[ "$flags" == *v* ]] && verbose=true
        [[ "$flags" == *c* ]] && cpu_only=true
        ;;
    esac
done

# Baseline Dir
LIBRARY_PATH=$(realpath "$0")
LIBRARY_DIR=$(dirname "$LIBRARY_PATH")

# Build AllFeature-VSLAM (Light_Glue_CPP and the other Thirdparty libraries are CMake subdirectories)
library_name="AllFeature-VSLAM"
source_folder="${LIBRARY_DIR}"
build_library ${library_name} ${source_folder} ${verbose} ${force_build}
