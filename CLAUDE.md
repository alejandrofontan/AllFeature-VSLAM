# AllFeature-VSLAM — Project Context

## What This Is

A multi-feature Visual SLAM system (RSS 2024) built on ORB-SLAM2. It automatically switches between 11 feature types during SLAM execution. Runs as a baseline within the **VSLAM-LAB** framework.

**Authors:** Alejandro Fontan, Javier Civera, Michael Milford

---

## Build

```bash
bash build.sh   # builds submodules then main library
```

Build order: DBoW2 → Light_Glue_CPP → SuperPoint-LightGlue-TensorRT → main

- **Standard:** C++17, `-O3 -march=native`
- **Output:** `lib/libAllFeature-VSLAM.so`, `bin/vslamlab_allfeature_mono`, `bin/vslamlab_allfeature_mono_stream`
- **Build dir:** `build/` (CMake 3.16+, Ninja)

---

## External Dependencies

| Dependency | Purpose |
|---|---|
| OpenCV | Image processing, classical feature detection |
| Eigen3 | Linear algebra |
| Pangolin | 3D visualization |
| yaml-cpp | Config file parsing |
| g2o | Graph optimization (BA, pose graph) |
| brisk / akaze / SiftGPU | Classical feature backends |
| OpenMP | Multi-threading |

**Git Submodules:**
- `DBoW2` — Bag-of-Words place recognition
- `Light_Glue_CPP` — C++ LightGlue neural matcher
- `Thirdparty/SuperPoint-LightGlue-TensorRT` — SuperPoint + LightGlue via TensorRT

---

## Architecture

Multi-threaded pipeline:

```
System
├── Tracking Thread       ← feature extraction, pose estimation, keyframe decision
├── LocalMapping Thread   ← triangulation, map point culling, local BA
├── LoopClosing Thread    ← DBoW2 loop detection, Sim3, global BA
└── Viewer Thread         ← Pangolin 3D visualization
```

---

## Supported Feature Types

Defined in `include/Types.h` as `FeatureType` enum:

| ID | Name | Descriptor |
|---|---|---|
| 0 | orb32 | 32-bit binary |
| 1 | akaze61 | 61-bit binary |
| 2 | brisk48 | 48-bit binary |
| 3 | sift128 | 128-dim float |
| 4 | kaze64 | 64-bit |
| 5 | surf64 | 64-bit float |
| 6 | r2d2_128 | 128-dim learned |
| 7 | anyFeatBin | binary, arbitrary |
| 8 | anyFeatNonBin | float, arbitrary |
| 9 | aliked128 | 128-dim learned |
| 10 | superpoint256 | 256-dim, TensorRT |

Settings YAML per feature type are in `settings/`.

---

## Matching Strategies

- **Method 0:** NN-ratio brute-force
- **Method 1:** LightGlue neural network
- **Method 2:** Robust RANSAC + epipolar constraints

Implemented in `src/FeatureMatcher.cc`, `src/FeatureMatcher_lightglue.cpp`, `src/FeatureMatcher_superglue.cpp`.

---

## Key Files

| File | Role |
|---|---|
| `include/Types.h` | Enums (FeatureType, DescriptorType, VerbosityLevel) and Eigen typedefs |
| `include/Feature.h` / `src/Feature.cpp` | Base feature class |
| `include/Feature_superpoint256.h` | SuperPoint feature (TensorRT) |
| `src/Utils.cpp` | Printing, colors, statistics helpers |
| `src/Tracking.cc` | Main tracking loop |
| `src/FeatureMatcher.cc` | All matching logic |
| `src/Optimizer.cc` | g2o-based BA and pose optimization |
| `CMakeLists.txt` | Full build configuration |

---

## Usage

```bash
./bin/vslamlab_allfeature_mono "sequence_path:docs/toy_sequence"
```

Vocabulary files are required for DBoW2 place recognition (see README for creation).

---

## Notes

- License: GPLv3 (inherited from ORB-SLAM2)
- `Feature.h` and `Feature.cpp` are untracked new files (in-progress work)
- `Thirdparty/SuperPoint-LightGlue-TensorRT` submodule recently replaced SuperPoint-SuperGlue-TensorRT
