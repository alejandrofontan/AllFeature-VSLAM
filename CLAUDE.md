# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A multi-feature Visual SLAM system (RSS 2024, "AnyFeature-VSLAM") built on ORB-SLAM2. It can be configured to switch between different classical and learned feature types during SLAM execution. This repo (`AllFeature-VSLAM-DEV`) is the **dev** variant of the baseline, run as a plugin within the parent **VSLAM-LAB** framework — see `Baselines/baseline_files/baseline_allfeature.py` (`ALLFEATURE_baseline_dev`) for how VSLAM-LAB invokes it (installation check, vocabulary download, command construction).

**Authors:** Alejandro Fontan, Javier Civera, Michael Milford

---

## Build

```bash
bash build.sh          # builds submodules then main library (skips already-built dirs)
bash build.sh -f       # force: delete build/bin/lib dirs first, then rebuild everything
bash build.sh -v       # verbose: show full cmake/ninja output (default suppresses it)
```

Build order: `Thirdparty/DBoW2` → `Thirdparty/Light_Glue_CPP` → main library (each is `cmake -G Ninja` + `cmake --build`, `-DCMAKE_INSTALL_PREFIX` set to its own source folder). `Thirdparty/SuperPoint-LightGlue-TensorRT` is *not* built by `build.sh` separately — it's pulled in as a CMake `add_subdirectory` from the main `CMakeLists.txt`.

This project only builds correctly inside the `allfeature-dev` pixi environment (from the parent `VSLAM-LAB` repo), which provides CUDA/TensorRT, OpenCV, Eigen3, Pangolin, yaml-cpp, g2o, and the fontan-channel packages (`brisk-vslamlab`, `akaze-vslamlab`, `siftgpu-vslamlab`, `g2o`) — see `pixi.toml`. There's no local build task defined here beyond `bash build.sh`; run it via `pixi run -e allfeature-dev bash Baselines/AllFeature-VSLAM-DEV/build.sh` from the parent repo, or `pixi run build` from this directory if already inside that shell/env.

- **Standard:** C++17, `-Wall -Wextra -Wpedantic -O3 -march=native`
- **Output:** `lib/libAllFeature-VSLAM.so`, `bin/vslamlab_allfeature_mono`, `bin/vslamlab_allfeature_mono_stream`
- **Build dir:** `build/` (CMake 3.16+, Ninja) — `build/`, `bin/`, `lib/` are all gitignored/untracked

There is no test suite or linter configured in this repo.

---

## Running

```bash
pixi run execute-mono    # ./bin/vslamlab_allfeature_mono, no args (prints usage)
pixi run stream          # ./bin/vslamlab_allfeature_mono_stream
```

The mono executable takes `key:value` style positional args (no flags), parsed by substring match in `src/vslamlab_allfeature_mono.cpp`'s `main()`:

```bash
./bin/vslamlab_allfeature_mono \
  sequence_path:<path> calibration_yaml:<path> rgb_csv:<path> \
  exp_folder:<path> exp_id:<id> settings_yaml:<path> \
  feature:<name> vocabulary_folder:<path> verbose:<0|1>
```

`settings_yaml` (e.g. `vslamlab_allfeature-dev_settings.yaml`) sets `features: [...]` (candidate feature list), camera name, and viewer/RGB-D params. `feature:` selects the active one at startup. Vocabulary files (DBoW2, one per feature family) live in `allfeature_vocabulary/` (gitignored — downloaded on demand by the Python baseline wrapper from the `fontan/anyfeature_vocabulary` HF dataset repo).

---

## Architecture

Multi-threaded pipeline (mirrors ORB-SLAM2's structure):

```
System
├── Tracking Thread       ← feature extraction, pose estimation, keyframe decision
├── LocalMapping Thread   ← triangulation, map point culling, local BA
├── LoopClosing Thread    ← DBoW2 loop detection, Sim3, global BA
└── Viewer Thread         ← Pangolin 3D visualization
```

### Feature abstraction

`Feature` (`include/Feature.h`) is the abstract base class every feature type implements: `getFeatureName()`, `getType()` (`FeatureType`), `getMatcherType()` (`MatcherType`), `getSettingsYamlFile()`, `descriptor_distance()`, `createExtractor()`. Concrete features live as `Feature_<name>.h`/`.cpp` pairs (`Feature_orb32`, `Feature_akaze61`, `Feature_brisk48`, `Feature_surf64`, `Feature_kaze64`, `Feature_sift128`, `Feature_aliked128`, `Feature_superpoint256`). `include/FeatureFactory.h` maps `FeatureType`/name strings to concrete instances (`get_feature`, `get_feature_type`) — this is the single place to register a new feature type.

### Feature types

Defined in `include/Types.h` as the `FeatureType` enum (8 types, values 0–7):

| ID | Name | Notes |
|---|---|---|
| 0 | orb32 | binary, classical |
| 1 | akaze61 | binary, classical |
| 2 | brisk48 | binary, classical |
| 3 | surf64 | float, classical |
| 4 | kaze64 | classical |
| 5 | sift128 | float, classical |
| 6 | aliked128 | float, learned |
| 7 | superpoint256 | float, learned, TensorRT |

Per-feature settings YAML files are in `settings/`.

### Matching strategies

`MatcherType` enum (`include/Types.h`): `BF_HAMMING`, `BF_L2`, `LIGHTGLUE_SIFT`, `LIGHTGLUE_ALIKED`, `LIGHTGLUE_SUPERPOINT`. Each `Feature` reports its own `MatcherType` via `getMatcherType()`; `FeatureMatcher.cpp` switches on it per-call (`SearchByProjection`, `SearchForTriangulation`, etc.) — binary/classical descriptors go through brute-force Hamming/L2 NN-ratio matching, the three learned feature types route through the LightGlue matcher (`FeatureMatcher_lightglue.cpp` / `FeatureMatcher_superglue.cpp`).

---

## Key Files

| File | Role |
|---|---|
| `include/Types.h` | `FeatureType`/`MatcherType`/`VerbosityLevel` enums and Eigen typedefs |
| `include/Feature.h` / `src/Feature.cpp` | Abstract feature base class |
| `include/FeatureFactory.h` | Name/enum ↔ concrete `Feature` subclass registry |
| `include/Feature_superpoint256.h` | SuperPoint feature (TensorRT-backed) |
| `src/Utils.cpp` | Printing, colors, statistics helpers |
| `src/Tracking.cc` | Main tracking loop |
| `src/FeatureMatcher.cpp` | All matching logic, dispatches on `MatcherType` |
| `src/Optimizer.cc` | g2o-based BA and pose optimization |
| `src/vslamlab_allfeature_mono.cpp` / `_stream.cpp` | CLI entry points (offline / streaming) |
| `src/createVocabulary.cpp` | DBoW2 vocabulary-building tool |
| `CMakeLists.txt` | Full build configuration |
| `Baselines/baseline_files/baseline_allfeature.py` (parent repo) | VSLAM-LAB integration: install check, vocabulary download, command building |

---

## External Dependencies

| Dependency | Purpose |
|---|---|
| OpenCV | Image processing, classical feature detection |
| Eigen3 | Linear algebra |
| Pangolin | 3D visualization |
| yaml-cpp | Config file parsing |
| g2o | Graph optimization (BA, pose graph) |
| brisk / akaze / SiftGPU (fontan channel packages) | Classical feature backends |
| TensorRT (via `Thirdparty/SuperPoint-LightGlue-TensorRT`) | SuperPoint feature + LightGlue matching, CUDA 12.x |
| OpenMP | Multi-threading |

**Git Submodules** (`.gitmodules`):
- `Thirdparty/DBoW2` — Bag-of-Words place recognition
- `Thirdparty/Light_Glue_CPP` — C++ LightGlue neural matcher
- `Thirdparty/SuperPoint-LightGlue-TensorRT` — SuperPoint + LightGlue via TensorRT (replaced the older SuperPoint-SuperGlue-TensorRT submodule)

---

## Notes

- License: GPLv3 (inherited from ORB-SLAM2)
- `bin/`, `lib/`, `build/`, and `allfeature_vocabulary/` are all build/runtime artifacts, not source — never assume content there is checked in
