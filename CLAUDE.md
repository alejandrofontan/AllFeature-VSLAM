# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A multi-feature Visual SLAM system built on ORB-SLAM2 (RSS 2024, "AnyFeature-VSLAM"). It runs several classical and learned feature types side by side and selects the place-recognition, keyframe and depth policies through a settings YAML. This repo (`AllFeature-VSLAM-DEV`) is the **dev** variant of the baseline, run as a plugin within the parent **VSLAM-LAB** framework — see `Baselines/baseline_files/baseline_allfeature.py` (`ALLFEATURE_baseline_dev`) for how VSLAM-LAB invokes it (installation check, model downloads, command construction).

**Authors:** Alejandro Fontan, Javier Civera, Michael Milford

## Documentation map

Everything that is not guidance for Claude lives under `docs/` (plan: `docs/plans/documentation_plan.md`). Read the page that matches the task before touching code:

| Where | What | When to update |
|---|---|---|
| `docs/reference/<File>.md` | what each source file does today (format: `docs/reference/README.md`; draft with `/document-file <Name>`) | in the same commit as a behaviour change, or say why not |
| `docs/review/<File>.md` | the author's dated reading notes, pinned to a commit (format: `docs/review/README.md`) | only by the author, append-only; Claude refreshes checklists with `docs/tools/review_checklist.py` |
| `docs/gym/` | measured effect of changes on VSLAM-LAB sequences: `protocol.md`, `board.md`, `entries/` | one entry per measured change (`docs/tools/gym_compare.py` prints the table) |
| `docs/notes/<date>_<slug>.md` | dated investigations | new file per investigation; never edit an old one, add a follow-up |
| `docs/plans/<slug>.md` | design plans with a status line | as the plan evolves |
| `allfeature.md` | index of the reference pages + the system diagram | when a reference page is added |
| GitHub issues | anything actionable (`gh issue list`) | file instead of writing TODOs into files |

Do **not** append investigations, audits or status updates to this file; write a `docs/notes/` or `docs/review/` page and, if it changes how to work here, one line in this file pointing at it. `python docs/tools/resolve_links.py` refreshes the `#L<n>` source links in every doc before a docs commit.

## Build

```bash
bash build.sh          # builds submodules then main library (skips already-built dirs)
bash build.sh -f       # force: delete build/bin/lib dirs first, then rebuild everything
bash build.sh -v       # verbose: show full cmake/ninja output (default suppresses it)
bash build.sh -c       # GPU-free: no TensorRT segmentation (ALLFEATURE_WITH_TENSORRT=OFF), in build-cpu/ (same bin/ and lib/); flags combine (-cv, -fcv)
```

**Learned models run on ONNX Runtime on every platform** (`docs/plans/learned_features_onnxruntime.md`, phase 4): `superpoint256`, `aliked128` and the three LightGlue matchers (`include/onnx/`, one `OrtSession` class + one thin wrapper per model, the process-wide `FeatureModels.h` configuration), `sift128` on OpenCV's CPU SIFT (`SiftCpu.h`), and the MegaLoc VPR backend through `placecell::megaloc` (`ALLFEATURE_WITH_MEGALOC`, ON everywhere). The execution provider — CUDA on Linux, CoreML or CPU on macOS — is a runtime setting (`features_provider`, `megaloc_provider`). The one remaining CMake backend option is `ALLFEATURE_WITH_TENSORRT` for online segmentation (`Thirdparty/Segmentation-TensorRT`, phase 5 ports it): ON on Linux, OFF on macOS; a `segmentation:` request without it is a clear `AF_ERROR` at startup (`SegmentationSettings.h`). The libtorch (`Light_Glue_CPP`), TensorRT SuperPoint-LightGlue and SiftGPU backends left in phase 4 step 6. `brisk` and `akaze` are found with `find_package(... QUIET)` and define `ALLFEATURE_WITH_BRISK` / `ALLFEATURE_WITH_AKAZE` when present (conda packages 1.0.1 from `VSLAM-LAB/conda-recipes` for linux-64, osx-arm64 and linux-aarch64, so every environment has them). The `cpu` pixi environment has none of the GPU packages (what the Mac has) and no tasks of its own, because a task name defined in two features makes a bare `pixi run <task>` ambiguous and breaks the parent's `pixi run --frozen execute-mono` forward: `pixi run -e cpu ./build.sh -c` builds there, `pixi run -e cpu ./bin/vslamlab_allfeature_mono ...` runs.

One CMake project (`cmake -G Ninja` + `cmake --build --parallel 4`, `-DCMAKE_INSTALL_PREFIX` set to its own source folder): `Thirdparty/Segmentation-TensorRT`, `Thirdparty/g2o`, `Thirdparty/PoseLib`, `Thirdparty/mimalloc` and `Thirdparty/placecell` are `add_subdirectory` of the main `CMakeLists.txt` (`Thirdparty/Light_Glue_CPP` and `Thirdparty/SuperPoint-LightGlue-TensorRT` are no longer built; phase 4 step 9 removes the submodules). `build.sh` passes `-DALLFEATURE_NATIVE=ON` (every C/C++ target, vendored ones included, built with `-march=native`). Off, the project builds for `x86-64-v3`; `cmake --install` lays out the executables, libraries, headers (`include/AllFeature-VSLAM/`) and reference settings (`share/allfeature-vslamlab/`). That is the conda package `allfeature`: `[package]` in `pixi.toml` (pixi-build-cmake, `ALLFEATURE_NATIVE=OFF`, `ALLFEATURE_OUTPUT_IN_SOURCE=OFF` so a package build never touches `bin/`/`lib/`), built on pushes and published to prefix.dev/vslamlab/vslamlab on `vX.Y.Z` tags by `.github/workflows/conda-package.yml`; locally `pixi publish --clean --to <dir>`.

This project only builds correctly inside the `allfeature-dev` pixi environment (from the parent `VSLAM-LAB` repo) or this repo's own `build` environment (`pixi run build` here; `pixi run setup` also downloads the models and exports SuperPoint); the GPU-free configuration also builds in the `cpu` environment (`pixi run -e cpu ./build.sh -c`), on linux-64 and osx-arm64. Both provide ONNX Runtime 1.31 (CUDA provider on linux-64, CoreML + CPU on osx-arm64), OpenCV 4.12, Eigen 3.4, Pangolin (`pangolin-opengl`), yaml-cpp and the brisk/akaze feature backends (prefix.dev/vslamlab/vslamlab, built from `VSLAM-LAB/conda-recipes`); linux-64 adds CUDA 12.9 and TensorRT 10.3 for segmentation — see `pixi.toml`.

- **Standard:** C++17, `-Wall -Wextra -Wpedantic -O3 -march=native` with `ALLFEATURE_NATIVE=ON` (off: `-march=x86-64-v3` on x86-64, the compiler default on arm64); vendored code is compiled with `-w`
- **Output:** `lib/libAllFeature-VSLAM.so`, `bin/vslamlab_allfeature_mono`, `bin/vslamlab_allfeature_rgbd`, `bin/vslamlab_allfeature_mono_stream`, `bin/vslamlab_allfeature_{mono,rgbd}_arstream` (with `libarstream`), plus test/benchmark binaries (`bin/test_bfmatcher_parity`, `bin/test_segmentation`, …)
- **Build dir:** `build/` (CMake 3.16+, Ninja) — `build/`, `bin/`, `lib/` and the `*_models/` folders are gitignored

There is no automated test suite or linter. Parity harnesses exist for the brute-force matcher and the segmentation engine (`src/test_bfmatcher_parity.cpp`, `bin/test_segmentation`). Building fills swap; the parent framework's swap watchdog then kills runs (VSLAM-LAB#119) — `pixi run kill-all` there resets it.

## Running

```bash
pixi run execute-mono           # ./bin/vslamlab_allfeature_mono (no args: prints usage)
pixi run execute-rgbd           # ./bin/vslamlab_allfeature_rgbd
pixi run stream                 # ./bin/vslamlab_allfeature_mono_stream (RTMP video input, e.g. Larix)
pixi run execute-mono-stream    # ./bin/vslamlab_allfeature_mono_arstream (ARStream relay, VSLAM-LAB `stream-iphone --run allfeature-dev`)
pixi run execute-rgbd-stream    # ./bin/vslamlab_allfeature_rgbd_arstream (... --mode rgbd, LiDAR depth)
```

The `*_arstream` entry points are built only when the `libarstream` conda package (prefix.dev/vslamlab/vslamlab, header-only ZeroMQ subscriber from github.com/VSLAM-LAB/arstream) is in the environment; they take `stream_url:` instead of `sequence_path:`/`rgb_csv:` and stop at the relay's end-of-stream message or Ctrl-C. The Viewer's `Pause` toggle (`System::set_paused`) blocks `System::Track` before the frame; the tracker relocalizes on resume (`Tracking::request_relocalization`).

The executables take `key:value` positional arguments, parsed by prefix in `src/vslamlab_allfeature_mono.cpp` / `_rgbd.cpp`:

```bash
./bin/vslamlab_allfeature_mono \
  sequence_path:<path> calibration_yaml:<path> rgb_csv:<path> \
  exp_folder:<path> exp_id:<id> settings_yaml:<path> verbose:<0|1>
```

`settings_yaml` (`vslamlab_allfeature-dev_settings.yaml`) is the single configuration file: `features: [...]`, `vpr: megaloc|none`, `feature_vpr`, `segmentation`, and the flat `Viewer.*`, `Optimizer.*`, `Tracking.*`, `PlaceRecognition.*`, `PlaceCell.*`, `LocalMapping.*`, `LoopClosing.*` blocks. Every block key is optional: a missing key keeps the compiled-in default (each class's `LoadParameters` reads only the keys present). Keep quotes on string values — `cv::FileStorage` swallows a trailing `# comment` into an unquoted string (issue #28).

Learned-model assets are resolved relative to the process cwd and fetched by the pixi tasks (`pixi run setup` runs them all): `megaloc_models/` (HF `vslamlab/megaloc-models`), `segmentation_models/` (HF `vslamlab/allfeature-vslamlab`), `feature_models/` (`download-feature-models`: the ALIKED and LightGlue exports + sidecars from the same HF repo; `export-superpoint` in the `export-features` env: SuperPoint is exported locally from Magic Leap's weights, which cannot be redistributed — `tools/export_features.py`, `docs/plans/learned_features_onnxruntime.md` Q6). The settings `features_models_dir` / `features_provider` / `features_threads` / `features_coreml_units` configure them (`include/onnx/FeatureModels.h`); `features_static_shapes: true` loads the fixed-shape exports instead (`<stem>_<W>x<H>.onnx` for the frame size, `lightglue_<f>_L<L>.onnx` padded to L keypoints — `export-features --static WxH --static-length L`), the Core ML path: it compiles only bounded shapes. The segmentation TensorRT engine is built on first run and cached next to its ONNX. There is no DBoW2 vocabulary any more; place recognition is `vpr: megaloc` (MegaLoc embeddings on ONNX Runtime through `Thirdparty/placecell`: `megaloc_onnx` names the file to load, the fp32 `megaloc_models/megaloc_322x322.onnx` or its `_fp16` twin for the CUDA / CoreML providers, on the provider `megaloc_provider` selects) or `none`.

## Architecture

Four threads, mirroring ORB-SLAM2 (`docs/reference/` has one page per file):

```
System                    ← owns Map, PlaceRecognition, the threads; saves trajectories / point cloud / placecell dump
├── Tracking Thread       ← feature extraction (+ online segmentation mask), pose estimation, information-based keyframe decision
├── LocalMapping Thread   ← triangulation + depth-seeded points, map-point culling, local BA, keyframe culling (heuristic | information)
├── LoopClosing Thread    ← VPR loop detection (MegaLoc), Sim3 verification, loop correction, global BA (not started with vpr: none)
└── Viewer Thread         ← Pangolin 3D view, frame view, optional placecell window (verbose:1 only)
```

On macOS the Viewer owns the process main thread instead (Cocoa requirement): the entry points hand their whole sequence run to `System::Run(work)`, which there runs `work` in an 8 MB-stack thread while the Viewer loop runs on main, and elsewhere just calls `work()` (same pattern as ORB-SLAM3-DEV).

Cross-cutting components: `Feature`/`FeatureExtractor`/`FeatureMatcher` (multi-feature abstraction, see below), `Optimizer` (g2o; mono, stereo and custom RGB-D inverse-depth edges), `PlaceRecognition` (`include/PlaceRecognition.h`: `megaloc` / `none` backends; retrieval for relocalization and loops, unexplained-information queries for keyframe insertion and culling), `Frame`/`KeyFrame`/`MapPoint`/`Map`.

### Feature abstraction

`Feature` (`include/Feature.h`) is the abstract base class every feature type implements: `getFeatureName()`, `getType()` (`FeatureType`), `getMatcherType()` (`MatcherType`), `getSettingsYamlFile()`, `descriptor_distance()`, `createExtractor()`. Concrete features live as `Feature_<name>.h`/`.cpp` pairs (`Feature_orb32`, `Feature_akaze61`, `Feature_brisk48`, `Feature_surf64`, `Feature_kaze64`, `Feature_sift128`, `Feature_aliked128`, `Feature_superpoint256`). `include/FeatureFactory.h` maps `FeatureType`/name strings to concrete instances (`get_feature`, `get_feature_type`) — this is the single place to register a new feature type. Per-feature settings YAML files are in `settings/`.

`FeatureType` (`include/Types.h`): 0 orb32 · 1 akaze61 · 2 brisk48 · 3 surf64 · 4 kaze64 · 5 sift128 · 6 aliked128 · 7 superpoint256 (the last two learned; surf64 is non-functional, issue #1). The order of `features:` in the settings matters: several code paths treat `features[0]` specially (see the review pages before changing it).

### Matching strategies

`MatcherType` enum (`include/Types.h`): `BF_HAMMING`, `BF_L2`, `LIGHTGLUE_SIFT`, `LIGHTGLUE_ALIKED`, `LIGHTGLUE_SUPERPOINT`. Each `Feature` reports its own `MatcherType`; `FeatureMatcher.cpp` switches on it per call — binary/classical descriptors go through the SIMD/OpenMP brute-force matcher (`BruteForceMatcher`, cross-checked, bit-exact with `cv::BFMatcher`), the three `LIGHTGLUE_*` types through the process-wide LightGlue sessions on ONNX Runtime (`onnx::shared_lightglue`, one per variant, keypoints normalised with the real image size; the descriptor-only path of every float descriptor is the brute force). Robust geometric filtering uses PoseLib (`filter_matches_by_fundamental`: fundamental → homography → pass-through).

## Key Files

| File | Role |
|---|---|
| `include/Types.h` | `FeatureType`/`MatcherType`/`VerbosityLevel` enums and Eigen typedefs |
| `src/System.cc` | wiring, thread start/shutdown, output files |
| `src/Tracking.cc`, `src/Tracking_aux.cc` | per-frame tracking, initialization, relocalization, keyframe decision |
| `src/LocalMapping.cc`, `src/LocalMapping_aux.cc` | keyframe processing, point creation, culling, local BA |
| `src/LoopClosing.cc`, `src/LoopClosing_aux.cc` | loop detection, Sim3, correction, global BA |
| `src/FeatureMatcher.cpp` | all matching logic, dispatches on `MatcherType` |
| `include/onnx/OrtSession.h`, `SuperPointOnnx.h`, `AlikedOnnx.h`, `LightGlueOnnx.h`, `FeatureModels.h`, `include/SiftCpu.h` | the learned feature models on ONNX Runtime and SIFT on the CPU (`tools/export_features.py` exports the models; `bin/test_feature_parity` checks them against the Python references) |
| `src/Optimizer.cc` | g2o-based BA, pose optimization, Sim3, essential graph; `OptimizerParameters` |
| `include/PlaceRecognition.h`, `src/PlaceRecognitionMegaLoc.cpp` | VPR backend interface and the MegaLoc backend (placecell) |
| `include/PlaceCellSettings.h`, `include/SegmentationSettings.h` | settings blocks for placecell diagnostics and online segmentation |
| `src/vslamlab_allfeature_mono.cpp`, `_rgbd.cpp`, `_mono_stream.cpp`, `_mono_arstream.cpp`, `_rgbd_arstream.cpp` | CLI entry points (offline, RTMP video, ARStream relay) |
| `include/afvslam_log.hpp` | `AF_INFO`/`AF_WARN`/`AF_ERROR` logging (`AF_WARN` goes to stderr and is unbuffered; `AF_INFO` to stdout, fully buffered under the runner's redirect — flush after one-off lines) |
| `CMakeLists.txt`, `build.sh`, `pixi.toml` | build configuration |
| `Baselines/baseline_files/baseline_allfeature.py` (parent repo) | VSLAM-LAB integration |

## External Dependencies

| Dependency | Purpose |
|---|---|
| OpenCV (headless) | image processing, classical feature detection |
| Eigen 3.4 | linear algebra (5.x is incompatible with the vendored g2o) |
| Pangolin (conda-forge `pangolin-opengl`) | 3D visualization |
| yaml-cpp | settings parsing in the entry points |
| ONNX Runtime (conda-forge `onnxruntime-cpp`; CUDA provider on linux-64, CoreML + CPU on osx-arm64) | SuperPoint, ALIKED, the LightGlue matchers (`include/onnx/`), MegaLoc (via `Thirdparty/placecell`) |
| TensorRT, CUDA 12 (linux-64) | EfficientViT segmentation |
| brisk / akaze (prefix.dev/vslamlab/vslamlab, `VSLAM-LAB/conda-recipes`) | classical feature backends (SIFT is OpenCV's) |
| OpenMP | multi-threading (brute-force matcher, local mapping loops) |

**Git submodules** (`.gitmodules`): `Thirdparty/Light_Glue_CPP` and `Thirdparty/SuperPoint-LightGlue-TensorRT` (no longer built; removed in phase 4 step 9), `Thirdparty/g2o` (`alejandrofontan/g2o` fork: RGB-D inverse-depth edges, Levenberg acceptance fix), `Thirdparty/PoseLib`, `Thirdparty/mimalloc` (process allocator), `Thirdparty/placecell` (`alejandrofontan/placecell`: keyframe-lifecycle library, MegaLoc embedder, information culling; has its own `CLAUDE.md`). `Thirdparty/Segmentation-TensorRT` is not a submodule.

## Working conventions

- **Never `git push`**, never mutate the remote (`gh pr merge`, …). Commit only on the user's request; feature branches off `main`, one commit per step when asked to implement a plan.
- **The user runs the experiments.** Do not launch `pixi run vslamlab`/`execute-*` runs to verify a change; build, report, and let the user run and share logs. Building is fine when wrapped in swap monitoring.
- **Logging:** `AF_INFO`/`AF_WARN`/`AF_ERROR` from `afvslam_log.hpp`; one line per event, no per-frame prints outside `PROFILING_EXHAUSTIVE`.
- **Style:** snake_case methods and members, trailing underscore on protected/private state (issue #32), `const`-correct queries, by-reference `shared_ptr` parameters. Newer files carry a module header docstring (Author/Version/Created/Updated) — issue #38 tracks the rest.
- **Settings keys:** every new tunable goes through the class's `LoadParameters` with a compiled default, is documented in the dev YAML with its default, and in the file's reference page settings table.
- **Vendored code:** fix `Thirdparty/*` upstream or in the fork the submodule points at, never in place.
- **Determinism:** `Tracking.Sequential: 1` + `features: [orb32]` + `vpr: none` + `verbose: 0` is the byte-identical configuration (issue #16); anything with learned features or loop closing is not.

## Notes

- License: GPLv3 (inherited from ORB-SLAM2)
- `../ORB-SLAM2-DEV` next to this checkout (a plain `--depth 1` clone of `raulmur/ORB_SLAM2`, gitignored by the parent) is the stock ORB-SLAM2 source the reference pages compare against; `../ORB-SLAM2` is the conda-packaged baseline and has no sources
- `bin/`, `lib/`, `build/` and `*_models/` are build/runtime artifacts, not source — never assume content there is checked in
