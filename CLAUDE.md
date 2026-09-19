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
```

Build order: `Thirdparty/Light_Glue_CPP` → main library (`cmake -G Ninja` + `cmake --build --parallel 4`, `-DCMAKE_INSTALL_PREFIX` set to its own source folder). `Thirdparty/SuperPoint-LightGlue-TensorRT`, `Thirdparty/Segmentation-TensorRT`, `Thirdparty/g2o`, `Thirdparty/PoseLib`, `Thirdparty/mimalloc` and `Thirdparty/placecell` are pulled in as CMake `add_subdirectory` from the main `CMakeLists.txt`.

This project only builds correctly inside the `allfeature-dev` pixi environment (from the parent `VSLAM-LAB` repo) or this repo's own `build` environment (`pixi run build` here; `pixi run setup` also downloads the models). Both provide CUDA/TensorRT, libtorch, OpenCV (headless), Eigen 3.4, Pangolin, yaml-cpp and the fontan-channel feature backends — see `pixi.toml`.

- **Standard:** C++17, `-Wall -Wextra -Wpedantic -Werror -O3 -march=native` (vendored code is compiled with `-w`)
- **Output:** `lib/libAllFeature-VSLAM.so`, `bin/vslamlab_allfeature_mono`, `bin/vslamlab_allfeature_rgbd`, `bin/vslamlab_allfeature_mono_stream`, plus test/benchmark binaries (`bin/test_bfmatcher_parity`, `bin/test_segmentation`, …)
- **Build dir:** `build/` (CMake 3.16+, Ninja) — `build/`, `bin/`, `lib/` and the `*_models/` folders are gitignored

There is no automated test suite or linter. Parity harnesses exist for the brute-force matcher and the segmentation engine (`src/test_bfmatcher_parity.cpp`, `bin/test_segmentation`). Building fills swap; the parent framework's swap watchdog then kills runs (VSLAM-LAB#119) — `pixi run kill-all` there resets it.

## Running

```bash
pixi run execute-mono    # ./bin/vslamlab_allfeature_mono (no args: prints usage)
pixi run execute-rgbd    # ./bin/vslamlab_allfeature_rgbd
pixi run stream          # ./bin/vslamlab_allfeature_mono_stream
```

The executables take `key:value` positional arguments, parsed by prefix in `src/vslamlab_allfeature_mono.cpp` / `_rgbd.cpp`:

```bash
./bin/vslamlab_allfeature_mono \
  sequence_path:<path> calibration_yaml:<path> rgb_csv:<path> \
  exp_folder:<path> exp_id:<id> settings_yaml:<path> verbose:<0|1>
```

`settings_yaml` (`vslamlab_allfeature-dev_settings.yaml`) is the single configuration file: `features: [...]`, `vpr: megaloc|none`, `feature_vpr`, `segmentation`, and the flat `Viewer.*`, `Optimizer.*`, `Tracking.*`, `PlaceRecognition.*`, `PlaceCell.*`, `LocalMapping.*`, `LoopClosing.*` blocks. Every block key is optional: a missing key keeps the compiled-in default (each class's `LoadParameters` reads only the keys present). Keep quotes on string values — `cv::FileStorage` swallows a trailing `# comment` into an unquoted string (issue #28).

Learned-model assets are resolved relative to the process cwd and downloaded by the pixi tasks (`pixi run setup` runs them all): `megaloc_models/` (HF `vslamlab/megaloc-models`), `segmentation_models/`, `superpoint_models/`, `lightglue_models/` (HF `vslamlab/allfeature-vslamlab`). TensorRT engines are built on first run and cached next to the ONNX. There is no DBoW2 vocabulary any more; place recognition is `vpr: megaloc` (TensorRT MegaLoc embeddings through `Thirdparty/placecell`) or `none`.

## Architecture

Four threads, mirroring ORB-SLAM2 (`docs/reference/` has one page per file):

```
System                    ← owns Map, PlaceRecognition, the threads; saves trajectories / point cloud / placecell dump
├── Tracking Thread       ← feature extraction (+ online segmentation mask), pose estimation, information-based keyframe decision
├── LocalMapping Thread   ← triangulation + depth-seeded points, map-point culling, local BA, keyframe culling (heuristic | information)
├── LoopClosing Thread    ← VPR loop detection (MegaLoc), Sim3 verification, loop correction, global BA (not started with vpr: none)
└── Viewer Thread         ← Pangolin 3D view, frame view, optional placecell window (verbose:1 only)
```

Cross-cutting components: `Feature`/`FeatureExtractor`/`FeatureMatcher` (multi-feature abstraction, see below), `Optimizer` (g2o; mono, stereo and custom RGB-D inverse-depth edges), `PlaceRecognition` (`include/PlaceRecognition.h`: `megaloc` / `none` backends; retrieval for relocalization and loops, unexplained-information queries for keyframe insertion and culling), `Frame`/`KeyFrame`/`MapPoint`/`Map`.

### Feature abstraction

`Feature` (`include/Feature.h`) is the abstract base class every feature type implements: `getFeatureName()`, `getType()` (`FeatureType`), `getMatcherType()` (`MatcherType`), `getSettingsYamlFile()`, `descriptor_distance()`, `createExtractor()`. Concrete features live as `Feature_<name>.h`/`.cpp` pairs (`Feature_orb32`, `Feature_akaze61`, `Feature_brisk48`, `Feature_surf64`, `Feature_kaze64`, `Feature_sift128`, `Feature_aliked128`, `Feature_superpoint256`). `include/FeatureFactory.h` maps `FeatureType`/name strings to concrete instances (`get_feature`, `get_feature_type`) — this is the single place to register a new feature type. Per-feature settings YAML files are in `settings/`.

`FeatureType` (`include/Types.h`): 0 orb32 · 1 akaze61 · 2 brisk48 · 3 surf64 · 4 kaze64 · 5 sift128 · 6 aliked128 · 7 superpoint256 (the last two learned; surf64 is non-functional, issue #1). The order of `features:` in the settings matters: several code paths treat `features[0]` specially (see the review pages before changing it).

### Matching strategies

`MatcherType` enum (`include/Types.h`): `BF_HAMMING`, `BF_L2`, `LIGHTGLUE_SIFT`, `LIGHTGLUE_ALIKED`, `LIGHTGLUE_SUPERPOINT`. Each `Feature` reports its own `MatcherType`; `FeatureMatcher.cpp` switches on it per call — binary/classical descriptors go through the SIMD/OpenMP brute-force matcher (`BruteForceMatcher`, cross-checked, bit-exact with `cv::BFMatcher`), learned features through LightGlue (`FeatureMatcher_lightglue.cpp`). Robust geometric filtering uses PoseLib (`filter_matches_by_fundamental`: fundamental → homography → pass-through).

## Key Files

| File | Role |
|---|---|
| `include/Types.h` | `FeatureType`/`MatcherType`/`VerbosityLevel` enums and Eigen typedefs |
| `src/System.cc` | wiring, thread start/shutdown, output files |
| `src/Tracking.cc`, `src/Tracking_aux.cc` | per-frame tracking, initialization, relocalization, keyframe decision |
| `src/LocalMapping.cc`, `src/LocalMapping_aux.cc` | keyframe processing, point creation, culling, local BA |
| `src/LoopClosing.cc`, `src/LoopClosing_aux.cc` | loop detection, Sim3, correction, global BA |
| `src/FeatureMatcher.cpp` | all matching logic, dispatches on `MatcherType` |
| `src/Optimizer.cc` | g2o-based BA, pose optimization, Sim3, essential graph; `OptimizerParameters` |
| `include/PlaceRecognition.h`, `src/PlaceRecognitionMegaLoc.cpp` | VPR backend interface and the MegaLoc backend (placecell) |
| `include/PlaceCellSettings.h`, `include/SegmentationSettings.h` | settings blocks for placecell diagnostics and online segmentation |
| `src/vslamlab_allfeature_mono.cpp`, `_rgbd.cpp`, `_mono_stream.cpp` | CLI entry points |
| `include/afvslam_log.hpp` | `AF_INFO`/`AF_WARN`/`AF_ERROR` logging (`AF_WARN` goes to stderr and is unbuffered; `AF_INFO` to stdout, fully buffered under the runner's redirect — flush after one-off lines) |
| `CMakeLists.txt`, `build.sh`, `pixi.toml` | build configuration |
| `Baselines/baseline_files/baseline_allfeature.py` (parent repo) | VSLAM-LAB integration |

## External Dependencies

| Dependency | Purpose |
|---|---|
| OpenCV (headless) | image processing, classical feature detection |
| Eigen 3.4 | linear algebra (5.x is incompatible with the vendored g2o) |
| Pangolin (fontan channel) | 3D visualization |
| yaml-cpp | settings parsing in the entry points |
| libtorch (pytorch-gpu) | LightGlue / ALIKED (via `Light_Glue_CPP`) |
| TensorRT, CUDA 12 | SuperPoint + LightGlue, MegaLoc, EfficientViT segmentation |
| brisk / akaze / SiftGPU (fontan channel) | classical feature backends |
| OpenMP | multi-threading (brute-force matcher, local mapping loops) |

**Git submodules** (`.gitmodules`): `Thirdparty/Light_Glue_CPP` (C++ LightGlue, VSLAM-LAB fork), `Thirdparty/SuperPoint-LightGlue-TensorRT`, `Thirdparty/g2o` (`alejandrofontan/g2o` fork: RGB-D inverse-depth edges, Levenberg acceptance fix), `Thirdparty/PoseLib`, `Thirdparty/mimalloc` (process allocator), `Thirdparty/placecell` (`alejandrofontan/placecell`: keyframe-lifecycle library, MegaLoc embedder, information culling; has its own `CLAUDE.md`). `Thirdparty/Segmentation-TensorRT` is not a submodule.

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
