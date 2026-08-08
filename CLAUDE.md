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

## RGB-D Depth Integration Opportunities

This codebase is a fork of ORB-SLAM2, and a lot of ORB-SLAM2's original RGB-D scaffolding is still physically present but inert: `Frame::UnprojectStereo`/`ComputeStereoFromRGBD` and `KeyFrame::UnprojectStereo` are `std::terminate()` stubs (with the original depth-unprojection code visible commented out in `KeyFrame::UnprojectStereo`); `Frame`/`KeyFrame` already carry `mvuRight`/`mvDepth` (per-keypoint depth storage, always `-1` today); `Tracking` already has `mThDepth`, `mbf`, `mDepthMapFactor` members and a `GrabImageRGBD()` heuristic comment block, but no `GrabImageRGBD()` function exists — `System::Track()` (`src/System.cc`) unconditionally calls `tracker->GrabImageMonocular(im, timestamp)` regardless of `mSensor`; `Initializer.h` literally says *"THIS IS THE INITIALIZER FOR MONOCULAR SLAM. NOT USED IN THE STEREO OR RGBD CASE"*; `Sim3Solver`/`LoopClosing`/`Optimizer::OptimizeSim3`/`OptimizeEssentialGraph` already thread through an `mbFixScale` flag that `System.cc` already sets `true` for any non-monocular sensor; `MapPoint::increasePointObservability` already checks `mvuRight >= 0` to weight depth-verified observations double; `vslamlab_allfeature-dev_settings.yaml` already declares `Camera.bf`/`ThDepth`/`DepthMapFactor`. In short: depth integration here is mostly *finishing* an existing design, not inventing one from scratch.

**Cross-cutting constraint for every row below**: gate every new depth code path behind an explicit check (`mSensor==RGBD` and/or the specific `cv::Mat` being non-empty — same pattern already used for `img.mask`/`FilterKeypointsByMask`'s `if (mask.empty()) return;`). `Frame`/`KeyFrame`/`Tracking`/`LocalMapping`/etc. are shared by both sensor modes, so a change that isn't defensively gated risks breaking monocular, not just failing to help RGB-D. None of the monocular-only code (`Initializer`, `MonocularInitialization`, the 2-view `CreateNewMapPoints` triangulation path, etc.) should be removed — RGB-D paths should be added alongside it, selected at runtime by sensor mode, so monocular keeps working exactly as it does today if depth is missing, low quality, or the sensor is monocular.

Ranked by expected impact on tracking accuracy/robustness (highest first):

| # | Where (file:function) | What depth enables | Why it matters | Key challenges | Advantages | Drawbacks / risks |
|---|---|---|---|---|---|---|
| 1 | `Tracking::MonocularInitialization`/`CreateInitialMapMonocular` (`src/Tracking.cc`); `Initializer`/`CheckHomography`/`CheckFundamental`/`CheckRT` (`src/Initializer.cc`, explicitly "not used in RGBD case") | **Prefer depth-*robustified* monocular init over pure instant depth init.** Two mechanisms, not one: (a) fold a depth-consistency vote into `CheckHomography`/`CheckFundamental`/`CheckRT`'s existing inlier scoring — an independent check (not derived from the match itself) that also breaks monocular's classic H-vs-F ambiguity on near-planar/degenerate scenes; (b) replace `CreateInitialMapMonocular`'s arbitrary `invMedianDepth=1` scale step with a depth-informed one (robust statistic — e.g. median sensor-depth/triangulated-depth ratio — over the *already geometrically-verified* inlier points). Fall back to pure instant depth-only init (back-project every valid-depth keypoint, no second frame) only when monocular init can't find enough parallax within a frame budget | Keeps monocular init's built-in cross-verification (bad correspondences get rejected by the geometric model fit, not just trusted blindly) while still getting real metric scale from frame 0 and fixing monocular's actual weak point (planar/degenerate-motion scenes) — without inheriting pure instant depth init's risk of silently baking in sensor bias/noise with zero self-correction | Two init paths to maintain (depth-scaled monocular as primary, instant depth-only as fallback) plus fallback-trigger logic (frame/time budget on failed parallax); depth-consistency terms in `CheckHomography`/`CheckFundamental`/`CheckRT` must stay soft/RANSAC-style, not hard gates, so one bad depth pixel doesn't reject an otherwise-good correspondence; `Tracking::Track()`'s `NOT_INITIALIZED` branch needs the sensor-mode + fallback-trigger logic; needs a valid-depth-pixel policy (min/max range vs. `mThDepth`) for both paths | Preserves monocular's existing robustness guarantee while adding metric scale; directly attacks monocular's actual failure mode (planar/degenerate scenes) instead of just giving it units; still guarantees a start via the fallback when parallax genuinely isn't available; a bad/noisy depth pixel biases at most one vote, not the whole reconstruction | More implementation surface than either pure approach — two init paths, fallback-trigger logic, plus new depth-consistency scoring wired into three existing verification functions; monocular `Initializer`'s existing behavior must stay unchanged when depth is unavailable; more tuning surface (consistency threshold, fallback trigger budget) than a scale-only change |
| 2 | `Tracking::CreateNewKeyFrame` (`src/Tracking.cc`) — **not** `LocalMapping::CreateNewMapPoints`. Confirmed against stock ORB-SLAM2 (`ORB-SLAM2-DEV/src/LocalMapping.cc` has zero sensor-mode branches at all): `CreateNewMapPoints`'s epipolar/SVD triangulation stays fully sensor-agnostic and untouched; depth-based point creation is a separate, additive step in `Tracking::CreateNewKeyFrame`, run immediately at keyframe-creation time | The instant a keyframe is created, back-project every close (`depth < mThDepth`), still-unmatched keypoint straight from *that keyframe's own* depth map (sorted by depth, capped around ~100 close points if fewer exist) — no matching second keyframe or parallax required, mirrors row 1's `StereoInitialization` logic | Directly fixes monocular's most common practical robustness failure: sparse/empty map growth during rotation-only or slow-forward motion (corridors, indoor turns) | New code in `Tracking::CreateNewKeyFrame` (gated on `mSensor==RGBD`), reusing `KeyFrame::UnprojectStereo` from row 3; needs the same depth-sort-and-cap loop as row 1; `LocalMapping::CreateNewMapPoints` needs no changes at all and keeps contributing triangulated points on top | Denser, more robust map growth in exactly the scenarios monocular struggles most; smaller/more contained change than modifying `LocalMapping`'s neighbor-search triangulation would have been; composes with (doesn't replace) the existing 2-view triangulation | Depth sensor noise/quantization becomes map-point noise directly (loses the cross-view averaging 2-view triangulation gets "for free"); more created points = more culling/BA load |
| 3 | `Frame::UnprojectStereo`/`ComputeStereoFromRGBD` (`src/Frame.cc`), `KeyFrame::UnprojectStereo` (`src/KeyFrame.cc`) — all three currently `std::terminate()` stubs. Reference implementations confirmed nearly verbatim in `ORB-SLAM2-DEV/src/Frame.cc`/`KeyFrame.cc` | Every current-frame keypoint gets an immediate 3D position — the foundational primitive rows 1, 2, and 6 all build on | This is the one missing piece of plumbing everything else in this table needs; `mvuRight`/`mvDepth` are already declared in `Frame.h`/`KeyFrame.h` (always `-1` today) | Convert `Image::depthImg` via `DepthMapFactor` (declared in the settings YAML and `Tracking::mDepthMapFactor`, currently unused anywhere) into metric depth, write into `mvDepth`/synthetic `mvuRight = u - mbf/depth`; needs a real `GrabImageRGBD`-equivalent entry point (doesn't exist — `System::Track` always calls `GrabImageMonocular`); **gotcha confirmed from the reference**: `ComputeStereoFromRGBD` must sample the depth image at each keypoint's *distorted* pixel coordinates (`mvKeys`, matching how the depth image itself is indexed), while `UnprojectStereo` must back-project using the *undistorted* coordinates (`keypoints`/`mvKeysUn`) — easy to swap by mistake | Unlocks rows 1, 2, 6 with one shared implementation; matches the exact convention the rest of the codebase (row 9, row 8) already expects | Touches core `Frame`/`KeyFrame` construction — highest blast-radius item here, since both classes are shared by both sensor modes; needs careful regression testing against the monocular path |
| 4 | Cross-cutting consequence of rows 1–3; visible today in `CreateInitialMapMonocular`'s `invMedianDepth` scale normalization (arbitrary units) and `System::SaveTrajectoryTUM`/`SaveTrajectoryKITTI`'s `mSensor==MONOCULAR` refuse-to-run guard | Trajectory/map coordinates in real-world meters instead of arbitrary, drifting monocular scale | This is what RGB-D benchmarks actually score (ATE in meters) — without it, RGB-D mode only offers robustness, not the accuracy win it's known for | Not one code change — the emergent result of rows 1–3 being metric everywhere, with no leftover `invMedianDepth`-style rescaling anywhere in the pipeline | Directly comparable/usable trajectories and maps; unlocks `SaveTrajectoryTUM`/`SaveTrajectoryKITTI` for the RGBD executable | A single leftover monocular-style rescale anywhere silently reintroduces scale drift — needs an explicit audit once rows 1–3 land, not just "add depth and done" |
| 5 | `Sim3Solver` (already defaults `bFixScale=true`), `LoopClosing::ComputeSim3`/`CorrectLoop`, `Optimizer::OptimizeSim3`/`OptimizeEssentialGraph` — all already receive `mbFixScale`, already set `true` by `System.cc` for any non-monocular sensor | 6-DoF SE3 loop correction instead of 7-DoF Sim3 (no separate scale-drift correction to solve for) | Already wired and *active today* the instant `AF_VSLAM::System::RGBD` is used — but its value is currently theoretical, since the map it's "fixing the scale of" isn't metric yet | None to implement — this row is about not accidentally losing something already correct; its practical benefit is entirely gated on rows 1–3 | Free — already implemented and enabled; fewer DoF means faster, more numerically stable loop-closure optimization once scale is real | None on its own; risks reading as "RGB-D is already wired up" in isolation — necessary but not sufficient |
| 6 | `Tracking::Relocalization` (`src/Tracking.cc`), `PnPsolver` (`src/PnPsolver.cc`) | Every candidate keyframe match gets an immediately-known 3D point (via row 3) instead of relying only on points that survived into the map via prior triangulation | Relocalization after track loss is one of the harder monocular problems (few usable 3D-2D correspondences); depth densifies candidates for the existing PnP-RANSAC | `PnPsolver` needs no algorithm change (EPnP already consumes arbitrary 3D-2D pairs) — the work is sourcing extra correspondences from row 3's unprojection in `Tracking::Relocalization` | Reuses existing, unmodified PnP-RANSAC machinery; faster/more reliable recovery after track loss | Marginal benefit without rows 1–3 already in place (same dependency as row 5) |
| 7 | New logic — natural home is `Tracking::TrackLocalMap`/`SearchLocalPoints` or `LocalMapping::MapPointCulling`, alongside the just-added `FeatureExtractor::FilterKeypointsByMask` | Flag a tracked map point as an outlier when its predicted depth (from the current pose) diverges from the live depth reading at that pixel — catches motion-based dynamic objects the *mask* (category-based, static per-frame) can miss | Complements, doesn't duplicate, the mask-filtering work already landed: masks reject by *what* an object is, depth-consistency rejects by *whether it's moving* | Plumb a per-observation depth check into the existing outlier-marking path (`currentFrame.mvbOutlier`) without conflicting with the Huber-based reprojection outlier logic already there; needs a threshold relative to `mThDepth`/depth noise, not a fixed pixel count | Purely additive quality gate, easy to disable/tune independently of mask filtering; reuses depth already loaded for rows 1–3 | False positives near depth-map edges/occlusion boundaries (a common depth-sensor artifact) could wrongly reject valid static points if the threshold is too tight |
| 8 | `Tracking::NeedNewKeyFrame` (`src/Tracking.cc`) — `nTrackedClose`/`nNonTrackedClose` are declared and used in `bNeedToInsertClose` but hardcoded to `0`, never populated | Insert a keyframe more eagerly when too few nearby ("close", within `mThDepth`) points are tracked but many more could be created — the literal ORB-SLAM2 RGBD heuristic, already scaffolded but inert | Keyframe-insertion timing directly affects map density and tracking robustness; small, contained, low-risk fix once rows 1–3 exist | Compute `nTrackedClose`/`nNonTrackedClose` from `currentFrame.mvDepth`/`mThDepth` (both already present) instead of hardcoding `0` — small, localized change | Small, self-contained, matches original ORB-SLAM2 semantics closely — low implementation risk | No effect until rows 1–3 make `mvDepth` real; wrong `mThDepth` tuning for a given sensor could over-trigger keyframe insertion (perf cost) |
| 9 | `MapPoint::increasePointObservability`/`decreasePointObservability` (`src/MapPoint.cc`) — already implemented, checks `mvuRight.at(featureType)[projIndex] >= 0` to add 2 to `nObs` instead of 1 | Depth-verified observations count as stronger evidence than bearing-only monocular ones throughout the pipeline (`MapPointCulling`, `TrackedMapPoints`, `NeedNewKeyFrame`'s `nRefMatches` all read `nObs`) | Free correctness improvement across several quality gates simultaneously — dead logic waiting on row 3's `mvuRight` to be non–`-1` | None beyond row 3 landing — literally already implemented | Zero implementation cost; improves several downstream decisions at once | Could subtly change existing tuning constants' effective meaning (e.g. `MAP_POINT_CULLING_MIN_NUM_OBSERVATIONS`) once `nObs` starts incrementing by 2 — worth re-checking those thresholds after enabling |
| 10 | `Optimizer.cc`/`Optimizer.h` — currently only monocular reprojection edges are used in `PoseOptimization`/`LocalBundleAdjustment`/`BundleAdjustment`/`GlobalBundleAdjustemnt`; no stereo/RGBD g2o edge type exists. **Confirmed cheaper than initially estimated**: `g2o::EdgeStereoSE3ProjectXYZ`/`EdgeStereoSE3ProjectXYZOnlyPose` are standard types that ship with g2o itself (already a project dependency) — no custom edge to write, only wiring | A per-observation depth-residual term (as in stock ORB-SLAM2 RGBD) that pulls optimized 3D points toward their measured depth during BA, not just their reprojected pixel position | Refines convergence speed/accuracy, especially for points seen from few keyframes — a refinement on top of rows 1–3's initial depth-seeded positions, not a new capability on its own | Wire the existing g2o stereo edge types into all four optimization entry points, selecting per-observation exactly like the reference's `PoseOptimization`: `if(mvuRight[i]<0) { monocular 2-D edge } else { stereo/RGBD 3-D edge, e->bf = mbf }` — confirmed this is genuinely per-keypoint in stock ORB-SLAM2, not per-sequence, so a keypoint with invalid depth silently still gets the monocular edge even in RGB-D mode | Standard, well-understood technique (exactly what stock ORB-SLAM2 RGBD does); no new edge type to implement, just wiring; the per-observation dispatch *is* the fallback-safety principle from this section's intro, already proven at the finest possible granularity in the reference codebase | Lowest impact-per-effort item here given rows 1–3 already inject strong depth priors; touches the most performance-sensitive code path (BA runs continuously) |
| 11 | `Viewer.cc`/`MapDrawer.cc`, `System::SavePointCloudVSLAMLAB` (`src/System.cc`) | Export/visualize a dense or semi-dense point cloud from depth, not just the sparse feature-based map | Diagnostic/downstream-application value (mapping, robotics) — no effect on tracking accuracy or robustness | Mostly plumbing: sample `im.depthImg` per saved frame, transform to world frame with the already-tracked pose; independent of the rest of this list | Low risk, easy to gate behind a flag, doesn't touch tracking/mapping internals at all | Lowest priority — cosmetic/diagnostic only, no accuracy or robustness benefit; larger output files |

---

## Notes

- License: GPLv3 (inherited from ORB-SLAM2)
- `bin/`, `lib/`, `build/`, and `allfeature_vocabulary/` are all build/runtime artifacts, not source — never assume content there is checked in
- See `TODO.md` for tracked cleanup items (bugs, dead code, duplication) found in the CLI entry points
