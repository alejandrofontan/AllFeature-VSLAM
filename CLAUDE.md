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

**Row 10 status update (post-review, see the Optimizer section below):** substantially implemented — `PoseOptimization`/`BundleAdjustment`/`LocalBundleAdjustment` all now dispatch per-observation across mono/stereo/RGBD edges. One correction to row 10's original premise: the RGB-D depth channel is **not** wired through the stock g2o stereo edge (`EdgeStereoSE3ProjectXYZ`, disparity-based, needs `bf`). It uses two **new custom edges**, `g2o::EdgeRGBDSE3ProjectXYZ`/`EdgeRGBDSE3ProjectXYZOnlyPose` (inverse-depth residual, no `bf`), vendored into a local, untracked `Thirdparty/g2o/` copy — see finding **P1** below, since this changes the risk profile row 10 originally assumed ("no custom edge to write, only wiring").

**Row 2 status update (2026-08-10):** landed, but not where row 2 originally proposed, and a real bug was found and fixed along the way. Someone had already added depth-based point creation to `LocalMapping::CreateNewMapPoints` (not `Tracking::CreateNewKeyFrame` as row 2 suggested) — but nested it *inside* the neighbor-matched-pairs loop (`vMatchedIndices`, from `match_keyframes_for_triangulation`), so a keypoint only reached the depth check if it *already* had a 2D feature match against some covisible neighbor. A keypoint with a perfectly valid sensor-depth reading but no such match (textureless region, repeated pattern, fast motion — exactly the cases row 2 called out RGB-D depth as most valuable for) was silently dropped and never became a map point, defeating the point of not needing a match at all. Fixed by adding a second, independent pass at the end of `CreateNewMapPoints` that iterates every keypoint of the current keyframe directly (not the matched-pairs list), skips ones with no valid depth or an existing map point (via the newly-exposed `KeyFrame::get_map_point`), and back-projects the rest with the single-observation `KeyFrame::CreateMapPoint` (already existed, previously unused/dead code — exactly the primitive this needed). No `mThDepth`/close-far range gate or point-count cap was added, deliberately matching the existing (already-shipped) matched-pairs depth branch's own criterion of "any `invDepth>0` is trusted" — row 2's original "capped around ~100" idea was aimed at a from-scratch dense-backprojection design; here it would just be an inconsistency against the sibling code path already in production.

---

## Optimizer.cc / Optimizer.h Review (2026-08-09)

Findings from a read-through of `src/Optimizer.cc` + `include/Optimizer.h`, cross-checked against `Frame.h`/`Frame.cc`, `KeyFrame.h`/`KeyFrame.cc`, `Tracking.cc`, `System.cc`, `CMakeLists.txt`, and the vendored `Thirdparty/g2o` edge types. No code was changed for this pass — this is a TODO list to work through deliberately, in roughly priority order within each group. `invDepth`-based RGB-D edges referenced below are already live (see rows 210-243, 424-440, 660-678 of `Optimizer.cc`), i.e. table row 10 above is further along than it looked before this review.

### P — Process / build risk (found while tracing the g2o edge types)

| # | Where | What | Why it matters | Recommendation |
|---|---|---|---|---|
| P1 | `CMakeLists.txt` (`git diff` shows `find_package(g2o REQUIRED)` replaced by `add_subdirectory(Thirdparty/g2o ...)`); `Thirdparty/g2o/` is untracked (`git status`: `?? Thirdparty/g2o/`), not in `.gitmodules`, not in `.gitignore` | The parent repo's `CLAUDE.md` lists `g2o` as a fontan-channel **pixi package** dependency (`find_package(g2o)`). That's been switched for a locally vendored, hand-patched g2o source tree so it can host two new edge types (`EdgeRGBDSE3ProjectXYZ`, `EdgeRGBDSE3ProjectXYZOnlyPose`, in `Thirdparty/g2o/g2o/types/types_six_dof_expmap.{h,cpp}`) that don't exist upstream | This is real, working, uncommitted code sitting outside git's tracking entirely. A fresh clone, a `pixi run install-baseline`, or anyone else's checkout gets none of it — and there's nothing to signal that a hard dependency swap happened, since `find_package(g2o)` failing silently falls back to whatever g2o the pixi env still provides (link errors on the new symbols, not "g2o missing") | Turn `Thirdparty/g2o` into a proper submodule (mirroring `Thirdparty/DBoW2`'s `alejandrofontan/DBoW2` fork pattern — fork g2o, add the two edge types there, point `.gitmodules` at the fork) before this depth work is considered done. Until then, treat every `EdgeRGBDSE3ProjectXYZ*` reference in `Optimizer.cc` as depending on uncommitted local state |

### B — Correctness bugs / risks

| # | Where | What | Why it matters | Recommendation |
|---|---|---|---|---|
| B1 | `Optimizer.h:46-49`, `Optimizer.cc:46-49` — `chi2_2dof`/`chi2_3dof`/`thHuber_2dof`/`thHuber_3dof` declared `static float` (mutable, not `const`) and `thHuber_*` computed **once** as `sqrtf(chi2_*)` at static-init time | These four are independent static storage, not a derived accessor — if anything ever reassigns `chi2_2dof` after startup (which is exactly what the `OptimizerParameters` loader proposed below would do), `thHuber_2dof` goes stale and silently desyncs from it (Huber delta no longer `sqrt(chi2)`) | Directly relevant to the `OptimizerParameters` design below: compute `thHuber_*` **inside** the loader right after reading `chi2_*`, never store them as independently-settable YAML keys |
| B2 | `Optimizer.cc:479-493`, `PoseOptimization`'s 4-pass loop: `if(optimizer.edges().size()<10) break;` | `optimizer.edges()` returns every edge ever added to the graph, regardless of `setLevel(1)` (outlier) status — outlier edges are never `removeEdge`'d, only disabled. So this count is identical across all 4 iterations; it can only ever fire on the very first check (already implied by the earlier `nInitialCorrespondences<3` guard) or never. Inherited near-verbatim from stock ORB-SLAM2, so not newly introduced, but it's dead/misleading logic worth either removing or replacing with an actual per-iteration inlier count (`nInitialCorrespondences - nBad`) | Low risk, cosmetic-but-confusing; worth a one-line fix (`if(nInitialCorrespondences - nBad < 10) break;`) next time this function is touched, not urgent on its own |
| B3 | `Optimizer.cc:161,383,578` — `constexpr double invDepthInfo = 20000.0;` duplicated verbatim (value + comment) in `BundleAdjustment`, `PoseOptimization`, `LocalBundleAdjustment` | Unlike the x/y information (`GetKeyPt2DInf`/`GetKeyPt3DInf`, scale-aware per keypoint), the RGB-D depth channel's information is one hardcoded global constant, already self-flagged in the existing comment as "a starting guess, not yet validated against real sequences (see issue #3)". Three copies means a future re-tune has to touch three call sites in lockstep or drift apart | This is the headline motivating case for the `OptimizerParameters` struct below — one source of truth instead of three `constexpr` copies |
| B4 | `Optimizer.cc:882-982`, `OptimizeEssentialGraph`'s "Set normal edges" loop iterates `vpKFs` with **no** `if(pKF->is_bad()) continue;` (contrast with the "Set KeyFrame vertices" loop just above it, which does skip bad KFs and thus never populates their `vScw[nIDi]` entry — left at whatever `g2o::Sim3`'s default constructor gives, i.e. identity) | If a bad keyframe reaches this loop still holding a live `GetParent()`/covisibility link, `Swi = vScw[nIDi].inverse()` for it resolves to an uninitialized identity Sim3 instead of a real pose, silently injecting a wrong constraint. This mirrors stock ORB-SLAM2's structure closely enough that it may be relying on an invariant elsewhere (e.g. `KeyFrameCulling` always clearing parent/child/covisibility links before marking a KF bad, under `mMutexMapUpdate`) that I did not fully trace end-to-end | Flagging as **audit, not confirmed live bug** — worth 30 minutes tracing `KeyFrame::SetBadFlag` to confirm the invariant holds before deciding whether a defensive `is_bad()` skip is needed here |

### C — Cleanup / duplication

| # | Where | What | Recommendation |
|---|---|---|---|
| C1 | `Optimizer.cc:35-41` | `<random>`, `<fstream>`, `<iomanip>`, `<iostream>` are `#include`d but nothing in the file uses `mt19937`/`uniform_*`, `ofstream`/`ifstream`, `setprecision`/`setw`, or `cout`/`cerr` | Delete the four unused includes |
| C2 | Same as B3 | `invDepthInfo` triplicated | Fold into `OptimizerParameters` (see below) |
| C3 | `Optimizer.cc:612-637` (`LocalBundleAdjustment`) vs. `Optimizer.cc:393-404` (`PoseOptimization`) | `LocalBundleAdjustment` keeps 9 flat parallel vectors (`vpEdgesMono`/`vpEdgeKFMono`/`vpMapPointEdgeMono`, ×3 for stereo, ×3 for RGBD) where `PoseOptimization` already uses `std::map<FeatureType, vector<...>>`-keyed containers via the newer `CreateBAEdge`/`MarkBAOutliers`/`CollectBAOutliers` template helpers. The two functions now use two different container idioms for the same per-edge-type bookkeeping problem | Not urgent, but if `LocalBundleAdjustment` is touched again, consider a small `struct Observation{EdgeT* edge; Keyframe kf; Pt mp;}` per category to cut 9 vectors to 3 and match the tidier pattern already established elsewhere in this file |

### M — Memory / speed

| # | Where | What | Recommendation |
|---|---|---|---|
| M1 | `Optimizer.cc:610` — `const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*llocalPts.size();` then reserved for all 9 vectors in C3 | This is the KF×point upper bound, not the actual edge count (most local points aren't seen by most local/fixed KFs) — inherited from stock ORB-SLAM2, but the multi-feature architecture here means `llocalPts` can span up to 8 feature types at once, making local maps (and this over-reservation) proportionally larger than plain ORB-SLAM2's | Low priority — only worth revisiting if `LocalBundleAdjustment` shows up in a memory profile on a large multi-feature map |
| M2 | `Optimizer.cc:393-404`, `PoseOptimization`'s six `std::map<FeatureType, vector<...>>` containers, indexed via `.at(ft)` inside the per-keypoint hot loop (lines 408-465) | `PoseOptimization` runs once per tracked frame — it's on the real-time critical path. Map (`.at()`) lookups per keypoint, inside a loop already iterating every keypoint of every feature type, cost more than plain vector indexing would | Minor; each `for (auto& [ft, pts] : pFrame->pts)` outer iteration could resolve `vpEdgesMono.at(ft)`/`vnIndexEdgeMono.at(ft)`/etc. **once** into local references before the inner `for(int i...)` loop, instead of re-doing the `.at(ft)` lookup on every `push_back` |
| M3 | `new EdgeT()`/`new g2o::RobustKernelHuber` per observation throughout | One heap allocation per edge/kernel is inherent to g2o's ownership model (`SparseOptimizer` frees vertices/edges in its destructor) — not a leak, not realistically avoidable without changing g2o itself. Documenting so it isn't mistaken for a bug later, not proposing a change | No action |

### A — Accuracy / robustness

| # | Where | What | Recommendation |
|---|---|---|---|
| A1 | `chi2_3dof`/`thHuber_3dof` shared between stereo-disparity edges and RGB-D inverse-depth edges (`Optimizer.cc:222,458,673` use `thHuber_3dof` for RGBD same as stereo) | Statistically fine *if* both channels' information matrices are correctly calibrated into the same dimensionless Mahalanobis scale — but `invDepthInfo` (B3) is explicitly an unvalidated placeholder, so any miscalibration there skews RGB-D inlier/outlier decisions while looking identical to (already-tuned) stereo/mono thresholds in the code | Once `invDepthInfo` is calibrated against real sequences, consider splitting an independently-tunable `chi2RGBD`/`thHuberRGBD` (already scaffolded as separate `chi2RGBD[4]` array in `PoseOptimization` line 475, just currently initialized to the same value as `chi2_3dof`) |
| A2 | Huber is the only robust kernel used anywhere in this file (`g2o::RobustKernelHuber`) | g2o also ships `RobustKernelCauchy`/`RobustKernelDCS`/`RobustKernelTukey`. Depth-sensor error (structured-light/ToF quantization, multipath, "flying pixels" at depth discontinuities) tends to be heavier-tailed than reprojection pixel noise, and Dynamic Covariance Scaling (DCS) in particular is a common choice in RGB-D SLAM/pose-graph literature for exactly this failure mode | Speculative — worth an experiment only after A1's calibration lands, not before; flagging so it isn't forgotten |
| A3 | `include/g2o` custom edges `EdgeRGBDSE3ProjectXYZ`/`EdgeRGBDSE3ProjectXYZOnlyPose` (`Thirdparty/g2o/g2o/types/types_six_dof_expmap.{h,cpp}`) | Hand-verified their `linearizeOplus()` against the analytic chain rule for `error = obs - (u, v, 1/z)` during this review — the derivation is consistent with the existing (field-tested) `EdgeStereoSE3ProjectXYZ` pattern, so no math bug found. But unlike upstream g2o types, these are brand new and have **no automated regression test** (e.g. finite-difference Jacobian check) — and per P1 they aren't even committed yet | Recommend a small standalone unit test that perturbs pose/point by ε and compares against `_jacobianOplusXi`/`_jacobianOplusXj` before this ships, since these edges sit on the critical path for every RGB-D BA/pose-optimization call |

### `OptimizerParameters` — proposed static struct

Goal: replace the 9 scattered `static` tunables in `Optimizer.h` (5 "Heuristics", 8 "Constants" — 4 of the "Heuristics" have the `thHuber`-desync footgun in B1) plus the 3 duplicated `invDepthInfo` locals (B3) with **one** static, YAML-loadable struct, mirroring how `Tracking`/`FeatureExtractorSettings` already load their settings from `strSettingsFile` via `cv::FileStorage` (see `Tracking::ChangeCalibration`, `src/Tracking.cc:1141-1172`, which reads flat `fSettings["Camera.bf"]`-style keys — the same style `vslamlab_allfeature-dev_settings.yaml` already uses for `Camera.bf`/`ThDepth`/`DepthMapFactor`).

**Struct sketch** (in `include/Optimizer.h`, inside `class Optimizer` or alongside it in `namespace AF_VSLAM`):

```cpp
struct OptimizerParameters
{
    // Robust-kernel / outlier thresholds (thHuber_* are DERIVED, not settable directly — see B1)
    float chi2_2dof{5.991f};
    float chi2_3dof{7.815f};
    float chi2_3dof_rgbd{7.815f};   // new: split from chi2_3dof per A1, same default until calibrated
    float thHuber_2dof{sqrtf(chi2_2dof)};
    float thHuber_3dof{sqrtf(chi2_3dof)};
    float thHuber_3dof_rgbd{sqrtf(chi2_3dof_rgbd)};

    // RGB-D depth-channel information (replaces the 3x-duplicated invDepthInfo constant, B3/C2)
    double invDepthInfo{20000.0};

    // PoseOptimization()
    int numItPoseOpt{10};

    // OptimizeEssentialGraph()
    double userLambdaInit{1e-16};
    int minFeat{100};
    int numItEssGraphOpt{20};

    // OptimizeSim3()
    int numItSim3Opt{5};
    int nMoreItHigh{10};
    int nMoreItLow{5};
    int minNCorrespondences{10};
};
```

**Loading, in `class Optimizer`:**

```cpp
static OptimizerParameters params;
static void LoadParameters(const cv::FileStorage& fSettings);
```

`LoadParameters` reads each key defensively (only overwrite the compiled-in default `if(!fSettings["Optimizer.XYZ"].empty())`), so existing settings YAMLs without an `Optimizer.*` block — including today's `vslamlab_allfeature-dev_settings.yaml`, which has none — keep working unchanged. It recomputes `thHuber_*` from `chi2_*` at the end of the function, closing B1. Proposed key names, flat-style to match the existing `Camera.bf`/`ThDepth` convention: `Optimizer.Chi2Mono`, `Optimizer.Chi2Stereo`, `Optimizer.Chi2RGBD`, `Optimizer.InvDepthInfo`, `Optimizer.PoseOptIterations`, `Optimizer.EssGraphLambdaInit`, `Optimizer.EssGraphMinFeat`, `Optimizer.EssGraphIterations`, `Optimizer.Sim3Iterations`, `Optimizer.Sim3MoreIterationsHigh`, `Optimizer.Sim3MoreIterationsLow`, `Optimizer.Sim3MinCorrespondences`.

**Call site:** `System::System()` (`src/System.cc`), right after the existing `fsSettings.isOpened()` check (line 47) and before `tracker = make_shared<Tracking>(...)` (line 84) — `Optimizer::LoadParameters(fsSettings);`. This guarantees every thread spawned later in the same constructor (`LocalMapping`, `LoopClosing`, both of which call into `Optimizer::LocalBundleAdjustment`/`OptimizeEssentialGraph`/`OptimizeSim3`) observes fully-populated params before doing any work, since none of those threads start running until the constructor's `make_shared<thread>(...)` calls near the end.

**Migration:** every bare reference in `Optimizer.cc` (`chi2_2dof`, `thHuber_3dof`, `minFeat`, `userLambdaInit`, ... — roughly 20 call sites) becomes `params.chi2_2dof`, `params.thHuber_3dof`, etc. Mechanical but should land as one focused pass/PR rather than mixed into unrelated changes.

**Benefits:** one source of truth instead of 9 scattered statics + 3 duplicated locals; fixes the `thHuber` staleness footgun (B1) as a side effect; makes future re-tuning (the `invDepthInfo` autotuner already mentioned in the existing "issue #3" code comments) a YAML edit instead of a recompile; documents every optimizer magic number in one place.

**Drawbacks/risks:** touches ~20 call sites across a file that's on every hot path (tracking, local mapping, loop closing) — needs careful review/testing, not a drive-by edit; turns former `constexpr`/`const` values into runtime-mutable state (negligible perf cost, since these are read once per BA/pose-opt call, not per inner-solver iteration); must stay backward-compatible for every other settings YAML in the repo (baselines/datasets that don't know about `Optimizer.*` keys yet) — silent-default-on-missing-key is a hard requirement, not a nice-to-have.

**Status update (2026-08-10): `OptimizerParameters` implemented, and B3/C2/A1's `invDepthInfo` gap closed.** The struct above (`include/Optimizer.h`) and `Optimizer::LoadParameters` are now real code, not just a proposal — `chi2_*`/`thHuber_*`/`invDepthInfo`/etc. are all `params.*` reads today, so most of this section is historical context for *why* the struct looks the way it does rather than a pending TODO.

`invDepthInfo` itself (B3/C2) is no longer a single hardcoded global shared by every RGB-D observation. `Frame`/`KeyFrame` gained a `sigma2invDepth` member (`std::map<FeatureType, std::vector<float>>`, same shape as `invDepth`), computed in `Frame::GetDepth` right alongside `invDepth` from a quadratic depth-sensor noise model (`sigma_depth(z) = k·z²`, `k=0.0028` starting estimate) and propagated through Frame's copy constructor and the `KeyFrame(Frame&, ...)` constructor exactly like `invDepth` already was. `Optimizer.cc`'s three RGB-D edge sites (`BundleAdjustment`, `PoseOptimization`, `LocalBundleAdjustment`) now call a small `RGBDInvDepthInformation(sigma2invDepth_i)` helper (right next to `CreateBAEdge`) that returns `1/sigma2invDepth_i` per observation, falling back to `params.invDepthInfo` only if `sigma2invDepth_i` is non-positive (defensive; shouldn't happen since both fields are always set together). `params.invDepthInfo` itself remains as that fallback default, still YAML-configurable via `Optimizer.InvDepthInfo`.

A1's premise changes slightly as a result: RGB-D inverse-depth information is no longer *uniformly* miscalibratable alongside stereo's — each observation now carries its own variance from the sensor-noise model, so a bad calibration would show up as a *shape* mismatch (wrong quadratic coefficient `k`) rather than a single global constant being off. `k=0.0028` is still an unvalidated placeholder (same caveat B3 originally flagged for the old constant) — worth revisiting once real RGB-D sequences are available to fit against.

---

## Build-Time Memory/Swap Investigation (2026-08-09)

User-reported symptom: system memory/swap fills up while compiling (`bash build.sh -fv` via `pixi run build -fv`), which reads like "a memory leak" even though nothing is running yet — it's the *build* consuming the memory, not the SLAM program. Investigated read-only: no builds were run, no files changed. Inputs used: the user's build log (`/home/alejandro/VSLAM-LAB/test.txt`, a full `-fv` run that did complete, 82/82 ninja steps), every `CMakeLists.txt` in this repo and its three git-submodule Thirdparty trees, the existing (already-built) `build/` directories' `compile_commands.json` and `CMakeCache.txt`, a dry `ninja -t commands`/`ninja -t targets` query (prints commands without executing them), and direct binary inspection (`ar`/`readelf`) of an already-built static library.

**Machine facts** (`free -h`, `nproc`, `/proc/swaps` at investigation time): 28 cores, 31 GiB RAM, 8 GiB swap. Baseline (before any build) already showed ~18 GiB "used", leaving only ~9.3 GiB free / ~12 GiB "available" — the build doesn't need to be enormous in absolute terms to spill into swap, it just needs to exceed that headroom.

### Primary finding (high confidence): unthrottled LTO in the vendored `Light_Glue_CPP` submodule, consumed by a non-LTO final link

`Thirdparty/Light_Glue_CPP/CMakeLists.txt:13-18` turns on whole-program optimization for any Release build:
```cmake
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)
if(IPO_SUPPORTED AND CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()
```
`git log -p` on this file (submodule history) shows this as generic CMake-template boilerplate (commit messages like "CMakeLists.txt for better debugging", "reorganize code") — nothing suggests it was added and measured as a deliberate performance win, which matters for how safe it is to change.

Confirmed via `compile_commands.json` that the real compile line for e.g. `ALIKED.cpp` includes `-flto=auto -fno-fat-lto-objects -fPIC -O3 -march=native -mtune=native -ffast-math`. Confirmed via `ar`/`readelf` on the already-built `Thirdparty/Light_Glue_CPP/build/lib/release/libLightGlue_lib.a` (52 MB — individual `.o` members are 4-5.8 MB each, versus the sub-MB size a normal compiled `.o` this size would be) that its object files are **slim LTO objects**: dozens of `.gnu.lto_*` ELF sections, no fallback native machine code (`-fno-fat-lto-objects` means there's nothing else in there).

This static library is linked into `libAllFeature-VSLAM.so` (`CMakeLists.txt:128`, `target_link_libraries(${PROJECT_NAME} ... LightGlue::lib ...)`). Confirmed via `ninja -t commands libAllFeature-VSLAM.so` that this specific link command contains **zero** `-flto` flags of its own. That combination — slim-LTO objects pulled into a link that doesn't request LTO itself — is exactly the scenario where GCC's linker plugin (`liblto_plugin.so`, auto-loaded by `collect2`/`ld` whenever it detects `.gnu.lto_*` sections in *any* input, including ones buried inside a static archive) transparently fires up `lto1`/`lto-wrapper` to do real codegen for those objects, *regardless of the invoking command line*.

Why that spikes memory specifically here: GCC's LTO backend (LTRANS) parallelism defaults to the machine's detected hardware-thread count (`nproc` → 28) when not explicitly capped, and — critically — **Ninja has no GNU Make jobserver support** (confirmed: `ninja --version` → 1.10.1, jobserver protocol was never implemented in Ninja). GCC's `lto-wrapper` tries to cooperate with an enclosing job pool via that jobserver protocol; under Ninja there's nothing to cooperate with, so it falls back to spawning up to `nproc` parallel `lto1`/codegen workers **on its own**, completely ignoring `build.sh`'s explicit `cmake --build ... --parallel 8` cap. Each such worker re-runs expensive `-O3 -march=native -ffast-math` codegen on its partition of the combined whole-program IR from a Torch/CUDA-heavy static library (libtorch headers are themselves notorious for multi-GB single-TU compile memory). Up to 28 such processes competing for RAM, on a machine that already had ~18 GiB "used" before the build started, is a very plausible mechanism for the reported swap-filling — and it happens inside **one** ninja step (`[78/82] Linking CXX shared library libAllFeature-VSLAM.so`), so `--parallel 8` never had a chance to constrain it in the first place.

I did not instrument an actual build run to capture live RSS numbers (per standing instruction not to run builds myself) — this is a strong, evidence-backed hypothesis built entirely from static inspection, not a directly observed measurement. If you want to confirm it empirically before applying a fix, the cheapest check is watching `free -s 1` or `ps --sort=-rss` in another terminal specifically during the `[78/82] Linking CXX shared library libAllFeature-VSLAM.so` step (or counting `lto1`-named processes with `pgrep -fc lto1` at that moment).

### Other candidate causes considered and downgraded

| Cause | Evidence checked | Verdict |
|---|---|---|
| LTO/IPO enabled elsewhere (main project, vendored `g2o`, `SuperPoint-LightGlue-TensorRT`, `DBoW2`) | Grepped every `CMakeLists.txt` in the repo for `INTERPROCEDURAL_OPTIMIZATION`/`check_ipo_supported`/`-flto` | Not found anywhere else — confirmed isolated to `Light_Glue_CPP` |
| `-march=native`+`-O3` template-heavy compiles (Eigen/g2o) under 8-way parallelism | Main `CMakeLists.txt:16` and vendored `Thirdparty/g2o/CMakeLists.txt:16-17` both use `-O3 -march=native` (no LTO) | Real but minor contributor at most — this is a well-known but comparatively modest per-TU memory multiplier (~1.5-2x), not the order-of-magnitude LTO's cross-file WPA pass can cause; `--parallel 8` is an explicit, sane cap for this class of cost |
| CUDA device-code linking (`nvlink`) for `SuperPoint-LightGlue-TensorRT` (`CUDA_SEPARABLE_COMPILATION ON`, `CUDA_RESOLVE_DEVICE_SYMBOLS ON` also set in `Light_Glue_CPP/CMakeLists.txt:187-188`) | Present, but this is inference-only TensorRT/CUDA code, not the much heavier libtorch autodiff-capable headers; the build log shows its link steps (`[59/82]`, `[60/82]`) without incident | Lower probability secondary contributor; worth remembering if the LTO fix alone doesn't fully resolve the symptom |
| Build log itself showing a crash/OOM kill | Searched `test.txt` for `killed`/`oom`/`bad_alloc`/`internal compiler error`/etc. | None found — this specific captured run completed successfully (82/82). The reported "memory leak" is a resource-pressure/slowdown symptom (swap thrashing), not a hard crash, so it wouldn't necessarily appear as an error in the log at all |
| Ambient system load (browser/IDE already using ~18 GiB before the build starts) | `free -h` baseline captured during this investigation | Real compounding factor, not a code bug — worth mentioning as practical advice (close memory-heavy apps before a full `-f` rebuild) regardless of the LTO fix |

### Fix options, ranked

| # | Fix | Where | Risk | Notes |
|---|---|---|---|---|
| 1 | Add `-ffat-lto-objects` alongside the existing IPO setting in `Light_Glue_CPP/CMakeLists.txt` (or override via `target_compile_options`) | `Thirdparty/Light_Glue_CPP/CMakeLists.txt` | Low | Keeps LTO's benefit for anything that explicitly opts into `-flto` itself (nothing currently does at the top level), but embeds regular machine code alongside the IR so a non-LTO consumer (today's `libAllFeature-VSLAM.so` link) uses the plain fallback code and never invokes `lto1` at all. Most targeted fix — addresses exactly the mechanism found, without touching anyone's optimization flags |
| 2 | Cap LTRANS parallelism explicitly, e.g. `-flto=4` instead of the default `-flto=auto`, on whatever compiles these objects | `Thirdparty/Light_Glue_CPP/CMakeLists.txt` (RELEASE_FLAGS section) | Low-Medium | Preserves true whole-program optimization but caps worker count to something the machine can actually afford concurrently with everything else; needs care since CMake's `CMAKE_INTERPROCEDURAL_OPTIMIZATION` property doesn't expose a job-count knob directly — would need a manual `-flto=N` compile option instead of (or carefully combined with) the CMake IPO property |
| 3 | Turn LTO off entirely for `Light_Glue_CPP` (`set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)` or delete lines 13-18) | `Thirdparty/Light_Glue_CPP/CMakeLists.txt` | Low, *if* it's genuinely unvalidated boilerplate as the submodule's commit history suggests | Simplest possible fix; the downside is only real if someone deliberately measured and relied on LTO's runtime speedup for LightGlue inference, which nothing in the history suggests — worth a quick sanity check (a before/after inference-time comparison) rather than assuming, since this is third-party vendored code not authored in this session |
| 4 | Reduce `build.sh`'s `--parallel 8` further, or add a Ninja link pool (`CMAKE_JOB_POOL_LINK`) capping concurrent *ninja-scheduled* link steps | `build.sh` / `CMakeLists.txt` | Low effort, low payoff | Does not address the actual mechanism (the blowup happens *inside* one link step's internal LTO fork, not from too many concurrent ninja-scheduled jobs) — a defensive/secondary measure at best, not a fix on its own |
| 5 | Increase swap size | System-level, outside any repo | Trivial | Pure stopgap — turns a possible crash into guaranteed severe slowdown instead of fixing the cause; the thrashing itself is what reads as "a leak," so this doesn't really resolve the complaint, just makes it survivable |

Recommended order if you want to act on this: try **#1** first (lowest risk, most directly targeted at the confirmed mechanism), verify a subsequent `-fv` rebuild no longer shows the swap spike, and only reach for **#2**/**#3** if #1 turns out insufficient or if you'd rather not carry any LTO risk at all in vendored code nobody here has validated the benefit of.

**Resolution (2026-08-09, later same day):** applied **#3** (deleted the IPO block from `Thirdparty/Light_Glue_CPP/CMakeLists.txt` entirely rather than #1's fat-object compromise — its own example executable isn't part of this repo's build output, so there was no real capability worth preserving) and, separately, dropped `build.sh`'s `--parallel 8` to `--parallel 4` (table's item #4 above, which on its own was "low payoff" against the LTO mechanism specifically, but is a real win against the *residual* aggregate-parallel-compile pressure once LTO was gone). Both were verified empirically with live, correctly-tracked monitoring (`free`/`vmstat`/`pgrep` sampled every 4s across a full `-fv` rebuild, with an armed 90%-swap auto-kill safety net that never had to fire):
- `libLightGlue_lib.a` dropped from 52MB (slim-LTO IR objects, confirmed via `readelf` showing dozens of `.gnu.lto_*` sections) to 3.5MB (plain machine code) — direct proof the LTO mechanism is gone.
- Peak swap usage across a full clean rebuild: **706MB/8191MB (~8.6%)**, down from **~65%** measured at `-j8` post-LTO-fix, down from the original uncontrolled/dangerous behavior pre-fix. Total build time ≈3 minutes at `-j4` (vs ≈2 minutes at `-j8`, vs several times longer pre-fix — LTO was a real time cost too, not just a memory one).
- One process-monitoring pitfall hit and fixed along the way, noted here in case it recurs: launching the build via `setsid cmd &` and capturing `$!` gives the PID of the `setsid` wrapper, not the actual long-running process it execs — that wrapper can exit immediately, making a naive `kill -0 $PID` liveness check falsely report the build as finished seconds in. Track the real process via `pgrep -g <pid>`/direct `ps` inspection of the actual command tree instead, or just capture `$!` from a plain `cmd &` (no `setsid`) — bash already makes a backgrounded job its own process group leader without needing `setsid`.

---

## Compiler Warnings Audit (2026-08-09)

Full `-fv` rebuild (post the LTO/parallelism fixes above), captured complete stdout+stderr (~8800 lines) and swept it for every `warning:`. Raw count: **1710 warning instances**, but that collapses to **161 distinct source locations** (a header included by many translation units re-warns once per TU) — **145 in this repo's own code, 16 in vendored `Thirdparty/` submodules**. No code was changed for this pass, per request — this is the full list with causes and proposed fixes, to work through one at a time.

### Real bugs found while triaging "just warnings" — fix these first, they're not cosmetic

| # | Where | What the warning caught | Why it's a real bug | Proposed fix |
|---|---|---|---|---|
| W1 | `src/Sim3Solver.cc:254-257` (flagged by `-Wuninitialized` on `Nmatrix`'s internal storage) | Building the symmetric 4×4 `N` matrix for Sim3 rotation extraction (Horn's method), rows 0-2 correctly mirror the upper triangle into the lower triangle (`Nmatrix(1,0) = Nmatrix(0,1);` etc.), but the last row does it **backwards**: `Nmatrix(0,3) = Nmatrix(3,0);` reads `Nmatrix(3,0)` — which was never written — instead of writing it. Compared against stock ORB-SLAM2's `cv::Mat`-based original (`N.at<float>(3,0) = N.at<float>(0,3);`), this is a transcription bug from porting to Eigen/`mat4f` | This silently overwrites the correctly-computed `(0,3)/(1,3)/(2,3)` entries with uninitialized stack garbage, and leaves `(3,0)/(3,1)/(3,2)` uninitialized too — corrupting the eigenvector computation (`eigSolver.compute(Nmatrix)` right after) that **every loop closure's relative Sim3/rotation estimate depends on**. This is the single highest-severity finding in this whole sweep | Swap the assignment direction to match rows 0-2's pattern and the original algorithm: `Nmatrix(3,0) = Nmatrix(0,3); Nmatrix(3,1) = Nmatrix(1,3); Nmatrix(3,2) = Nmatrix(2,3);` |
| W2 | `src/Vocabulary.cpp` — 5 `switch(featureType)` statements (`createVocabulary` L15, `loadFromTextFile` L45, `size()` L87, `score()` L105, `transform()` L124), flagged by `-Wswitch` ("enumeration value 'FEAT_SUPERPOINT256'/'FEAT_ALIKED128' not handled") | None of the 5 switches has a case for the two "learned" feature types. Compounded by `-Wreturn-type` ("control reaches end of non-void function") at lines 84, 102, 120: `loadFromTextFile()` (returns `bool`), `size()` (returns `unsigned int`), and `score()` (returns `double`) all fall through with **no return statement** if `featureType` is `FEAT_ALIKED128`/`FEAT_SUPERPOINT256` | This repo's own default `vslamlab_allfeature-dev_settings.yaml` lists `features: ["orb32", "superpoint256"]` — meaning `superpoint256` is the actively-used second feature, and any `Vocabulary` call for it hits real undefined behavior (returns whatever garbage was in the return register/stack). `createVocabulary()`/`transform()` are `void`, so they don't UB, they just silently do nothing for those feature types (place-recognition vocabulary never gets built/used for superpoint/aliked keypoints) | Add `FEAT_ALIKED128`/`FEAT_SUPERPOINT256` cases to all 5 switches (presumably wiring up an `alikedVocabulary`/`superpointVocabulary` member analogous to the existing 6, if one doesn't already exist — worth checking `Vocabulary.h`'s member list first) |
| W3 | `src/Tracking.cc:789-798` (flagged by `-Wunused-but-set-variable` on `radiusTh`) | An adaptive search radius is computed (`radiusTh_low_slp`, bumped to `_medium_slp` for RGBD, bumped further to `_high_slp` after a recent relocalization) but the call that should consume it has the argument commented out: `matcher->match_map_points_to_frame(currentFrame, localPts);//, radiusTh);`. Checked `match_map_points_to_frame`'s current signature (`include/FeatureMatcher.h:83`) — it only takes 2 parameters today, no radius slot at all, so this isn't a simple "uncomment it" fix; the callee was refactored at some point and this caller-side logic was orphaned | Relocalization-aware coarser search and RGBD-specific radius tuning currently have **zero effect** on matching — dead logic that looks load-bearing. Not necessarily a regression (maybe the matcher's internal fixed radius was judged sufficient) but worth a deliberate decision rather than silent staleness | Either (a) restore radius-adaptivity by adding a radius parameter to `match_map_points_to_frame` and wiring `radiusTh` through, or (b) if the fixed internal radius is intentionally sufficient now, delete the dead computation (lines 789-796) along with the stale comment |
| W4 | `src/MapPoint.cc:320` (flagged by `-Wvla`) | `Descriptor_Distance_Type Distances[N][N];` — a 2D **variable-length array on the stack**, where `N` is the number of observations of this map point. VLAs are a GCC/Clang extension, not standard C++ | Real robustness risk, not just portability: for a well-observed long-lived map point (`N` in the hundreds), this allocates `N²` stack bytes in one call — a map point with e.g. `N=1000` observations would try to stack-allocate several MB, risking stack overflow on threads with default-sized stacks. This runs inside `MapPoint::ComputeDistinctiveDescriptors()`, called from map-maintenance code paths that aren't rare | Replace with a heap-backed container — `std::vector<std::vector<Descriptor_Distance_Type>> Distances(N, std::vector<Descriptor_Distance_Type>(N));` (simplest) or a single flat `std::vector<Descriptor_Distance_Type>` of size `N*N` indexed manually (more cache-friendly, matches the existing tight double loop) |
| W5 | `include/FeatureExtractor.h:96-97` (flagged by `-Wreturn-type`, "no return statement in function returning non-void") | Base-class virtuals `GetKeypointOctave()`/`GetKeypointSize()` are declared `virtual` (not pure `= 0`) with empty `{}` bodies, but return `int`/`float` — calling either on a base-class-typed object with no override would return garbage | Currently safe only because *every* concrete `Feature_*` subclass happens to override both (verified via `grep` — all 8 do) — but that invariant is enforced by convention, not the compiler. A new feature type that forgets to override either would silently compile and return undefined values | Make both pure virtual (`= 0;`, no body) — since every existing subclass already overrides them, this is a zero-behavior-change compiler-enforced guarantee going forward |
| W6 | `src/MapPoint.cc:303-311` + `:348` (flagged by `-Wunused-but-set-variable` on `latestIndex`) | `latestIndex` (index of the observation with the highest `frame_id`) is computed every call, but its only consumer is `//BestIdx = latestIndex; // Prefer the latest one` — commented out | Not a bug, just an incompletely-reverted experiment: someone tried "prefer the most recent observation's descriptor" as an alternative to the current "prefer least-median-distance" strategy (still active a few lines below) and left the computation in place when disabling it | Either delete the `latestIndex` computation (lines 303-311) and the stale commented line 348 together, or leave a clearer note if the alternative strategy is intentionally kept around for future experimentation |

### Systematic/mechanical fixes (bulk cleanup, all in this repo's own code)

| # | Warning | Locations | Root cause | Fix |
|---|---|---|---|---|
| S1 | `-Wignored-qualifiers` — 500 raw instances, but only ~18 distinct declarations | `include/Feature.h:21-22` (`getType()`, `getMatcherType()`) plus the matching override in all 8 `include/Feature_*.h` subclasses | `virtual const FeatureType getType() const = 0;` — the **leading** `const` on a by-value return type is meaningless (the caller gets a copy regardless); the **trailing** `const` (making the method itself const) is correct and unrelated. By far the largest warning count in this whole sweep, from the smallest actual fix | Drop the leading `const` in all 9 files: `virtual FeatureType getType() const = 0;` / `FeatureType getType() const override { return FEAT_ORB32; }` etc. One mechanical, low-risk pass eliminates roughly a third of all raw warnings |
| S2 | `-Wreorder` — ~35 raw instances across ~14 classes | `FeatureMatcher`, `Frame`, `KeyFrame`, `LocalMapping`, `LoopClosing`, `MapPoint`, `Sim3Solver`, `System`, `Tracking`, `Vocabulary` (all in `include/*.h` + matching `src/*.cc`/`.cpp` constructors) | Constructor initializer-list order doesn't match member-declaration order (members are always initialized in declaration order regardless of initializer-list order, so a mismatch is at best confusing, at worst a footgun if one initializer depends on another) | Spot-checked several — none found where the value of one member's initializer expression depends on another member declared *after* it, so this looks purely cosmetic throughout, but worth a quick re-check per class while fixing (not just a blind reorder) since that's the one way this category could hide a real bug. Fix: reorder each initializer list to match declaration order (or vice versa) |
| S3 | `-Wunused-parameter` — ~250 raw instances | Concentrated in `include/FeatureExtractor.h`'s default/no-op virtuals, `src/FeatureMatcher_superglue.cpp`, and scattered elsewhere | Parameter names kept (presumably for documentation/interface-clarity) in stub or not-yet-implemented function bodies that never reference them | Case by case: drop the parameter name and keep only the type (`void setupImage(const Image&){}`), or annotate `[[maybe_unused]]` where the name genuinely aids readability |
| S4 | `-Wsign-compare` — ~20 raw instances | `src/Tracking.cc`, `src/Initializer.cc`, `src/FeatureMatcher.cpp`, `src/MapPoint.cc`, `src/Viewer.cc` | Almost all are `int i` looped/compared against a `.size()` call (`size_t`/unsigned) | Change the loop counter to `size_t` (matching the existing convention already used elsewhere in this codebase, e.g. `Optimizer.cc`'s `size_t i=0, iend=...`), or add an explicit cast at the comparison site |
| S5 | `-Wdeprecated-copy` — 6 raw instances, all `AF_VSLAM::Frame` | `include/Frame.h:55` declares a custom copy constructor (`Frame(const Frame &frame);`) but no copy-assignment operator — the compiler's implicitly-generated one is deprecated since C++11 once a copy constructor exists, precisely because it's a common bug source (memberwise assignment may not match what the custom copy constructor's logic intends) | `Frame` objects **are** copy-assigned today (`currentFrame = Frame(...)` in `Tracking::GrabImageMonocular`, every frame) — worth explicitly reviewing whether the implicit memberwise assignment actually produces correct results matching the custom copy constructor's intent, then either `=default` it (if memberwise really is fine) or write an explicit `operator=` mirroring the copy constructor |
| S6 | `-Wunused-function`/`-Wunused-variable` in our own code | `include/Feature_orb32.h`: `computeOrientation()`, `computeDescriptors()`, `bit_pattern_31_` | Dead ORB-related helper code, likely superseded by the actual extractor implementation elsewhere | Delete, once confirmed genuinely unused (a quick repo-wide grep, not just this file) |
| S7 | `-Wunused-parameter` on `Optimizer::LocalBundleAdjustment`'s `pbStopFlag` | `src/Optimizer.cc:518` | Connects back to the earlier Optimizer review in this document — the `pbStopFlag`-gated early-stop checks in this function are commented out (`// if(pbStopFlag) //   if(*pbStopFlag) //     return;`), so the parameter is accepted but never read | Out of scope for a pure warnings cleanup pass — either restore the early-stop logic or drop the parameter; tie this to whichever Optimizer.cc item gets picked up next rather than fixing in isolation |

### Vendored/third-party — do not fix by editing files under `Thirdparty/` directly

| Warning | Count | Location | Note |
|---|---|---|---|
| `-Wdeprecated-declarations` (`Eigen::AlignedBit`) | 263 | `Thirdparty/g2o/` | Already noted in this document's Optimizer review — g2o's `base_binary_edge.h`/`base_vertex.h` use a deprecated Eigen enum |
| `-Woverloaded-virtual` (`DBoW2::FClass::meanValue` hidden) | 260 | `Thirdparty/DBoW2/` | Stock DBoW2 issue, present upstream |
| `-Wvla` | 4 | `Thirdparty/g2o/g2o/core/sparse_block_matrix.hpp` | Same category as W4 above, but in vendored code |
| `-Wunused-function` (`VisualizeMatching`, `GetFileNames`) | 54 | `Thirdparty/SuperPoint-LightGlue-TensorRT/include/utils.h` | Dead helper code in the vendored TensorRT wrapper |
| `-Wsign-compare` | 2 | `Thirdparty/DBoW2/executables/createVocabulary.cpp` | Stock DBoW2 tool |
| `-Wpedantic` (`extra ';'`) | 19 | `Thirdparty/g2o/g2o/types/se3quat.h` | Stock g2o |

All of these would need fixing upstream or in a proper fork — per this document's **P1** finding, `Thirdparty/g2o` is already untracked/unforked local state, so fixing warnings there directly would just be more uncommitted drift on top of an already-flagged problem. Not touching these in this repo.

---

## Tracking-Lost Audit (2026-08-10)

Read-through of `src/Tracking.cc` (cross-checked against `include/Tracking.h` for threshold defaults) to enumerate every place tracking can transition to `LOST`, ahead of a planned change to throw/catch a typed exception at each site so the system can print *why* tracking was lost plus the specific statistics that broke, before falling back to relocalization. No code was changed for this pass.

**Implemented (2026-08-10, same day):** T1-T4 now throw a new `AF_VSLAM::TrackingLostException` (`include/TrackingLostException.h`, a thin `std::runtime_error` subclass kept distinct so its catch sites don't swallow unrelated errors) carrying a formatted reason + the relevant statistics, instead of a bare `return false;`. `Track()` catches it at both call sites (around `TrackReferenceKeyFrame()`/`TrackLocalMap()`), prints it via `AF_WARN`, stashes it in a new `Tracking::mLastTrackingLostReason` member, and continues exactly as before (`bOK=false` → `mState=LOST` → next frame's `Relocalization()` call is unchanged). The `N1` post-loss-reset branch (`Tracking.cc:263-271`-ish) now includes that stashed reason in its own `AF_WARN` instead of a plain `cout`. N2/N3 (init-time resets) were left untouched — different family, out of scope for this pass. Verified with a full `pixi run build` (clean compile, `ninja: no work to do` on a re-run) and a smoke-run of `bin/vslamlab_allfeature_mono` (expected usage error, no crash from the new code paths).

`Relocalization()` (R1, R6) deliberately got **no** printing on *failure*, despite an earlier version of this change adding `AF_WARN` diagnostics there — removed same-day per explicit feedback: `Relocalization()` is retried once per frame for as long as tracking stays lost, so a print at R1/R6 fires every single frame and quickly scrolls the one-time, actually-useful `TrackingLostException` reason (from T1-T4) out of the visible log. The original `OK→LOST` transition reason is the signal worth keeping visible; per-frame "still couldn't relocalize" noise is not.

`Relocalization()` *success* is a one-time event per loss episode though (the `while` loop breaks via `bMatch=true` as soon as one candidate crosses `nGood_high`), so it's safe to print there without the spam problem above — added an `AF_INFO` right at the `bMatch=true; break;` site (`Tracking.cc`, inside the RANSAC candidate loop), reporting `frame`, `feature`, the matched keyframe's id, and the achieved vs. required inlier count. This closes the loop with the earlier `AF_WARN`: one line when tracking is lost and why, one line when it's recovered and against which keyframe.

**Follow-up fix (same day):** the success line reportedly wasn't showing up. Root cause: `afvslam_log.hpp`'s `AF_WARN`/`AF_ERROR` write to `std::cerr`, which the standard implicitly flushes after every insertion (`unitbuf`), while `AF_INFO`/`AF_DEBUG` write to `std::cout`, which is only line-buffered on an actual terminal — once VSLAM-LAB's runner redirects the baseline's stdout into a log file (`Baselines/BaselineVSLAMLAB.py:273`, `subprocess.Popen(..., stdout=log_file, ...)`), `std::cout` becomes fully buffered and the line can sit unflushed for a long time (until the buffer fills or the process exits cleanly) — meanwhile every `AF_WARN` in the same file shows up immediately, since it never depended on `std::cout`'s buffer at all. Fixed with an explicit `std::cout.flush();` right after that one `AF_INFO` call, rather than touching the shared macro (which other, more performance-sensitive `AF_DEBUG`/`AF_INFO` call sites should keep buffered).

### The state transition itself

There is exactly **one** place `mState` is assigned `LOST`:

```cpp
// Tracking::Track(), line 203-206
if(bOK)
    mState = OK;
else
    mState=LOST;
```

`bOK` is computed a few lines earlier from one of two mutually-exclusive calls (`Track()` line 175-184), then optionally downgraded by a third:

```cpp
if(mState==OK)
    bOK = TrackReferenceKeyFrame();      // frame-to-frame tracking, normal case
else
    bOK = Relocalization(...);            // already LOST, trying to recover
...
if(bOK)
    bOK = TrackLocalMap();                // further refines/can still fail
```

So every root cause of a `LOST` transition (or of *staying* `LOST`) bottoms out in one of these three functions returning `false`. Below is every such return site.

### `TrackReferenceKeyFrame()` — normal frame-to-frame tracking (state was `OK`)

| # | Where | How (exact trigger) | Why (what it means) | Stats available to log |
|---|---|---|---|---|
| T1 | `Tracking.cc:578-579` | `nmatches < TRACK_REFERENCE_KEYFRAME_MIN_MATCHES_HIGH` (default **15**) — checked right after `matcher->match_keyframe_to_frame(refKeyframe, currentFrame, ...)`, before any pose optimization | Feature matching against the reference keyframe itself failed to find enough correspondences — fast motion, large appearance/viewpoint change, textureless scene, or a feature-type mismatch between `refKeyframe` and `currentFrame` | `nmatches` (total), per-feature breakdown from `nmatches_ft` map, `refKeyframe->keyId`, `currentFrame.mnId` |
| T2 | `Tracking.cc:620` | `return nmatchesMap >= TRACK_REFERENCE_KEYFRAME_MIN_MATCHES_LOW;` (default **10**) — evaluated after `Optimizer::PoseOptimization(&currentFrame)`, counting only inlier (non-outlier) matches with `number_of_observations() > 0` | Enough raw matches existed (passed T1), but the chi2-based outlier rejection inside `PoseOptimization` threw out most of them — usually means the initial pose guess (`lastFrame.Tcw`, i.e. the motion model) was a bad prior for this frame, or the matches themselves were geometrically inconsistent | `nmatchesMap` vs. `nmatches` (pre-optimization) to show the outlier-rejection ratio, `currentFrame.mnId` |

### `TrackLocalMap()` — runs after either of the above two succeed

| # | Where | How (exact trigger) | Why (what it means) | Stats available to log |
|---|---|---|---|---|
| T3 | `Tracking.cc:667-668` | `currentFrame.mnId < lastRelocFrameId + maxFrames && mnMatchesInliers < minMatches_trackLocalMap_high` (default **50**) — the *stricter* threshold, active only within `maxFrames` (≈1 second at the sequence's fps) of the last relocalization | Track was just recovered via relocalization and hasn't yet proven itself stable enough against the local map — a deliberate "don't trust a fresh reloc too quickly" guard | `mnMatchesInliers`, `currentFrame.mnId - lastRelocFrameId`, `maxFrames` |
| T4 | `Tracking.cc:670-671` | `mnMatchesInliers < minMatches_trackLocalMap_low` (default **30**) — the general-case threshold, evaluated after `UpdateLocalMap()` + `SearchLocalPoints()` + `Optimizer::PoseOptimization(&currentFrame)` | Not enough of the *local map* (not just the reference keyframe) projects into and matches the current frame after pose refinement — broader loss of map overlap, e.g. viewpoint drifted away from all nearby keyframes, occlusion, or motion blur degrading the whole frame's features | `mnMatchesInliers` vs. thresholds, size of `localPts`, count of points that were `isInFrustum` (candidates fed to `SearchLocalPoints`) vs. actually matched |

### `Relocalization()` — runs when state is already `LOST`; failure here means tracking *stays* lost

`Relocalization()` doesn't cause the `OK→LOST` transition, but every reason it fails to recover is equally in scope for "why is the system lost right now" logging.

| # | Where | How (exact trigger) | Why (what it means) | Stats available to log |
|---|---|---|---|---|
| R1 | `Tracking.cc:923-924` | `vpCandidateKFs.empty()` — `keyFrameDB->DetectRelocalizationCandidates(&currentFrame)` returned zero candidates | DBoW2 place-recognition query found no keyframe similar enough to the current frame's BoW vector — either a genuinely novel/unmapped location, or a vocabulary/descriptor mismatch for the active `featureType` | `currentFrame.mnId`, `featureType` used for `ComputeBoW` |
| R2 | `Tracking.cc:950-954` (per-candidate, inside the `for(i<nKFs)` loop) | `nmatches_ft[featureType] < minNmatches` (default **15**) — candidate keyframe discarded before a `PnPsolver` is even built for it | This candidate KF looked similar enough for DBoW2 to surface it, but direct feature matching against it found too few correspondences | Count of candidates discarded here vs. `nKFs` total, per-candidate `nmatches_ft[featureType]` |
| R3 | `Tracking.cc:988-992` (per-candidate, inside the RANSAC `while` loop) | `bNoMore` from `pSolver->iterate(...)` — P4P RANSAC exhausted `ransac_maxIterations` (default **3000**) without reaching `ransac_probability`/`ransac_minInliers`-consistent consensus | The 2D-3D correspondences for this candidate were too noisy/inconsistent for RANSAC to find a supported pose within its iteration budget | Iterations used, `nInliers` at time of `bNoMore`, candidate index |
| R4 | `Tracking.cc:1015-1016` | `nGood < nGood_low` (default **10**) — after a PnP pose hypothesis passed RANSAC and `Optimizer::PoseOptimization(&currentFrame)` ran, too few inliers survived; `continue`s to the next candidate | The RANSAC pose looked plausible enough to optimize, but full pose optimization (all matched points, not just the minimal RANSAC set) rejected almost all of them as outliers | `nGood`, candidate index, `sFound.size()` before optimization |
| R5 | `Tracking.cc:1023-1051` (two nested `SearchByProjection` escalation stages) | `nGood` lands between `nGood_low` (10) and `nGood_high` (50, default) after the first optimization — triggers a coarse (`radiusTh_high_reloc`=10.0) then, if still short, a narrow (`radiusTh_low_reloc`=3.0) `SearchByProjection` re-match + re-optimize, each requiring the running total to cross `nGood_high` to succeed | Middling confidence pose — worth spending extra matching effort before giving up on this candidate, rather than an outright failure; if both escalation stages still don't cross `nGood_high` the candidate is abandoned (falls through the `if(nGood >= nGood_high)` at line 1056 without setting `bMatch`) | `nGood` before/after each `SearchByProjection` call, `nadditional` returned by each, final `nGood` vs `nGood_high` |
| R6 | `Tracking.cc:1065-1068` | `if(!bMatch) return false;` — terminal failure, reached once `nCandidates` drops to 0 (all candidates discarded via R2/R3, or fell through R4/R5 without ever setting `bMatch=true`) | The actual "relocalization failed, still lost" signal callers see; by this point every candidate KF the place-recognition system could offer has been tried and rejected | `nKFs` (candidates initially returned by R1), how many were discarded at each of R2/R3/R4/R5 stage, best `nGood` achieved across all candidates (not currently tracked — would need adding) |

### Adjacent failure paths (not `LOST`, but same family — related "why did the system fail" moments)

| # | Where | How | Why | Note |
|---|---|---|---|---|
| N1 | `Tracking.cc:263-271` (`Track()`, right after the `mState=LOST` assignment) | `if(mState==LOST) { if(map->KeyFramesInMap() <= minKeyframesInMap) { ...; mpSystem->Reset(); return; } }` (default `minKeyframesInMap`=**5**) | Downstream *consequence* of a `LOST` transition, not a new cause — if tracking is lost while the map is still small/fresh, the system gives up on relocalization entirely and resets rather than trying to relocalize. Currently only logs a plain `cout` line, no reason for the original loss | This is the natural place to also print the *root cause* captured from T1-T4 once those throw/carry a reason, since this branch is the last thing that runs before the frame's `LOST` state is handed back to the caller |
| N2 | `Tracking.cc:459-462` (`CreateInitialMapMonocular`) | `medianDepth<0 \|\| pKFcur->TrackedMapPoints(1) < keyframeTrackedMapPoints` (default **100**) → `Reset()` | Not a `LOST` transition (state is still `NOT_INITIALIZED`/being set from init) — the two-view monocular initialization produced a degenerate map (bad median depth or too few well-tracked points) and the system resets rather than ever reaching `OK` | Different family (init-time, not tracking-time), but the same "silent reset with only a `cout` line" pattern as N1 — worth handling with the same mechanism if convenient |
| N3 | `Tracking.cc:375-379` (`MonocularInitialization`) | `nmatches < minMatches_monoInit` (default **100**) → `mpInitializer = nullptr;` (silently restarts init search, no message at all) | Two-view init never found enough matches between `mInitialFrame` and `currentFrame` to attempt initialization; state stays `NOT_INITIALIZED` and it just waits for a better frame pair | Not currently logged at all, unlike N2's `cout`; lowest priority of the three since it's an expected, frequent, non-error occurrence during startup |

### Summary of touch points for the planned exception-based rework

- **Real "why tracking is lost" throw sites**: T1, T2 (from `TrackReferenceKeyFrame`), T3, T4 (from `TrackLocalMap`) — these are what actually flips `mState` to `LOST` at `Tracking.cc:206`.
- **"Why relocalization couldn't recover" sites**: R1-R6 inside `Relocalization()` — R6 is the single terminal point, but the interesting statistics (which candidates failed and why) are only available earlier in the loop (R2-R5), so the exception/reason needs to be accumulated across the loop and attached at R6, not just thrown fresh at R6.
- All four T-sites and R6 are reachable through the single `bOK` chain in `Track()` (lines 175-206), which is the natural place to catch a thrown reason and both print it and confirm the transition to relocalization on the next call.

---

## Stop-Induced Keyframe Runaway Investigation (2026-08-12)

Started as a crash report (`pixi run vslamlab configs/exp_debug.yaml` on NSAVP `R0_FA0`, RGB-D mode with `fastfoundationstereo` depth: `terminate called after throwing an instance of 'cv::Exception' ... usac/estimator.cpp:353 !model.empty()`), ended as a diagnosis of a systemic keyframe-insertion feedback loop that affects **both** sensor modes. Three separate problems were found; the first is fixed, the other two are diagnosed with a fix plan below.

### Problem 1 (fixed): OpenCV 4.12 USAC_MAGSAC crash in `filter_matches_by_fundamental`

`FeatureMatcher::filter_matches_by_fundamental` (`src/FeatureMatcher.cpp`) calls `cv::findFundamentalMat(USAC_MAGSAC)` on every `match_keyframe_to_frame`. OpenCV 4.12's USAC has an internal bug: when the correspondences are H-degenerate (planar scene, rotation-only or near-zero-baseline motion) **with some gross outliers mixed in**, its degeneracy-handling path passes an empty model to `SampsonError::setModelParameters`, whose `CV_Assert(!model.empty())` throws `cv::Exception`. Nothing caught it → `std::terminate`. Reproduced standalone against the exact conda OpenCV 4.12 this repo links (fuzzer: planar + ~30% outliers, small N; clean degenerate input alone does *not* throw).

**Fix (landed):** three-tier filtering. Tier 1: `findFundamentalMat(USAC_MAGSAC)` unchanged. Tier 2 on `cv::Exception`: `findHomography(RANSAC)` — chosen over `FM_RANSAC` because on 10 regenerated crash inputs H-RANSAC kept 0 bad / rejected 0 good in all cases, while classic F kept 1-2 outliers each time (F's extra DoF absorbs outliers exactly in the H-degenerate regime that triggers the throw). Tier 3 if H also fails: pass matches through unfiltered (pose optimization's chi2 test is the downstream safety net). An `AF_WARN` logs each tier-2/3 activation. In practice the fallback fires regularly on driving sequences (stops = zero baseline) and is benign: keeps ~99% of matches, few ms cost.

### Problem 2 (diagnosed, fix plan below): emergency-keyframe feedback loop — the "hiccups when the car stops" and the tracking losses

With the crash gone, runs showed (a) visible stalls whenever the car stops, (b) eventual tracking loss with a bizarre signature (`TrackReferenceKeyFrame: 488 raw matches → 0 after pose optimization`; `TrackLocalMap: 0 inliers of 5639 local points`). Instrumentation was added (all `AF_WARN`): per-channel (mono/stereo/rgbd) breakdown in `PoseOptimization` when >50% of correspondences are rejected (removed again after serving its diagnostic purpose here — re-add something like it when working fix-plan item 3 or re-enabling RGB-D); trigger stats when `NeedNewKeyFrame` fires the emergency path (kept, along with `Track()`'s slow-frame stage timing and `match_keyframe_to_frame`'s match/filter timing).

**A/B result that frames everything: the entire syndrome reproduces in pure monocular mode** (`mode: mono`, no depth anywhere — every rejected edge in the logs is `mono N/M, rgbd 0/0`). The RGB-D depth channel is exonerated as the root cause. Observed mechanism, with numbers from the mono run:

1. `NeedNewKeyFrame` (`src/Tracking.cc`) triggers the **emergency keyframe** path when `mnMatchesInliers < 0.5 * nRefMatches` while LocalMapping is busy; `Track()` then **blocks the tracking thread** in a wait loop (`Tracking.cc:320-332`) until LocalMapping is idle. This stall is the user-visible hiccup.
2. Each inserted keyframe gets hundreds of points triangulated into it by LocalMapping, inflating `nRefMatches` (observed climbing 591 → 712 → 759 across consecutive emergency KFs) while an ordinary frame only ever matches ~250-350 — so the `0.5 ×` condition re-arms itself and fires every ~2-6 frames. 16 emergency KFs in one run.
3. During a **stop** (zero baseline — conveniently flagged by Problem 1's fallback warnings firing), those spammed keyframe pairs have near-zero parallax, so 2-view triangulation mints ill-conditioned map points. They reproject fine at their creation KF but fail chi2 as soon as the viewpoint shifts: recurring `PoseOptimization: rejected ~50-60%` bursts (33 in one run), escalating to 95% (`376/394`) right before the loss at frame 1257 (`18 < 30` inliers, local map bloated to 10,152 points).
4. Same signature immediately after init in both modes (`511 → 243` tracked inliers within 3 frames): the low-parallax init map has the same ill-conditioned-point problem.

So: bad points → mass rejection → inlier sag → emergency KF (+stall) → more zero-baseline triangulation → worse points. RGB-D just changed how the spiral ends (its runs died faster; one segfaulted — see Problem 3).

**Fix plan, in priority order:**

| # | Fix | Where | Notes |
|---|---|---|---|
| 1 | **Stationarity gate on keyframe insertion.** When the camera is essentially not moving (median keypoint displacement/parallax vs. the last KF below a threshold, or motion-model translation ~0), insert no keyframes at all — normal or emergency | `Tracking::NeedNewKeyFrame` | Directly removes the stop-time trigger; stock ORB-SLAM2 survives stops precisely because its stricter conditions rarely fire when nothing changes |
| 2 | **Replace the self-inflating emergency condition.** `0.5 * nRefMatches` compares against a stat that grows after every insertion (post-hoc triangulation). Compare against a stable reference (e.g. recent frames' own inlier history) and/or add a minimum-frame cooldown between emergency KFs | `Tracking::NeedNewKeyFrame` | The observed `nRefMatches` climb (591→759 while inliers stayed ~300) is the smoking gun |
| 3 | **Minimum-parallax gate on 2-view triangulation.** Verify `LocalMapping::CreateNewMapPoints` enforces a real parallax check (stock ORB-SLAM2's `cosParallaxRays < 0.9998`-style) so near-zero-baseline KF pairs yield zero triangulated points rather than garbage | `LocalMapping::CreateNewMapPoints` | Also re-examine the init path — the `511→243` post-init attrition suggests the initial map suffers the same conditioning problem |
| 4 | **Unblock the tracking thread.** The emergency wait loop stalls frame consumption indefinitely; once 1-2 fix the spam it fires rarely, but it should still have a bounded wait (or be removed — `InsertKeyFrame` already sets `mbAbortBA`) | `Tracking::Track` (`Tracking.cc:320-332`) | Turns any future recurrence from a stall into a non-event |
| 5 | **Re-check chi2/information calibration only after 1-3.** The mass rejections are (per this diagnosis) mostly *correct* rejections of bad points; retuning thresholds first would mask the real problem | `Optimizer`/`GetKeyPt2DInf` | The per-feature-type split of the rejection breakdown is the next instrumentation step if rejections persist after the map stops being poisoned |
| 6 | **Then re-enable RGB-D.** Depth-seeded points need no parallax, so once stops no longer poison the map, RGB-D should *outperform* mono exactly there | — | Re-run the original `mode: rgbd` config as the regression test |

### Profiling verdict (2026-08-12, later same day): the stop hiccups are `cv::findFundamentalMat(USAC_MAGSAC)` grinding, and the frame budget is blown by brute-force matching

Stage timing was added to `Tracking::Track` (per-frame `AF_WARN` breakdown over 150 ms: map-mutex wait / trackRef / localMap / emergencyWait / other) and to `FeatureMatcher::match_keyframe_to_frame` (per-feature match time + filter time, warned over 100 ms). Findings from instrumented mono runs on NSAVP `R0_FA0`:

- **Stop hiccups**: at stop sections, `filter_matches_by_fundamental`'s `cv::findFundamentalMat(USAC_MAGSAC)` call takes **1.2–4.0 seconds per frame without throwing** (measured `filter=1183/1630/3646/4023ms` while `match=13ms`) — MAGSAC's iteration/degeneracy schedule explodes on real zero-baseline correspondences. Synthetic zero-baseline benchmarks (5–9 ms) do *not* reproduce this; only real stop data does. So OpenCV's estimator causes both the original crash (throwing path, rare) and the stop hiccups (grinding path, every stop). Neither map-mutex contention nor the emergency wait loop contributes (`mapMutexWait=0`, `emergencyWait=0` throughout).
- **Baseline overload**: outside stops, `TrackReferenceKeyFrame` costs ~100–170 ms/frame (vs. a 50 ms budget at 20 fps), dominated by brute-force orb32 matching (60–155 ms for ~5400 KF keypoints × ~2000 frame keypoints; aliked128 adds 10–50 ms; the filter is 1–3 ms when not degenerate). `TrackLocalMap` contributes 110–190 ms spikes in dense-keyframe regions (local map ~10k points).
- **Decision executed (same day): PoseLib replaces OpenCV robust estimation.** `Thirdparty/PoseLib` added as a git submodule (upstream `PoseLib/PoseLib`, v2.0.5+, built via `add_subdirectory` with `target_compile_options(PoseLib PRIVATE -w)` since vendored code can't survive this project's global `-Werror`). `filter_matches_by_fundamental` now runs `poselib::estimate_fundamental` (LO-RANSAC, `max_error=3.0`, iterations bounded 100–1000) with `estimate_homography` as the degenerate-consensus fallback and unfiltered pass-through as the last tier; the `outlierMethod` parameter is gone. **Migration gotcha that actually bit:** two call sites still passed `cv::FM_LMEDS` as the old 4th int argument, which silently bound to the new `maxForRansac` parameter (=4), truncating matching to 4 matches and breaking monocular initialization entirely (zero maps in a full 5001-frame run) — compiles clean, fails silently; check *all* call sites when changing int-parameter signatures. Verified on NSAVP `R0_FA0` mono: max filter time **51 ms** across the whole run including stop sections (was 3.6–4.0 s with OpenCV), init map 718 points (was ~510), worst frame 345 ms (was 4.2 s), tracked through the previously fatal stop section. Remaining losses (e.g. frame 1551, 288 raw matches → 8 after pose opt) are the fix-plan item 3 map-quality problem, not the filter.
- **Also seen**: `terminate called without an active exception` at natural process exit (after GBA + trajectory save — results unaffected). Pre-existing shutdown bug (a joinable thread destroyed at teardown), only visible now that runs complete instead of crashing earlier; untracked, worth its own small fix.
- **Additional fix-plan item**: an early tracking loss (map > `minKeyframesInMap` KFs but only seconds old) leaves the system retrying relocalization forever against a stale map while the vehicle drives away — there is no reset budget on failed relocalization. Observed live: loss at frame 22, no reset, permanently lost.

### Problem 3 (open): racy segfault under emergency-KF bursts, RGB-D mode

One RGB-D run segfaulted (`Segmentation fault (core dumped)`) a couple of frames after an emergency-keyframe insertion; a rerun of the identical command lost tracking at a different frame instead (non-deterministic), and two gdb-supervised attempts didn't reproduce it. Working hypothesis: a race between Tracking and LocalMapping over the emergency keyframe's contents under insertion spam — plausibly moot once fixes 1-2 eliminate the spam, but not proven. If it recurs: run under gdb (`gdb -batch -ex run -ex bt --args ./bin/vslamlab_allfeature_rgbd ...` with the args from `system_output_*.txt`'s invoking command) or a TSan build.

## Scattered Tracking-Loss Investigation (2026-08-13)

User-reported: `pixi run vslamlab configs/exp_debug.yaml` (NSAVP `R0_FA0`, RGB-D, fastfoundationstereo depth, mask2former masks, 19103 frames) loses tracking non-deterministically at scattered frames. Diagnosed on branch `trackinglost` with repeated instrumented full-range runs (two concurrent processes per round to double the sample rate). Four distinct defects found; all fixed on this branch.

### Fixed 1: `verbose:0` crashed every run on frame 1 (null-viewer profiling calls)

With `verbose:0` no `Viewer` is created, yet `Tracking::GrabImageMonocular` (Tracking.cc:127) and `LocalMapping::Run` (LocalMapping.cc:83) called `viewer->set_*_time_median()` unconditionally through the null `shared_ptr`. `Viewer::mutexProfileStats` is the class's first data member (offset 0), so the inlined `unique_lock` received a null `std::mutex*` — libstdc++ throws `std::system_error` EPERM ("Operation not permitted") for that, uncaught → `terminate` on the first frame, every run. (These call sites date to April; all prior debug runs must have used `verbose:1`.) Guarded both sites like `Tracking::Reset` already does. Also fixed `build.sh`'s `dirname "LIBRARY_PATH"` (missing `$`) which made the build silently a no-op when invoked from the parent repo.

### Fixed 2 (the Problem-3 segfault): wild-pointer UB in `MapPoint::EraseObservation`

Run 2 died with a kernel-journal general protection fault at `libAllFeature-VSLAM.so+0x120b7e` → resolved (by rebuilding the exact commit's `.so` and `addr2line`) to `MapPoint::EraseObservation`. After erasing a point's **last** observation the code ran `mpRefKF = observations.begin()->second->projKeyframe;` on the now-**empty** map — copying a `shared_ptr` out of `end()` garbage, i.e. a refcount increment through a wild pointer: silent heap corruption on most occurrences, occasional immediate GPF. RGB-D makes this hot: the depth-seeded pass in `LocalMapping::CreateNewMapPoints` creates single-observation points by the thousand, and `KeyFrameCulling → KeyFrame::SetBadFlag` erases exactly such last observations en masse. Stock ORB-SLAM2 has the same latent code shape but rarely holds single-observation points there. Fix: only re-point `mpRefKF` when observations remain; the empty case falls through to `SetBadFlag` immediately below.

### Fixed 3 (the actual scattered tracking losses): pose-optimization divergence from a constant-position seed

The recurring loss signature (`TrackReferenceKeyFrame`: hundreds of raw matches → 0–5 inliers after pose optimization; relocalization succeeds **on the next frame** with >170 inliers, proving frame and map were fine) was captured with per-pass instrumentation at frame 3325: `passInliers=[0,-,-,-]`, `poseDelta trans=1.48m rot=3.5°`, rejected medians pxErr≈31–43px while inverse-depth error stayed ≈0.7σ. Mechanism: this codebase dropped ORB-SLAM2's `TrackWithMotionModel` (`mVelocity` was computed but never consumed — issue #8), so pose optimization seeded from `lastFrame.Tcw`. At 15–19 m/s that starts every pixel residual ~a full frame of motion (20–40px) large → the 2D terms begin Huber-saturated (bounded gradients) while the RGB-D inverse-depth terms (information `1/σ²≈1.3e5`) are still quadratic. Depth constrains only the z-component — it is blind to rotation/lateral translation — so pass-1 LM occasionally walks into a wrong rotation basin, ends with depth satisfied and all pixels ~30px off, and classifies **every** correspondence as an outlier → `LOST`. Non-deterministic because it needs a particular mix of match set/geometry, hence "fully scattered" losses. Two-part fix in `TrackReferenceKeyFrame`: seed with `mVelocity * lastFrame.Tcw` when valid (restores the small-initial-residual precondition the 4-pass scheme assumes), and on a collapse (<10 inliers from ≥45 raw matches) reset flags, re-seed, and re-run `PoseOptimization(pFrame, useDepthChannel=false)` once — pure 2D, the configuration the scheme was tuned for.

### Observed, filed as issues (not fixed here)

- **#9** post-relocalization keyframe embargo (`lastRelocFrameId + maxFrames`) starves the tracker at driving speed: reloc at 4380 → raw matches decayed 276→124 against the frozen reference KF → re-lost at 4396.
- **#10** no recovery budget when lost: a loss at frame 2327 left the run grinding relocalization for the remaining ~17k frames.
- **#11** `MapPointCulling` found-ratio test only applies to `featureTypes[0]`.
- **#12** shutdown `terminate called without an active exception` (pre-existing, from the 2026-08-12 notes, now tracked).
- **#8** motion model unused (the seeding half is now fixed on this branch; the issue remains as the umbrella for a full `TrackWithMotionModel` port).

### Diagnostics kept in the code (all cheap, `AF_WARN`/`AF_INFO`)

- `PoseOptimization`: on >50% rejection — per-channel breakdown (mono/stereo/rgbd), per-pass inlier counts, initial→final pose delta, rejected-edge medians (pixel error, inverse-depth error, sensor depth). The pixel-vs-depth split attributes a burst to 2D geometry vs the depth channel in one line.
- `Tracking::Track`: 100-frame heartbeat (inliers / localPts / KFs / mapPts) so post-mortems see the trend leading into a loss.

## Notes

- License: GPLv3 (inherited from ORB-SLAM2)
- `bin/`, `lib/`, `build/`, and `allfeature_vocabulary/` are all build/runtime artifacts, not source — never assume content there is checked in
- See `TODO.md` for tracked cleanup items (bugs, dead code, duplication) found in the CLI entry points
