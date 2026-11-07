# Cleanup

## Parallelism on macOS

**Problem.** AppleClang ships no OpenMP runtime. Homebrew/conda `libomp` works, but OpenCV and ONNX
Runtime may link their own — two OpenMP runtimes in one process gives `OMP: Error #15` or silent
oversubscription. CMake now links it optionally, so Mac already builds and runs serially.

**Solution.** Replace OpenMP with `std::thread`. All five `#pragma omp` sites are the same shape —
run N independent items, join — so one `parallel_for` helper (atomic counter over worker threads)
covers them, and the OpenMP dependency goes away entirely. The counter reproduces
`schedule(dynamic,1)`; `critical` becomes a `std::mutex`. Portable, dependency-free, C++17.

**Watch.** Keep the catch-into-`exception_ptr`-and-rethrow — an exception escaping a `std::thread`
calls `terminate`. Cap extractor intra-op threads to 1 when fanning out across feature types.

## Frame (include/Frame.h, src/Frame.cc)

**Hyperparameters and heuristics reachable from Frame (2026-11-07)**

| # | Knob | Value | Where it lives | Configurable? | Notes |
|---|---|---|---|---|---|
| 1 | `viewing_cos_limit` | `0.5` (60°) | `Tracking.h:118`, `Tracking.cc:1028` | ✅ `Tracking.ViewingCosLimit` | The only one of these already exposed. **But the same gate is hardcoded at `FeatureMatcher.cpp:507` and `:794`** — one policy, two sources |
| 2 | Scale band `1.2` / `0.8` | — | `MapPoint.cc:388`, `:394` | ❌ hardcoded | `project_if_visible` rejects outside `[0.8·min, 1.2·max]`. `0.8 ≠ 1/1.2` — an inherited asymmetry, not a design choice |
| 3 | `maxKeyPtSize` (band width) | `pow(1.2, 7) ≈ 3.583` | `MapPoint.cc:379`, from `FeatureExtractor.cpp:34` | ❌ hardcoded | `min_distance_ = max_distance_ / 3.583`. Identical for every feature type — ORB's pyramid span applied to all 8 |
| 4 | `depth_noise_coeff` | `0.0028` | `Calibration::depth_noise_coeff` | ✅ `calibration.yaml` → `depth_noise_coeff` | `Var(1/z) = k²` → sets the depth channel's weight (`1/k² ≈ 127,551`) in all three optimizers. **No dataset emits it yet**, so every sensor gets the Kinect-era default (issue #3) |
| 5 | Search radius `r` | see below | passed into `get_features_in_area` | partly | Four live callers, four different radii |
| 5a | `flow_association_radius` | `2.0` px | `Tracking.h:101` | ✅ | optical-flow association |
| 5b | `reloc_search_radius_coarse` / `_narrow` | `10.0` / `3.0` px | `Tracking.h:135-136` | ✅ | relocalization, two passes |
| 5c | `radiusTh_factor` | `12.0` | `FeatureMatcher.h:108` | ❌ `const` member | fuse into keyframe |
| 5d | `radiusScale` | `1.15` | `FeatureMatcher.cpp:34` | ❌ static | multiplies 5c and the projection radii |
| 5e | `projection_match_radius_th` | `20.0` px | `FeatureMatcher.h:71` | ❌ `const` member | now a direct distance test, not a grid query |
| 6 | `FRAME_GRID_ROWS` / `COLS` | `48` / `64` | `Frame.h:21-22` | ❌ compile-time | Pure performance — changes cost, never the answer. ~4× too fine: `s_opt = ∛(2·r·c_cell/(c_pt·ρ))`, so `s ∝ r^(1/3)` |

**The two structural problems**

**Radii are absolute pixels, so they mean different things per camera.** Pose error produces a
pixel displacement of `f·θ`, so a correct window scales with focal length. A fixed 20 px covers
twice the scene on a half-focal-length camera. The principled form is `r = f·θ` with `θ` (an angle)
in config — which is also what would make cross-dataset comparison fair.

**Half of these aren't reachable from a config file at all.** `1.2`, `0.8`, `3.583`, `12`.
`20.0` and the grid dimensions are all compile-time. For a project whose purpose is comparing eight
feature types, the scale and radius constants are exactly the ones most likely to need pe
values — and they're the ones you can't set.

openmp mac?

# Frame / KeyFrame — outstanding items (2026-11-07)

From the 18-lens pass, in order of importance. 14 items were fixed; these were not.

| # | Item | Why not done | Recorded |
|---|---|---|---|
| 1 | **The runs.** One `table_3` determinism run for the byte-identical batch; one before/after accuracy run for the behaviour-changing one (grid floor/circle/clamp, FeatureMatcher distance tests, Calibration bounds) | Yours to run — blocks confidence in everything else | `frame_review` |
| 2 | **Colour order bug.** `is_rgb_` is always true (no `cam_type: bgr` in 254 sequences) and `imread` always returns BGR, so `RGB2GRAY` is applied to BGR data on every colour sequence. Changes which keypoints are detected | Behaviour-changing everywhere; confirm whether `cam_type` describes the file or the memory layout first | `colour_order_bug` |
| 3 | **Match cache leak.** ~90% of writes are unlookupable (written frame-keyed, all readers look up keyframe ids); old keyframes grow unbounded via the neighbour path — hundreds of MB in a loopy run | Lifetime reasoning needs care; `cache_mutex_` is a leaf lock taken pairwise | `match_cache_leak` |
| 4 | **`keyPtsSigma2`/`keyPtsInf` are both `keyPtsSize[i]` in disguise** — 32 bytes/keypoint storing one scalar, ~64 KB per type per frame, kept for every keyframe's life. `get_keypoint_sigma` and `get_keypoint_size` also return the identical number | The `mat2f` machinery is a placeholder for anisotropic covariance that was never built. Removing it commits you to isotropy — a design call | `frame_review` |
| 5 | **`pts` + `outliers` are parallel arrays** encoding one 3-state value (unmatched / inlier / outlier) | ~100 sites across 4 files plus `KeyFrame::map_points_`. The live hazard was already removed | `frame_review` |
| 6 | **`using namespace std;`** in Frame.h and KeyFrame.h, reaching nearly every TU | Measured: 22 TUs, ~190 errors. Both headers must go together. Frame.h and Frame.cc are already clean | `frame_review` |
| 7 | **Grid layout and size.** 3072 `std::vector`s = 72 KB per keyframe held for life (CSR ≈ 20 KB, 2 allocations). Separately, 48×64 is ~4× too fine: `s_opt = ∛(2·r·c_cell/(c_pt·ρ))` | CSR first — it moves the cost ratio, so tuning before it means re-tuning after | `grid_tuning` (**stale**, assumes per-frame queries) |
| 8 | **Scale/view-angle gates as hyperparameters** — `viewing_cos_limit` is a parameter in `project_if_visible` but hardcoded at FeatureMatcher.cpp:507/:794; same for MapPoint's `1.2`/`0.8` | Deferred to the MapPoint pass by decision | `frame_review` |
| 9 | **`sigma2_inv_depth` holds one constant** — one bit that `inv_depth != 0` already carries | It's the hook for per-pixel depth confidence; zero plumbing once real values exist (issue #3) | `frame_review` |
| 10 | **No keypoint-count logging.** Can't tell "no keypoints detected" from "detected but matching failed"; the heartbeat is `#ifdef PROFILING_EXHAUSTIVE` | Belongs in Tracking, which can rate-limit — not Frame | `frame_review` |
| 11 | **`KeyFrame.cc:34` `-Wreorder`** — `grid_` declared after `map_` but initialised before. The only warning left in the build | Trivial: move `grid_(...)` after `map_(pMap)` | — |
| 12 | **55 `const FeatureType&` parameters** project-wide (KeyFrame, FeatureMatcher, Map so KeyFrame's accessors no longer match Frame's in style | Mechanical sweep, better in one pass than file-by-file | — |
| 13 | **`take_grid()`** — `Frame::grid` now has only two external touches, both at KeyFrake it private, replace the comma-operator trick, and name the build-then-move contract(currently undocumented) | Offered, not taken | — |
| 14 | **`get_features_in_area` reserves `keypoints.size()`** (~16 KB) to return a handfur on the per-frame path | `frame_review` |
| 15 | **`grid_tuning` note is stale** — assumes per-frame grid queries, which the lazy build removed | Correct it before a future session acts on it | itself |

## Start here

**11** is one line and byte-identical, and leaves the build warning-free apart from vendored PoseLib.
**15** is a two-minute correction that stops a wrong premise being acted on.
Both are free; everything above them needs **1** first.