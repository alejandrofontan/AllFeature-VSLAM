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

## Keyframe (include/Keyframe.h, src/Keyframe.cc)

## Frame (include/Frame.h, src/Frame.cc)

`FRAME_GRID_ROWS = 48` , `FRAME_GRID_ROWS = 64`

`project_if_visible`:
- `viewing_cos_limit` = `0.5` (60°) — ⚠️ **one policy, three sources**: `Tracking.ViewingCosLimit` ✅,
  `FeatureMatcher::cos_viewing_angle_th` (`const` member) ❌, and a bare `-1.0f` at the
  relocalization site meaning *gate off*.
- `pt->get_max_distance_invariance()` / `get_min_distance_invariance()` = `[d/1.2, d·1.2]` around
  the point's **reference** distance — `MapPoint::depth_invariance_ratio`, ❌ `static constexpr`.

`get_features_in_area` — `r`, the search radius. Four live callers, four radii, all absolute pixels:
- `flow_association_radius` = `2.0` px — `Tracking.h` ✅
- `reloc_search_radius_coarse` / `_narrow` = `10.0` / `3.0` px — `Tracking.h` ✅
- `radiusTh_factor` = `12.0` — `FeatureMatcher.h:110` ❌ `const` member
- `radiusScale` = `1.15` — `FeatureMatcher.cpp:34` ❌ static, multiplies the above
- `projection_match_radius_th` = `20.0` px — `FeatureMatcher.h:73` ❌ `const` member, now a direct
  distance test rather than a grid query

`get_depth` — the only place to filter and smooth depths and their covariances:
- `calibration->depth_noise_coeff` = `0.0028` — ✅ `calibration.yaml`

## Keyframe (include/Keyframe.h, src/Keyframe.cc)

`update_connections` + `rebuild_ordered_connections`:
- `min_covisibility_weight` = `15` — `KeyFrame.h:110`, ✅ `KeyFrame.MinCovisibilityWeight`.
  An **absolute** count inherited from ORB-SLAM2 at ~1000 features; the learned extractors run at
  2000+, so the same number is a much weaker criterion for them.
- Never-orphan fallback: if nothing reaches the threshold, keep the single strongest edge. ❌

`compute_scene_median_depth` — `q = 2` (the median). Was a parameter, now fixed; every caller
passed 2. ❌

`get_best_covisibility_keyframes(N)`

`get_covisibles_by_weight(w)`

## MapPoint (include/MapPoint.h, src/MapPoint.cc)
`get_max_distance_invariance` and `get_min_distance_invariance`:
- `depth_invariance_ratio`

`compute_distinctive_descriptors`: THINK ON THIS

`increase_observability` / `decrease_observability`:
- `depth_observation_weight` = `2` — `MapPoint.h`, ✅ `MapPoint.DepthObservationWeight`. What one
  observation adds to `num_observations_`: this value if that keyframe measured sensor depth at
  the keypoint (`inv_depth > 0`, i.e. RGB-D), 1 otherwise. ORB-SLAM2's stereo/RGB-D rule — a
  bearing-only observation fixes a ray, one with depth fixes a point — inherited with its value
  and no derivation behind it. Both callers go through one `observation_weight()` helper, so the
  add and the erase cannot disagree and leave the count permanently wrong.

  The weighted count is what the **quality gates** read: map-point culling
  (`LocalMapping.cc:182`), keyframe-culling redundancy (`:561`),
  `KeyFrame::count_tracked_map_points` (`KeyFrame.cc:291`) and which of two points survives a
  fuse (`FeatureMatcher.cpp:541`). So **the same threshold means something different on RGB-D
  than on mono**: a gate of "≥3" needs three keyframes on mono but only two on RGB-D.

`erase_observation`:
- `min_observing_keyframes` = `2` — `MapPoint.h`, ✅ `MapPoint.MinObservingKeyframes`. The point is
  discarded once this many keyframes or fewer still observe it. Two is the triangulation minimum,
  so a point at the threshold has no redundancy left: lose one more observer and it cannot be
  re-triangulated. Counted as **keyframes**, deliberately not as the depth-weighted total above —
  so "how well observed is this point" has two answers depending on which rule is asking.

`update_normal_and_depth`:
- `reference_keypoint_size` = `1.5` — `MapPoint.h:180` ❌ `static constexpr`. Assigned to
  `ref_size_` and read nowhere else.
- `ref_distance_` — distance to the **reference** keyframe. Not what sets the depth window
  (that uses the local `dist` directly); read only by `predict_size`.
- `ref_size_` — a member that only ever holds `reference_keypoint_size`.

`predict_size(d) = ref_size_ · ref_distance_ / d` — multiplies the search radius at
`FeatureMatcher.cpp:760` (Sim3 fuse, **loop closing**) and `:848` (**relocalization**). Never runs
on the tracking or local-mapping path. `project_if_visible` has already clamped `d` to
`[ref_distance_/1.2, ref_distance_·1.2]`, so the result is pinned to `[1.25, 1.8]` — 1.5 ± 20%.
With `radiusScale` = `1.15` that is a near-constant ~1.7× on the radius, while the local-mapping
fuse at `:487` passes its radius **raw**. One policy, two conventions, no comment either way.

**Deleting the chain** removes all three members above plus `predict_size`, the `predictedSize`
and `dist3D` locals at both call sites, `Projection::dist` (those locals are its only readers) and
therefore the `Projection` struct itself, and `radiusScale` — if both sites stop multiplying by it.
Three files. The decision it hangs on is what the two sites use instead: `radius_th` raw (~1.7×
narrower windows), or `radius_th` raw **with the config defaults multiplied by 1.725**
(`fuse_radius`, `reloc_search_radius_coarse`/`_narrow`), which keeps today's searched window and
makes the radius literally the number in the config. The latter preserves the mean, not the
wobble. **Not byte-identical either way**, and it lands on loop closing and relocalization.

# Frame / KeyFrame — outstanding items (2026-11-07)

From the 18-lens pass, in order of importance. 14 items were fixed; these were not.

| # | Item | Why not done | Recorded |
|---|---|---|---|
| 2 | **Colour order bug.** `is_rgb_` is always true (no `cam_type: bgr` in 254 sequences) and `imread` always returns BGR, so `RGB2GRAY` is applied to BGR data on every colour sequence. Changes which keypoints are detected | Behaviour-changing everywhere; confirm whether `cam_type` describes the file or the memory layout first | `colour_order_bug` |
| 3 | **Match cache leak.** ~90% of writes are unlookupable (written frame-keyed, all readers look up keyframe ids); old keyframes grow unbounded via the neighbour path — hundreds of MB in a loopy run | Lifetime reasoning needs care; `cache_mutex_` is a leaf lock taken pairwise | `match_cache_leak` |
| 5 | **`pts` + `outliers` are parallel arrays** encoding one 3-state value (unmatched / inlier / outlier) | ~100 sites across 4 files plus `KeyFrame::map_points_`. The live hazard was already removed | `frame_review` |
| 7 | **Grid layout and size.** 3072 `std::vector`s = 72 KB per keyframe held for life (CSR ≈ 20 KB, 2 allocations). Separately, 48×64 is ~4× too fine: `s_opt = ∛(2·r·c_cell/(c_pt·ρ))` | CSR first — it moves the cost ratio, so tuning before it means re-tuning after | `grid_tuning` (**stale**, assumes per-frame queries) |
| 8 | **Scale/view-angle gates as hyperparameters** — `viewing_cos_limit` is a parameter in `project_if_visible` but hardcoded at FeatureMatcher.cpp:507/:794; same for MapPoint's `1.2`/`0.8` | Deferred to the MapPoint pass by decision | `frame_review` |
| 9 | **`sigma2_inv_depth` holds one constant** — one bit that `inv_depth != 0` already carries | It's the hook for per-pixel depth confidence; zero plumbing once real values exist (issue #3) | `frame_review` |
| 10 | **No keypoint-count logging.** Can't tell "no keypoints detected" from "detected but matching failed"; the heartbeat is `#ifdef PROFILING_EXHAUSTIVE` | Belongs in Tracking, which can rate-limit — not Frame | `frame_review` |
| 12 | **55 `const FeatureType&` parameters** project-wide (KeyFrame, FeatureMatcher, Map so KeyFrame's accessors no longer match Frame's in style | Mechanical sweep, better in one pass than file-by-file | — |
| 13 | **`take_grid()`** — `Frame::grid` now has only two external touches, both at KeyFrake it private, replace the comma-operator trick, and name the build-then-move contract(currently undocumented) | Offered, not taken | — |
| 14 | **`get_features_in_area` reserves `keypoints.size()`** (~16 KB) to return a handfur on the per-frame path | `frame_review` |
| 15 | **`grid_tuning` note is stale** — assumes per-frame grid queries, which the lazy build removed | Correct it before a future session acts on it | itself |

