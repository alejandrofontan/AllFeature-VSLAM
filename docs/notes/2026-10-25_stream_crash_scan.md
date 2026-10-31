# Crash and freeze scan: live ARStream runs (2026-10-25)

Branch `debug`. Symptom: sporadic segfaults and freezes of `vslamlab_allfeature_mono_arstream` on a live iPhone
stream (orb32 + superpoint256, LightGlue on Core ML, vpr: megaloc, optical-flow tracking), no backtraces. Method: a
read-only review of the whole system in five slices (loop closure; tracking + relocalization; map data structures;
local mapping + optimizer; placecell and its integration) plus the lifecycle / viewer / ONNX / entry-point code,
every finding re-checked against the code before a fix. Nothing was run: sanitizer builds (`ALLFEATURE_SANITIZE`,
below) are for the next live run.

## Fixed

Most likely to explain the reported symptoms first.

| # | Defect | Symptom | Fix |
|---|---|---|---|
| 1 | `KeyFrame::cache_matched_pairs{,_feat_type}`: plain `std::map`s written concurrently by Tracking (`match_keyframe_to_frame`, every reference-keyframe fallback and relocalization), LocalMapping (`cache_neighbor_matches`, `match_keyframes`) and LoopClosing (`match_keyframes`; an `operator[]` insert while reading in `search_by_projection_for_compute_sim3`). The header claimed single-thread access. | Red-black-tree corruption: segfault later in an unrelated `find`/insert/destructor, heap corruption (a destroyed-looking mutex), or a `find` spinning forever. | `KeyFrame::cache_mutex_` (leaf lock) around every access; two keyframes with `std::scoped_lock`; cached vectors copied out under the lock; `find` instead of `operator[]`. |
| 2 | Same caches: the per-type maps stored stacked (cross-type offset) indices from two writers and per-type indices from the third; the Sim3 projection search compares them with per-type indices and writes `pts_matched[bestIdx]`. | Wrong loop matches; out-of-range write when `feature_vpr` is not the first feature type. | Per-type indices everywhere; `bestIdx` bounds-checked. |
| 3 | `Tracking::matches_per_feature_` survived a new reference frame / a dropped initializer / a reset, while the FrameDrawer drew them against the current frame's keypoints (no bounds check). | Out-of-range read on the viewer (the macOS main thread) when the lens is covered or the image goes blank during initialization, or right after the start-pause. | Cleared at every initialization attempt and in `reset()`; `DrawFrame` skips matches outside either keypoint set. |
| 4 | `Optimizer::OptimizeSim3` dereferenced the current keyframe's point before its null check; LocalMapping keeps erasing points during the Sim3 search. | Segfault on a loop candidate that reaches Sim3 consensus. | Null check first, snapshot size checked. |
| 5 | `KeyFrame` created `descriptors[ft]` only for types with keypoints; LocalMapping / fusion / loop matching read `descriptors.at(ft)` for every type (inside an OpenMP region in `cache_neighbor_matches`). | `std::out_of_range` -> `std::terminate` when a keyframe has no keypoints of one type (dark / low texture). | Every type gets an entry (empty Mat); `match_descriptors` returns no matches for an empty side. |
| 6 | `KeyFrame::set_bad_flag` held its connections lock while locking children and parent; LocalMapping culls while LoopClosing runs deferred culls (`set_erase`). | Deadlock (parent and child culled concurrently), then every `is_bad()` on either blocks: full freeze. | Process-wide `KeyFrame::set_bad_mutex_` around the destructive part; idempotent (`bad_` early return); `set_erase` reads `to_be_erased_` under its lock. |
| 7 | `correct_loop` requested the mapper stop before waiting on `gba_mutex_`; a finishing global BA's `release()` cleared that request; the wait then never ended. | LoopClosing livelock; a later reset froze Tracking, Shutdown hung. | Stop request re-asserted inside the wait loops (also in `apply_gba_correction`). |
| 8 | A superseded global BA was detached and then shared `gba_stop_` with the next one (reset to false), so it kept writing `Tcw_gba`/`position_gba` next to the new BA and could outlive Shutdown; reset never stopped a running BA before `map_->clear()`. | Wrong poses; use-after-free at reset or at exit (static mutexes destroyed under a running thread: "mutex lock failed"). | `abort_global_bundle_adjustment()`: stop flag + new generation + join, used by `correct_loop`, the reset and the thread's own exit; never detached. |
| 9 | `apply_gba_correction`'s spanning-tree BFS had no visited set and applied uninitialized `Tcw_gba` / `Tcw_before_gba` (Eigen does not zero them) to keyframes it reached without a BA result. | Endless loop under `gba_mutex_` + `map_update_mutex_` (full freeze) on a tree cycle; NaN / garbage map. | Visited set; only keyframes with a result are moved, points only through corrected keyframes; both matrices initialized to identity. |
| 10 | `OptimizeEssentialGraph` added edges to keyframes without a vertex (bad parents, loop connections, loop edges) and recovered poses from null vertices. | Null `Vertex*` dereference in g2o `addEdge`: segfault during loop correction. | One edge helper that skips missing vertices; recovery and point correction skip missing ones. |
| 11 | `update_connections` adopted a keyframe being culled as first parent (never re-hung afterwards). | Orphan subtrees: the root cause of #10 / #9 inputs. | First parent = strongest non-bad covisible. |
| 12 | Reset ran inside `track()` under `map_update_mutex_` (degenerate initialization) while waiting for LocalMapping / LoopClosing; a GBA correction waiting for that lock with the mapper stopped deadlocked. Reset also waited forever for a finished viewer, and left `ref_keyframe_`, `last_keyframe_`, `local_keyframes_`, `last_frame_`, ... of the wiped map (ids restart at 0). | Freeze; stale keyframes aliasing new ids. | Deferred to the start of the next `track()`, outside the lock; finished viewer not waited for; per-run state cleared. |
| 13 | Emergency-keyframe wait spun on `accepts_keyframes()`, false for a whole loop correction, and forever when the insertion itself had been refused. | Tracking freeze for seconds (or for good). | Flag cleared when insertion is refused; wait ends when the mapper is stopped or finished. |
| 14 | Exceptions with no handler on worker threads: LightGlue / MegaLoc embed failures (Core ML / ORT), OpenCV asserts and `.at()` in tracking stages, an undecodable stream message. | `std::terminate` (abort with the terminate-handler stack). | Failed LightGlue match -> no matches; failed MegaLoc embed -> no query / no VPR entry / no information value; any exception in a tracking stage (now also optical flow and relocalization) -> frame LOST; undecodable stream messages skipped (50 in a row stop the run). |
| 15 | Local BA wrote non-finite estimates back. | NaN poses / points spreading into tracking. | Non-finite estimates skipped. |
| 16 | `Map::clear()` took no lock; LoopClosing ran detection on keyframes culled while queued (re-adding them to the VPR store). | Use-after-free if anything reads during a reset; stale VPR rows. | `clear()` under `mMutexMap` (contents released after it); bad queued keyframes skipped. |

## Not changed (reported)

- Relocalization descriptor thresholds swapped (`FeatureMatcher.cpp`, `descDistTh_{low,high}_reloc` read from the other key)
  and `initRotationHistogram` using `rotFactor = 1/30` instead of `30/360`: wrong results (harder relocalization,
  looser orientation filter), behaviour change -> measure on the gym first.
- `match_map_points_to_frame` validates against stale projections of points not in the frustum this frame; a
  rejected relocalization hypothesis leaves its pose in the LOST frame (returned to the caller).
- Monocular initializer flattens keypoints in `FeatureType` order, Tracking in `features:` order: identical for
  `[orb32, superpoint256]`, wrong correspondences if a later enum value is listed first (no out-of-range).
- LocalMapping can stop with a keyframe still queued (non-atomic `has_new_keyframes` / `stop_if_requested`), and
  `release()` drops the queue: that keyframe is never mapped.
- placecell: O(n^2) cull / centring / visualizer work over every keyframe ever inserted (append-only store) ->
  progressive slowdown and memory growth on long streams; `PlaceCell.VisualizeCentred: 0` halves the viewer part.
  Latent: `clear()` invalidates `descriptor()` pointers, `items()` vs `set_items()`, `add()` check-then-act across
  two lock scopes (unreachable from the host today). Fixes belong in the placecell fork.
- Viewer reads `tracker->state_` / `num_tracked_frames_` unsynchronized (benign).

## Sanitizer builds

`ALLFEATURE_SANITIZE=address|thread` (CMakeLists.txt): instruments every target, -O1 -g, executables without
mimalloc. Built into `build-asan/` and `build-tsan/` with `ALLFEATURE_OUTPUT_IN_SOURCE=OFF`; run them from the repo
root (models resolve from the cwd), e.g.
`./build-asan/vslamlab_allfeature_mono_arstream stream_url:tcp://localhost:5556 calibration_yaml:... exp_folder:...`.
TSan does not see libomp's synchronisation: reports inside OpenMP regions need checking by hand.
