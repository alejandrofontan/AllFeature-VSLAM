# Scattered tracking-loss investigation (NSAVP R0_FA0)

- **Date:** 2026-08-13
- **Kind:** note
- **Provenance:** moved verbatim from `CLAUDE.md` § *Scattered Tracking-Loss Investigation (2026-08-13)* on 2026-09-19 (documentation plan, `docs/plans/documentation_plan.md`). File/line references are as of the original date.


User-reported: `pixi run vslamlab configs/exp_debug.yaml` (NSAVP `R0_FA0`, RGB-D, fastfoundationstereo depth, mask2former masks, 19103 frames) loses tracking non-deterministically at scattered frames. Diagnosed on branch `trackinglost` with repeated instrumented full-range runs (two concurrent processes per round to double the sample rate). Four distinct defects found; all fixed on this branch.

### Fixed 1: `verbose:0` crashed every run on frame 1 (null-viewer profiling calls)

With `verbose:0` no `Viewer` is created, yet `Tracking::GrabImageMonocular` (Tracking.cc:127) and `LocalMapping::Run` (LocalMapping.cc:83) called `viewer->set_*_time_median()` unconditionally through the null `shared_ptr`. `Viewer::mutexProfileStats` is the class's first data member (offset 0), so the inlined `unique_lock` received a null `std::mutex*` — libstdc++ throws `std::system_error` EPERM ("Operation not permitted") for that, uncaught → `terminate` on the first frame, every run. (These call sites date to April; all prior debug runs must have used `verbose:1`.) Guarded both sites like `Tracking::Reset` already does. Also fixed `build.sh`'s `dirname "LIBRARY_PATH"` (missing `$`) which made the build silently a no-op when invoked from the parent repo.

### Fixed 2 (the Problem-3 segfault): wild-pointer UB in `MapPoint::EraseObservation`

Run 2 died with a kernel-journal general protection fault at `libAllFeature-VSLAM.so+0x120b7e` → resolved (by rebuilding the exact commit's `.so` and `addr2line`) to `MapPoint::EraseObservation`. After erasing a point's **last** observation the code ran `mpRefKF = observations.begin()->second->projKeyframe;` on the now-**empty** map — copying a `shared_ptr` out of `end()` garbage, i.e. a refcount increment through a wild pointer: silent heap corruption on most occurrences, occasional immediate GPF. RGB-D makes this hot: the depth-seeded pass in `LocalMapping::CreateNewMapPoints` creates single-observation points by the thousand, and `KeyFrameCulling → KeyFrame::SetBadFlag` erases exactly such last observations en masse. Stock ORB-SLAM2 has the same latent code shape but rarely holds single-observation points there. Fix: only re-point `mpRefKF` when observations remain; the empty case falls through to `SetBadFlag` immediately below.

### Fixed 3 (the actual scattered tracking losses): pose-optimization divergence from a constant-position seed

The recurring loss signature (`TrackReferenceKeyFrame`: hundreds of raw matches → 0–5 inliers after pose optimization; relocalization succeeds **on the next frame** with >170 inliers, proving frame and map were fine) was captured with per-pass instrumentation at frame 3325: `passInliers=[0,-,-,-]`, `poseDelta trans=1.48m rot=3.5°`, rejected medians pxErr≈31–43px while inverse-depth error stayed ≈0.7σ. Mechanism: this codebase dropped ORB-SLAM2's `TrackWithMotionModel` (`mVelocity` was computed but never consumed — issue #8), so pose optimization seeded from `lastFrame.Tcw`. At 15–19 m/s that starts every pixel residual ~a full frame of motion (20–40px) large → the 2D terms begin Huber-saturated (bounded gradients) while the RGB-D inverse-depth terms (information `1/σ²≈1.3e5`) are still quadratic. Depth constrains only the z-component — it is blind to rotation/lateral translation — so pass-1 LM occasionally walks into a wrong rotation basin, ends with depth satisfied and all pixels ~30px off, and classifies **every** correspondence as an outlier → `LOST`. Non-deterministic because it needs a particular mix of match set/geometry, hence "fully scattered" losses. Two-part fix in `TrackReferenceKeyFrame`: seed with `mVelocity * lastFrame.Tcw` when valid (restores the small-initial-residual precondition the 4-pass scheme assumes), and on a collapse (<10 inliers from ≥45 raw matches) reset flags, re-seed, and re-run `PoseOptimization(pFrame, useDepthChannel=false)` once — pure 2D, the configuration the scheme was tuned for.

### Fixed 4: g2o LM accepted cost-increasing trial steps, and outlier classification read stale errors

Even with fixes 1–3 plus motion-model seeding, collapses persisted. The collapse dumps proved the match set was near-perfect at the prior (median 1.5px residual, unimodal) yet the optimizer ended 2.3° away with all matches rejected — impossible under monotone LM, which exposed two defects in the vendored `Thirdparty/g2o`'s Levenberg loop and its callers:

- **Acceptance test sign bug** (`optimization_algorithm_levenberg.cpp`): `rho = (currentChi−tempChi)/scale` also passes when *both* numerator and `computeScale()` are negative — a degenerate linear-solve step (much likelier with RGB-D's mixed-information system: pixel info ~1 vs inverse-depth info ~1.3e5) can be **accepted while increasing the cost**, walking the estimate out of a converged basin. Fixed in the g2o fork (submodule commit `6a60ea7`): require `tempChi < currentChi` explicitly.
- **Stale-chi2 classification**: after a rejected trial, g2o `pop()`s the vertex estimates but leaves each edge's cached `_error` at the rejected pose. `PoseOptimization`'s per-pass classification, LBA's `MarkBAOutliers`/`CollectBAOutliers` (which **erase observations from the map** — silent map damage, mono mode included), and `OptimizeSim3`'s inlier checks all read `chi2()` from that stale state. Fixed by recomputing every edge's error before thresholding.

### Fixed 5: garbage depth-seeded points + ungated global matching dominate pose optimization

Last observed loss mode (frame 17166: 469 matches, median residual 2px at the prior, but 10 matches at 10²–6.5×10⁵ px dragged the pose 25°, surviving even the depth-free retry). The 10 were all single-observation depth-seeded points created by the *current* reference keyframe from spiky near-road sensor depths; at ~18.5 m/s they cross the camera plane within 1–2 frames, and `match_keyframe_to_frame` (global descriptor matching + F-filter, which never sees the 3D position) happily keeps them — stock ORB-SLAM2's projection-windowed search can't produce such matches by construction. Huber's linear tail × the exploding `fx/z` Jacobian gives them unbounded influence. Fixed with a **prior-consistency gate** in `TrackReferenceKeyFrame`: drop matches whose map point projects behind/near the camera plane (z<0.2m) or >80px from the motion prior (genuine matches measure ≤11px there in every collapse dump). Deeper hardening of depth-seeded creation itself (spike rejection) belongs to issue #5's depth-consistency work.

### Verification

Full-range (19103-frame) RGB-D runs previously lost tracking at frames 2327/3325/4379/4456 (four different runs, never surviving past ~4500). With the fixes, four full-range runs were completed:

- pre-gate build: **zero losses** (3 pose-opt collapses, all caught by the depth-free rescue);
- gate build ×2: one with zero losses, one with two transient 1-2-frame dips (frames 428/450, driving under a large overhanging canopy — dark/dappled imagery; both self-recovered via instant relocalization, the second dip being the #9 embargo echo that motivated the bypass);
- final build (all fixes incl. embargo bypass): **zero losses, zero collapses** across all 19103 frames, 3 benign emergency keyframes, median tracking time 73 ms, ATE RMSE 16.4 m over the full route (2587 keyframes) — accuracy tuning (e.g. issue #3 depth-info calibration) is follow-up work; this investigation's target was the losses.

Note for reruns: the parent framework's swap watchdog measures absolute system swap, so stale swap from any earlier memory-heavy episode makes it kill new runs instantly (VSLAM-LAB/VSLAM-LAB#119) — the final run above was executed by invoking `pixi run --frozen -e allfeature-dev execute-rgbd ...` directly, then `pixi run evaluate`.

### Observed, filed as issues (not fixed here)

- **#9** post-relocalization keyframe embargo (`lastRelocFrameId + maxFrames`) starves the tracker at driving speed: reloc at 4380 → raw matches decayed 276→124 against the frozen reference KF → re-lost at 4396.
- **#10** no recovery budget when lost: a loss at frame 2327 left the run grinding relocalization for the remaining ~17k frames.
- **#11** `MapPointCulling` found-ratio test only applies to `featureTypes[0]`.
- **#12** shutdown `terminate called without an active exception` (pre-existing, from the 2026-08-12 notes, now tracked).
- **#8** motion model unused (the seeding half is now fixed on this branch; the issue remains as the umbrella for a full `TrackWithMotionModel` port).

### Diagnostics kept in the code (all cheap, `AF_WARN`/`AF_INFO`)

- `PoseOptimization`: on >50% rejection — per-channel breakdown (mono/stereo/rgbd), per-pass inlier counts, initial→final pose delta, rejected-edge medians (pixel error, inverse-depth error, sensor depth). The pixel-vs-depth split attributes a burst to 2D geometry vs the depth channel in one line.
- `Tracking::Track`: 100-frame heartbeat (inliers / localPts / KFs / mapPts) so post-mortems see the trend leading into a loss.
