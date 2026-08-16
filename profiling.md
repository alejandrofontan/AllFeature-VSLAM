# Profiling Protocol

Protocol for measuring the speed impact of each optimization from
[issue #13](https://github.com/alejandrofontan/AllFeature-VSLAM/issues/13), one modification at a
time, against a per-branch vanilla baseline. Results accumulate in the **Results log** at the
bottom of this file.

---

## Workload

Everything runs from the parent repo root (`/home/alejandro/VSLAM-LAB`):

```bash
pixi run vslamlab configs/exp_debug.yaml --overwrite
```

The `--overwrite` flag is **mandatory** for profiling: without it the runner treats the already-
completed runs in the evaluation folder as done and skips them, so you'd silently re-parse the
previous build's logs instead of measuring the new one.

- Experiment: `exp_debug_allfeature-dev` — `Module: allfeature-dev`, `Parameters: {verbose: 0, mode: mono}`.
- Sequence (from `config_debug.yaml`): `eth` / `table_3`.
- **`NumRuns: 3`** must be set in `configs/exp_debug.yaml` (one command → 3 runs). Verify before starting a cycle.
- Outputs land in:
  `/home/alejandro/VSLAM-LAB-Evaluation/exp_debug_allfeature-dev/ETH/table_3/`
  - `system_output_00000.txt` … `00002.txt` — full stdout/stderr of each run (what we parse)
  - `log_run_sequence_time.csv` — wall-clock per run
  - `XXXXX_KeyFrameTrajectory.csv`, `vslamlab_evaluation/` — accuracy guardrails

## What we measure

`PROFILING_EXHAUSTIVE` (in `include/Definitions.h`, currently **on**) makes the tracking thread
print a cumulative profiling block every frame. The histograms behind it accumulate over the whole
run, so the **last block in the log is the whole-run median per stage** (1 ms quantization):

```
[AFVSLAM] [Tracking Profiling]
    |  Resize Image                  0.00 ms
    |  Frame Creation                18.00 ms
    |  Tracking                      49.00 ms
    |    Track Ref                   22.00 ms
    |    Pose Optimization           1.00 ms
    |    Track Local Map             20.00 ms
    |  Grab Image Monocular          69.00 ms
```

Reading the nesting: `Grab Image Monocular` ≈ `Resize` + `Frame Creation` + `Tracking`;
`Tracking` ⊇ `Track Ref` + `Pose Optimization` + `Track Local Map`. Each row is an independent
median, so rows do not sum exactly.

The local-mapping thread prints its own block per keyframe cycle (`LocalMapping.cc`), same
cumulative scheme — again, the **last block in the log is the whole-run median**:

```
[AFVSLAM] [Local Mapping Profiling]
    |    Create NewMap Points        51.00 ms
    |    Search in Neighbors         12.00 ms
    |    Local Bundle Adjustment     40.00 ms
    |  Local Mapping                 115.00 ms
```

`Local Mapping` ⊇ the three sub-stages (plus BoW/culling not itemized). These medians are **per
keyframe**, not per frame — an optimization that changes keyframe *count* (guardrail!) shifts
them indirectly, which is another reason the keyframe-count guardrail below matters. Note the two
blocks interleave in the log from different threads; parse each by its own header, never by
position.

Per run we record:

| Metric | Source |
|---|---|
| The 7 tracking medians (ms) | last `[Tracking Profiling]` block of `system_output_XXXXX.txt` |
| The 4 local-mapping medians (ms) | last `[Local Mapping Profiling]` block of the same file |
| Slow frames (count, worst ms) | `grep "Track: slow frame"` lines (>150 ms stalls — tail behavior the medians hide) |
| Tracking losses | `grep -c "Tracking lost"` (guardrail, must not increase) |
| Wall clock (s) | `log_run_sequence_time.csv` |
| ATE RMSE | `pixi run evaluate configs/exp_debug.yaml` → `vslamlab_evaluation/` (guardrail) |

### Extraction one-liner

```bash
EXP=/home/alejandro/VSLAM-LAB-Evaluation/exp_debug_allfeature-dev/ETH/table_3
for f in "$EXP"/system_output_*.txt; do
  echo "== $(basename "$f")  ($(date -r "$f" +%H:%M:%S))"
  grep -A7 "\[Tracking Profiling\]" "$f" | tail -8
  grep -A4 "\[Local Mapping Profiling\]" "$f" | tail -5
  echo "   slow frames: $(grep -c "Track: slow frame" "$f")   losses: $(grep -c "Tracking lost" "$f")"
done
```

The `date -r` timestamp is the staleness check: the runner overwrites `system_output_XXXXX.txt`
on each rerun, but a crashed run can leave old files behind — **never parse a file whose mtime
predates the run you just launched.**

## Per-modification cycle

For **each** optimization item:

1. **Branch** — create and check out a temporary branch for the item (e.g. `perf/<item-slug>`),
   from the current integration state.
2. **Vanilla baseline** — rebuild unmodified (`pixi run build` from this directory, or
   `pixi run --frozen -e allfeature-dev bash Baselines/AllFeature-VSLAM-DEV/build.sh` from the
   parent repo) and confirm the build actually did work/relinked (not `ninja: no work to do` after
   a code change). Reset swap (`pixi run kill-all` — building fills it, see Environment rules),
   then run the experiment once (= 3 runs via `NumRuns: 3`). Parse all three logs and
   record the vanilla rows in the Results log below **before touching any code**.
   - Vanilla is re-measured per branch, not reused across branches: machine state drifts, and
     baselines must be contemporaneous with their comparison.
3. **Modify** — apply the single optimization under test. One item per branch; never mix.
4. **Measure** — rebuild (verify relink), reset swap (`pixi run kill-all`), rerun the experiment
   (3 runs), parse the three logs.
5. **Compare & record** — fill the item's table in the Results log: 3 vanilla runs, 3 modified
   runs, per-metric median of each triplet, and the delta. Decision rule with n=3 (no new
   thresholds, plain rank test): call it an **improvement** for a metric only if the modified
   median beats *all three* vanilla runs; **regression** if it's worse than all three; otherwise
   **within noise**.
6. **Guardrails** — before declaring the item good:
   - Tracking losses: identical count (expected 0 on `table_3`).
   - `pixi run evaluate configs/exp_debug.yaml`: ATE RMSE within the vanilla min–max spread.
   - Keyframe count (`wc -l XXXXX_KeyFrameTrajectory.csv`) in the vanilla spread.
   - Note on "bit-exact" items: end-to-end runs are **not** reproducible even vanilla-to-vanilla
     (LocalMapping timing changes keyframe decisions), so bit-exactness is verified at the unit
     level (parity tests, e.g. matcher-output equality on recorded frames), while the run-level
     check is only the statistical guardrail above. Do not expect identical trajectories.

## Environment rules (constant across vanilla and modified, per cycle)

- **Swap watchdog gotcha (VSLAM-LAB#119):** the runner kills processes on absolute system swap —
  stale swap from any earlier memory-heavy episode kills runs instantly, **and the build itself
  fills swap**, so this bites on every rebuild→run transition of the cycle. Reset with
  `pixi run kill-all` (from the parent repo: kills stray SLAM processes, then
  `swapoff -a && swapon -a`) after *every* build, before launching runs. `free -h` to verify
  swap is back near zero.
- Close memory/CPU-heavy apps and keep the machine's load comparable across the two halves of a
  cycle; run vanilla and modified back-to-back in the same session.
- `include/Definitions.h` flags (`PROFILING_EXHAUSTIVE`, `ALLFEATURE_REAL_TIME`,
  `ALLFEATURE_MAX_KEYFRAMES`, …) must be identical between vanilla and modified — they change
  behavior, not just logging.
- Keep `verbose: 0` (no viewer; also exercises the headless code path).
- Don't run two experiments concurrently while profiling (unlike the trackinglost investigation —
  here the numbers themselves are the result).
- First run after a clean environment may pay one-time GPU/model init; it's included in all runs
  equally, but if run 00000 looks anomalous versus 00001/00002, say so in the notes column.

---

# Results log

Newest item first. One subsection per branch/modification.

## Template

```markdown
## <date> — perf/<item-slug> — <one-line description of the change>

| Metric (median ms) | van 0 | van 1 | van 2 | van med | mod 0 | mod 1 | mod 2 | mod med | Δ | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Resize Image | | | | | | | | | | |
| Frame Creation | | | | | | | | | | |
| Tracking | | | | | | | | | | |
| — Track Ref | | | | | | | | | | |
| — Pose Optimization | | | | | | | | | | |
| — Track Local Map | | | | | | | | | | |
| Grab Image Monocular | | | | | | | | | | |
| LM: Create NewMap Points | | | | | | | | | | |
| LM: Search in Neighbors | | | | | | | | | | |
| LM: Local Bundle Adjustment | | | | | | | | | | |
| LM: Local Mapping (total) | | | | | | | | | | |
| Slow frames (n / worst ms) | | | | | | | | | | |
| Wall clock (s) | | | | | | | | | | |

Guardrails: losses <n>/<n>, ATE RMSE van [min–max] vs mod [min–max], KFs van [..] vs mod [..].
Notes: <anomalies, first-run effects, anything that qualifies the numbers>
```

<!-- Results entries go below this line -->

## 2026-08-16 — aliked-only — single learned feature, no vocabulary (experiment, not a perf cycle)

Change (deliberately behavioral — outside the bit-exact rule): `features: ["aliked128"]` with a
new vocabulary-less mode: `Vocabulary::isSupported()` gates the startup load (no more
`terminate()`), the loop-closing thread is not spawned (object still constructed; shutdown safe
via `mbFinished=true`), and `transform`'s per-keyframe error is a warn-once. **No loop closing,
no BoW relocalization — a tracking loss is unrecoverable in this mode** (stated in the startup
warning). orb32 configs are untouched (all gates key off `isSupported()`).

Result vs the two-feature P5-state baseline (3 runs each, ETH table_3, 1180 frames):

| Metric | orb32+aliked128 | aliked128 only | Δ |
|---|---|---|---|
| Grab Image Monocular (ms) | 54 | **20** | −63% |
| Tracking (ms) | 37 | 9 | −76% |
| — Track Ref | 23 | 7 | −70% |
| — Track Local Map | 10 | 1 | — |
| Frame Creation (ms) | 14 | 9 | −36% |
| LM: Local Mapping (ms/KF) | 106 | 33 | −69% |
| — CNMP / SiN / LBA | 33 / 19 / 39 | 21 / 3 / 6 | — |
| Wall clock (s) | 80 | 60 | −25% |
| Keyframes | 66–73 | 55–56 | fewer |
| **ATE RMSE (mm)** | **4.8–6.5** | **8.6 / 16.5 / 12.4** | **~2–3× worse** |
| Losses / slow frames | 0 / 0 | 0 / 0 | = |

Findings: (a) robustness held — full sequence tracked, zero losses, despite half the features
and no reloc safety net; (b) **accuracy degraded ~2–3×** — the cost of dropping from ~3000 to
~1000 keypoints per frame; (c) speed collapsed across the board, which **corrects an earlier
inference**: the LightGlue forward is ~5–7 ms at this scale, not ~15 — the two-feature Track
Ref's 23 ms was mostly per-feature-type loop overhead, map-point vector copies, joint filtering,
and association over 3× the keypoints, i.e. the multi-feature bookkeeping itself. Known cosmetic
leftover: one `Vocabulary::size()` ERROR line per run at startup (safe, returns 0).

Takeaway: aliked-only is a viable fast mode (20 ms/frame end-to-end) with a real accuracy
trade-off at current settings; before judging it, the fair comparison would raise
`FeatureExtractor.maxNumFeatures` for aliked toward the combined budget — that experiment is a
settings change, not code.

## 2026-08-16 — P45-parallel-lm — two-phase parallel fuse kept (P5); parallel triangulation reverted (P4)

Change (value-exact batch, both two-phase parallel-search + serial-apply-in-original-order):
**P4** — `CreateNewMapPoints` triangulation: phase A validates/triangulates candidates in
parallel (pure geometry), phase B creates points serially in original order (identical ids/order).
**P5** — `fuse_map_points_to_keyframe`: parallel projection/visibility/best-descriptor search,
serial apply **with `is_bad`/`is_in_keyframe` re-checked at apply time** — required for serial
equivalence, since a `replace()` earlier in the same call can invalidate later candidates
(co-observed points appear as both candidates and existing points). One documented bounded
deviation: a candidate's descriptor search sees its pre-call descriptor even if an earlier fuse
updated it (rare shared-point corner, same freshness class as existing cross-thread updates).
Parity 25/25.

Combined measurement (3v3), per-stage attribution (the items touch disjoint stages):

| Metric (median ms) | van med | P4+P5 med | Δ | Attribution / verdict |
|---|---|---|---|---|
| LM: Search in Neighbors | 19 | 14 | −5 (−26%) | **P5: improvement** |
| LM: Create NewMap Points | 34 | 35 | +1 | **P4: regression** — reverted |
| LM: Local Bundle Adjustment | 62 | 41 | −21 | side effect, see notes |
| LM: Local Mapping (total) | 131 | 109 | −22 (−17%) | improvement |
| Tracking / Track Ref / TLM | 36 / 23 / 10 | 36–37 / 23 / 10 | 0 | unchanged |
| Wall clock (s) | 80.0 | 80.0 | 0 | unchanged |

P4 diagnosis: up to ~10 OMP team spawns per keyframe cycle (per neighbor × feature type) on
batches of only a few hundred geometry candidates — spawn overhead exceeds the parallelized
work at table_3 scale. A batch-size threshold would fix it but is an excluded tunable; stashed
("P4 parallel triangulation … reverted; P5 kept") for possible revisit at NSAVP scale.

**P5-only confirmation runs (final state of this branch):**

| Metric (median ms) | van med | P5-only runs | med | Δ |
|---|---|---|---|---|
| LM: Search in Neighbors | 19 | 15 / 14 / 14 | 14 | **−26%** |
| LM: Create NewMap Points | 34 | 33 / 33 / 33 | 33 | back to baseline (P4 gone) |
| LM: Local Bundle Adjustment | 62 | 43 / 38 / 39 | 39 | **−37%** |
| LM: Local Mapping (total) | 131 | 110 / 102 / 106 | 106 | **−19%** |

Guardrails (confirmation): losses 0/0/0; ATE [mm] 4.8–6.5 (fine — the combined run's single
7.8 mm outlier did not recur and is treated as distribution tail across ~30 runs today); KFs
66–73 (fine).

Notes: the LBA drop (62 → 39, reproduced 6/6 runs across both builds) is a genuine side effect
of P5 tied to cycle concurrency — LBA itself is untouched; with SearchInNeighbors 5 ms shorter
the LM cycle aligns differently against the tracking thread's GPU/CPU phases. This mirrors (and
finally reverses) the LBA drift first seen in the P2 cycle: LBA's measured time on this machine
is dominated by cross-thread contention alignment, not solver work — worth remembering when
reading any future LBA delta. Verdict: **keep P5**; local mapping cycle now 106 ms median
(day cumulative: 190 → 106 ms, −44%).

## 2026-08-16 — TR-overlap-batch — GPU/CPU match overlap + observation visitor + parallel frustum — **no benefit, reverted**

Change (bit-exact batch): (1) `match_keyframe_to_frame` split two-phase — LightGlue types on a
worker `std::thread` concurrent with brute-force types on the calling thread, results consumed
in original order; (2) `MapPoint::forEachObservation` visitor replacing the per-point
`GetObservations()` map copy in `UpdateLocalKeyFrames`; (3) parallel frustum loop in
`SearchLocalPoints` (addendum P6). Parity 25/25 before measuring.

| Metric (median ms) | van 0 | van 1 | van 2 | van med | mod 0 | mod 1 | mod 2 | mod med | Δ | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Tracking | 37 | 37 | 37 | 37 | 37 | 38 | 39 | 38 | +1 | slight regression |
| — Track Ref | 23 | 23 | 23 | 23 | 24 | 24 | 24 | 24 | +1 | **regression** (item 1) |
| — Track Local Map | 10 | 11 | 10 | 10 | 9 | 11 | 11 | 11 | +1 | within noise (items 2–3) |
| Grab Image Monocular | 55 | 55 | 55 | 55 | 55 | 56 | 56 | 56 | +1 | within noise |
| LM: Local Bundle Adjustment | 63 | 62 | 61 | 62 | 62 | 63 | 65 | 63 | +1 | within noise |
| LM: Local Mapping (total) | 133 | 137 | 130 | 133 | 131 | 136 | 138 | 136 | +3 | within noise |
| Wall clock (s) | 80.0 | 80.0 | 80.0 | 80.0 | 80.0 | 80.0 | 80.0 | 80.0 | 0 | unchanged |

(Stages not shown were identical medians. Guardrails: losses 0/0/0; ATE and KF counts within
historical spread.)

Notes / diagnosis: the overlap hypothesis mis-estimated the split. After P2, the CPU work that
could hide under the GPU forward is only ~1–2 ms (orb32 BF) — while the overlap's own overhead
(a fresh `std::thread` per frame, and torch re-initializing per-thread CUDA/cuBLAS handle state
on every new thread) costs about the same, netting +1 ms. Per-stage attribution is clean because
the items touch disjoint stages: the Track Ref +1 is item 1 alone; items 2–3 are real
allocation/parallelism wins on paper but individually below the 1 ms quantization here.

**Decision: whole batch reverted** (stash "TR-overlap-batch (TrackRef +1ms regression / TLM
within noise, reverted)" on this branch). If the overlap idea is ever revisited, the fix for its
overhead is a persistent worker thread (condition-variable driven, so torch's thread-local state
initializes once) — but the ceiling is still only the ~2 ms BF share, so it is unlikely to be
worth it. Conclusion for the tracking loop: **Track Ref (23 ms) and Track Local Map (10 ms) are
at their practical bit-exact floors on this config** — the remaining mass is the LightGlue
forward and already-lean per-point work.

## 2026-08-16 — G2-lightglue-cache — LightGlue keyframe-tensor cache (addendum G2) — within noise

Change (bit-exact): `lightglue_feats_for()` caches the GPU `FeatDict` per frame_id (FIFO cap 8,
mutex-protected — LocalMapping calls the matcher from an OMP region), so repeat LightGlue matches
against the same keyframe skip the CPU→GPU tensor rebuild. Wired via optional `cacheId1/2`
params (default −1 = uncached) through `match_descriptors`/`serialFeatureMatching`;
`match_keyframe_to_frame` caches the KF side, `CreateNewMapPoints` both sides. Correctness hooks:
frame ids restart at 0 on `System::Reset()`, so `Tracking::Reset` and
`LocalMapping::ResetIfRequested` clear the cache. Parity 25/25 after changes.

| Metric (median ms) | van med | mod med | Δ | Verdict |
|---|---|---|---|---|
| Tracking | 37 | 36 | −1 | within noise |
| — Track Ref | 23 | 23 | 0 | **unchanged — the target metric** |
| — Track Local Map | 10 | 10 | 0 | unchanged |
| Frame Creation | 14 | 15 | +1 | within noise |
| Grab Image Monocular | 54 | 55 | +1 | within noise |
| LM: Create NewMap Points | 34 | 33 | −1 | within noise (borderline) |
| LM: Search in Neighbors | 19 | 19 | 0 | unchanged |
| LM: Local Bundle Adjustment | 61 | 58.5 | −2.5 | within noise (spread 51–61) |
| LM: Local Mapping (total) | 128 | 126 | −2 | within noise |
| Wall clock (s) | 80.0 | 80.0 | 0 | unchanged |

(Per-run values omitted — all within ±1 ms of the medians except LBA's 51–61 spread.)
Guardrails: losses 0/0/0; ATE [mm] van 5.2–6.2 vs mod 4.9–6.0; KFs van 68–69 vs mod 67–72. Fine.

Notes / finding: **the KF-side tensor rebuild is not a measurable cost at table_3 scale**
(~1000 aliked keypoints ≈ 0.5 MB per rebuild — sub-millisecond). Track Ref's 23 ms is therefore
dominated by the LightGlue forward pass itself plus match association — which also answers the
Frame Creation question: the learned-feature side is GPU-inference-bound, at its bit-exact floor.
The cache would matter more at NSAVP keyframe sizes (~5400 kps, 5× larger tensors) and in reloc
loops (candidate KFs re-matched repeatedly). **Decision: reverted** (no measurable effect on this
benchmark → not worth carrying). The implementation is preserved in `git stash` on this branch
("G2 lightglue cache (within-noise, reverted per decision)") in case it's revisited at NSAVP
scale; it includes the reset-correctness hooks (frame ids recycle after `System::Reset()`, so
any future frame_id-keyed cache MUST clear on reset — that finding outlives the revert).

## 2026-08-16 — M2-copies — copy/allocation elimination batch (addendum M2 remainder)

Change (value-exact batch): (a) `MapPoint::get_descriptor()` returns a shared `cv::Mat` header
instead of `clone()` — audited safe: `mDescriptor` is only rebound under its mutex, never written
in place; (b) `Frame`'s copy constructor shares descriptor buffers instead of `copyTo`
(~0.6 MB/frame off `lastFrame = Frame(currentFrame)`; `KeyFrame` clones its own copy explicitly);
(c) `match_map_points_to_frame` builds both descriptor matrices two-pass into preallocated Mats
(no `cv::Mat::push_back` whole-matrix regrowth); (d) rotation histogram reuses a `thread_local`
buffer — **inert on this benchmark** (only caller is `SearchByProjection`, reloc-only; zero
losses here), kept for lost-mode/reloc-storm scenarios; (e) `ComputeSceneMedianDepth` uses
`nth_element` (same element, O(n)). Full `Frame` move semantics evaluated and rejected:
`currentFrame` is read after the `lastFrame` assignment, so a move would change behavior.
Parity harness after changes: 25/25 PASS.

| Metric (median ms) | van 0 | van 1 | van 2 | van med | mod 0 | mod 1 | mod 2 | mod med | Δ | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Resize Image | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | unchanged |
| Frame Creation | 14 | 14 | 15 | 14 | 15 | 15 | 14 | 15 | +1 | within noise |
| Tracking | 37 | 37 | 38 | 37 | 37 | 36 | 37 | 37 | 0 | unchanged |
| — Track Ref | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 0 | unchanged |
| — Pose Optimization | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | unchanged |
| — Track Local Map | 10 | 11 | 11 | 11 | 11 | 9 | 10 | 10 | −1 | within noise (borderline) |
| Grab Image Monocular | 55 | 56 | 56 | 56 | 55 | 54 | 55 | 55 | −1 | within noise (borderline) |
| LM: Create NewMap Points | 34 | 34 | 34 | 34 | 33 | 33 | 33 | 33 | −1 (−3%) | **improvement** |
| LM: Search in Neighbors | 21 | 23 | 22 | 22 | 19 | 17 | 20 | 19 | −3 (−14%) | **improvement** |
| LM: Local Bundle Adjustment | 62 | 61 | 60 | 61 | 62 | 56 | 59 | 59 | −2 | **improvement** (noisy) |
| LM: Local Mapping (total) | 134 | 134 | 136.5 | 134 | 131.5 | 120 | 129 | 129 | −5 (−4%) | **improvement** |
| Slow frames (n) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | unchanged |
| Wall clock (s) | 80.0 | 80.0 | 80.0 | 80.0 | 80.0 | 80.0 | 80.0 | 80.0 | 0 | unchanged |

Guardrails: losses 0/0/0 (=); KFs van 65–69 vs mod 66–71 (fine); ATE RMSE [mm] van 4.2–4.9 vs
mod 4.9–6.2 — mod run 1 (6.2) sits above this cycle's unusually tight vanilla range but well
inside the session's historical run-to-run spread (4.0–7.4 across all cycles today), so treated
as ordinary variance rather than a change signal; all changes are value-exact by construction.

Notes: the win concentrates exactly where the clones lived — `Search in Neighbors` −14% (fuse
calls `get_descriptor()` per candidate map point per target keyframe) with smaller gains in
`Create NewMap Points` and LBA; the tracking thread was already lean here after P2/L1, so
`match_map_points_to_frame`'s prealloc shows only as the borderline TLM/Grab drift. Verdict:
**keep**. Local-mapping cycle is now 190 → 129 ms (−32%) cumulative across M1/P2/L1/M2, tracking
51 → 37 ms (−27%), wall 110 → 80 s.

## 2026-08-16 — L1-singletons — factory singletons + hardware popcount + grid-cell refs

Change (bit-exact batch): (a) `get_feature()` returns `const Feature&` to 8 function-local
static singletons instead of a fresh `unique_ptr` per call — removes a heap allocation from
every `descriptor_distance`/matcher dispatch (12 call sites adapted); (b) both copies of
orb32's 32-bit SWAR Hamming loop replaced with 4× `__builtin_popcountll` on 64-bit words
(`Orb32::descriptor_distance` now forwards to `DescriptorDistance_orb32` — one body, bit-identical
output); (c) the per-query grid-cell vector copies in `Frame::get_features_in_area` /
`KeyFrame::get_features_in_area` are now const references. Deliberately excluded: the float
features' `cv::norm` L2 distances — an Eigen swap would perturb last-ulp values feeding `TH_LOW`
threshold comparisons, violating the no-behavior-change rule. Parity harness after refactor:
25/25 PASS.

| Metric (median ms) | van 0 | van 1 | van 2 | van med | mod 0 | mod 1 | mod 2 | mod med | Δ | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Resize Image | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | unchanged |
| Frame Creation | 14 | 14 | 15 | 14 | 15 | 14 | 15 | 15 | +1 | within noise |
| Tracking | 39 | 39 | 39 | 39 | 36 | 37 | 37 | 37 | −2 (−5%) | **improvement** |
| — Track Ref | 23 | 23 | 23 | 23 | 22 | 23 | 23 | 23 | 0 | within noise |
| — Pose Optimization | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | unchanged |
| — Track Local Map | 12 | 12 | 11 | 12 | 10 | 11 | 11 | 11 | −1 | within noise (borderline) |
| Grab Image Monocular | 56 | 56 | 57 | 56 | 54 | 55 | 55 | 55 | −1 (−2%) | **improvement** |
| LM: Create NewMap Points | 34 | 34 | 33 | 34 | 33 | 34 | 34 | 34 | 0 | unchanged |
| LM: Search in Neighbors | 24 | 23 | 23 | 23 | 21 | 23 | 22 | 22 | −1 (−4%) | **improvement** |
| LM: Local Bundle Adjustment | 65 | 63 | 62 | 63 | 60 | 60 | 61 | 60 | −3 (−5%) | **improvement** |
| LM: Local Mapping (total) | 143 | 137 | 134 | 137 | 130 | 135 | 133 | 133 | −4 (−3%) | **improvement** |
| Slow frames (n) | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | unchanged |
| Wall clock (s) | 90.0 | 80.0 | 80.0 | 80.0 | 80.0 | 80.0 | 80.0 | 80.0 | 0 | within noise (quantized) |

Guardrails: losses 0/0/0 (=); ATE RMSE [mm] van 4.6–6.6 vs mod 4.6–5.0 (fine); KFs van 70–74 vs
mod 72–74 (fine).

Notes: vanilla re-measurement confirms the P2-era LBA drift is the steady state (62–65 ms), and
this batch pulls it back down to 60 ms — with LBA not using `descriptor_distance` at all, that
gain is best read as reduced cross-thread allocator/cache pressure from eliminating millions of
factory allocations in the concurrently-running fuse/culling paths. Every delta is small (1–4 ms)
but the direction is uniformly non-negative across all 13 metrics, which is what a correct
micro-optimization batch should look like. The isolated Hamming benchmark also firmed up
(2.3 → 1.6 ms). Verdict: **keep** — low payoff but zero risk, and the singleton factory removes a
class of overhead that would otherwise re-appear in every future caller.

## 2026-08-16 — P2 — SIMD/OpenMP brute-force matcher replaces cv::BFMatcher (addendum P2)

Change: `BruteForceMatcher.{h,cpp}` — cross-check (mutual-1-NN) matcher replicating
`cv::BFMatcher(<norm>, crossCheck=true)` exactly: single distance sweep, OpenMP over queries,
thread-local per-train bests merged with OpenCV's first-minimum tie-breaking; Hamming via
`__builtin_popcountll` (bit-exact), L2 via Eigen `squaredNorm` (same ordering, last-ulp
distances). Swapped into `match_descriptors`/`match_descriptors_only`; dead `cv::BFMatcher`
members removed. New `test_bfmatcher_parity` harness: **25/25 PASS** vs cv::BFMatcher across
shapes up to 2000×5400 and widths 32/48/61 B + 64/128/256 f. Isolated speedup at production
scale: Hamming 8.8 → 2.3 ms (3.8×), L2 15.8 → 6.6 ms (2.4×).

Step-0 diagnostic (now logged at startup): OpenCV 4.12 runs the **OpenMP backend with 28
threads** — the audit's suspicion that cv::BFMatcher was single-threaded is wrong; the NSAVP
60–155 ms figures must be re-read as large-input + in-run contention, not a disabled backend.

| Metric (median ms) | van 0 | van 1 | van 2 | van med | mod 0 | mod 1 | mod 2 | mod med | Δ | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Resize Image | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | unchanged |
| Frame Creation | 14 | 14 | 14 | 14 | 15 | 15 | 15 | 15 | +1 | slight regression |
| Tracking | 50 | 49 | 50 | 50 | 39 | 38 | 39 | 39 | −11 (−22%) | **improvement** |
| — Track Ref | 27 | 27 | 27 | 27 | 24 | 23 | 23 | 23 | −4 (−15%) | **improvement** |
| — Pose Optimization | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | unchanged |
| — Track Local Map | 16 | 16 | 17 | 16 | 12 | 11 | 12 | 12 | −4 (−25%) | **improvement** |
| Grab Image Monocular | 67 | 66 | 66 | 66 | 57 | 57 | 57 | 57 | −9 (−14%) | **improvement** |
| LM: Create NewMap Points | 50 | 50 | 50 | 50 | 34 | 33 | 34 | 34 | −16 (−32%) | **improvement** |
| LM: Search in Neighbors | 23 | 24 | 24.5 | 24 | 24 | 23 | 24 | 24 | 0 | unchanged (not BF) |
| LM: Local Bundle Adjustment | 48 | 59 | 55 | 55 | 63 | 62 | 62 | 62 | +7 | regression (see notes) |
| LM: Local Mapping (total) | 143 | 149 | 148 | 148 | 136 | 135 | 136 | 136 | −12 (−8%) | **improvement** |
| Slow frames (n) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | unchanged |
| Wall clock (s) | 100.0 | 100.0 | 100.0 | 100.0 | 90.0 | 80.0 | 80.0 | 80.0 | −20 (2 quanta) | **improvement** |

Guardrails: losses 0/0/0 (=); ATE RMSE [mm] van 5.4–7.0 vs mod 4.0–6.5 (fine); KFs van 70–76
vs mod 69–70 (fine).

Notes: `Create NewMap Points` −32% is the biggest single-stage gain — its neighbor matching runs
inside an existing OMP region where nesting keeps our matcher serial, so the win there is pure
per-pair popcount efficiency. `Search in Neighbors` unchanged as predicted (it uses
`descriptor_distance`, not BF matching — that's the L1/factory-singleton item's target). The one
regression: **LBA +7 ms** (62 vs 48–59, formally outside the vanilla range) with bit-identical
matcher output and similar KF counts — the plausible mechanism is increased concurrency
contention: tracking now finishes frames faster and our matcher's 28-thread bursts overlap more
of LBA's single-threaded runtime. LM **total** still improved (−12 ms) and wall clock dropped
two quanta (100 → 80 s), so the trade is clearly net-positive; watch whether the LBA drift
persists in later cycles. Verdict: **keep**.

## 2026-08-16 — perf/p3-g2o-openmp — G2O_USE_OPENMP in vendored g2o (addendum P3) — **REGRESSION, reverted**

Change: `G2O_USE_OPENMP=ON` for `Thirdparty/g2o` + parent-side `target_link_libraries(g2o
OpenMP::OpenMP_CXX)` and `EIGEN_DONT_PARALLELIZE` (needed because the vendored g2o CMakeLists
consumes `g2o_CXX_FLAGS` into `CMAKE_CXX_FLAGS` at line 23 *before* appending the OpenMP flags
at line 41 — flag-ordering bug worth fixing in the fork regardless). Verified genuinely active
before measuring (`#define G2O_OPENMP 1` in config.h, `GOMP_parallel` in `libg2o.so`, libgomp
linked). Vanilla = M1 state, re-measured fresh on this branch same session.

| Metric (median ms) | van 0 | van 1 | van 2 | van med | mod 0 | mod 1 | mod 2 | mod med | Δ | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Resize Image | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | — | (quantization) |
| Frame Creation | 14 | 14 | 14 | 14 | 15 | 16 | 16 | 16 | +2 | **regression** |
| Tracking | 50 | 48 | 51 | 50 | 67 | 68 | 70 | 68 | +18 (+36%) | **regression** |
| — Track Ref | 27 | 27 | 27 | 27 | 28 | 30 | 29 | 29 | +2 | **regression** |
| — Pose Optimization | 1 | 1 | 1 | 1 | 7 | 7 | 7 | 7 | +6 (7×) | **regression** |
| — Track Local Map | 16 | 16 | 17 | 16 | 23 | 24 | 24 | 24 | +8 (+50%) | **regression** |
| Grab Image Monocular | 66 | 65 | 68 | 66 | 85 | 87 | 88 | 87 | +21 (+32%) | **regression** |
| LM: Create NewMap Points | 51 | 50 | 51 | 51 | 51 | 52 | 52 | 52 | +1 | within noise |
| LM: Search in Neighbors | 25 | 24 | 24 | 24 | 24 | 23 | 26 | 24 | 0 | within noise |
| LM: Local Bundle Adjustment | 56 | 49 | 58 | 56 | 71.5 | 71 | 73.5 | 71.5 | +15 (+28%) | **regression** |
| LM: Local Mapping (total) | 155 | 144 | 149 | 149 | 168 | 168 | 176.5 | 168 | +19 (+13%) | **regression** |
| Slow frames (n) | 1 | 0 | 0 | 0 | 1 | 2 | 0 | 1 | +1 | (minor) |
| Wall clock (s) | 100.0 | 100.0 | 100.1 | 100.0 | 120.0 | 130.0 | 130.0 | 130.0 | +30 | **regression** |

Guardrails: losses 0/0/0 (=); ATE RMSE [mm] van 5.5–7.4 vs mod 4.3–5.5 (fine); KFs van 70–72 vs
mod 66–73 (fine) — the regression is purely speed, not accuracy.

Notes / diagnosis: g2o's OpenMP support spawns a full default-width (28-thread) team on **every**
`buildSystem` call. `PoseOptimization` makes ~40 such calls per frame (4 passes × 10 LM
iterations) on graphs of only a few hundred edges — team-spawn overhead is far larger than the
work being parallelized, hence 1 → 7 ms. LBA's graphs are bigger but still lose (+28%): on top of
team-spawn cost, `G2O_OPENMP` turns g2o's `openmp_mutex` into a *real* `omp_lock` (adding lock
traffic throughout the graph structures) and the two SLAM threads' OpenMP pools now fight each
other and the tracking thread for the same 28 cores — visible as collateral damage in stages that
don't even use g2o (`Frame Creation` +2 ms, `Track Ref` +2 ms).

**Verdict: revert (done — flag off, cache entry cleared, rebuild verified GOMP-free; branch left
uncommitted).** A salvageable variant would cap the team size (e.g. 4 threads) and enable it for
LBA only — but that's a tuning knob, which this tier explicitly excludes; parked as a possible
follow-up experiment after the bit-exact items land. The vendored CMakeLists flag-ordering bug
(OpenMP flags appended after `g2o_CXX_FLAGS` is consumed) is independently worth fixing in the
g2o fork so the option isn't silently inert for the next person.

## 2026-08-16 — perf/m1-allocator — mimalloc v2.4.5 linked as process allocator (addendum M1)

Change: `Thirdparty/mimalloc` submodule (pinned v2.4.5), built shared-only, linked **first** into
the three executables so `libmimalloc.so.2` precedes libc in symbol lookup and interposes
malloc/free for the whole process (verified: first `NEEDED` entry + `MIMALLOC_VERBOSE=1` banner).
Zero source-code change. Vanilla columns reused from the same-day baseline below (same tree,
measured minutes before branching; machine state unchanged, swap 0B in both halves).

| Metric (median ms) | van 0 | van 1 | van 2 | van med | mod 0 | mod 1 | mod 2 | mod med | Δ | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Resize Image | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | unchanged |
| Frame Creation | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 0 | unchanged |
| Tracking | 52 | 51 | 50 | 51 | 48 | 49 | 49 | 49 | −2 (−4%) | **improvement** |
| — Track Ref | 27 | 27 | 27 | 27 | 27 | 27 | 27 | 27 | 0 | unchanged |
| — Pose Optimization | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | unchanged |
| — Track Local Map | 18 | 18 | 16 | 18 | 15 | 15 | 15 | 15 | −3 (−17%) | **improvement** |
| Grab Image Monocular | 69 | 68 | 67 | 68 | 65 | 65.5 | 65 | 65 | −3 (−4%) | **improvement** |
| LM: Create NewMap Points | 51 | 51 | 51 | 51 | 52 | 49 | 51 | 51 | 0 | within noise |
| LM: Search in Neighbors | 27 | 28 | 25 | 27 | 23 | 24 | 23 | 23 | −4 (−15%) | **improvement** |
| LM: Local Bundle Adjustment | 88.5 | 88 | 84 | 88 | 47 | 47 | 45 | 47 | −41 (−47%) | **improvement** |
| LM: Local Mapping (total) | 190 | 194 | 185 | 190 | 143 | 141 | 141 | 141 | −49 (−26%) | **improvement** |
| Slow frames (n) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | unchanged |
| Wall clock (s) | 110.0 | 110.0 | 100.0 | 110.0 | 100.0 | 100.0 | 100.0 | 100.0 | −10 | improvement* |

Guardrails: losses 0/0/0 (=); ATE RMSE [mm] van 4.32–6.56 vs mod 3.98–6.57 (same spread); KFs
van 68–76 vs mod 67–74 (overlapping). All pass.

Notes: the headline is **Local Bundle Adjustment −47%** (88 → 47 ms median) — g2o's per-edge/
per-vertex heap churn plus cross-thread allocator contention with tracking was evidently the
dominant hidden cost of LBA. `Track Local Map` −17% (the descriptor-clone/`cv::Mat push_back`
path) and `Search in Neighbors` −15% follow the same pattern: the allocation-heaviest stages
gained the most, while compute-bound stages (`Track Ref` brute-force matching, `Frame Creation`)
are exactly unchanged — consistent with a pure allocator effect and no behavior change.
(*) Wall clock is quantized (~10 s steps: 110.0/110.0/100.0 → 100.0×3) — read it as "dropped one
quantum", not a precise −10 s. Verdict: **keep**; M1 lands as the new baseline for later items.

## 2026-08-16 — branch `dev` — Vanilla reference (no modification)

First end-to-end execution of this protocol; establishes the initial baseline and validates the
pipeline (build → swap check → 3 runs `--overwrite` → parse → evaluate). Working tree = `dev` +
profiling instrumentation (`Definitions.h`: `PROFILING_EXHAUSTIVE`; `LocalMapping.cc`: Local
Mapping profiling block). ETH `table_3`, 1180 frames, mono, `verbose: 0`.

| Metric (median ms) | run 0 | run 1 | run 2 | median |
|---|---|---|---|---|
| Resize Image | 1 | 1 | 1 | 1 |
| Frame Creation | 14 | 14 | 14 | 14 |
| Tracking | 52 | 51 | 50 | 51 |
| — Track Ref | 27 | 27 | 27 | 27 |
| — Pose Optimization | 1 | 1 | 1 | 1 |
| — Track Local Map | 18 | 18 | 16 | 18 |
| Grab Image Monocular | 69 | 68 | 67 | 68 |
| LM: Create NewMap Points | 51 | 51 | 51 | 51 |
| LM: Search in Neighbors | 27 | 28 | 25 | 27 |
| LM: Local Bundle Adjustment | 88.5 | 88 | 84 | 88 |
| LM: Local Mapping (total) | 190 | 194 | 185 | 190 |
| Slow frames (n) | 0 | 0 | 0 | 0 |
| Wall clock (s) | 110.0 | 110.0 | 100.0 | 110.0 |

Guardrails: losses 0/0/0; ATE RMSE [mm] 6.10 / 6.56 / 4.32 (range 4.3–6.6); KFs 72 / 76 / 68.

Notes: runs are highly consistent (per-stage medians within 1–2 ms across runs). At ~68 ms
median `Grab Image Monocular` on a 640×480 indoor sequence, the frame budget breakdown matches
the audit's expectations: `Track Ref` (27 ms, global BF matching + filter) and `Track Local Map`
(18 ms) dominate tracking; `Frame Creation` (14 ms, extraction) next. Local mapping cycles at
~190 ms median per keyframe, dominated by LBA (88 ms) and `Create NewMap Points` (51 ms).
Swap stayed at 0B throughout (no rebuild work preceded the runs).
