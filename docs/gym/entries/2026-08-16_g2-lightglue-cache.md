# 2026-08-16 — G2-lightglue-cache — LightGlue keyframe-tensor cache (addendum G2) — within noise

| | |
|---|---|
| Sequence | `eth` / `table_3` (1180 frames, mono, `verbose: 0`) |
| Config | `configs/exp_debug.yaml` → `exp_debug_allfeature-dev`, `NumRuns: 3` (the pre-gym profiling workload) |
| Baseline | per-branch vanilla re-measured the same day, see the table |
| Kind | speed (profiling protocol of issue #13); ATE / losses / keyframes as guardrails |
| Verdict | reverted (within noise at table_3 scale) |
| Issue | issue #13 |

*Moved verbatim from `profiling.md`'s results log on 2026-09-19 (documentation plan). Written before the gym entry format existed: trigger / hypothesis / change / effect / verdict are inside the text rather than under their own headings.*

## Record

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
