# 2026-08-16 — aliked-only — single learned feature, no vocabulary (experiment, not a perf cycle)

| | |
|---|---|
| Sequence | `eth` / `table_3` (1180 frames, mono, `verbose: 0`) |
| Config | `configs/exp_debug.yaml` → `exp_debug_allfeature-dev`, `NumRuns: 3` (the pre-gym profiling workload) |
| Baseline | per-branch vanilla re-measured the same day, see the table |
| Kind | speed (profiling protocol of issue #13); ATE / losses / keyframes as guardrails |
| Verdict | experiment recorded; aliked-only kept as an optional mode, not adopted as default |
| Issue | issue #13 (speed audit) |

*Moved verbatim from `profiling.md`'s results log on 2026-09-19 (documentation plan). Written before the gym entry format existed: trigger / hypothesis / change / effect / verdict are inside the text rather than under their own headings.*

## Record

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
