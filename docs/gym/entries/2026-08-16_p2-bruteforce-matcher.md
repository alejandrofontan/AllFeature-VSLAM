# 2026-08-16 — P2 — SIMD/OpenMP brute-force matcher replaces cv::BFMatcher (addendum P2)

| | |
|---|---|
| Sequence | `eth` / `table_3` (1180 frames, mono, `verbose: 0`) |
| Config | `configs/exp_debug.yaml` → `exp_debug_allfeature-dev`, `NumRuns: 3` (the pre-gym profiling workload) |
| Baseline | per-branch vanilla re-measured the same day, see the table |
| Kind | speed (profiling protocol of issue #13); ATE / losses / keyframes as guardrails |
| Verdict | kept |
| Issue | issue #13 |

*Moved verbatim from `profiling.md`'s results log on 2026-09-19 (documentation plan). Written before the gym entry format existed: trigger / hypothesis / change / effect / verdict are inside the text rather than under their own headings.*

## Record

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
