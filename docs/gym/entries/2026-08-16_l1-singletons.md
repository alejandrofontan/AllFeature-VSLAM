# 2026-08-16 — L1-singletons — factory singletons + hardware popcount + grid-cell refs

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
