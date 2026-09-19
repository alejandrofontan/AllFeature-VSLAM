# 2026-08-16 — P45-parallel-lm — two-phase parallel fuse kept (P5); parallel triangulation reverted (P4)

| | |
|---|---|
| Sequence | `eth` / `table_3` (1180 frames, mono, `verbose: 0`) |
| Config | `configs/exp_debug.yaml` → `exp_debug_allfeature-dev`, `NumRuns: 3` (the pre-gym profiling workload) |
| Baseline | per-branch vanilla re-measured the same day, see the table |
| Kind | speed (profiling protocol of issue #13); ATE / losses / keyframes as guardrails |
| Verdict | kept (P5 two-phase parallel fuse); P4 parallel triangulation reverted |
| Issue | issue #13 |

*Moved verbatim from `profiling.md`'s results log on 2026-09-19 (documentation plan). Written before the gym entry format existed: trigger / hypothesis / change / effect / verdict are inside the text rather than under their own headings.*

## Record

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
