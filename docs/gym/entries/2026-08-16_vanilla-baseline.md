# 2026-08-16 — branch `dev` — Vanilla reference (no modification)

| | |
|---|---|
| Sequence | `eth` / `table_3` (1180 frames, mono, `verbose: 0`) |
| Config | `configs/exp_debug.yaml` → `exp_debug_allfeature-dev`, `NumRuns: 3` (the pre-gym profiling workload) |
| Baseline | per-branch vanilla re-measured the same day, see the table |
| Kind | speed (profiling protocol of issue #13); ATE / losses / keyframes as guardrails |
| Verdict | baseline measurement, no change |
| Issue | issue #13 |

*Moved verbatim from `profiling.md`'s results log on 2026-09-19 (documentation plan). Written before the gym entry format existed: trigger / hypothesis / change / effect / verdict are inside the text rather than under their own headings.*

## Record

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
