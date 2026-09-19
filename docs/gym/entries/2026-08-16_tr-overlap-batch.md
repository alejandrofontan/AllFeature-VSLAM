# 2026-08-16 — TR-overlap-batch — GPU/CPU match overlap + observation visitor + parallel frustum — **no benefit, reverted**

| | |
|---|---|
| Sequence | `eth` / `table_3` (1180 frames, mono, `verbose: 0`) |
| Config | `configs/exp_debug.yaml` → `exp_debug_allfeature-dev`, `NumRuns: 3` (the pre-gym profiling workload) |
| Baseline | per-branch vanilla re-measured the same day, see the table |
| Kind | speed (profiling protocol of issue #13); ATE / losses / keyframes as guardrails |
| Verdict | reverted (Track Ref +1 ms, rest within noise) |
| Issue | issue #13 |

*Moved verbatim from `profiling.md`'s results log on 2026-09-19 (documentation plan). Written before the gym entry format existed: trigger / hypothesis / change / effect / verdict are inside the text rather than under their own headings.*

## Record

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
