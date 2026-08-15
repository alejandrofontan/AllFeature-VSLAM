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
