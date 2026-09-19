# Gym protocol

The gym is where AllFeature-VSLAM is run against VSLAM-LAB sequences to find failures and to measure
what a change does about them. Its output is a **log of changes with measured effects**
(`entries/`), a **board** of the gym sequences with their current numbers (`board.md`), and the
issues that runs expose. This page is the procedure; it grew out of the 2026-08 speed-profiling
protocol of issue #13 (its results are the first ten entries).

Rules that make the log trustworthy:

1. **One change per entry**, measured against the same baseline commit, in the same session. Two
   changes in one entry cannot be attributed.
2. **Three runs or more** per configuration. Only `Tracking.Sequential: 1` + `features: [orb32]` +
   `vpr: none` + `verbose: 0` is deterministic (issue #16); everything else needs the rank test below.
3. **Fixed configs.** Gym sequences and experiments live in VSLAM-LAB's `configs/config_gym.yaml`
   and `configs/exp_gym.yaml`; an entry cites the experiment name, never a description of it.
4. **`--overwrite`, always.** Without it the runner treats the completed runs in the evaluation
   folder as done and skips them: you would re-parse the previous build's logs.
5. **Gym ≠ benchmark.** The gym set is small and run daily; a change kept in the gym gets a
   regression check on the benchmark set (issue #18, 100 sequences) before it counts as final, and
   the entry records that check. This is the guard against tuning to two sequences.

## Workload

From the VSLAM-LAB root:

```bash
pixi run kill-all                                       # reset swap after a build (see Environment)
pixi run vslamlab configs/exp_gym.yaml --overwrite      # NumRuns: 3 → three runs per sequence
pixi run evaluate configs/exp_gym.yaml                  # ATE → <seq>/vslamlab_evaluation/ate.csv
```

| Gym sequence | Why it is in the set | Config |
|---|---|---|
| `eth` / `table_3` | indoor tabletop, 1180 frames, ~80 s per run; loop closure at the end; the speed-profiling workload since 2026-08 | `exp_gym_mono` (mono, `verbose: 0`) |

Outputs land in `VSLAM-LAB-Evaluation/<exp>/<DATASET>/<sequence>/`: `system_output_<run>.txt`
(full stdout/stderr, what the tools parse), `<run>_KeyFrameTrajectory.csv`,
`log_run_sequence_time.csv`, `vslamlab_evaluation/ate.csv`, `<run>_placecell/` (when
`PlaceCell.Dump: 1`), and `vslamlab_exp_log.csv` at the experiment root (RAM/swap peaks).

**Staleness check:** the runner overwrites `system_output_<run>.txt` on each rerun, but a crashed
run leaves old files behind. Never read a file whose mtime predates the run you launched
(`date -r <file>`).

## What is measured

`docs/tools/gym_compare.py <before_exp> <after_exp>` prints the whole table from two experiment
folders; the columns are fixed so entries compare across time.

| Metric | Source | Direction |
|---|---|---|
| ATE RMSE (m) | `vslamlab_evaluation/ate.csv`, `rmse` per run | lower |
| evaluated / tracked frames | same file (`num_evaluated_frames`, `num_tracked_frames`) | higher |
| keyframes | rows of `<run>_KeyFrameTrajectory.csv` | context (a guardrail, not a target) |
| tracking losses | `Tracking lost` lines in `system_output_<run>.txt` | lower, 0 expected on `table_3` |
| emergency keyframes | `emergency keyframe` lines | lower |
| median tracking time (ms) | `median tracking time:` printed by the executable at exit | lower |
| wall clock (s) | `log_run_sequence_time.csv` | lower (quantised in ~10 s steps by the runner) |
| peak RAM / swap (GB) | `vslamlab_exp_log.csv` | lower |
| per-stage medians (ms) | last `[Tracking Profiling]` / `[Local Mapping Profiling]` block | lower; only with `PROFILING_EXHAUSTIVE` |

Per-stage profiling: `PROFILING_EXHAUSTIVE` in `include/Definitions.h` (off by default) makes the
tracking thread print a cumulative profiling block every frame and the local-mapping thread one
per keyframe cycle; the histograms accumulate over the run, so the **last block is the whole-run
median per stage** (1 ms quantisation). Rows are independent medians and do not sum exactly;
`Grab Image Monocular` ≈ `Resize` + `Frame Creation` + `Tracking`; `Tracking` ⊇ `Track Ref` +
`Pose Optimization` + `Track Local Map`; `Local Mapping` ⊇ `Create NewMap Points` +
`Search in Neighbors` + `Local Bundle Adjustment` (per keyframe, not per frame, so a change in
keyframe count shifts them indirectly). The two blocks interleave in the log; parse each by its
header, never by position. Turn the flag on for speed entries and keep it identical between the
two halves of a cycle; it changes behaviour, not just logging.

**Verdict rule** (n runs, no distribution assumed): a metric is an *improvement* when the modified
median beats every baseline run, a *regression* when it is worse than every baseline run, *within
noise* otherwise. Wall clock is quantised: read it as "dropped one quantum", not as seconds.

**Guardrails** for a speed or robustness change that should not move accuracy: losses identical
(0 on `table_3`), ATE RMSE within the baseline min–max, keyframe count within the baseline spread.
End-to-end runs are not reproducible even vanilla-to-vanilla (LocalMapping timing changes keyframe
decisions), so "bit-exact" is verified at the unit level (parity harnesses such as
`bin/test_bfmatcher_parity`) and only the statistical guardrail at run level.

## Cycle

1. **Trigger.** A failure or a suspect number on a gym or benchmark run. Record the sequence,
   config, commit and the signature (log line, frame number, metric) — this becomes the entry's
   `Trigger` and often an issue.
2. **Baseline.** Branch from the integration state (`gym/<slug>` or the issue's branch). Rebuild
   unmodified (`pixi run build` here; confirm it relinked, not `ninja: no work to do`), reset swap,
   run the gym experiment. Copy the experiment folder aside or rename the experiment
   (`exp_gym_mono` → `exp_gym_mono_<slug>_before`) so the "after" run does not overwrite it.
   Baselines are re-measured per cycle, never reused across days: machine state drifts.
3. **Hypothesis.** One or two sentences on the cause, written *before* the change.
4. **Change.** One change. Commit it on the branch.
5. **Measure.** Rebuild (verify relink), reset swap, run the gym experiment, evaluate.
6. **Compare.** `python docs/tools/gym_compare.py <before> <after>`; paste the table under
   `## Effect`; apply the verdict rule and the guardrails.
7. **Decide.** Kept, reverted, or kept with a known cost. A kept change is not final until the
   benchmark check (rule 5) is recorded in the entry.
8. **Record.** One file `entries/<date>_<slug>.md` in the format below, one row update in
   `board.md`, one commit `Gym: <slug> — <verdict>`.

## Entry format

````markdown
# <date> — <slug> — <one line: what changed>

| | |
|---|---|
| Sequence | `eth` / `table_3` |
| Config | `configs/exp_gym.yaml` → `exp_gym_mono` |
| Baseline | `<commit>` (`<before experiment folder>`) |
| Change | `<commit>` on `<branch>` (`<after experiment folder>`) |
| Kind | accuracy \| robustness \| speed \| memory \| tuning |
| Verdict | kept \| reverted \| kept with a known cost (<which>) |
| Benchmark check | not yet \| <date>, <exp>, <result> |
| Issue | #NN |

## Trigger
Where it was seen and the signature (log line, frame, metric).

## Hypothesis
Why it happens, written before the change.

## Change
What was done, which files/functions, and the reference pages updated.

## Effect
<paste of gym_compare.py>

## Verdict
What was decided and why; what remains open.
````

Entries are immutable once committed except for the `Benchmark check` cell and a dated
`## Follow-up` section. A revisit is a new entry that links the old one.

## Environment

- **Swap watchdog (VSLAM-LAB#119).** The runner kills processes on *absolute* system swap; a build
  fills swap, so every build → run transition needs `pixi run kill-all` from the VSLAM-LAB root
  (kills stray SLAM processes, then `swapoff -a && swapon -a`). `free -h` to confirm.
- Close memory/CPU-heavy applications; run the two halves of a cycle back to back in one session.
- `include/Definitions.h` flags (`PROFILING_EXHAUSTIVE`, `ALLFEATURE_MAX_KEYFRAMES`, …) identical
  between the two halves.
- `verbose: 0` (no viewer; exercises the headless path).
- One experiment at a time while measuring: the numbers are the result.
- The first run after a clean environment pays one-time GPU/TensorRT engine builds; it is in both
  halves equally, but say so in the entry if run 00 looks anomalous.

## History

The protocol's first form (2026-08-16, speed-only, `exp_debug_allfeature-dev`) and its ten result
entries are the `entries/2026-08-16_*.md` files; the tracking-loss investigations of 2026-08-12/13
on NSAVP `R0_FA0` (`docs/notes/`) predate the entry format and are the model for a *robustness*
entry.
