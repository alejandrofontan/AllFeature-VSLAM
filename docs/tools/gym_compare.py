#!/usr/bin/env python3
"""Print the before/after table of a gym entry from two VSLAM-LAB experiment folders.

    python docs/tools/gym_compare.py <before_exp> <after_exp> [--sequence ETH/table_3] [--eval-root PATH]

`<before_exp>` / `<after_exp>` are experiment names under the evaluation root (default
`../../../VSLAM-LAB-Evaluation`, i.e. the sibling of VSLAM-LAB) or paths to experiment folders.
Each run of a sequence contributes one column; the table shows every run, the median, the delta of
the medians and a plain rank verdict (n runs, no distribution assumptions):

    improvement  the "after" median beats every "before" run
    regression   the "after" median is worse than every "before" run
    within noise otherwise

Metrics, all read from what VSLAM-LAB and the executable already write:

| metric | source |
|---|---|
| ATE RMSE (m) | `<seq>/vslamlab_evaluation/ate.csv` (`rmse`, row per `<run>_KeyFrameTrajectory.txt`) |
| evaluated / tracked frames | same file (`num_evaluated_frames`, `num_tracked_frames`, `num_frames`) |
| keyframes | rows of `<seq>/<run>_KeyFrameTrajectory.csv` |
| tracking losses | `Tracking lost` lines in `<seq>/system_output_<run>.txt` |
| emergency keyframes | `emergency keyframe` lines in the same file |
| median tracking time (ms) | `median tracking time:` line printed by the executable at the end |
| wall clock (s) | `<seq>/log_run_sequence_time.csv` |
| peak RAM / swap (GB) | `vslamlab_exp_log.csv` (`RAM`, `SWAP` columns) |
| profiling medians (ms) | last `[Tracking Profiling]` / `[Local Mapping Profiling]` block, when `PROFILING_EXHAUSTIVE` is on |

Lower is better for every metric except evaluated / tracked frames. The script prints Markdown,
ready to paste under `## Effect` of `docs/gym/entries/<date>_<slug>.md`. It does not run anything.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import re
import statistics
import sys

import os

HERE = pathlib.Path(__file__).resolve()
# docs/tools → docs → AllFeature-VSLAM-DEV → Baselines → VSLAM-LAB → its parent, where
# VSLAM-LAB-Evaluation lives by default (path_constants.py in VSLAM-LAB); VSLAMLAB_EVALUATION overrides.
DEFAULT_EVAL_ROOT = pathlib.Path(os.environ.get("VSLAMLAB_EVALUATION", HERE.parents[5] / "VSLAM-LAB-Evaluation"))

HIGHER_IS_BETTER = {"evaluated frames", "tracked frames"}
PROFILE_BLOCKS = {
    "[Tracking Profiling]": "T",
    "[Local Mapping Profiling]": "LM",
}


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def count_lines(path: pathlib.Path, pattern: str) -> int | None:
    if not path.exists():
        return None
    rx = re.compile(pattern)
    return sum(1 for line in path.open(errors="replace") if rx.search(line))


def median_tracking_time_ms(path: pathlib.Path) -> float | None:
    if not path.exists():
        return None
    val = None
    for line in path.open(errors="replace"):
        m = re.search(r"median tracking time:\s*([0-9.eE+-]+)", line)
        if m:
            val = float(m.group(1)) * 1000.0
    return val


def last_profile_block(path: pathlib.Path, header: str) -> dict[str, float]:
    """The last `header` block: `|  Stage name   12.00 ms` rows → {stage: ms}."""
    if not path.exists():
        return {}
    lines = path.read_text(errors="replace").splitlines()
    starts = [i for i, l in enumerate(lines) if header in l]
    if not starts:
        return {}
    out: dict[str, float] = {}
    for l in lines[starts[-1] + 1:]:
        m = re.match(r"\s*\|\s+(.*?)\s{2,}([0-9.]+)\s*ms", l)
        if not m:
            break
        out[m.group(1).strip()] = float(m.group(2))
    return out


def collect(exp_dir: pathlib.Path, sequence: str) -> tuple[dict[str, dict[str, float]], list[str]]:
    """{run_id: {metric: value}} for one sequence of one experiment, plus the run ids in order."""
    seq_dir = exp_dir / sequence
    if not seq_dir.is_dir():
        sys.exit(f"{seq_dir} does not exist")
    runs = sorted(p.name[: -len("_KeyFrameTrajectory.csv")] for p in seq_dir.glob("*_KeyFrameTrajectory.csv"))
    metrics: dict[str, dict[str, float]] = {r: {} for r in runs}

    for row in read_csv(seq_dir / "vslamlab_evaluation" / "ate.csv"):
        run = row.get("traj_name", "").split("_")[0]
        if run in metrics:
            for col, name in (("rmse", "ATE RMSE (m)"), ("num_evaluated_frames", "evaluated frames"),
                              ("num_tracked_frames", "tracked frames")):
                if row.get(col):
                    metrics[run][name] = float(row[col])

    for row in read_csv(seq_dir / "log_run_sequence_time.csv"):
        run = f"{int(row['experiment_id']):05d}"
        if run in metrics:
            metrics[run]["wall clock (s)"] = float(row["runtime"])

    dataset, seq_name = sequence.split("/", 1)
    for row in read_csv(exp_dir / "vslamlab_exp_log.csv"):
        if row.get("sequence_name") != seq_name or row.get("dataset_name", "").lower() != dataset.lower():
            continue
        run = f"{int(row['exp_it']):05d}"
        if run in metrics:
            if row.get("RAM"):
                metrics[run]["peak RAM (GB)"] = float(row["RAM"])
            if row.get("SWAP"):
                metrics[run]["peak swap (GB)"] = float(row["SWAP"])

    for run in runs:
        traj = seq_dir / f"{run}_KeyFrameTrajectory.csv"
        metrics[run]["keyframes"] = sum(1 for _ in traj.open()) - 1
        log = seq_dir / f"system_output_{run}.txt"
        losses = count_lines(log, r"Tracking lost")
        if losses is not None:
            metrics[run]["tracking losses"] = losses
        emerg = count_lines(log, r"emergency keyframe")
        if emerg is not None:
            metrics[run]["emergency keyframes"] = emerg
        mtt = median_tracking_time_ms(log)
        if mtt is not None:
            metrics[run]["median tracking time (ms)"] = mtt
        for header, prefix in PROFILE_BLOCKS.items():
            for stage, ms in last_profile_block(log, header).items():
                metrics[run][f"{prefix}: {stage} (ms)"] = ms
    return metrics, runs


def fmt(v: float | None) -> str:
    if v is None:
        return "—"
    if abs(v) >= 100 or float(v).is_integer():
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.4f}"


def verdict(before: list[float], after_med: float, higher_better: bool) -> str:
    if not before:
        return "—"
    better = (lambda a, b: a > b) if higher_better else (lambda a, b: a < b)
    if all(better(after_med, b) for b in before):
        return "**improvement**"
    if all(better(b, after_med) for b in before):
        return "**regression**"
    return "within noise"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("before", help="experiment name or folder (baseline)")
    ap.add_argument("after", help="experiment name or folder (modified)")
    ap.add_argument("--sequence", help="DATASET/sequence folder, e.g. ETH/table_3 (default: the only one present)")
    ap.add_argument("--eval-root", type=pathlib.Path, default=DEFAULT_EVAL_ROOT)
    args = ap.parse_args()

    def resolve(x: str) -> pathlib.Path:
        p = pathlib.Path(x)
        return p if p.is_dir() else args.eval_root / x

    before_dir, after_dir = resolve(args.before), resolve(args.after)
    for d in (before_dir, after_dir):
        if not d.is_dir():
            sys.exit(f"experiment folder not found: {d}")

    sequence = args.sequence
    if not sequence:
        seqs = sorted(str(p.relative_to(before_dir)) for p in before_dir.glob("*/*") if p.is_dir() and (p / "log_run_sequence_time.csv").exists())
        if len(seqs) != 1:
            sys.exit(f"--sequence needed; candidates in {before_dir.name}: {seqs}")
        sequence = seqs[0]

    b, b_runs = collect(before_dir, sequence)
    a, a_runs = collect(after_dir, sequence)
    names: list[str] = []
    for d in (*b.values(), *a.values()):
        for k in d:
            if k not in names:
                names.append(k)

    print(f"Sequence `{sequence}` — before: `{before_dir.name}` ({len(b_runs)} runs), after: `{after_dir.name}` ({len(a_runs)} runs)\n")
    head = ["Metric"] + [f"before {r[-2:]}" for r in b_runs] + ["before med"] + [f"after {r[-2:]}" for r in a_runs] + ["after med", "Δ", "Verdict"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for n in names:
        bv = [b[r][n] for r in b_runs if n in b[r]]
        av = [a[r][n] for r in a_runs if n in a[r]]
        bm = statistics.median(bv) if bv else None
        am = statistics.median(av) if av else None
        delta = (am - bm) if (am is not None and bm is not None) else None
        row = [n] + [fmt(b[r].get(n)) for r in b_runs] + [fmt(bm)] + [fmt(a[r].get(n)) for r in a_runs] + [fmt(am)]
        row.append(("+" if delta is not None and delta > 0 else "") + fmt(delta) if delta is not None else "—")
        row.append(verdict(bv, am, n in HIGHER_IS_BETTER) if am is not None else "—")
        print("| " + " | ".join(row) + " |")
    print("\nVerdict rule: n-run rank test — the after-median must beat (or lose to) every before run. "
          "Lower is better except evaluated/tracked frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
