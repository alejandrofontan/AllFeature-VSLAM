#!/usr/bin/env python
"""
Module: AllFeature-VSLAM - tools/feature_lab/plot_feature_benchmark.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5.2)
- Version: 0.5 (random VPR pairs: the measures against the pair's MegaLoc similarity; a figure with the times, a figure with the descriptor distances and the thresholds they imply, a figure with the geometric errors)
- Created: 2026-11-12
- License: GPLv3 License

Plot the per-feature csv files that bin/feature_benchmark writes (<results_dir>/<feature>.csv, one row per image
pair: image_0, image_1, index_0, index_1, vpr_similarity, keypoints_0, keypoints_1, matches, matched_percent,
correct_matches, correct_percent, geometry, timings).

The benchmark draws random pairs whose MegaLoc similarity is above a threshold, so the x axis is that similarity
(how alike the two images look: high = nearly the same view, low = barely overlapping), cut into --bins equal
bins; each point is the mean over the pairs in the bin, drawn at the bin's centre. A bin with no pairs leaves a
gap in the line. Four panels, one line per feature. Top row, on one shared scale so they compare directly: the
matcher's raw matches (left) and the ones that pass the geometric check (right). Bottom row: the keypoints extracted
per image (left, on its own scale) and the correct share (correct matches as a share of the matches). The grey bars along the bottom of the first panel show how many pairs each bin holds - a point on few
pairs is noisy.

A second figure, feature_benchmark_times.png, holds the time statistics: one panel per stage (keypoint extraction
per image, matching per pair, geometric check per pair), one bar per feature at the median, with a thin line out to
the 90th percentile. Extraction is counted once per image, although an image's time repeats on every row it is in.

A third figure, feature_benchmark_distances.png, holds one histogram per feature: the descriptor distance of its
correct matches (those that pass the geometric check), read from <feature>_distances.csv. Each panel has its own
x axis - the features' distances are in different metrics (Hamming for the binary descriptors, L2 for the rest) -
and the same y axis, the share of the feature's correct matches per bin, so the shapes compare.
The histograms also carry the descriptor thresholds transferred from a reference feature (--reference, orb32 by
default): for each of its three FeatureMatcher thresholds (TH_LOW, descDistTh_low_reloc, descDistTh_high_reloc, read
from settings/<reference>_settings.yaml) the share of the reference's correct matches at or below it is measured,
and every feature gets a line at the distance below which the SAME share of its own correct matches lies - the
threshold that is as permissive for that feature as the reference's is for the reference. The legend of each panel
gives that distance next to the value the feature's settings file holds now; the table is printed as well.

A fourth figure, feature_benchmark_errors.png, holds one histogram per feature of the geometric error of its
correct matches, in pixels (the error_px column of <feature>_distances.csv: the Sampson distance to the pair's
fundamental matrix, or the transfer error under its homography - an epipolar error, not a 3D reprojection error).
The correct matches are those under the check's 3 px threshold, so every histogram ends there. Each panel prints
the mean, the median and the standard deviation of the errors it shows, and the sigma of the Gaussian they would
come from: the histogram is of error MAGNITUDES, so if the signed error is a zero-mean Gaussian of standard
deviation sigma the magnitudes are half-normal, with median 0.6745 sigma, mean sqrt(2/pi) sigma = 0.798 sigma and
RMS sigma. The panel prints the three estimates (from the median, from the mean, the RMS); they agree only if the
errors really are Gaussian, and the median-based one is the one the 3 px cut and the heavy tail disturb least. The number of pairs behind a point shrinks as the gap grows (N - k pairs at the
k-th gap), so the right end of every line is noisier than the left. Each feature keeps the same colour whichever
features are plotted.

    python tools/feature_lab/plot_feature_benchmark.py                                  # reads and writes tools/feature_lab/experiments
    python tools/feature_lab/plot_feature_benchmark.py <results_dir>                    # reads another folder, writes the PNG there
    python tools/feature_lab/plot_feature_benchmark.py --output-dir <dir>               # writes <dir>/feature_benchmark.png
    python tools/feature_lab/plot_feature_benchmark.py --features orb32 aliked128
    python tools/feature_lab/plot_feature_benchmark.py --show                           # also open a window

Without arguments both the csv files and the PNG live in the `experiments` folder next to this script - the default
--output-dir of feature_benchmark when it is run from the repository root.

Needs matplotlib only. This repository's pixi environments do not carry it; VSLAM-LAB's does:
    (from the VSLAM-LAB root)  pixi run -e vslamlab python Baselines/AllFeature-VSLAM-DEV/tools/feature_lab/plot_feature_benchmark.py
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# One fixed colour per feature (colour follows the feature, never its position in the plot): a categorical palette
# checked for colour-vision-deficiency separation between neighbours, in this order.
FEATURE_COLORS = {
    "orb32": "#2a78d6",          # blue
    "sift128": "#eb6834",        # orange
    "aliked128": "#1baf7a",      # aqua
    "superpoint256": "#eda100",  # yellow
    "akaze61": "#e87ba4",        # magenta
    "brisk48": "#008300",        # green
    "kaze64": "#6250d6",         # violet
    "surf64": "#e34948",         # red
}
SURFACE, TEXT, TEXT_MUTED, GRID = "#fcfcfb", "#1a1a19", "#6b6a63", "#e6e5e0"
DEFAULT_DIR = Path(__file__).resolve().parent / "experiments"
SETTINGS_DIR = Path(__file__).resolve().parents[2] / "settings"
# The descriptor thresholds of a feature's settings yaml, with the line style each is drawn in
THRESHOLDS = (("TH_LOW", "--"), ("descDistTh_low_reloc", "-."), ("descDistTh_high_reloc", ":"))


def feature_of(label: str) -> str | None:
    """The feature a run belongs to: the label itself, or the feature name a variant's label starts with."""
    matches = [f for f in FEATURE_COLORS if label == f or label.startswith(f + "_")]
    return max(matches, key=len) if matches else None


def read_thresholds(feature: str | None) -> dict[str, float]:
    """The FeatureMatcher thresholds of settings/<feature>_settings.yaml ({} if the file or the feature is unknown)."""
    path = SETTINGS_DIR / f"{feature}_settings.yaml"
    if feature is None or not path.exists():
        return {}
    found = dict(re.findall(r"^FeatureMatcher\.(\w+):\s*([-+0-9.eE]+)", path.read_text(), flags=re.MULTILINE))
    return {key: float(found[key]) for key, _ in THRESHOLDS if key in found}
COLUMNS = ("vpr_similarity", "keypoints_0", "keypoints_1", "matches", "correct_matches",
           "correct_percent")


def load_results(path: Path, edges: list[float] | None, bins: int) -> dict:
    """One feature's csv: per similarity bin, the mean of the four plotted measures (None for an empty bin) and the
    number of pairs; plus the overall means for the legend. `edges` None: the bins span this file's similarities."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing} (written by an older feature_benchmark? run it again)")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no pairs")
    similarities = [float(r["vpr_similarity"]) for r in rows]
    if edges is None:
        low, high = min(similarities), max(similarities)
        high = high if high > low else low + 1e-6
        edges = [low + (high - low) * k / bins for k in range(bins + 1)]
    measures = ("matches", "keypoints", "correct_matches", "correct_percent")   # keypoints: the mean of the pair's two
    binned: list[list[tuple[float, ...]]] = [[] for _ in range(bins)]
    for r, similarity in zip(rows, similarities):
        k = min(int((similarity - edges[0]) / (edges[-1] - edges[0]) * bins), bins - 1)
        values = {**r, "keypoints": (int(r["keypoints_0"]) + int(r["keypoints_1"])) / 2}
        binned[max(k, 0)].append(tuple(float(values[m]) for m in measures))
    result = {m: [mean([values[k] for values in b]) if b else None for b in binned] for k, m in enumerate(measures)}
    result.update({
        "edges": edges,
        "centres": [(edges[k] + edges[k + 1]) / 2 for k in range(bins)],
        "counts": [len(b) for b in binned],
        "pairs": len(rows),
        "mean_keypoints": mean([(int(r["keypoints_0"]) + int(r["keypoints_1"])) / 2 for r in rows]),
        "mean_matches": mean([float(r["matches"]) for r in rows]),
        "mean_correct": mean([float(r["correct_matches"]) for r in rows]),
    })
    return result


def load_times(path: Path) -> dict[str, list[float]]:
    """One feature's csv as three samples in ms: extraction per image (each image once), matching and the geometric
    check per pair."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    needed = ("extract_ms_0", "extract_ms_1", "match_ms", "geometry_ms")
    if not rows or any(c not in rows[0] for c in needed):
        raise ValueError(f"{path}: no timing columns {needed} (written by an older feature_benchmark? run it again)")
    per_image: dict[str, float] = {}
    for r in rows:
        per_image[r["image_0"]] = float(r["extract_ms_0"])
        per_image[r["image_1"]] = float(r["extract_ms_1"])
    return {"extract": list(per_image.values()), "match": [float(r["match_ms"]) for r in rows],
            "geometry": [float(r["geometry_ms"]) for r in rows]}


def percentile(values: list[float], q: float) -> float:
    """The q-th percentile (0-100) by linear interpolation between the sorted samples."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def plot_times(plt, found: dict[str, Path], names: list[str], colors: dict[str, str], title: str, output: Path) -> None:
    """The time figure: one panel per stage, one horizontal bar per feature at the median, a thin line to the 90th
    percentile, the median written beside it. Each panel has its own scale: the stages differ by two orders."""
    times = {name: load_times(found[name]) for name in names}
    stages = (("extract", "Keypoint extraction (ms per image)"), ("match", "Matching (ms per pair)"),
              ("geometry", "Geometric check (ms per pair)"))
    fig, axes = plt.subplots(1, len(stages), figsize=(14, 1.6 + 0.55 * len(names)), sharey=True, facecolor=SURFACE)
    rows = list(range(len(names)))[::-1]   # the first feature on top
    for ax, (key, stage_title) in zip(axes, stages):
        ax.set_facecolor(SURFACE)
        medians = [percentile(times[name][key], 50) for name in names]
        p90s = [percentile(times[name][key], 90) for name in names]
        ax.barh(rows, medians, height=0.55, color=[colors[name] for name in names])
        longest = max(p90s)
        for row, median, p90 in zip(rows, medians, p90s):
            ax.plot([median, p90], [row, row], color=TEXT_MUTED, linewidth=1.2, solid_capstyle="butt")
            ax.text(p90 + 0.02 * longest, row, f"{median:.1f}", va="center", ha="left", fontsize=9, color=TEXT)
        ax.set_xlim(0, longest * 1.18)
        ax.set_title(stage_title, loc="left", fontsize=11, color=TEXT, pad=6)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_yticks(rows)
    axes[0].set_yticklabels(names, color=TEXT)
    fig.suptitle(title, x=0.01, y=0.98, ha="left", fontsize=13, color=TEXT)
    fig.text(0.99, 0.975, "bar and number: median   ·   thin line: out to the 90th percentile", ha="right", va="top",
             fontsize=8, color=TEXT_MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output, dpi=150, facecolor=SURFACE)


def plot_distances(plt, results_dir: Path, names: list[str], colors: dict[str, str], title: str, output: Path,
                   reference: str) -> bool:
    """The descriptor-distance figure: per feature, the histogram of the distances of its correct matches, with the
    median and the reference feature's thresholds transferred to it (same share of correct matches at or below).
    False (and no file) when a feature has no <feature>_distances.csv."""
    from matplotlib.lines import Line2D

    correct: dict[str, list[float]] = {}
    for name in names:
        path = results_dir / f"{name}_distances.csv"
        if not path.exists():
            return False
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader)
            correct[name] = sorted(float(row[0]) for row in reader if row[1] == "1")

    # The reference's thresholds as shares of its correct matches (the share at or below each), then the distance at
    # that share for every feature
    shares: dict[str, float] = {}
    reference_thresholds = read_thresholds(feature_of(reference))
    if reference in correct and correct[reference] and reference_thresholds:
        for key, value in reference_thresholds.items():
            shares[key] = 100.0 * sum(1 for d in correct[reference] if d <= value) / len(correct[reference])
    else:
        print(f"plot_feature_benchmark: no thresholds on the histograms ('{reference}' is not among the plotted runs, or "
              f"{SETTINGS_DIR}/{reference}_settings.yaml is missing)", file=sys.stderr)
    transferred = {name: {key: percentile(correct[name], share) for key, share in shares.items()} if correct[name] else {}
                   for name in names}
    if shares:
        print(f"descriptor thresholds transferred from {reference} (the distance with the same share of correct matches at or below):")
        print(f"  {'feature':26s}" + "".join(f"{key + f' ({shares[key]:.1f}%)':>36s}" for key, _ in THRESHOLDS if key in shares))
        for name in names:
            now = read_thresholds(feature_of(name))
            cells = [f"{transferred[name][key]:.4g} (now {now[key]:.4g})" if key in now else f"{transferred[name][key]:.4g}"
                     for key, _ in THRESHOLDS if key in transferred[name]]
            print(f"  {name:26s}" + "".join(f"{cell:>36s}" for cell in cells))

    columns = min(len(names), 3)
    rows = (len(names) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.9 * columns, 1.0 + 3.3 * rows), sharey=True, facecolor=SURFACE,
                             squeeze=False)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    for ax, name in zip(axes.flat, names):
        ax.set_facecolor(SURFACE)
        values = correct[name]
        ax.set_title(f"{name}   {len(values):,} correct matches", loc="left", fontsize=11, color=TEXT, pad=6)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0, labelleft=True)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        if not values:
            continue
        # 40 bins from 0 to the 99.9th percentile (the last bin takes the tail); an integer metric (Hamming) gets
        # integer-wide bins so that no bin is empty by construction
        top = percentile(values, 99.9)
        integer = all(v == int(v) for v in values[:2000])
        width = max(1, round(top / 40)) if integer else top / 40
        edges = [k * width for k in range(int(top / width) + 2)]
        clipped = [min(v, edges[-1] - width / 2) for v in values]
        weights = [100.0 / len(values)] * len(values)
        ax.hist(clipped, bins=edges, weights=weights, color=colors[name], edgecolor=SURFACE, linewidth=0.6)
        number = (lambda v: f"{v:.0f}") if integer else (lambda v: f"{v:.2f}")
        median = percentile(values, 50)
        ax.axvline(median, color=TEXT, linewidth=1.2)
        handles = [Line2D([], [], color=TEXT, linewidth=1.2)]
        labels = [f"median: {number(median)}"]
        now = read_thresholds(feature_of(name))
        for key, style in THRESHOLDS:
            if key not in transferred[name]:
                continue
            value = transferred[name][key]
            ax.axvline(value, color=TEXT, linewidth=1.2, linestyle=style)
            handles.append(Line2D([], [], color=TEXT, linewidth=1.2, linestyle=style))
            labels.append(f"{key.replace('descDistTh_', '')} ({shares[key]:.1f}%): {number(value)}"
                          + (f"  [now {number(now[key])}]" if key in now else ""))
        ax.set_xlim(0, max([edges[-1]] + list(transferred[name].values())) * 1.02)
        ax.set_xlabel("Hamming distance (bits)" if integer else "L2 distance")
        # opaque, so the threshold lines pass behind the text instead of through it
        legend = ax.legend(handles, labels, loc="upper right", frameon=True, facecolor=SURFACE, edgecolor="none",
                           framealpha=1.0, fontsize=7.5, handlelength=2.2, labelcolor=TEXT)
        legend.set_zorder(5)
    for row in axes:
        row[0].set_ylabel("share of the correct matches (%)")
    fig.suptitle(title, x=0.01, y=0.985, ha="left", fontsize=13, color=TEXT)
    if shares:
        fig.text(0.99, 0.98, f"dashed / dotted lines: {reference}'s thresholds, placed at the same share of each feature's "
                             "correct matches\n(share at or below the threshold in brackets; [now]: the feature's settings file)",
                 ha="right", va="top", fontsize=8, color=TEXT_MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=150, facecolor=SURFACE)
    return True


def plot_errors(plt, results_dir: Path, names: list[str], colors: dict[str, str], title: str, output: Path) -> bool:
    """The geometric-error figure: per feature, the histogram of the error (px) of its correct matches against the
    pair's two-view model, with the mean, median and standard deviation printed in the panel. False (and no file)
    when a feature's <feature>_distances.csv is missing or has no error_px column."""
    errors: dict[str, list[float]] = {}
    for name in names:
        path = results_dir / f"{name}_distances.csv"
        if not path.exists():
            return False
        with open(path, newline="") as f:
            reader = csv.reader(f)
            if "error_px" not in next(reader):
                return False
            errors[name] = [float(row[2]) for row in reader if row[1] == "1"]
    columns = min(len(names), 3)
    rows = (len(names) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.9 * columns, 1.0 + 3.3 * rows), sharex=True, sharey=True,
                             facecolor=SURFACE, squeeze=False)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    top = 3.0   # the geometric check's inlier threshold: no correct match lies beyond it
    edges = [top * k / 30 for k in range(31)]
    print("geometric error of the correct matches (px):")
    for ax, name in zip(axes.flat, names):
        ax.set_facecolor(SURFACE)
        values = errors[name]
        ax.set_title(f"{name}   {len(values):,} correct matches", loc="left", fontsize=11, color=TEXT, pad=6)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0, labelleft=True, labelbottom=True)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(0, top)
        ax.set_xlabel("error against the pair's two-view model (px)")
        if not values:
            continue
        ax.hist([min(v, top - 1e-6) for v in values], bins=edges, weights=[100.0 / len(values)] * len(values),
                color=colors[name], edgecolor=SURFACE, linewidth=0.6)
        average = mean(values)
        deviation = (sum((v - average) ** 2 for v in values) / len(values)) ** 0.5
        median = percentile(values, 50)
        # The Gaussian the magnitudes would come from (half-normal): three estimates of its sigma
        sigma_median = median / 0.6745
        sigma_mean = average * (3.141592653589793 / 2) ** 0.5
        sigma_rms = (sum(v * v for v in values) / len(values)) ** 0.5
        ax.text(0.97, 0.95, f"mean {average:.2f} px\nmedian {median:.2f} px\nstd {deviation:.2f} px", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, color=TEXT, linespacing=1.5)
        ax.text(0.97, 0.60, f"Gaussian sigma (px)\nfrom the median {sigma_median:.2f}\nfrom the mean {sigma_mean:.2f}\n"
                            f"RMS {sigma_rms:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=TEXT_MUTED, linespacing=1.5)
        print(f"  {name:26s} mean {average:.3f}  median {median:.3f}  std {deviation:.3f}  | Gaussian sigma: from the median "
              f"{sigma_median:.3f}, from the mean {sigma_mean:.3f}, RMS {sigma_rms:.3f}  ({len(values):,} matches)")
    for row in axes:
        row[0].set_ylabel("share of the correct matches (%)")
    fig.suptitle(title, x=0.01, y=0.985, ha="left", fontsize=13, color=TEXT)
    fig.text(0.99, 0.98, "Sampson distance to the pair's fundamental matrix (transfer error for a homography);\n"
                         "correct matches are those under the check's 3 px threshold", ha="right", va="top", fontsize=8,
             color=TEXT_MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=150, facecolor=SURFACE)
    return True


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path, nargs="?", default=DEFAULT_DIR,
                        help="the --output-dir of a feature_benchmark run (default: the experiments folder next to this script)")
    parser.add_argument("--features", nargs="+", default=None, help="plot only these features (default: every csv found)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="folder the PNG is saved in, created if needed (default: results_dir)")
    parser.add_argument("--reference", default="orb32",
                        help="the run whose descriptor thresholds are transferred to every feature in the distance "
                             "histograms (default: orb32)")
    parser.add_argument("--bins", type=int, default=10, help="similarity bins along the x axis (default: 10)")
    parser.add_argument("--show", action="store_true", help="also open an interactive window")
    args = parser.parse_args()

    try:
        import matplotlib
        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("plot_feature_benchmark: matplotlib is not installed in this environment (see the header for one that has it)",
              file=sys.stderr)
        return 1

    # The csv files to plot, in the palette's fixed order
    # (a --variant run is named by its label: it takes one of the colours no plotted feature is using, in label order)
    found = {p.stem: p for p in sorted(args.results_dir.glob("*.csv"))
             if p.stem != "vpr_megaloc_images" and not p.stem.endswith("_distances")}
    if args.features:
        unknown = [f for f in args.features if f not in found]
        if unknown:
            print(f"plot_feature_benchmark: no csv for {unknown} in {args.results_dir}", file=sys.stderr)
            return 1
    wanted = [f for f in found if not args.features or f in args.features]
    names = [f for f in FEATURE_COLORS if f in wanted] + sorted(f for f in wanted if f not in FEATURE_COLORS)
    spare = [c for f, c in FEATURE_COLORS.items() if f not in wanted]
    variants = [f for f in names if f not in FEATURE_COLORS]
    if len(variants) > len(spare):
        print(f"plot_feature_benchmark: {len(names)} runs but only {len(FEATURE_COLORS)} colours; plot fewer with --features",
              file=sys.stderr)
        return 1
    colors = {**FEATURE_COLORS, **dict(zip(variants, spare))}
    if not names:
        print(f"plot_feature_benchmark: no <feature>.csv found in {args.results_dir}", file=sys.stderr)
        return 1
    if args.bins < 1:
        print("plot_feature_benchmark: --bins must be >= 1", file=sys.stderr)
        return 1
    # Every feature holds the same pairs, so the first file's bins serve all of them
    first = load_results(found[names[0]], None, args.bins)
    results = {name: first if name == names[0] else load_results(found[name], first["edges"], args.bins) for name in names}

    panels = (("matches", "Matched keypoints"), ("correct_matches", "Correct matches (pass the geometric check)"),
              ("keypoints", "Keypoints per image (mean of the pair's two images)"),
              ("correct_percent", "Correct share (% of the matches)"))
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "text.color": TEXT, "axes.labelcolor": TEXT_MUTED,
                         "xtick.color": TEXT_MUTED, "ytick.color": TEXT_MUTED})
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, facecolor=SURFACE)
    for ax, (key, title) in zip(axes.flat, panels):
        ax.set_facecolor(SURFACE)
        for name in names:
            values = [float("nan") if v is None else v for v in results[name][key]]
            ax.plot(results[name]["centres"], values, color=colors[name], linewidth=2, marker="o", markersize=4,
                    solid_capstyle="round", label=name)
        ax.set_title(title, loc="left", fontsize=11, color=TEXT, pad=6)
        if key.endswith("_percent"):
            ax.set_ylim(0, 100)   # the share on the full scale
        else:
            ax.set_ylim(bottom=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0, labelleft=True)
        ax.margins(x=0.01)
    # The two counts on one scale, so raw and correct matches compare directly
    top = max(v for name in names for key in ("matches", "correct_matches") for v in results[name][key] if v is not None)
    for ax in axes[0]:
        ax.set_ylim(0, top * 1.05)
    for ax in axes[-1]:
        ax.set_xlabel("MegaLoc similarity of the pair (right = more alike)")

    # How many pairs stand behind each point: grey bars along the bottom of the first panel
    counts, centres, edges = first["counts"], first["centres"], first["edges"]
    width = (edges[-1] - edges[0]) / len(counts) * 0.8
    scale = 0.12 * top * 1.05 / max(counts)
    axes[0][0].bar(centres, [c * scale for c in counts], width=width, color=GRID, zorder=0)
    for centre, count in zip(centres, counts):
        if count:
            axes[0][0].text(centre, count * scale, str(count), ha="center", va="bottom", fontsize=7, color=TEXT_MUTED)
    axes[0][0].set_title("Matched keypoints   (grey bars: pairs per bin)", loc="left", fontsize=11, color=TEXT, pad=6)

    # One legend for the four panels, with each feature's overall means so the lines have numbers behind them
    handles, _ = axes[0][0].get_legend_handles_labels()
    labels = [f"{name}   {results[name]['mean_keypoints']:.0f} · {results[name]['mean_matches']:.0f} · "
              f"{results[name]['mean_correct']:.0f}" for name in names]
    legend = fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.975), frameon=False,
                        ncol=min(len(names), 4), fontsize=9, handlelength=1.6, columnspacing=1.6,
                        title="feature   mean keypoints per image · mean matches · mean correct matches (all pairs)",
                        title_fontsize=8)
    legend.get_title().set_color(TEXT_MUTED)
    pairs = max(results[name]["pairs"] for name in names)
    fig.suptitle(f"Feature benchmark - {args.results_dir.name} ({pairs} image pairs)", x=0.01, y=0.985, ha="left",
                 fontsize=13, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.92 if len(names) <= 4 else 0.9))

    output_dir = args.output_dir or args.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "feature_benchmark.png"
    fig.savefig(output, dpi=150, facecolor=SURFACE)
    times_output = output_dir / "feature_benchmark_times.png"
    plot_times(plt, found, names, colors, f"Feature benchmark times - {args.results_dir.name} ({pairs} image pairs)",
               times_output)
    distances_output = output_dir / "feature_benchmark_distances.png"
    has_distances = plot_distances(plt, args.results_dir, names, colors,
                                   f"Descriptor distance of the correct matches - {args.results_dir.name} ({pairs} image pairs)",
                                   distances_output, args.reference)
    errors_output = output_dir / "feature_benchmark_errors.png"
    has_errors = has_distances and plot_errors(
        plt, args.results_dir, names, colors,
        f"Geometric error of the correct matches - {args.results_dir.name} ({pairs} image pairs)", errors_output)
    if has_distances and not has_errors:
        print("plot_feature_benchmark: no errors figure (no error_px column: written by an older feature_benchmark)",
              file=sys.stderr)
    print(f"plot_feature_benchmark: {len(names)} features, {pairs} pairs -> {output}, {times_output.name}"
          + (f", {errors_output.name}" if has_errors else "")
          + (f" and {distances_output.name}" if has_distances else
             " (no distances figure: no <feature>_distances.csv, written by a newer feature_benchmark)"))
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
