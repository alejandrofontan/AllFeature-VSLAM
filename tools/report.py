#!/usr/bin/env python3
"""Plots of a run's report (<exp_id>_Report.csv, written by AF_REPORT in the C++ code).

    pixi run -e report report <Report.csv>                        every module, function and variable recorded
    pixi run -e report report <module> <Report.csv>               the functions and variables of one module
    pixi run -e report report <module> <function> <Report.csv>    one plot per variable, against frame_id

A module is the thread that reported (tracking, local_mapping, loop_closing); a function is the C++ function the
entry was written from. One figure opens per call, one subplot per variable, one line per feature (orb32, ...).
A function that reports several times in the same frame is drawn as one value per frame, with its single calls as
faint dots behind. How the calls of a frame combine depends on the variable: see aggregation_of().
"""

import argparse
import csv
import difflib
import math
import os
import sys
from collections import defaultdict

COLUMNS = ["thread", "frame_id", "call", "function", "key", "feature", "value"]

# Fixed order: a feature keeps its colour whatever else is in the plot. Entries with no feature are drawn in ink.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, SECONDARY_INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e0"


def read_report(path):
    """Rows of the report as dicts, with frame_id, call and value as numbers."""
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            sys.exit(f"report: {path} is not a report file: its header is {reader.fieldnames}, expected {COLUMNS}")
        rows = []
        for row in reader:
            try:
                row["frame_id"] = int(row["frame_id"])
                row["call"] = int(row["call"])
                row["value"] = float(row["value"])
            except ValueError:
                continue   # a truncated last line of a run that was killed
            rows.append(row)
    if not rows:
        sys.exit(f"report: {path} has no entries")
    return rows


def catalogue(rows):
    """{module: {function: {key: {feature: number of entries}}}}"""
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
    for row in rows:
        tree[row["thread"]][row["function"]][row["key"]][row["feature"]] += 1
    return tree


def print_catalogue(tree, path, only_module=None):
    print(f"\n{path}\n")
    for module in sorted(tree):
        if only_module is not None and module != only_module:
            continue
        print(f"{module or '(no module)'}")
        for function in sorted(tree[module]):
            keys = tree[module][function]
            entries = sum(sum(features.values()) for features in keys.values())
            print(f"  {function}   ({entries} entries)")
            for key in sorted(keys, key=key_order):
                features = keys[key]
                named = sorted(feature for feature in features if feature)
                per_feature = f"  [{', '.join(named)}]" if named else ""
                print(f"      {key:<28s}{per_feature}")
        print()
    print("Plot one function with:  pixi run -e report report <module> <function> " + path + "\n")


def key_order(key):
    """Times last, the function's own time_ms at the very end; the rest alphabetical."""
    return (key == "time_ms", key.endswith("time_ms"), key)


def aggregation_of(key):
    """How several calls of one frame become one value: (name, function).

    Counts and times add up (the work done in that frame); a minimum or maximum keeps its meaning; a value that is
    a property and not an amount (an id, a model flag, a ratio, a fitted parameter) is averaged."""
    last = key.rsplit(".", 1)[-1]
    if last.endswith("_min") or last == "min":
        return "min", min
    if last.endswith("_max") or last == "max":
        return "max", max
    if any(word in last for word in ("ratio", "sigma", "nu", "model", "frame_id", "median", "mean", "q95", "deg")):
        return "mean", lambda values: sum(values) / len(values)
    return "sum", sum


def closest(name, options):
    matches = difflib.get_close_matches(name, options, n=3, cutoff=0.4)
    return f" Did you mean: {', '.join(matches)}?" if matches else ""


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])


def format_number(value):
    if value == 0 or 0.01 <= abs(value) < 1e6:
        return f"{value:.3g}" if abs(value) < 100 else f"{value:,.0f}"
    return f"{value:.2e}"


def apply_style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": SECONDARY_INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": SECONDARY_INK, "ytick.labelcolor": SECONDARY_INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    })


def has_display():
    return not sys.platform.startswith("linux") or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def plot_function(rows, module, function, path, save=None, show=True):
    selected = [row for row in rows if row["thread"] == module and row["function"] == function]

    # key -> feature -> frame -> the values of every call in that frame
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in selected:
        data[row["key"]][row["feature"]][row["frame_id"]].append(row["value"])
    keys = sorted(data, key=key_order)

    # One colour per feature for the whole figure, in a fixed order
    features = sorted({feature for key in data for feature in data[key] if feature})
    if len(features) > len(SERIES_COLORS):
        sys.exit(f"report: {len(features)} features in one function, more than the {len(SERIES_COLORS)} colours "
                 "that can be told apart")
    color_of = {feature: SERIES_COLORS[i] for i, feature in enumerate(features)}
    color_of[""] = INK

    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    apply_style(plt)

    columns = 1 if len(keys) <= 3 else 2 if len(keys) <= 8 else 3
    rows_of_plots = math.ceil(len(keys) / columns)
    figure, axes = plt.subplots(rows_of_plots, columns, sharex=True, squeeze=False,
                                figsize=(6.4 * columns, 0.9 + 2.6 * rows_of_plots))

    for index, key in enumerate(keys):
        axis = axes[index // columns][index % columns]
        aggregation_name, aggregate = aggregation_of(key)
        several_calls = False
        summary = []
        for feature in sorted(data[key]):
            by_frame = data[key][feature]
            frames = sorted(by_frame)
            values = [aggregate(by_frame[frame]) for frame in frames]
            color = color_of[feature]
            if any(len(by_frame[frame]) > 1 for frame in frames):
                several_calls = True
                axis.scatter([frame for frame in frames for _ in by_frame[frame]],
                             [value for frame in frames for value in by_frame[frame]],
                             s=5, color=color, alpha=0.18, linewidths=0, zorder=1)
            # A handful of frames (the initialization, a loop) reads better with its points marked
            axis.plot(frames, values, color=color, linewidth=1.6, zorder=2,
                      marker="o" if len(frames) <= 40 else None, markersize=3.5)
            summary.append((f"{feature} " if feature else "") + f"median {format_number(median(values))}")
        # Variable name, and under it what the line is: the medians, and how a frame's calls were combined
        if several_calls:
            summary.append(f"{aggregation_name} per frame, dots: single calls")
        axis.set_title(key, loc="left", color=INK, fontweight="bold", pad=17)
        axis.text(0.0, 1.02, "   ·   ".join(summary), transform=axis.transAxes, ha="left", va="bottom",
                  color=SECONDARY_INK, fontsize=8).set_in_layout(False)   # a long line must not shrink the plot
        axis.margins(x=0.01)
        if key.endswith("time_ms"):
            axis.set_ylabel("ms")
            axis.set_ylim(bottom=0)

    for index in range(len(keys), rows_of_plots * columns):
        axes[index // columns][index % columns].set_visible(False)
    for column in range(columns):
        visible = [axes[row][column] for row in range(rows_of_plots) if axes[row][column].get_visible()]
        if visible:
            visible[-1].set_xlabel("frame_id")
            visible[-1].tick_params(labelbottom=True)

    frames_seen = sorted({row["frame_id"] for row in selected})
    figure.suptitle(f"{module} / {function}", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    short_path = os.sep.join(os.path.normpath(path).split(os.sep)[-3:])   # the full one does not fit a narrow figure
    figure.text(0.01, 0.985 - 0.36 / figure.get_figheight(),
                f"{len(selected)} entries over {len(frames_seen)} frames ({frames_seen[0]} to {frames_seen[-1]})   ·   {short_path}",
                ha="left", va="top", fontsize=8, color=SECONDARY_INK)
    if features:
        handles = [Line2D([0], [0], color=color_of[feature], linewidth=2, label=feature) for feature in features]
        figure.legend(handles=handles, loc="upper right", ncol=len(features), frameon=False,
                      bbox_to_anchor=(0.995, 0.995), labelcolor=SECONDARY_INK)
    figure.tight_layout(rect=(0, 0, 1, 1 - 0.6 / figure.get_figheight()))

    if save:
        figure.savefig(save, dpi=130)
        print(f"report: saved {save}")
    if show:
        try:
            figure.canvas.manager.set_window_title(f"{module} / {function}")
        except AttributeError:
            pass
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        prog="pixi run -e report report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("\n\n", 1)[1],
        epilog="Give -h together with a Report.csv to also list the modules, functions and variables it holds:\n"
               "    pixi run -e report report -h <Report.csv>",
        add_help=False)
    parser.add_argument("-h", "--help", action="store_true", help="show this help (and, with a Report.csv, its contents)")
    parser.add_argument("names", nargs="*", metavar="[module] [function] Report.csv",
                        help="the report file, optionally preceded by a module, or by a module and a function")
    parser.add_argument("--save", metavar="FILE.png", help="also write the figure to this file")
    parser.add_argument("--no-show", action="store_true", help="do not open a window (use with --save)")
    arguments = parser.parse_args()

    names = arguments.names
    path = names[-1] if names and (names[-1].endswith(".csv") or os.path.isfile(names[-1])) else None

    if arguments.help or path is None:
        parser.print_help()
        if path is None:
            if not arguments.help:
                sys.exit("\nreport: the last argument must be a Report.csv")
            return
    if not os.path.isfile(path):
        sys.exit(f"report: no such file: {path}")
    if len(names) > 3:
        sys.exit("report: expected [module] [function] Report.csv")

    rows = read_report(path)
    tree = catalogue(rows)
    selection = names[:-1]

    if selection and selection[0] not in tree:
        sys.exit(f"report: no module '{selection[0]}' in {path}. Modules: {', '.join(sorted(tree))}."
                 + closest(selection[0], list(tree)))
    if arguments.help or len(selection) < 2:
        print_catalogue(tree, path, selection[0] if selection else None)
        return

    module, function = selection
    if function not in tree[module]:
        elsewhere = [other for other in tree if function in tree[other]]
        hint = f" It is reported by: {', '.join(elsewhere)}." if elsewhere else closest(function, list(tree[module]))
        sys.exit(f"report: no function '{function}' in module '{module}'. Functions: "
                 f"{', '.join(sorted(tree[module]))}." + hint)

    show = not arguments.no_show
    if show and not has_display():
        show = False
        if not arguments.save:
            arguments.save = f"{module}_{function}.png"
        print("report: no display, so no window: the figure is saved instead")
    plot_function(rows, module, function, path, save=arguments.save, show=show)


if __name__ == "__main__":
    main()
