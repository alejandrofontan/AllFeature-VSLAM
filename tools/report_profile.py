#!/usr/bin/env python3
"""Where each thread's time goes, from a run's report (<exp_id>_Report.csv).

    pixi run -e report profile <Report.csv>             one row per thread, its stages stacked per frame
    pixi run -e report profile --depth 2 <Report.csv>   each stage split into the scopes it calls

A stage is a function with a report scope (AF_REPORT_SCOPE). A thread's outermost scope is its total for a frame
(tracking: grab_image, local mapping: process_keyframe, loop closing: run); the stages are the scopes it calls, at
the depth asked. A scope's own time (what is not inside a deeper scope) is drawn as "<function> (self)", and the
outermost scope's own time as "other". Stacks add up to the thread's total. The x axis is the frame the work
belongs to (for local mapping and loop closing, the keyframe's frame), not when it ran.
"""

import argparse
import os
import sys
from collections import defaultdict

from report import INK, MUTED, SECONDARY_INK, SERIES_COLORS, apply_style, format_number, has_display, read_report

THREAD_ORDER = ["tracking", "local_mapping", "loop_closing"]
OTHER_COLOR = "#c3c2bc"
MAX_STAGES = len(SERIES_COLORS)   # beyond that the smallest fold into "rest"


def read_scopes(rows):
    """{(thread, call): {function, frame_id, time_ms, parent}} from the scopes' time_ms and parent_call entries."""
    scopes = {}
    parents = {}
    for row in rows:
        if row["key"] == "time_ms":
            scopes[(row["thread"], row["call"])] = {"function": row["function"], "frame_id": row["frame_id"],
                                                     "time_ms": row["value"]}
        elif row["key"] == "parent_call":
            parents[(row["thread"], row["call"])] = int(row["value"])
    if not parents:
        sys.exit("profile: this report has no parent_call entries, so its scopes cannot be nested; it was written by "
                 "an older build. Rebuild and run again.")
    for (thread, call), scope in scopes.items():
        scope["parent"] = parents.get((thread, call), 0)
    return scopes


def stage_times(scopes, depth):
    """{thread: {stage: {frame_id: ms}}}, and {thread: {frame_id: total ms}}"""
    children = defaultdict(list)
    for key, scope in scopes.items():
        if scope["parent"]:
            children[(key[0], scope["parent"])].append(key)

    stages = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    totals = defaultdict(lambda: defaultdict(float))

    def visit(key, level, frame_id):
        scope = scopes[key]
        thread = key[0]
        if level == depth or (level > 0 and not children[key]):
            stages[thread][scope["function"]][frame_id] += scope["time_ms"]
            return
        inner = sum(scopes[child]["time_ms"] for child in children[key])
        own = max(scope["time_ms"] - inner, 0.0)
        stages[thread]["other" if level == 0 else f"{scope['function']} (self)"][frame_id] += own
        for child in children[key]:
            visit(child, level + 1, frame_id)

    for key, scope in scopes.items():
        if scope["parent"] == 0:
            totals[key[0]][scope["frame_id"]] += scope["time_ms"]
            visit(key, 0, scope["frame_id"])
    return stages, totals


def order_stages(stage_frames, num_frames):
    """Stages by mean ms per frame, largest first, "other" last; the smallest beyond MAX_STAGES fold into "rest"."""
    mean = {stage: sum(frames.values()) / num_frames for stage, frames in stage_frames.items()}
    named = sorted((stage for stage in mean if stage != "other"), key=lambda stage: -mean[stage])
    if len(named) > MAX_STAGES:
        rest = defaultdict(float)
        for stage in named[MAX_STAGES - 1:]:
            for frame_id, ms in stage_frames[stage].items():
                rest[frame_id] += ms
        stage_frames = {stage: stage_frames[stage] for stage in named[:MAX_STAGES - 1] + ["other"] if stage in mean}
        stage_frames["rest"] = rest
        mean["rest"] = sum(rest.values()) / num_frames
        named = named[:MAX_STAGES - 1] + ["rest"]
    return named + (["other"] if "other" in mean else []), stage_frames, mean


def plot_profile(stages, totals, path, depth, save=None, show=True):
    threads = sorted(totals, key=lambda thread: (THREAD_ORDER.index(thread) if thread in THREAD_ORDER else 99, thread))

    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style(plt)
    figure = plt.figure(figsize=(15, 0.9 + 3.1 * len(threads)))
    grid = figure.add_gridspec(len(threads), 2, width_ratios=[3, 1])
    axes = []
    for row in range(len(threads)):
        left = figure.add_subplot(grid[row, 0], sharex=axes[0][0] if axes else None)
        axes.append([left, figure.add_subplot(grid[row, 1])])

    for row, thread in enumerate(threads):
        frames = sorted(totals[thread])
        order, stage_frames, mean = order_stages(stages[thread], len(frames))
        color_of = {stage: SERIES_COLORS[i] for i, stage in enumerate(stage for stage in order if stage != "other")}
        color_of["other"] = OTHER_COLOR

        # Left: the stages stacked per frame, largest at the bottom
        axis = axes[row][0]
        series = [[stage_frames[stage].get(frame_id, 0.0) for frame_id in frames] for stage in order]
        if len(frames) > 300:
            axis.stackplot(frames, series, colors=[color_of[stage] for stage in order], linewidth=0)
        else:
            gaps = [b - a for a, b in zip(frames, frames[1:])]
            width = 0.8 * (sorted(gaps)[len(gaps) // 2] if gaps else 1)
            bottom = [0.0] * len(frames)
            for stage, values in zip(order, series):
                axis.bar(frames, values, width=width, bottom=bottom, color=color_of[stage], linewidth=0)
                bottom = [b + v for b, v in zip(bottom, values)]
        total_values = [totals[thread][frame_id] for frame_id in frames]
        ordered = sorted(total_values)
        median_total = ordered[len(ordered) // 2]
        axis.set_title(thread, loc="left", color=INK, fontweight="bold", pad=17)
        axis.text(0.0, 1.02, f"{len(frames)} frames   ·   total per frame: median {format_number(median_total)} ms, "
                             f"max {format_number(ordered[-1])} ms   ·   sum {format_number(sum(ordered) / 1000)} s",
                  transform=axis.transAxes, ha="left", va="bottom", color=SECONDARY_INK, fontsize=8)
        axis.set_ylabel("ms")
        axis.set_ylim(bottom=0)
        axis.margins(x=0.01)
        if row < len(threads) - 1:
            axis.tick_params(labelbottom=False)

        # Right, also the legend: mean ms per frame of each stage, and its share of the total
        axis = axes[row][1]
        positions = list(range(len(order)))[::-1]
        axis.barh(positions, [mean[stage] for stage in order], color=[color_of[stage] for stage in order], height=0.7)
        total_mean = sum(mean[stage] for stage in order)
        for position, stage in zip(positions, order):
            axis.text(mean[stage], position, f"  {stage}   {format_number(mean[stage])} ms · "
                                             f"{100 * mean[stage] / total_mean:.0f}%",
                      va="center", ha="left", fontsize=7.5, color=SECONDARY_INK)
        axis.set_yticks([])
        axis.set_xlim(0, 2.8 * max(mean.values()))
        axis.grid(axis="y", visible=False)
        axis.spines["left"].set_visible(False)
        axis.set_title("mean per frame", loc="left", color=MUTED, fontsize=8.5, pad=17)

    axes[-1][0].set_xlabel("frame_id")
    axes[-1][1].set_xlabel("ms")
    short_path = os.sep.join(os.path.normpath(path).split(os.sep)[-3:])
    figure.suptitle(f"Profile (depth {depth})", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    figure.text(0.01, 0.985 - 0.36 / figure.get_figheight(), short_path, ha="left", va="top", fontsize=8,
                color=SECONDARY_INK)
    figure.tight_layout(rect=(0, 0, 1, 1 - 0.6 / figure.get_figheight()))

    if save:
        figure.savefig(save, dpi=130)
        print(f"profile: saved {save}")
    if show:
        try:
            figure.canvas.manager.set_window_title("Profile")
        except AttributeError:
            pass
        plt.show()


def main():
    parser = argparse.ArgumentParser(prog="pixi run -e report profile", description=__doc__.split("\n\n", 1)[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", metavar="Report.csv")
    parser.add_argument("--depth", type=int, default=1, help="scope depth of the stages (default 1: the scopes the "
                                                             "thread's total calls directly)")
    parser.add_argument("--save", metavar="FILE.png", help="also write the figure to this file")
    parser.add_argument("--no-show", action="store_true", help="do not open a window (use with --save)")
    arguments = parser.parse_args()
    if not os.path.isfile(arguments.path):
        sys.exit(f"profile: no such file: {arguments.path}")
    if arguments.depth < 1:
        sys.exit("profile: --depth must be at least 1")

    stages, totals = stage_times(read_scopes(read_report(arguments.path)), arguments.depth)

    show = not arguments.no_show
    if show and not has_display():
        show = False
        arguments.save = arguments.save or "profile.png"
        print("profile: no display, so no window: the figure is saved instead")
    plot_profile(stages, totals, arguments.path, arguments.depth, save=arguments.save, show=show)


if __name__ == "__main__":
    main()
