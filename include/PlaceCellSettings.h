/**
 * Module: AllFeature-VSLAM - PlaceCellSettings.h
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.0
 * - Created: 2026-09-05
 * - License: GPLv3 License
 *
 * Settings for placecell's three diagnostic managers (Thirdparty/placecell: the Logger
 * / print manager, the Profiler and the Recorder + viz visualizer), read from the
 * `PlaceCell.*` keys of the settings YAML. Same pattern as TrackingParameters /
 * LocalMappingParameters: compiled-in defaults, overridden only by keys present in the
 * file, so settings YAMLs without a PlaceCell.* block keep working unchanged.
 *
 * Only meaningful with `vpr: megaloc` (the placecell store exists only then). System
 * loads them once, turns them into placecell::PlaceCell::Options when constructing the
 * store, prints the profile table at shutdown (PrintProfile), dumps kernel + history +
 * plots next to the run's results (Dump), and the Viewer renders the three visualizer
 * images in a second Pangolin window (Visualize).
 *
 * This header deliberately includes nothing from placecell so System.h / Viewer.h stay
 * light; the conversion to placecell types happens in System.cc.
 */

#ifndef AF_VSLAM_PLACECELL_SETTINGS_H
#define AF_VSLAM_PLACECELL_SETTINGS_H

#include <string>

#include <opencv2/core/core.hpp>

namespace AF_VSLAM
{

struct PlaceCellSettings
{
    // ---- Print manager (placecell::Logger, process-wide) -------------------------------
    // Log level name: off | error | warn | info | debug | trace (or 0-5). The environment
    // variable PLACECELL_VERBOSITY wins over this key (placecell contract), so a single
    // run can be made verbose without editing the YAML.
    std::string verbosity{"warn"};
    // Prefix every placecell log line with the seconds elapsed since the logger started
    bool log_elapsed{false};

    // ---- Profiler (per store) -----------------------------------------------------------
    // Time the main entry points (add, unexplained_information, cull_keyframes + stages,
    // embedding); off reduces each measurement to one atomic load
    bool profile{true};
    // Print the count / median / p95 / max table (Logger::print, bypasses the verbosity)
    // when the system shuts down
    bool print_profile{false};

    // ---- Recorder (per store) -----------------------------------------------------------
    // Keep the query / cull / keyframe-decision history. Dump and Visualize need it.
    bool record{true};
    // At the end of the run write kernel.npy, kernel_centred.npy, views.csv, the
    // recorder CSVs, profile.csv and the three plots as PNGs into
    // <exp_folder>/<exp_id>_placecell/ (offline twin: Thirdparty/placecell/tools/plot_placecell.py)
    bool dump{false};

    // ---- Visualizer (placecell::viz, rendered into a second Pangolin window) -------------
    // Show the kernel heatmap, the unexplained-information history and the alive-
    // information strip in their own Pangolin window, driven by the Viewer thread (needs
    // the Viewer, i.e. verbose:1). Toggle "PlaceCell Window" in the Viewer menu at runtime.
    bool visualize{false};
    // Re-render the panels at most this often (0 = every viewer frame)
    double visualize_max_hz{2.0};
    // Kernel heatmap: the centred kernel (what the culler marginalises) or the raw cosine
    bool visualize_centred{true};
    // Information history: only the last N queries (0 = all)
    int visualize_history_last_n{0};

    // Read the PlaceCell.* keys present in the file over the compiled-in defaults
    static PlaceCellSettings Load(const cv::FileStorage& fSettings);
};

} // namespace AF_VSLAM

#endif // AF_VSLAM_PLACECELL_SETTINGS_H
