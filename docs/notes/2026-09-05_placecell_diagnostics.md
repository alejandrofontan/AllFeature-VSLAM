# placecell diagnostics: print manager, profiler, recorder + visualizer

- **Date:** 2026-09-05
- **Kind:** note
- **Provenance:** moved verbatim from `CLAUDE.md` § *placecell diagnostics: print manager, profiler, recorder + visualizer (2026-09-05)* on 2026-09-19 (documentation plan, `docs/plans/documentation_plan.md`). File/line references are as of the original date.


placecell (submodule, see its `CLAUDE.md`) gained three always-compiled, runtime-switched managers: a
process-wide `Logger`, and a `Profiler` and `Recorder` per store, plus the optional `placecell::viz` module
(OpenCV) that renders the store + recorder into three images. The host side wires them through the
`PlaceCell.*` settings keys (`include/PlaceCellSettings.h`, loaded in `System`'s constructor like the
`Tracking.*` / `LocalMapping.*` blocks — every key optional, compiled defaults otherwise; only meaningful with
`vpr: megaloc`):

| key | default | what it drives |
|---|---|---|
| `PlaceCell.Verbosity` | `warn` | `Logger` level via `PlaceCell::Options::verbosity` (`PLACECELL_VERBOSITY` in the environment wins, by placecell's contract; an unknown name warns and keeps the default) |
| `PlaceCell.LogElapsed` | 0 | `Logger::set_show_elapsed` |
| `PlaceCell.Profile` | 1 | `Options::profile` |
| `PlaceCell.PrintProfile` | 0 | `System::Shutdown()` prints the profile table once every placecell caller has stopped (stdout is flushed first so the stderr table lands after the AF_ lines in a redirected log) |
| `PlaceCell.Record` | 1 | `Options::record`; Dump and Visualize need it |
| `PlaceCell.Dump` | 0 | `System::SavePlaceCellDiagnostics(<exp_folder>/<exp_id>_placecell)` — called by the mono and rgbd CLIs after the trajectory/point cloud saves: `PlaceCell::dump` (kernels .npy, views.csv, recorder CSVs, profile.csv) + the three plots as PNGs at placecell's default sizes |
| `PlaceCell.Visualize` | 0 | second Pangolin window (below); Viewer checkbox "PlaceCell Window", offered only when a store exists |
| `PlaceCell.VisualizeMaxHz` | 2.0 | re-render rate (throttled on the host side, `Visualizer::Options::max_hz` is 0) |
| `PlaceCell.VisualizeCentred` | 1 | kernel heatmap of the centred kernel vs raw cosine (panels and dump) |
| `PlaceCell.VisualizeHistoryLastN` | 0 | information history window (panels and dump) |

- **Recorder feed.** `PlaceRecognition` gained two default-no-op hooks, `record_keyframe_thresholds(tau,
  min_information)` and `record_keyframe_decision(frame_id, inserted, unexplained, reason)`;
  `PlaceRecognitionMegaLoc` forwards them to the store's `Recorder` (`set_thresholds` dedupes repeats, so the
  history only grows when the Viewer slider moves; `record_decision` is a no-op while `Record` is off).
  `Tracking::need_new_keyframe` calls the first once per tracked frame and the second inside its `decide`
  lambda, so every decision (skips included, with the same reason string the AF_INFO line prints) is in
  `decisions.csv` and insertions show as triangles on the information plot. Culls need no host code:
  `PlaceCell::cull_keyframes` records itself.
- **A second Pangolin window, not OpenCV windows.** The pixi env ships conda-forge's *headless* OpenCV
  (`opencv-4.12.0-headless_*`), so `Visualizer::Options::windows` (`cv::imshow`) would throw here. The Viewer
  therefore constructs a windowless `Visualizer` and shows its images in its own Pangolin window
  (`"<main title> | placecell"`, 1500×600) driven from the same viewer thread: Pangolin keeps one GL context per
  named window, so each loop iteration renders the main window, `BindToContext`s the placecell one, uploads
  `kernel_image()` / `information_image()` / `alive_image()` (BGR) as `GL_BGR` textures into three views (kernel
  heatmap left, information history above the alive strip right, all 1:1 pixels — the kernel side is the window
  height minus margins and the 26 px title strip, the plots get the remaining width split 64/36), `FinishFrame`s
  it and binds back. The window, its views and its textures (GL objects belong to that context) are created
  lazily when the "PlaceCell Window" toggle is on and destroyed (`DestroyWindow`) when it is off, when the user
  closes the window (`ShouldQuit` on that context unticks the toggle) and at viewer exit. `update()` is called at
  most `VisualizeMaxHz`. Any exception from the visualizer disables the window with one `AF_WARN` instead of
  taking the viewer down. `PLACECELL_WITH_VIZ` is ON in `CMakeLists.txt` and the SLAM library links
  `placecell::viz`.
- Known placecell-side cost worth remembering: `Visualizer::update()`'s change check copies the whole
  `Recorder::decisions()` vector (strings) every call, and a centred-kernel render is O(n²) in stored keyframes —
  the host-side throttle exists for exactly that; lower `VisualizeMaxHz` on long sequences.
- The dev YAML turns on `PrintProfile`, `Dump` and `Visualize`; the compiled defaults leave all three off so
  other settings files are unaffected.
