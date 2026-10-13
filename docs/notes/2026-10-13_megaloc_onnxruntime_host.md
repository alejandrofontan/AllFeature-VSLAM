# MegaLoc on ONNX Runtime inside the SLAM: first table_3 runs on Linux (CUDA) and on the Mac (Core ML)

- **Date:** 2026-10-13
- **Kind:** note (single runs, not gym entries: the gym protocol wants three runs of `exp_gym_mono` with
  `--overwrite`; this records the first end-to-end results of `docs/plans/megaloc_onnxruntime.md`, tests T5–T7)
- **Code:** AllFeature `megaloc-onnx` at the step-6 commit + the `PlaceRecognitionMegaLoc.cc` CMake fix; placecell
  `onnxruntime` branch (backend numbers: placecell `docs/notes/2026-10-13_megaloc_onnxruntime_backend.md`)

## Runs

| | T5 — Linux, CUDA | T6 — Mac, Core ML | T7 — Linux, `vpr: none` |
|---|---|---|---|
| machine | workstation, RTX 4000 Ada | MacBook Pro M1 Pro, 16 GB | workstation |
| config | `configs/test_exp_megaloc_onnx.yaml` (`config_demo.yaml` = `eth/table_3`, `NumRuns: 1`, `verbose: 0`) | the Mac's `exp_demo.yaml` on `table_3`, `verbose: 1` | direct `vslamlab_allfeature_mono`, the deterministic settings of issue #16 |
| features | `orb32 + aliked128` (the shipped dev default) | `orb32` | `orb32` |
| MegaLoc | `megaloc_models/megaloc_322x322.onnx` (fp32), provider auto → `onnxruntime/cuda` | `Thirdparty/placecell/megaloc_models/megaloc_322x322_fp16.onnx`, provider auto → `onnxruntime/coreml`, `CPUAndGPU` | none |
| `add_image` median (placecell profile) | **11.9 ms** (462 keyframes) | **60.5 ms** (347 keyframes) | — |
| median tracking time | 53 ms | 37 ms | (unchanged) |
| keyframes inserted / alive at the end | 462 / 59 | 347 / 60 | 43 (information) |
| tracking losses | 0 | 0 | 0 |
| ATE RMSE (`vslamlab_evaluation/ate.csv`) | **4.39 mm** (59 keyframes, 44 evaluated) | not evaluated on the Mac | — |
| `[LoopClosing]` lines | **none** | **none** | n/a |
| result vs reference | — | — | **byte-identical** to the pre-phase-1 run A1 (trajectories, point set, stripped log) |

Board context (`docs/gym/board.md`, 2026-08-16, pre-MegaLoc, `orb32+aliked128`): ATE 4.8–6.5 mm, 66–73 keyframes,
no losses. T5 sits inside that range with the new VPR backend active.

## What it says

- **The backend is transparent to the SLAM.** Both machines run `table_3` end to end with MegaLoc embedding every
  keyframe on the mapping thread: 12 ms per keyframe on CUDA, 60 ms on Core ML (46 ms of inference, the rest
  preprocessing and the store update). Neither touches the tracking thread's budget.
- **No loop was detected on either machine, and that is the YAML, not the backend.** Linux with CUDA fp32 and
  the Mac with Core ML fp16 agree: zero `[LoopClosing]` lines. The shipped dev YAML runs `information` culling on
  the `covisibility` kernel, which inserts 350–460 keyframes and keeps ~60 (the runaway recorded in phase 1,
  `docs/notes/2026-10-12_cpu_only_build_identity.md`); the loop at the end of `table_3` is not detected under that
  policy. The backend comparison the plan wanted (ORT vs TensorRT on a closing loop) therefore needs a settings
  file with `LocalMapping.KeyframeCullingMethod: heuristic` (77 keyframes, no runaway) — a follow-up run, three
  times each, as a gym entry.
- **T7 holds:** with `vpr: none` the ONNX Runtime backend is never constructed and the output is byte-identical to
  before phase 1, so nothing in the host wiring leaks into the deterministic path.
- The Linux run loaded AllFeature's own `megaloc_models/megaloc_322x322.onnx`, which at that moment was still the
  old tracer export (`pixi run setup` had downloaded it before the re-export); same maths to 4e-7 (placecell's export
  note), so the numbers stand, but the folder should be refreshed from Hugging Face (`pixi run setup`) now that the
  new files are published.

## Follow-ups

1. Gym entry: `exp_gym_mono` three runs with `vpr: megaloc` on ONNX Runtime, heuristic culling, Linux; the same
   three runs on `ALLFEATURE_MEGALOC_BACKEND=tensorrt` for the one-time ORT-vs-TensorRT comparison (decision 3),
   then TensorRT goes.
2. Mac: evaluate the T6 trajectory (ATE) once the Mac has the evaluation environment, and re-run with heuristic
   culling for the loop.
3. The covisibility-kernel runaway is the covisibility plan's pending gym step, not this plan's.
