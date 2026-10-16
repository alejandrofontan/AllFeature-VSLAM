# Learned features on ONNX Runtime, the C++ side (2026-10-16, step 4; extended by steps 5–7)

- **Plan:** `docs/plans/learned_features_onnxruntime.md`, step 4 (T2, T4) — later steps append their sections.
- **Code:** `include/onnx/OrtSession.h` (one session class, provider auto|cuda|coreml|cpu, per-call shapes),
  `include/onnx/SuperPointOnnx.h` (dense graph + the C++ tail), `include/onnx/LightGlueOnnx.h` (one class, the
  three variants from the sidecar), `include/onnx/AlikedOnnx.h` (full graph, no tail), `include/SiftCpu.h`
  (cv::SIFT + upstream duplicate filter + RootSIFT); harness `src/test_feature_parity.cpp` (`bin/test_feature_parity`).
- **References:** the PyTorch dumps of `tools/export_features.py` on the four `table_3` frames
  (`docs/notes/2026-10-16_learned_features_onnx_export.md`). Machine: the Linux workstation (CUDA), 4 ORT threads.

## T2 — C++ on ONNX Runtime vs upstream PyTorch, `table_3` 739×458

| Model, file | Provider | Keypoints / matches vs reference | Descriptor cos | Time per call (median of 10) | Verdict |
|---|---|---|---|---|---|
| SuperPoint fp32 | CPU | 705/705, 709/709, 683/683, 628/628 at the same pixel | 1.0000000 | 97.5 ms graph, +1.5 ms tail | PASS, exact |
| SuperPoint fp32 | CUDA, TF32 on (provider default) | 99.2–100 % kept, scores ±1e-3 | 0.999996 | 8.5 ms | PASS, not exact |
| SuperPoint fp32 | **CUDA, TF32 off (kept)** | all four frames exact | 1.0000000 | 9.3 ms graph, 10.4 total | PASS, exact |
| SuperPoint fp16 | CUDA | 96.5–98.8 % kept (NMS ties) | ≥ 0.9995 | 5.4 ms | FAIL (T2 asks ≥ 99 %) |
| ALIKED fp32 | CPU | 440/440 within 1.4e-4 px | 1.0000000 | 217 ms | PASS, exact |
| ALIKED fp32 | CUDA | 440/440, 444/444, 515/515, 441/441 within 1.4e-4 px | 1.0000000 | 15.4–16.0 ms graph, 16.7–17.4 total | PASS, exact |
| ALIKED fp16 | CUDA | 438 vs 440, 99.5 % within 0.5 px, up to 0.37 px off | 0.9938 | 13.6 ms | FAIL (sub-pixel refinement in half) |
| SIFT (cv::SIFT) | CPU | 845/845, 860/860, 845/845, 846/846; scale ±5e-5 px, ori ±8e-7 rad | 1.0000000 | 21–23 ms | PASS, exact |
| LightGlue-SuperPoint fp32 | CUDA / CPU | 478/478 and 471/471 identical pairs | Δscore ≤ 1.4e-4 | 13.3–15.7 ms / 178 ms | PASS |
| LightGlue-ALIKED fp32 | CUDA / CPU | 308/308 and 314/314 | Δscore ≤ 1.2e-4 | 9.3–9.5 ms / 83 ms | PASS |
| LightGlue-SIFT fp32 | CUDA | 444/444 and 413/413 | Δscore ≤ 1.1e-4 | 17.5–18.8 ms | PASS |

Keypoint counts per frame: SuperPoint 628–709 of a 1000 budget, ALIKED 440–515 (threshold mode), SIFT 845–860.
A SuperPoint extraction + LightGlue match costs ~25 ms on CUDA, ~280 ms on 4 CPU threads; ALIKED + LightGlue ~26 ms
on CUDA, ~300 ms on the CPU; SIFT + LightGlue-SIFT ~22 + 18 ms (the extractor on the CPU either way).

## Findings

1. **TF32.** ORT's CUDA provider runs float32 convolutions in TF32 by default on Ampere+ GPUs: scores ±1e-3, ~1 % of
   SuperPoint's NMS ties flip. `use_tf32=0` (OrtSession) restores bit-identity for ~1 ms per frame. Part of what phase 2
   attributed to TensorRT's fp16 may have been this.
2. **Env before providers.** `CreateCUDAProviderOptions` needs the Ort::Env's logger; the shared Env is created first
   in the OrtSession constructor (placecell had the right order by member initialisation).
3. **The exported contracts hold in C++ exactly** (fp32): the SuperPoint tail written from `superpoint_tail`, the
   LightGlue match extraction written from `filter_matches`, the SIFT duplicate filter from `filter_dog_point` —
   each reproduces the Python side to float rounding. The Python functions in the export tool are the specification.
4. **ALIKED on CUDA: 10 Memcpy nodes.** ORT warns that part of the graph (the `NonZero` / gather tail of the DKD
   head, `Range`, shape arithmetic) runs on the CPU with device copies in between; 16 ms per frame all included. A
   step-7 lever (the static top-k export variant would remove `NonZero`).
5. **fp16 twins on CUDA** are 1.2–1.7× faster and approximate, exactly as the Python column (d) predicted: SuperPoint
   loses ~2 % of keypoints to NMS ties, ALIKED's soft-argmax moves points by up to 0.4 px. Step-7 material with the
   block-list candidates of the export note.
6. **SIFT on the CPU is 21–23 ms** at 739×458 with OpenCV 4.12's parallel implementation — the same class as AKAZE,
   not the 40–80 ms feared (Q7's SiftGPU fallback is not needed).
7. **Concurrent `Run` calls on one CUDA session return wrong results** (found 2026-10-16 evening, from "unstable
   features" in the demo with aliked128 or superpoint256 alone). With `Tracking.Sequential: 0`, tracking was lost on
   the frame after almost every keyframe insertion — 38 losses in 41 keyframes for superpoint256, 39 in 53 for
   aliked128, none for orb32 — each time with exactly `num_matches=0` to the new reference keyframe and an immediate
   MegaLoc relocalization hiding it. A diagnostic showed LightGlue itself returning **zero raw matches** between a
   330-keypoint keyframe and its 330-keypoint successor frame, while the keypoints and map points were all valid, and
   only while LocalMapping was matching that same keyframe against its five neighbours in an OpenMP loop on the same
   shared session (the Sequential-1 smoke runs never lost). ONNX Runtime documents `Session::Run` as thread-safe; a
   mutex around it in `OrtSession::run` removed every loss (aliked128: 0 losses, 65 keyframes, 352 points, median
   tracking 29 ms instead of 40 ms with the relocalizations). The GPU serialises the kernels anyway, so the lock costs
   nothing measurable. Open question for an issue: per-thread sessions or `user_compute_stream` as the way back to
   overlapping calls, and whether the Core ML provider has the same problem (the Mac session reported no losses
   with static shapes, but was not looking for them). Tracking *does* call LightGlue once per frame
   (`match_keyframe_to_frame` → `match_descriptors`), which the step 5–6 paragraph above understated.

## Steps 5–6 — the host on the new path (2026-10-16)

Wiring: `Feature_superpoint256` → `SuperPointOnnx`, `Feature_aliked128` → `AlikedOnnx` (+ the octree budget as
before), `Feature_sift128` → `SiftCpu` (scale and orientation kept on the keypoints); `FeatureMatcher` dispatches
the three `LIGHTGLUE_*` types to `onnx::shared_lightglue(variant).match(...)` with the real image size, and its
descriptor-only path sends SIFT through the cross-checked brute force (Q7); one LightGlue session per variant per
process (`onnx/FeatureModels.h`), warmed up once; settings `features_models_dir / features_provider /
features_threads / features_coreml_units`; CMake options `ALLFEATURE_WITH_TORCH` and `ALLFEATURE_WITH_SIFTGPU` gone,
`ALLFEATURE_WITH_TENSORRT` = segmentation only; pixi `download-feature-models` + `export-superpoint`
(`export-features` env now also osx-arm64), `pytorch-gpu` / `siftgpu` / `glew` out of the dependencies.

Smoke (150 `table_3` frames, `vpr: none`, `Tracking.Sequential: 1`, information culling, CUDA, each feature alone):

| features | startup | keyframes inserted / points | median tracking s |
|---|---|---|---|
| `[superpoint256]` | 2 extractors (budgets 1000 / 4000 for the init frames) + 1 shared matcher, warm-up 21 ms | 138 / 455 | 0.137 |
| `[aliked128]` | same shape, warm-up 9 ms | 112 / 327 | 0.086 |
| `[sift128]` | cv::SIFT ×2 + 1 shared matcher, warm-up 57 ms | 131 / 2191 | 0.245 |

All three run to the end and save their trajectories; the tracking times include the sequential wait for local
mapping at ~1 keyframe per frame (the shipped YAML's covisibility-kernel keyframe runaway, phase 6 material), so
they are not per-frame front-end costs: the same superpoint256 run with `Tracking.Sequential: 0` (the repo default)
has a median tracking time of 0.011 s (mean 0.015 s, 526 points) — with that smoke YAML's keyframe policy, which
inserted 3 keyframes in 150 frames. **The Linux reference for the Mac's configuration** (repo YAML, superpoint256
only, `vpr: none`, Sequential 0, after finding 7's fix): median **34 ms** on the dynamic files (70 keyframes, 647
points) and **48 ms** with `features_static_shapes: true` (the 672×416 + L512/768/1024 files; the padded matcher
costs more on CUDA, 70 keyframes, 613 points). Three `FeatureMatcher`s log the same shared
session: by design. LightGlue is called once per frame by tracking (`track_reference_keyframe` →
`match_keyframe_to_frame`) and by local mapping for the 5 covisible keyframes of each new keyframe (OpenMP loop);
the local-map and motion-model stages match by projection on the CPU.

## Step 7 — the Mac (2026-10-16, M1 Pro 16 GB, macOS 26.6, `table_3` 739×458)

What Core ML needs, learned in this order (`docs/notes/2026-10-16_learned_features_onnx_export.md` has the export
side): (1) it compiles only **bounded shapes** — the dynamic exports fall back to the CPU provider whole, so
`--static WxH` (detectors) and `--static-length L` (matchers, padded with float attention masks) exist; (2) every
operator the EP has no mapping for is a **partition** (a separate Core ML model, a CPU round trip): SuperPoint's NMS
booleans (8 partitions → moved to the C++ tail), ALIKED's DeformConv (→ GridSample taps), its DKD/SDDH heads (16
partitions, see the plan's §7b lever), the matcher's Gather / Neg / Expand (→ Split, `* -1`, Concat); (3) the ORT
Core ML model cache is keyed by graph I/O only, so a re-export with the same signature served the stale compiled
model until the cache folder was keyed by file content hash (`OrtSession`); (4) a session that cannot be created on
the provider falls back to the CPU inside `OrtSession` (the dynamic matcher's case).

| Model, static file | Provider / units | Result vs reference | Time per call |
|---|---|---|---|
| SuperPoint fp32, dynamic | CPU, 4 threads | exact | 186 ms |
| SuperPoint fp32, `_739x458` | Core ML CPUAndGPU | exact, 2 partitions | 16.4 ms graph (`time_onnx.py`: 16.0) |
| SuperPoint fp16, `_739x458` | Core ML CPUAndGPU | approximate | 15.4 ms |
| SuperPoint, `_739x458` | Core ML ALL / CPUAndNeuralEngine | ANE compile fails → CPU | — |
| ALIKED fp32, `_739x458` top-2048, DeformConv node | Core ML CPUAndGPU | exact, 20 partitions | 153 ms |
| ALIKED fp32, `_739x458` top-2048, decomposed deform convs | Core ML CPUAndGPU | exact, 16 partitions | 136 ms |
| ALIKED trunk only, `_739x458` (measurement graph) | Core ML CPUAndGPU | 2 partitions (582/584 nodes) | 53.9 ms |
| LightGlue fp32, dynamic | CPU fallback | exact | 242 ms |
| LightGlue fp32, `_L1024` padded (Linux check) | CUDA / CPU 4 threads | exact match sets, Δscore ≤ 1.4e-4 | 21 ms / see below |
| LightGlue fp32, `_L1024` padded | Core ML CPUAndGPU | exact (478/478), 4 partitions, 1632/1650 nodes | 57.9 ms graph (`time_onnx.py`: 57.2) |
| LightGlue fp16, `_L1024_fp16` | Core ML CPUAndGPU | approximate (Linux column d) | 52.6 ms: not worth the accuracy |
| LightGlue fp32, `_L768` padded | Core ML CPUAndGPU | same graph, 4 partitions | 40.7 ms (the cost is near-linear in L, not quadratic: the per-token linear layers dominate, not attention) |
| LightGlue fp32, `_L512` padded (ALIKED variant) | Core ML CPUAndGPU | same graph, 4 partitions | 26.7 ms: 512 / 768 / 1024 ≈ 27 / 41 / 57 ms, linear in L |

The padded matcher costs O(L²) whatever the pair's size: 21 ms on CUDA at L=1024 against 9–19 ms for the dynamic
file, the price of a bounded graph (on the Mac the alternative is the 242 ms CPU path). The host setting
`features_static_shapes: true` makes the extractors load `<stem>_<W>x<H>.onnx` for the frame size on the first frame
and `shared_lightglue(variant, n_max)` pick the smallest `lightglue_<f>_L<L>.onnx` with L ≥ max(N0, N1) (dynamic file
beyond, one warning); a missing static file is a hard error naming the export command. That error fired on the first
Linux smoke run: the tracker resizes every frame to a nominal area (`NOMINAL_IMAGE_AREA`, `Tracking_aux.cc`) and crops it to a multiple of 32 (`Image::fix_image_size`), so
`table_3` runs at **672×416**, not the 739×458 of the parity frames — the static detector files the host needs are
`--static 672x416`; the 739×458 ones serve `test_feature_parity` and `time_onnx.py` only. With those files the two
150-frame smoke runs (`features_static_shapes: true`, CUDA) go through: superpoint256 on `superpoint_v1_672x416.onnx`
+ `lightglue_superpoint_L1024.onnx` (439 points, median tracking 0.141 s), aliked128 on `aliked_n16_672x416.onnx` +
`lightglue_aliked_L1024.onnx` (271 points, 0.131 s); no pair outgrew L=1024, so the dynamic fallback never fired.

## Open

- Mac (`docs/notes/2026-10-16_mac_tracking_latency.md`, measured on the Mac before finding 7's fix): median tracking
  73–84 ms with superpoint256, 39 ms with orb32; the per-frame LightGlue call of `track_reference_keyframe` (a padded
  L768 call, 41 ms isolated, 53–96 ms with the mapper's five calls in flight) is the gap, SuperPoint itself stays at
  16 ms. Fixed here: every padded session is created at startup (`preload_lightglue`; the lazy L1024 creation cost a
  9 s stall on the tracking thread) and the mapper's calls are serialised per session (finding 7). Remaining levers,
  phase 6: motion-model projection tracking instead of a LightGlue match per frame; L512 / fp16 for the tracker's
  call if it stays.
- ALIKED on the Mac: the head split (plan §7b), 136 → ≈ 60 ms.
- ALIKED CPU 217 ms: profile per op; the Memcpy partition on CUDA (10 nodes, 16 ms all included).
- The two ORT "nodes not assigned to the preferred provider" warnings print per session; keep (informative) or
  demote to INFO once the host creates four sessions at startup.
