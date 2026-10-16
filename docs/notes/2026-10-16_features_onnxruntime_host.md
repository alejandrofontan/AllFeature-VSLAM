# Learned features on ONNX Runtime, the C++ side (2026-10-16, step 4; extended by steps 5–8)

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

## Open

- Step 5–6: host wiring (`Feature_*` classes, `FeatureMatcher`, settings keys, model folder tasks), then T3–T5.
- ALIKED CPU 217 ms: profile per op (step 7), the Memcpy partition on CUDA (same).
- The two ORT "nodes not assigned to the preferred provider" warnings print per session; keep (informative) or
  demote to INFO once the host creates four sessions at startup.
