# Online Static/Dynamic Segmentation — Implementation Plan

- **Author:** Alejandro Fontan Villacampa (planned with Claude, Fable 5)
- **Created:** 2026-08-17
- **Status:** Planned, not started

## Goal

Replace the offline Python masking step (`Datasets/extra-files/mask2former.py` in the parent
VSLAM-LAB repo) with an **online, in-process C++** segmentation stage: every frame gets a binary
mask (`CV_8UC1`, **1 = static, 0 = dynamic**) generated inside the executable before `Track()`,
consumed by the already-existing `Image::mask` → `FeatureExtractor::FilterKeypointsByMask` path.

No classification output is needed — only the binary mask. Model parity with the offline
Mask2Former masks is explicitly **not** a goal (the offline path stays available for A/B).

## Decisions made (2026-08-17 debrief)

| Decision | Choice | Rationale |
|---|---|---|
| Model | **EfficientViT-Seg-L1, ADE20K checkpoint** (Cityscapes-L1 as a second ONNX for driving-only) | "As broad as possible" domain coverage: ADE20K's 150 classes span indoor + outdoor and include the movable ones. Apache-2.0, official ONNX export, ~2–5 ms FP16 on the RTX 4000 Ada. B1 is the smaller fallback if L1 disappoints on speed. |
| Runtime | **TensorRT** (not ONNX Runtime) | Two worked examples already in-tree (SuperPoint-LightGlue-TensorRT; DepthAnything-TensorRT in dangling commit `a063c89`) — best classroom + best performance. ONNX is still learned either way: it's the export format both runtimes consume. |
| Scheduling | **Synchronous, every frame** | <5 ms fits the ~50 ms budget; deterministic; stream-overlap with feature extraction is a later optimization (same bucket as the DA3 CUDA-sync follow-up). |
| Parity with offline masks | Not required | Frees the model choice; offline mask2former path untouched for comparison experiments. |
| Rejected: Mask2Former in C++ | — | No clean ONNX path (MSDeformAttn + masked-attention need custom TRT plugins), and swin-large isn't real-time anyway. |
| Rejected: YOLO11/YOLOv8-seg | Escape hatch only | Broadest class list (COCO) and ~2 ms, but AGPL-3.0 vs this repo's GPLv3, instance masks need union/thresholding, and instances buy nothing here. Revisit only if ADE20K masking proves inadequate on some dataset. |
| Rejected: Python sidecar / embedded interpreter | — | Keeps the heavy model + IPC fragility; not "in cpp". |
| Complementary (separate work) | Geometric depth-consistency gating (CLAUDE.md RGB-D table row 7) | Catches *actually moving* objects class-free (semantics kills parked cars, misses moving carts). Composes with, doesn't replace, this plan. |

## Existing infrastructure this rides on

- **Consumption side is done**: `Image::LoadMask` → resize/crop in `Image.cpp` →
  `FilterKeypointsByMask` with the `mask.empty()` no-op guard. Only mask *generation* is new.
- **Runtimes already in the `allfeature-dev` pixi env** (local `pixi.toml`): `tensorrt-cxx-full`,
  `onnxruntime-cpp` (CUDA), `pytorch-gpu` — no new dependencies.
- **Pattern precedent ×2**: SuperPoint (`Thirdparty/SuperPoint-LightGlue-TensorRT`, ONNX → cached
  `.engine` on first run) and DA3 (`Thirdparty/DepthAnything-TensorRT`, optional CLI arg → falls
  back cleanly when absent). The DA3 module — including the vendored Apache-2.0
  `tensorrt_common/` + `cuda_utils/` scaffolding from ika-rwth-aachen — survives in dangling
  commit **`a063c89`** (the `depthanything` branch was deleted).
- GPU: RTX 4000 Ada, 20 GB — all candidates run in single-digit ms FP16.

## Branch

New branch `segmentation` off `aliked-only` (tip `00c8c6b`). Independent of the pending `dev` merge.

---

## Phase 0 — Model export (Python, one-off)

1. Script `Thirdparty/Segmentation-TensorRT/convert2onnx/export_efficientvit_seg.py` (mirrors
   SuperPoint-LightGlue-TensorRT's `convert2onnx/` convention):
   - pip-installs/clones `mit-han-lab/efficientvit`, pulls the ADE20K EfficientViT-Seg-L1
     checkpoint from HF (49.2 mIoU; B1 = 42.8 mIoU fallback);
   - exports through a thin `nn.Module` wrapper that **folds `argmax` into the graph** so the
     ONNX outputs an **int32 class map** (not 150 float channels — tiny GPU→CPU copy);
   - also exports the Cityscapes-L1 checkpoint as a second `.onnx`.
2. Sidecar `<name>.classes.yaml` per ONNX: input size, normalization mean/std (recorded from the
   export, never hardcoded in C++), and the dynamic class-ID list
   (ADE20K: person, car, bus, truck, van, boat, airplane, bicycle, minibike, animal, ship;
   Cityscapes: the 8 movable classes). Swapping models never touches C++.
3. Artifacts in gitignored `segmentation_models/` (same treatment as `depth_anything_models/`).
   Start with **512×512 plain resize** (EfficientViT's ADE20K training resolution; aspect
   distortion is acceptable for masks). Revisit only if boundary quality disappoints.

## Phase 1 — C++ module `Thirdparty/Segmentation-TensorRT/`

*(Revised 2026-08-17: written **from scratch** against the TensorRT 10.3 API — the earlier idea
of restoring the DA3 `tensorrt_common`/`cuda_utils` scaffolding from dangling commit `a063c89`
was dropped in favor of a minimal, self-authored module, as a TensorRT learning exercise. The
DA3 code remains available at `a063c89` as a reference to consult, not to copy.)*

4. Minimal engine lifecycle, hand-written: `ILogger` subclass → builder + ONNX parser →
   FP16 `IBuilderConfig` → `buildSerializedNetwork`, cached to `<onnx>.fp16.engine` next to
   the ONNX (rebuild when missing; first build ~a minute, logged); at startup prefer
   deserializing the cache via `IRuntime`.
5. New `TensorRTSeg` class:
   - ctor(onnx path, classes yaml, precision) → build-or-load engine as above;
   - `cv::Mat inferMask(const cv::Mat& frame)`: replicate-to-3-channels if grayscale
     (NSAVP is mono — must match cv2.imread's BGR replication in the Python self-check),
     resize + normalize per the sidecar YAML → `enqueueV3` → int32 class map → LUT to 0/1 →
     **dilate the dynamic region ~3 px at network resolution** (keypoints concentrate on
     object boundaries) → nearest-upsample to frame size.
6. Own `CMakeLists.txt`, pulled in via `add_subdirectory` from the main one — same ~14-line
   pattern as `a063c89`'s CMake diff for DA3.
7. Standalone `test_segmentation` binary (mirroring `a063c89:src/test_depth_anything.cpp`):
   image in → mask + colored overlay PNGs out. Doubles as the TensorRT learning harness — small,
   no SLAM attached.

## Phase 2 — Wiring into the entry points

8. `vslamlab_allfeature_mono.cpp` / `_rgbd.cpp` parse new optional args:
   `segmentation_onnx:<path>`, `segmentation_classes:<path>` (default `<onnx>.classes.yaml`),
   `segmentation_precision:<fp16|fp32>` (default fp16). Segmenter constructed once at startup,
   next to where SuperPoint's engine spins up.
   *(Revised 2026-08-17: `segmentation_onnx` now DEFAULTS to
   `segmentation_models/efficientvit-seg-l1-ade20k_512x512.onnx` — online segmentation is ON
   by default. `segmentation_onnx:none` disables it; a missing default model degrades to
   no-mask with a warning, while an explicitly given missing path is a hard error.)*
9. In the frame loop, exactly where `im.LoadMask(...)` runs today: if the segmenter exists →
   `im.mask = segmenter->inferMask(im.img)`; `Image`'s existing resize/crop then handles it like
   any offline mask. If **both** an offline mask column and `segmentation_onnx:` are present,
   **online wins** with a one-time `AF_WARN`.
10. Add a `seg=` entry to the existing per-frame stage-timing warning so its cost is visible
    alongside trackRef/localMap.

## Phase 3 — VSLAM-LAB wrapper (parent repo)

11. `Baselines/baseline_files/baseline_allfeature.py` dev subclass forwards `segmentation_onnx`
    (+ classes/precision) from experiment-yaml `Parameters`, with an existence check and a
    pointer to the export script when the ONNX is missing — same shape as the `depth_onnx`
    forwarding was.

## Phase 4 — Verification

12. `test_segmentation` on a handful of NSAVP + ETH `table_3` frames; side-by-side + dynamic-region
    IoU vs the offline Mask2Former masks (sanity numbers, not parity targets).
13. Timing: expect ~2–5 ms/frame synchronous; confirm the tracking budget is unaffected.
14. `exp_debug` NSAVP three-way A/B: no mask vs offline mask2former vs online — compare ATE,
    tracked inliers, loss count. (Delete the `VSLAM-LAB-Evaluation` exp folder first or the
    framework skips the rerun.)
15. Regression: run **without** the arg → behavior bit-identical to today (mask stays empty).
    Then update `CLAUDE.md` with an "Online Segmentation" section.

## Risks / knowns

- **ArgMax in TensorRT**: if the int64 ArgMax output annoys TRT, cast to int32 inside the
  exported wrapper — standard workaround, handled at export time (Phase 0).
- **ADE20K granularity**: one generic "animal" class instead of COCO's species list — acceptable
  for "broad"; YOLO-seg remains the escape hatch.
- **Non-square/native resolutions**: ADE20K checkpoints are trained at 512×512; if distorted-
  aspect masks look poor on wide driving frames, re-export at a wide resolution (the export
  script takes `--resolution H W`) and re-evaluate.
- **GPU contention** with ALIKED/SuperPoint (and DA3, if combined later): the synchronous design
  sidesteps ordering issues; stream overlap is a later optimization.
- **Licenses**: EfficientViT code + weights Apache-2.0; vendored TRT scaffolding Apache-2.0 —
  both fine inside GPLv3.

## Landing order

Phases 0–1 + the `test_segmentation` binary first (standalone-verifiable; where the TensorRT
learning happens), then Phase 2 wiring, then Phase 3 wrapper + Phase 4 A/B.

## References

- [EfficientViT (mit-han-lab)](https://github.com/mit-han-lab/efficientvit) ·
  [EfficientViT-Seg README](https://github.com/mit-han-lab/efficientvit/blob/master/applications/efficientvit_seg/README.md)
- [PIDNet](https://arxiv.org/pdf/2206.02066) · [DDRNet](https://arxiv.org/pdf/2101.06085)
  (rejected: Cityscapes-only domains)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) ·
  [YOLOs-CPP-TensorRT](https://github.com/Geekgineer/YOLOs-CPP-TensorRT) ·
  [PLY-SLAM, YOLOv8-seg dynamic SLAM](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12196564/)
  (escape hatch)
- [Mask2Former TensorRT port](https://github.com/Luckydog-lhy/Tensorrt_Mask2Former) ·
  [mmdeploy Mask2Former export issue](https://github.com/open-mmlab/mmdeploy/issues/2663)
  (why C++ parity was rejected)
