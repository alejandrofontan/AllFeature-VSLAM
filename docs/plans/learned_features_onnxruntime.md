# Phase 4 — learned features on ONNX Runtime: SuperPoint, LightGlue, ALIKED (and SIFT on the CPU) from the git repos to C++ on Linux and macOS

- **Parent plan:** `docs/plans/mac_port.md` (phase 4). Decision 3 there: libtorch, `Light_Glue_CPP`'s CUDA kernels and
  SiftGPU leave the build once ORT parity is measured; decision 3b (ALIKED route) is re-opened below with what the
  study found (Q5). Phase 2's plan `docs/plans/megaloc_onnxruntime.md` is the template: same ten steps, same
  "the ONNX file is the contract" rule, same sidecar / reference / parity discipline.
- **Branches:** AllFeature `features-onnx` (off `main`). No placecell work. The two feature submodules
  (`Thirdparty/Light_Glue_CPP`, `Thirdparty/SuperPoint-LightGlue-TensorRT`) are removed at step 9, not forked.
- **Status:** draft 2026-10-16, nothing implemented. Questions for the user at the end.
- **Written as a walkthrough on purpose**, like phase 2: every step says what it does and why, so the path
  *git repo → ONNX → C++ → ONNX Runtime on two platforms* can be followed end to end for three models at once. What is
  new versus MegaLoc, and therefore where the learning is: (i) **variable-size outputs and inputs** — a detector
  returns N keypoints, a matcher takes N0 and N1 of them; (ii) an **operator the exporter does not know**
  (deformable convolution in ALIKED); (iii) **one runtime session class shared by several models**; (iv) models that
  are **one half of a pipeline** (extractor → matcher), so parity has to be measured at both seams.

## 0. The pieces and how they fit

| Piece | What it is | Today in AllFeature |
|---|---|---|
| **SuperPoint** (DeTone et al., 2018) | CNN detector + descriptor: a VGG-style encoder to 1/8 resolution, a 65-channel detector head (8×8 cells + "dustbin") softmaxed and pixel-shuffled back to full resolution, a 256-d descriptor map at 1/8 sampled bilinearly at the keypoints. Weights `superpoint_v1.pth` (5.2 MB) by Magic Leap, **non-commercial research licence, no redistribution** (see Decisions) | `superpoint256`: TensorRT engine from `superpoint_models/superpoint_v1_fixed.onnx` (opset 12, 87 nodes, dynamic H×W; dense `scores [1,H,W]` + `descriptors [1,256,H/8,W/8]`); threshold / border / top-k / bilinear sampling / L2 in C++ (`Thirdparty/SuperPoint-LightGlue-TensorRT/src/super_point.cpp`), **no NMS** |
| **LightGlue** (Lindenberger et al., 2023) | transformer matcher: 9 layers of self + cross attention over the two descriptor sets, rotary-style positional encoding of the normalised keypoint coordinates, a log-assignment head with a dustbin; adaptive depth (early exit) and width (point pruning) at inference. Apache-2.0 code **and** weights, one `.pth` per extractor (`superpoint_lightglue`, `aliked_lightglue`, `sift_lightglue`, …, 47 MB each, GitHub release `v0.1_arxiv`) | two copies: (a) `superpoint_models/superpoint_lightglue_fixed.onnx` (opset 17, PyTorch 2.1, 2301 nodes, 4 dynamic inputs → `scores [1,N0,N1]`) on TensorRT; mutual-NN + `exp > 0.1` in C++; **keypoints normalised with the config's 320×240 instead of the image size**; (b) `Light_Glue_CPP`'s libtorch port loading `lightglue_models/aliked_lightglue.pt` for **both** `aliked128` and `sift128` |
| **ALIKED** (Zhao et al., 2023) | CNN detector + descriptor with two **deformable-convolution** blocks (block3/4, 3×3, no modulation mask), a Differentiable Keypoint Detection head (NMS by max-pool, threshold or top-k, 5×5 soft-argmax refinement) and a Sparse Deformable Descriptor Head (learned offsets, `grid_sample`, 128-d). BSD-3-Clause; weights `aliked-n16.pth` (2.7 MB) in the upstream repo | `aliked128`: `Light_Glue_CPP` libtorch port; CUDA-only pieces: vendored `vision::ops::deform_conv2d` (`deform_conv2d_kernel.cu`) and `get_patches` (CPU twin exists, `accessor<long>` breaks on macOS); the host hands it the **BGR** `Image::img` as RGB |
| **SIFT** (Lowe, 2004) | classical DoG detector + 128-d gradient-histogram descriptor; RootSIFT = L1-normalise + sqrt. No model, no licence issue (`cv::SIFT` in OpenCV ≥ 4.4) | `sift128`: **SiftGPU** (CUDA) extractor + `SiftMatchGPU` ratio-test matcher for the descriptor-only path; the keypoint path uses the libtorch LightGlue **with ALIKED weights** and keypoints whose scale/orientation were zeroed |
| **ONNX / ONNX Runtime / CoreML** | as in phase 2 (`docs/plans/megaloc_onnxruntime.md` §0). New here: ORT runs graphs with **data-dependent shapes** (`NonZero`, `TopK` outputs) on CPU and CUDA; the CoreML EP takes dynamic shapes too (`RequireStaticInputShapes=0`) "but performance may suffer", and its MLProgram operator list has `GridSample`, `LayerNormalization`, `MatMul`, `Softmax`, `Erf`, `MaxPool` but **no `TopK`, `NonZero`, `Where`, `Gather`** — those nodes fall back to the CPU EP (partition boundaries, phase 2's 13-partition lesson). ORT 1.31 CPU **and** CUDA EPs have a `DeformConv` kernel (ONNX opset 19+), `GridSample` (16+), `NonZero`, `TopK` | `onnxruntime-cpp 1.31` already in every environment (phase 2) |
| **The upstream clone** | `cvg/LightGlue` is one repository that holds the matcher **and** PyTorch re-implementations of the three extractors (`lightglue/superpoint.py` with the Magic Leap weights, `lightglue/aliked.py`, `lightglue/sift.py` on `cv2.SIFT`) with their download URLs — so one pinned clone is the source of every export. Pins: `cvg/LightGlue` `eb42fee` (main, 2026-02-18); ALIKED weights from `Shiaoming/ALIKED` `683d7c6` (`models/aliked-n16.pth`) | nothing; today's ONNX came from the TensorRT repo's own `convert2onnx/` and the `.pt` files are libtorch state-dict pickles of unknown provenance |
| **Prior art** | `fabio-sim/LightGlue-ONNX` (Apache-2.0, `d12b4ba`, 2026-07-30): exports extractor+matcher pipelines end to end with the dynamo exporter, dynamic batch/H/W, fused attention, fp16/fp8 for TensorRT. Worth reading for the fused-pipeline and attention-fusion ideas; we do not depend on it (our contract is per-model files + sidecars, and we need SIFT and the scale/orientation inputs it does not export) | — |

Mental model to keep from phase 2: **the ONNX file is the contract**, and the Python self-check of step 3 produces the
reference the C++ side is measured against. Phase 4 adds a second rule: **the reference is upstream PyTorch, not
today's C++** — today's backends carry three bugs (320×240 normalisation, ALIKED-weights-for-SIFT, BGR into ALIKED),
so "parity with the old binary" would mean reproducing them. The old backends are measured too, for information and
for the gym baseline, but the pass criterion is against the Python reference.

## 1. Get the models out of the git repos — **new** (`tools/export_features.py`, pixi feature `export-features`)

Same shape as `placecell/tools/export_megaloc.py`'s `ensure_megaloc_clone` / `load_pretrained_megaloc`:

1. clone `cvg/LightGlue` into `tools/lightglue/` (gitignored), check out the pinned commit, `sys.path.insert`;
   `from lightglue import SuperPoint, ALIKED, SIFT, LightGlue`;
2. weights: upstream downloads them with `torch.hub.load_state_dict_from_url` into `~/.cache/torch/hub` — the script
   sets `TORCH_HOME` to `feature_models/.cache` first (the phase-2 habit: nothing lands in `~/.cache`). Sources:
   `superpoint_v1.pth` and `{superpoint,aliked,sift}_lightglue.pth` from the LightGlue release, `aliked-n16.pth` from
   the ALIKED repo (raw file at the pinned commit);
3. construct `SuperPoint(...)`, `ALIKED(model_name="aliked-n16", ...)`, `LightGlue(features=X)` with the adaptive
   behaviour off (`depth_confidence=-1`, `width_confidence=-1`, `flash=False`, `mp=False`) — the exported graph must
   be the full 9 layers with no data-dependent early exit; `eval()`.

Environment: a new `[feature.export-features]` in AllFeature's `pixi.toml` (pytorch-cpu 2.10, torchvision, kornia
— upstream `superpoint.py`/`aliked.py` import it for colour conversion —, onnx, onnxscript, onnxruntime,
onnxconverter-common, opencv, huggingface_hub, git; `PYTHONNOUSERSITE=1` like placecell's `export-megaloc`).
Linux only (the Mac runs the C++ side; the exports are files).

**Exercise:** read `lightglue/lightglue.py` (`features` table: SuperPoint 256-d, ALIKED 128-d, SIFT 128-d
**with `add_scale_ori=True`** → the SIFT matcher takes four inputs per side, not two; `normalize_keypoints`;
`filter_matches`), `lightglue/superpoint.py` (`forward` = encoder → 65-softmax → drop dustbin → pixel shuffle →
`simple_nms(radius 4)` → borders → threshold → top-k → `sample_descriptors(s=8, align_corners=True)` → L2), and
`lightglue/aliked.py` (`DeformableConv2d` → `torchvision.ops.deform_conv2d`, `DKD`, `SDDH`, `InputPadder` to
multiples of 32). Those three forwards are what steps 2–4 re-express.

## 2. Make the models exportable — **new** (`ExportWrapper`s in `tools/export_features.py`)

Dynamo exporter (`torch.onnx.export(dynamo=True)`, opset 18 for the attention graphs, **opset 19 where `DeformConv`
is used**), `optimize=True`, `external_data=False`, as decided in phase 2. One wrapper per model; what each has to
rewrite and why:

| Model | Upstream pattern | Problem | Rewrite / decision |
|---|---|---|---|
| SuperPoint | `rgb_to_grayscale` branch on channel count; list-per-image outputs after threshold/top-k | control flow on shapes; Python lists are not graph outputs | input is `[1,1,H,W]` grey in [0,1] (C++ feeds `Image::grayImg`); **graph ends at the dense maps** (Q4): `scores [1,H,W]` after softmax, pixel shuffle, **NMS** and border masking; `descriptors [1,256,H/8,W/8]` L2-normalised. Threshold, top-k and bilinear sampling stay in C++ (they exist, in `super_point.cpp`, and they are cheap); the graph gains the NMS today's path lacks. Dynamic `H`, `W` (multiples of 8; C++ crops to them, as the TRT path implicitly does) |
| LightGlue ×3 | `data` dicts, `image_size`, adaptive depth/width, `flash` attention, lists of matches | dict I/O and early exit are not exportable; pruning is data-dependent | inputs `keypoints_0/1 [1,N,2]` **already normalised** by the C++ side (centre and half-long-side of the real image — the 320×240 bug disappears by construction), `descriptors_0/1 [1,N,D]`, and for SIFT `scales_0/1 [1,N]`, `oris_0/1 [1,N]`; output `log_assignment [1,N0+1,N1+1]` (dustbin rows kept — step 4 does mutual-NN + `exp > 0.1` exactly as `filter_matches`); dynamic `N0`, `N1` (`dynamic_shapes=`). Eager attention (`flash=False`) — the plain `Softmax(QKᵀ)V` pattern every runtime fuses, phase 2's finding |
| ALIKED | `torchvision.ops.deform_conv2d` (no ONNX symbolic in torchvision, no onnxscript function); `nonzero()` threshold mode; `InputPadder` | the exporter has no translation for the op; the rest exports | `custom_translation_table={torch.ops.torchvision.deform_conv2d.default: <onnxscript fn>}` emitting **ONNX `DeformConv` (opset 19)**, which ORT's CPU and CUDA EPs implement (Q5; the GridSample+MatMul decomposition is the fallback and the CoreML lever). Full graph incl. DKD and SDDH: input `[1,3,H,W]` **RGB** in [0,1] (C++ converts BGR→RGB — the second bug disappears), H, W multiples of 32 (C++ pads like `InputPadder`, replicate), outputs `keypoints [N,2]` in pixels, `descriptors [N,128]`, `scores [N]`; `N` data-dependent (`NonZero`, threshold 0.2, `n_limit`) |
| SIFT | — | no model | `cv::SIFT` in C++ (step 4); the Python side only produces the reference (`lightglue/sift.py`, OpenCV backend, RootSIFT, `contrastThreshold 0.0066667`, `nOctaveLayers 4` — the same values `settings/sift128_settings.yaml` already carries) |

**Exercise:** export ALIKED once *without* the translation table and read the error — it names the op the exporter
cannot lower; then with it, and open the graph in Netron to find the two `DeformConv` nodes at 1/4 and 1/8
resolution. Export LightGlue once with `depth_confidence` left at 0.95 and watch the graph lose layers (or fail on the
data-dependent `break`): that is why step 1 turns the adaptive behaviour off.

## 3. Export, sidecars, and the references — **new**

Five files into `feature_models/` (one folder replaces `superpoint_models/` + `lightglue_models/`, Q3):
`superpoint_v1.onnx`, `lightglue_superpoint.onnx`, `aliked_n16.onnx`, `lightglue_aliked.onnx`, `lightglue_sift.onnx`,
each with a sidecar YAML (`source:` repo + commit + weights URL + sha256, `exporter: torch.export`, `opset`,
`precision`, input/output names and the preprocessing contract: colour order, value range, padding multiple,
keypoint normalisation, `descriptor_dim`, `add_scale_ori`). fp16 twins (`convert_float_to_float16`,
`keep_io_types=True`) for the two CNNs and the three matchers, measured in step 7 before anything ships them.

Self-check table per export (the phase-2 columns): (a) upstream PyTorch, (b) the wrapper in PyTorch, (c) ONNX on
ORT CPU fp32, (d) fp16 — for detectors: keypoint set equality (positions within 1e-3 px, same count) and descriptor
cosine at those keypoints; for matchers: identical match sets and max |Δ score|. Inputs: four `table_3` frames
(739×458 — not a multiple of 8 or 32, deliberately, so the crop/pad contract is exercised) and two frame pairs.

References for the C++ side: `feature_models/reference/` with, per frame, the Python keypoints/scores/descriptors
(`.txt`) of SuperPoint, ALIKED and SIFT, and per pair the Python matches of each LightGlue — produced by (a), the
upstream model, so step 8 measures C++ against PyTorch, never against today's binary.

## 4. The ONNX Runtime side in C++ — **new**

One session class and three thin model wrappers, all in AllFeature (Q2):

- `include/onnx/OrtSession.h` / `src/onnx/OrtSession.cpp`: what `placecell/src/megaloc/onnxruntime_megaloc.cpp`
  does minus MegaLoc — `Ort::Env` + logger bridge to `AF_INFO/AF_WARN`, `appendProviders(auto|cuda|coreml|cpu)` with
  the same CoreML options and cache directory, `checkIOTensors`, and a `run(inputs by name → outputs by name)` that
  **accepts variable shapes per call** (the MegaLoc class has one fixed input; here every call has a different N or
  H×W, so the wrapper builds the `Ort::Value`s from `(data, shape)` pairs each time and reads the output shapes back
  from `GetTensorTypeAndShapeInfo()`). Thread count and provider from settings (Q1).
- `SuperPointOnnx`: crop to multiples of 8, grey→float, run, then the C++ tail moved out of
  `super_point.cpp` and rewritten in float with the upstream semantics (threshold, borders, top-k = `maxNumFeatures`,
  bilinear sampling with `align_corners=True` and the `s=8` normalisation, L2).
- `AlikedOnnx`: BGR→RGB, replicate-pad to multiples of 32, run, un-pad the keypoint coordinates, hand `[N,2]`,
  `[N,128]`, `[N]` to the feature class.
- `LightGlueOnnx(features)`: normalise both keypoint sets with the real image size, pack `[1,N,2]`/`[1,N,D]` (+
  `scales`/`oris` for SIFT from `cv::KeyPoint::size`/`angle` in radians), run, mutual-NN + `exp > filter_threshold`
  on the `[N0+1,N1+1]` log-assignment, return `cv::DMatch` with `distance = 1 − score` (today's convention).
- `Feature_sift128` on `cv::SIFT::create(maxNumFeatures, nOctaveLayers, contrastThreshold=detectTh, edgeThreshold=10,
  sigma=1.6)` + RootSIFT, **keeping `size` and `angle` on the keypoints** (the LightGlue-SIFT inputs; the SLAM code
  that reads `GetKeypointOctave`/`GetKeypointSize` keeps returning 0/1 as today — that is a separate question for the
  gym, not this phase).

**Exercise:** write `OrtSession::run` first against `superpoint_v1.onnx` with the dense outputs dumped to disk, and
diff them against the Python `(c)` column before writing any post-processing — the same "descriptor first, SLAM
later" order phase 2 used.

## 5. Build system — **new**

- CMake: `ALLFEATURE_WITH_TORCH` and `ALLFEATURE_WITH_SIFTGPU` disappear (step 9); `ALLFEATURE_WITH_TENSORRT` keeps
  **only** `Thirdparty/Segmentation-TensorRT` until phase 5; the ORT pieces are unconditional (like MegaLoc's). The
  three features compile on every platform; a missing model file is a clear `AF_ERROR` at startup naming the pixi
  task that downloads or exports it.
- pixi: common `onnxruntime-cpp` already there; linux-64 loses `pytorch-gpu`, `siftgpu`, `glew` and the TensorRT
  packages stay for segmentation; the `cpu` feature changes nothing (it already has ORT). New
  `[feature.export-features]` (step 1) and tasks `export-features`, `download-feature-models`, `upload-feature-models`
  — the latter two per Q6 (what may be on the HF repo).
- `[package]` (conda `allfeature`): drops the libtorch/SiftGPU host deps; `LIGHTGLUE_MODELS_DIR` and friends go.

## 6. Host changes — **new**

- `Feature_superpoint256`, `Feature_aliked128`, `Feature_sift128`: own the three wrappers; no `NvInfer.h`, no
  `torch/torch.h` in any header (the `#ifdef CHECK` dance in `Feature_superpoint256.h` goes with it).
- `FeatureMatcher`: one `LightGlueOnnx` per learned matcher type actually requested (SuperPoint, ALIKED, SIFT), built
  in the constructor from the `feature_types` list; `match_descriptors` dispatches `LIGHTGLUE_*` to it,
  `match_descriptors_only` keeps `BF_L2` for the three (SIFT's descriptor-only path moves from `SiftMatchGPU`'s ratio
  test to the cross-checked brute force — a behaviour change to record in the gym).
- Settings (Q1): provider / threads / CoreML units for the feature models, documented in the dev YAML and
  `docs/reference/settings.md`; `maxNumFeatures` is the top-k of SuperPoint and ALIKED, `detectTh` their score
  threshold (SuperPoint 0.0005 and ALIKED 0.2 upstream — the per-feature YAMLs get the upstream defaults with a
  comment).
- `System.cc` startup log: one line per learned model loaded — file, provider, precision — like MegaLoc's.
- Docs: `docs/reference/FeatureMatcher.md`, `Feature_*.md`, `settings.md`, `CLAUDE.md` (build section, model
  folders, "no DBoW2" paragraph), `allfeature.md` diagram if the backend box changes.

## 7. The Apple side specifically — measured, not assumed

What phase 2 taught: Core ML (GPU) 46 ms vs CUDA 11 ms for one 322×322 ViT-B; first compile 90–175 s, cached 36 s;
ANE loses accuracy on this kind of graph. Expectations to test, in this order, on the M1 Pro:

1. **CPU EP first** for all five models (threads 4/6): CNNs at 739×458 may already be real-time-ish on the P-cores
   (SuperPoint is a small VGG at 1/8 resolution, ALIKED-n16 is smaller still); LightGlue with 1000×1000 keypoints is ~9 layers × attention — the number
   to beat is the libtorch CUDA path's ms per match call on Linux (step 8 records it).
2. **CoreML EP** per model with `ProfileComputePlan=1` once: which nodes stay on the CPU (`NonZero`/`TopK`/`Where`/
   `Gather` in ALIKED's DKD and in the matchers' index gymnastics, `DeformConv`) and how many partitions result.
   If the matcher's dynamic `N0`/`N1` makes CoreML recompile or crawl, the lever is a **static-padded export**
   (pad both sides to 1024 with the attention masks upstream's `compile` path uses) — written down, pulled only if
   measured.
3. fp16 twins where the fp32 CoreML path is accuracy-neutral (the phase-2 criterion, cosine ≥ 0.999 on the self-check
   columns; for detectors: same keypoint set).
4. Thread budget: 6 P-cores for tracking + local mapping + loop closing + viewer + ORT intra-op pools of up to four
   sessions — set `features_threads` explicitly, measure 2 vs 4.

## 7b. Speed levers, in the order to pull them (phase 6 material, written down now)

Fused extractor+matcher pipeline per pair (LightGlue-ONNX style: one session, no host round trip of descriptors);
attention fusion (`com.microsoft.MultiHeadAttention` on CUDA/CPU — not CoreML); int8 weights for the CNNs; a lower
`n_limit`/top-k for ALIKED; the static-padded LightGlue variant (§7.2); image downscale per device
(`Image::fix_image_size` exists); cache the per-frame SuperPoint/ALIKED outputs the matcher re-uses across the several
`match_descriptors` calls Tracking makes on the same frame.

## 8. Tests (what "done" means)

| | Test | Pass |
|---|---|---|
| T1 | Python self-check per export (step 3), columns (a)–(d) | detectors: same keypoint set, descriptor cosine ≥ 0.9999 fp32 / ≥ 0.999 fp16; matchers: identical match set fp32, ≥ 99 % overlap fp16 |
| T2 | `bin/test_feature_parity` (new, like `test_bfmatcher_parity`): C++ ORT vs the Python references of step 3 on the 4 frames / 2 pairs, **Linux CUDA and CPU, Mac CPU and CoreML** | keypoints within 0.5 px with the same count ±1 %, descriptor cosine as T1, match overlap ≥ 99 % (CPU/CUDA) / ≥ 97 % (CoreML, to be tightened once measured) |
| T3 | Same harness against today's backends on Linux (TensorRT SuperPoint pair, libtorch ALIKED + LightGlue, SiftGPU) | information only: quantifies the three bugs and the old NMS-less SuperPoint; recorded in the note |
| T4 | Timing per call (extract, match) per backend/provider, both machines | table in the note; Linux ORT-CUDA within 1.5× of today's TRT/libtorch, else explain |
| T5 | SLAM: `table_3` with `features: ["orb32", "aliked128"]` (the pre-Mac default) and with `["superpoint256"]` and `["sift128"]`, Linux, 3 runs each, old binary vs new | gym entries; ATE and keyframe counts within run-to-run variance of the old backend or better (the bug fixes may move them — that is a finding, not a failure) |
| T6 | Same three configurations on the Mac (CPU EP, then CoreML) | runs through; ms per stage recorded; real time or by how much it misses |
| T7 | Byte identity of the `orb32`-only deterministic configuration before and after the branch (D1/D2 references in `VSLAM-LAB-Evaluation/claude_refs/`) | identical — the learned-feature work must not touch the classical path |

## 9. After parity: libtorch, Light_Glue_CPP, SuperPoint-LightGlue-TensorRT and SiftGPU leave

`git rm` the two submodules, delete `ALLFEATURE_WITH_TORCH`/`ALLFEATURE_WITH_SIFTGPU`, the pixi packages, the
`lightglue_models/`/`superpoint_models/` download tasks, `Feature_sift128`'s GLEW include; `ALLFEATURE_WITH_TENSORRT`
is then segmentation-only and issue #41 (TensorRT 11 migration) shrinks to phase 5. HF repo
`vslamlab/allfeature-vslamlab`: `superpoint_models/` and `lightglue_models/` are retired per Q6.

## 10. The write-up (the learning deliverable)

`docs/notes/<date>_learned_features_onnx_export.md` (exporter findings per model: the deform-conv translation, the
dynamic-shape declarations, the SIFT scale/ori inputs, fp16 numbers), `docs/notes/<date>_features_onnxruntime_host.md`
(T2–T6 tables, CoreML partitions, open items), and a `docs/learned_features.md` page in the shape of placecell's
`docs/megaloc.md`: "from the git repo to C++ on two platforms", for three models.

## Order of work and commits

1. Plan + the three bug reports as GitHub issues (`bug`): SP-LightGlue 320×240 normalisation, `sift128` matched with
   ALIKED-LightGlue weights and zeroed scale/orientation, ALIKED fed BGR.
2. Step 1 + 2 + 3: `tools/export_features.py`, `export-features` env, five exports with sidecars and references;
   note on the export findings. **User runs the exports** (phase-2 convention).
3. Step 4: `OrtSession` + `SuperPointOnnx` + T2 for SuperPoint only (the static-shape warm-up).
4. `LightGlueOnnx` (SuperPoint weights) + T2 for the pair; then `Feature_sift128` on `cv::SIFT` + LightGlue-SIFT.
5. `AlikedOnnx` + LightGlue-ALIKED + T2.
6. Steps 5–6 host wiring, settings, docs; T3, T4, T5 on Linux (user runs T5).
7. Mac: build, T2/T4/T6 (user runs; `ProfileComputePlan` once).
8. Step 9 removals; step 10 write-up; `mac_port.md` status; PR into `main`, `dev` follows.

## Decisions (open — answers go here, like phase 2's)

- **Q1 — settings keys.** (a) one shared block for every ORT-hosted feature model, `features_provider`,
  `features_threads`, `features_coreml_units` next to the existing `megaloc_*` keys (recommended: the models share
  one answer per machine; MegaLoc keeps its own because its cost profile is different); (b) per-model keys; (c) fold
  the `megaloc_*` keys into one `OnnxRuntime.*` block now.
- **Q2 — where the session class lives.** (a) in AllFeature, `include/onnx/OrtSession.h`, with the provider code
  duplicated once from placecell (recommended: two consumers, two copies, unify when the segmentation port makes it
  three); (b) factor a header-only `placecell/ort_session.hpp` and have both use it (couples a keyframe library to
  feature extraction); (c) a new tiny repo/package.
- **Q3 — model folder.** (a) one `feature_models/` with sidecars (recommended); (b) keep `superpoint_models/` +
  `lightglue_models/`.
- **Q4 — SuperPoint graph boundary.** (a) dense maps out, NMS in, post-processing in C++ (recommended: CoreML-friendly
  graph, reuse of existing C++, gains the missing NMS); (b) the whole upstream forward incl. threshold/top-k/sampling
  in the graph (one less C++ tail, but `NonZero`/`TopK`/`GridSample` tail runs on the CPU EP on the Mac anyway).
- **Q5 — ALIKED deformable convolution.** (a) ONNX `DeformConv` (opset 19) via a custom translation (recommended:
  one node, native CPU and CUDA kernels in ORT 1.31; CoreML runs those two blocks on the CPU EP); (b) decomposition
  into 9 `GridSample`s + `MatMul` (exact for the unmasked case, CoreML-capable, bigger graph) — (b) stays as the phase-6
  lever if the (a) partitions hurt on the Mac. This replaces `mac_port.md` decision 3b's libtorch-CPU/MPS route: with
  ORT able to run the op natively, one runtime for everything is the lower-risk path now.
- **Q6 — SuperPoint weights and the public HF repo.** The Magic Leap licence allows non-commercial research use and
  forbids distribution; today's `superpoint_models/superpoint_v1_fixed.onnx` on the public `vslamlab/allfeature-vslamlab`
  repo is a derivative of those weights. (a) **do not upload our SuperPoint export**: `download-feature-models`
  fetches the four Apache/BSD models from HF and a `superpoint` sub-task clones + exports locally from the official
  URL on the user's machine (recommended; the export takes seconds on CPU); (b) private HF repo; (c) retire
  `superpoint256`. Either way the existing public file should come down.
- **Q7 — SIFT.** (a) `cv::SIFT` + RootSIFT, SiftGPU and its matcher leave, descriptor-only matching becomes
  cross-checked BF L2 (recommended; `mac_port.md` decision 3); (b) keep SiftGPU as an optional Linux backend.
- **Q8 — the three bugs.** (a) fixed by construction in the ORT path (the plan above), old behaviour recorded by T3
  and the gym baseline (recommended); (b) fix them in the old backends first for a cleaner A/B (more work on code
  that leaves at step 9).

## Risks

- The dynamo exporter on `NonZero`-shaped graphs (ALIKED threshold mode): data-dependent output shapes export, but
  `torch.export` may need `torch._check` hints; fallback is top-k mode (`maxNumFeatures` static) which is also the
  SLAM's actual need.
- `DeformConv` kernel maturity in ORT (a recent kernel, few users): verify numerics against torchvision in T1 (c) before trusting
  it; (b) of Q5 is the escape.
- CoreML on dynamic `N`: possible recompiles per shape; §7.2's static-padded variant is the escape.
- Behaviour drift from the bug fixes (correct normalisation, RGB, right SIFT weights, NMS) will move SLAM numbers;
  T3 and the gym separate "different because fixed" from "different because broken".
- Licence: Q6 must be answered before anything SuperPoint-derived is pushed anywhere public.
