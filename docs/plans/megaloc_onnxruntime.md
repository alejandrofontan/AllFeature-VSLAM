# Phase 2 — MegaLoc on ONNX Runtime: from the git repo to a C++ embedder on Linux and macOS

- **Parent plan:** `docs/plans/mac_port.md` (phase 2). Decisions already taken there: 3 (a) TensorRT leaves once
  ORT parity is measured; 4 (a) measure the 322×322 fp32 export first; 5 (a) ORT CUDA on Linux, CPU + CoreML on the Mac.
- **Branches:** AllFeature `megaloc-onnx` (off `main`), placecell `onnxruntime` (off `main`), pointer bump at the end.
- **Status:** draft 2026-10-13, nothing implemented. Questions for the user at the end.
- **Written as a walkthrough on purpose:** every step says what it does and why, including the steps that already
  exist (marked **exists**), so the whole path *git repo → ONNX → C++ library → ONNX Runtime on two platforms* can be
  followed, re-run and understood end to end. The steps marked **new** are the phase-2 work. The learning write-up
  itself goes into placecell's `docs/megaloc.md`, today a placeholder, as each step lands (step 10).

## 0. The pieces and how they fit

| Piece | What it is | Where |
|---|---|---|
| **MegaLoc** (Berton & Masone, 2025) | a visual place recognition model: DINOv2 ViT-B/14 backbone + an optimal-transport feature aggregator → one 8448-d L2-normalised descriptor per image, so cosine similarity = dot product. MIT, weights on Hugging Face (`gberton/MegaLoc`), code on GitHub (`gmberton/MegaLoc`), published usage is `torch.hub.load("gmberton/MegaLoc", "get_trained_model")` | `Thirdparty/placecell/tools/megaloc/` (pinned clone, gitignored) |
| **ONNX** | a file format for a computation graph: nodes are standard operators (MatMul, LayerNormalization, Softmax, Resize…) of a given *opset* version, edges are tensors, weights are embedded constants. Produced here by tracing the PyTorch model with a dummy input (`torch.onnx.export`). Static shapes: our file is `[1,3,322,322] float32 → [1,8448] float32` and valid only at that size | `megaloc_models/megaloc_322x322.onnx` (912 MB fp32) + sidecar `.yaml` |
| **ONNX Runtime (ORT)** | a library that loads an ONNX file and runs it. At session creation it partitions the graph among **execution providers (EPs)**: each EP claims the nodes it supports (CUDA EP on NVIDIA, CoreML EP on Apple, …) and the CPU EP takes the rest. One C++ API on every platform; the EP is a runtime choice | conda-forge `onnxruntime-cpp` 1.31: linux-64 `cuda134`/`cuda129` builds (CUDA EP), osx-arm64 `cpu` build **with the CoreML EP compiled in** |
| **TensorRT** (today's backend) | NVIDIA's ahead-of-time compiler: parses the ONNX, benchmarks kernels for *this* GPU for 1–2 min, serialises a machine-specific `.engine`. Fastest on NVIDIA, NVIDIA-only, and its 10.3 fuser miscompiles one attention pattern (see step 2) | `Thirdparty/placecell/src/megaloc/tensorrt_megaloc.{hpp,cpp}` |
| **CoreML** (Apple's equivalent) | Apple's ML runtime that dispatches a model to CPU, GPU and the Neural Engine. Reached from C++ *through ORT's CoreML EP*: ORT converts the subgraphs the EP supports into a CoreML model at session creation (`ModelFormat=MLProgram`) and caches it (`ModelCacheDirectory`). What CoreML cannot run stays on ORT's CPU EP | nothing yet |
| **placecell** | our keyframe-lifecycle library; its `MegaLocEmbedder` (public, backend-free pimpl) wraps the backend and `MegaLocPlaceCell` stores the descriptors and answers the VPR / information queries AllFeature uses | `Thirdparty/placecell` |
| **AllFeature host** | `System` builds the embedder from `megaloc_onnx` / `megaloc_precision`, `PlaceRecognitionMegaLoc` queries it for relocalisation and loops, LocalMapping/Tracking for the information kernel | `src/System.cc`, `include/PlaceRecognitionMegaLoc.h` |

Mental model to keep: **the ONNX file is the contract.** Everything upstream of it (steps 1–3) is Python and runs
once per export, on Linux. Everything downstream (steps 4–9) is C++ and must reproduce, for the same input, the same
descriptor the Python side produced; the `_reference.txt` of step 3 is what "the same" is measured against.

## 1. Get the model out of the git repo — **exists** (`tools/export_megaloc.py`, functions `ensure_megaloc_clone`, `load_pretrained_megaloc`)

What `torch.hub.load` does under the hood, done explicitly so it is reproducible and leaves nothing in `~/.cache`:

1. clone `gmberton/MegaLoc` next to the script and check out the pinned commit (`MEGALOC_COMMIT`, main of 2026-08-28);
2. `sys.path.insert(0, clone)` and `from megaloc_model import MegaLoc` — the upstream `hubconf.py`'s `get_trained_model`
   does the same import, then downloads weights;
3. download `model.safetensors` from HF `gberton/MegaLoc` into `megaloc_models/.cache` (`HF_HOME` set before importing
   `huggingface_hub`), `load_state_dict`, `eval()`.

Why pin a commit: an ONNX export is tied to the exact module code; a later upstream refactor would silently change
what the export means. The commit is written into the sidecar YAML (`source:`), so a model file always says where it
came from.

**Exercise (Linux, `pixi run -e export-megaloc`):** read `megaloc_model.py` in the clone — `MegaLoc.forward` is ~20
lines: backbone → `(cls_token, patch_tokens)` → aggregator → `F.normalize`. The export wrapper of step 2 re-expresses
exactly that forward.

## 2. Make the model exportable — **exists** (`ExportWrapper` in `tools/export_megaloc.py`)

`torch.onnx.export` *traces*: it runs the module once on a dummy tensor and records every tensor operation into the
graph. Three properties of upstream code fight tracing or the downstream runtime, and the wrapper rewrites them
without changing the maths (checked numerically in step 3 at max |diff| ~1e-6):

| Upstream pattern | Problem | Rewrite |
|---|---|---|
| `if H % 14: resize` and `interpolate_pos_encoding` (bicubic resize of the 37×37 positional grid to the input grid) at every forward | data-dependent control flow is not traceable; a `Resize` op with cubic antialias would have to be bit-reproduced by every runtime | the export is locked to one resolution (322 = 23 × 14); the interpolated positional embedding is computed **once** in `__init__` and stored as a constant buffer |
| Sinkhorn solver writing slices into `torch.empty()` in place, plus `.squeeze()` | in-place slice writes trace into scatter ops; `squeeze` is batch-ambiguous | the same three iterations written with `torch.cat` / `logsumexp` on explicit shapes |
| attention `qkv.reshape(B,N,3,heads,hd).permute(2,0,3,1,4)[0..2]` | TensorRT 10.3's Myelin fuser **miscompiles** this (cos 0.75 vs ORT; found by bisecting the graph with `trtexec --loadInputs/--exportOutput`) | q, k, v as three slices of the `[B,N,3C]` projection — same numbers, friendly to every runtime |

Everything else is called unchanged (`patch_embed`, each block's `norm1/attn/ls1/mlp`, `cluster_features`, `score`,
`token_features`, `linear`). Normalisation stays **outside** the graph (input is already `(x/255 - mean)/std`), which
is why the C++ side must do it and why the sidecar carries `mean`/`std`.

Export settings: `opset_version=17` (has `LayerNormalization`, which keeps the ViT norms as single nodes instead of
a dozen primitives), `do_constant_folding=True` (the positional buffer and the Sinkhorn constants become initialisers),
`dynamo=False` (the classic tracer; the newer dynamo exporter is not needed for a static graph and was less stable in
torch 2.7). After export, `onnx.checker.check_model` validates the file and the input/output shapes are printed.

## 3. Export, sidecar, and the reference the C++ side is measured against — **exists** (`export`, `write_sidecar_yaml`, `self_check`)

```bash
cd Thirdparty/placecell
pixi run -e export-megaloc export-models -- --test_images <img1> <img2> <img3>   # (Linux, GPU env)
```

- writes `megaloc_models/megaloc_322x322.onnx` (912 MB: ViT-B weights in fp32) and `megaloc_322x322.onnx.yaml`
  — the sidecar is the **only** place input size, normalisation, tensor names and descriptor dim live; the C++
  backends read it with yaml-cpp and cross-check it against the loaded model's tensor shapes;
- `--test_images` runs three descriptor sources per image and prints their cosines: (a) the original torch model with
  the README transform (**the reference**), (b) the wrapper with the OpenCV preprocessing the C++ side uses (area
  resize instead of torchvision's antialiased bilinear; measures the preprocessing delta), (c) ONNX Runtime in Python
  on the exported file. (a) is saved to `megaloc_322x322_reference.txt`, one line per image.

Pass criterion carried into every later step, restated after the 2026-10-13 measurement (placecell
`docs/notes/2026-10-13_megaloc_onnx_export.md`): every ONNX path agrees with the cv2-preprocessed wrapper to 1e-6,
and all of them sit at **0.9953–0.9958 against the README reference** — that floor is the preprocessing (OpenCV area
resize vs torchvision's antialiased bilinear, PIL vs OpenCV decoding), identical for every backend and irrelevant to
retrieval consistency inside the SLAM. So: **cos(backend, reference) ≥ 0.995** and **cos(backend, ORT-CPU fp32) ≥
0.9999** for fp32 providers, **≥ 0.999** for fp16/CoreML. The fp16 twin measured ≤ 6e-5 below fp32. Done on
2026-10-13 with `--exporter both --raw`: dynamo 478 nodes / 23 op types vs tracer 807 / 29 (same maths to 4e-7); the
untouched upstream module also exports under `torch.export` but with a runtime `Resize`, two scatter ops and 38
`Gather`s, so the wrapper's rewrites stay for the graph's sake. The published files live in HF
`vslamlab/megaloc-models` (`pixi run setup` downloads them in AllFeature).

**Exercise:** run the export with 3–4 `table_3` frames as test images and read the printed table; then open the ONNX
with `python -c "import onnx; m=onnx.load('…'); print(len(m.graph.node)); print({n.op_type for n in m.graph.node})"`
to see the ~30 operator types the runtimes must support.

## 4. The ONNX Runtime backend in C++ — **new** (`src/megaloc/onnxruntime_megaloc.{hpp,cpp}`)

Same method surface as `TensorRTMegaLoc`, so `MegaLocEmbedder::Impl` can hold either: constructor
`(onnx_path, precision, sidecar_path)`, `infer(cv::Mat) → std::vector<float>`, `descriptorDim/inputWidth/inputHeight`,
`enginePath()` (returns the ONNX path), `loadedFromCache()` (true), `describeIOTensors()`. Internally:

1. `Ort::Env` (one per process, holds the logger: ORT messages → placecell `Logger` component `ORT`, same mapping as
   the TensorRT logger).
2. `Ort::SessionOptions`: `SetGraphOptimizationLevel(ORT_ENABLE_ALL)`, `SetIntraOpNumThreads(n)` with `n` from a
   setting (default 4, **never** the core count: the tracking and mapping threads need the rest; on the M1 Pro this
   also keeps ORT on the performance cores), `SetInterOpNumThreads(1)`, `SetExecutionMode(ORT_SEQUENTIAL)`.
3. Execution providers, appended in priority order from a `provider` string (`auto|cuda|coreml|cpu`):
   `auto` = CUDA where compiled in, else CoreML where compiled in, else CPU; the CPU EP is always last and always
   present. CUDA: `OrtCUDAProviderOptionsV2` (`device_id=0`, `cudnn_conv_use_max_workspace=1`). CoreML:
   `so.AppendExecutionProvider("CoreML", {ModelFormat=MLProgram, MLComputeUnits=ALL, RequireStaticInputShapes=1,
   ModelCacheDirectory=<next to the onnx>})` — the static-shapes flag is true for us and lets CoreML compile more; the
   cache directory is ORT's equivalent of the `.engine` file (compilation of a 300-node graph takes seconds to a
   minute on first use).
4. `Ort::Session(env, path, so)`; read the input/output names and shapes with `GetInputNameAllocated` /
   `GetInputTypeInfo`, cross-check them against the sidecar exactly as `setupInference()` does.
5. `infer()`: the preprocessing block is moved **verbatim** out of `tensorrt_megaloc.cpp` into a shared
   `megaloc_preprocess.{hpp,cpp}` (BGR → area resize → RGB → /255 → (x-mean)/std → CHW float planar) so both backends
   and the test program use one copy; then `Ort::Value::CreateTensor<float>(memory_info_cpu, host_input, shape)`,
   `session.Run(Ort::RunOptions{}, &in_name, &in_tensor, 1, &out_name, 1)`, copy the `[1,D]` output, host-side
   re-normalisation (keep: CoreML fp16 paths leave |d| ≠ 1 just like TensorRT fp16). The input tensor wraps our own
   buffer, so there is no per-call allocation; ORT copies it to the device inside `Run` (an `IoBinding` with a
   pinned/device buffer is the later optimisation if profiling shows the H2D copy matters — it will not at 1.2 MB).
6. `precision` becomes a **provider hint**: CPU EP ignores it; CUDA EP: fp32 only (ORT's CUDA EP does not do mixed
   precision from a flag; an fp16 *model* would be a separate export — decision 4 says measure fp32 first); CoreML:
   the Neural Engine computes fp16 anyway, `AllowLowPrecisionAccumulationOnGPU=0` keeps accumulation fp32.

Thread-safety stays in `MegaLocEmbedder` (one mutex around `infer`); an `Ort::Session` is itself thread-safe for
concurrent `Run` but one at a time is what the host needs.

## 5. Build system — **new**

- placecell `CMakeLists.txt`: `PLACECELL_MEGALOC_BACKEND` = `onnxruntime` (default) | `tensorrt`. `onnxruntime`:
  `find_package(onnxruntime CONFIG)` (conda-forge's `onnxruntime-cpp` ships `onnxruntimeConfig.cmake` and the
  `onnxruntime::onnxruntime` target; a `find_path/find_library` fallback under `$CONDA_PREFIX` like today's TensorRT
  lookup if the config is missing), link it PUBLIC into `placecell_megaloc`, no CUDA toolkit needed at all —
  the CUDA EP is inside the ORT shared library and dlopens its CUDA libs at runtime. `tensorrt`: today's block,
  untouched, deleted after step 8.
- `megaloc_embedder.cpp`: `#if PLACECELL_MEGALOC_TENSORRT … #else … #endif` around the backend type; one line.
- `examples/test_megaloc.cpp`: `--backend` goes away (it is a CMake choice), the printed header names the backend and
  the providers the session actually got (`session.GetProviders()` is not in the C API; log the appended list).
- placecell `pixi.toml`: add `osx-arm64`; feature `megaloc-onnx` (onnxruntime-cpp, opencv, yaml-cpp; on linux-64 the
  `cuda129` build + `cudnn`), tasks `build-megaloc-onnx`, `megaloc-test-onnx`; the TensorRT feature keeps its name
  until deletion. The `export-megaloc` env stays linux-64 (torch GPU for the export).
- AllFeature `pixi.toml`: `onnxruntime-cpp` in the common dependencies (`1.31.*`, build `*cuda129*` pinned on
  linux-64 so CUDA 12.9 stays — decision 5), `cudnn` on linux-64; `[package]` host deps the same with `if(linux)`
  for the CUDA half; `ALLFEATURE_WITH_MEGALOC` CMake option (ON everywhere) replaces the `PLACECELL_WITH_MEGALOC =
  ALLFEATURE_WITH_TENSORRT` coupling of phase 1.

## 6. Host changes — **new** (AllFeature)

- `System.cc`: the `#ifndef ALLFEATURE_WITH_TENSORRT` refusal of `vpr: megaloc` becomes `#ifndef
  ALLFEATURE_WITH_MEGALOC`; the `.engine` probe and the "building TensorRT engine (1-2 min)" line become
  backend-neutral ("loading MegaLoc (<backend>, provider <p>) …"); `megaloc_precision` is documented as a provider
  hint; new key `megaloc_provider: "auto"` (`auto|cuda|coreml|cpu`) and `megaloc_threads: 4`, both passed to the
  embedder; the settings reference regenerated.
- `CLAUDE.md`, `docs/reference/System.md`, `docs/reference/settings.md` updated; `docs/plans/mac_port.md` phase-2
  status line.

## 7. The Apple side specifically — **new**, measured not assumed

CoreML is the "TensorRT equivalent" only in the sense of *where* the model runs (GPU / Neural Engine); it is reached
through ORT, which keeps the C++ identical. What to expect and check on the M1 Pro:

- **Operator coverage.** The CoreML EP supports a fixed operator list per `ModelFormat`; unsupported nodes fall back
  to the CPU EP, each fallback boundary costing a tensor copy. A ViT has ~12 × (LayerNorm, MatMul with a
  non-constant B for `q·kᵀ` and `scores·v`, Softmax, Gelu, Add) plus the Sinkhorn (`LogSumExp`/`Exp`/`ReduceMax`).
  `ModelFormat=MLProgram` supports far more of these than `NeuralNetwork`. The first run with
  `ProfileComputePlan=1` prints per-op dispatch (ANE / GPU / CPU) — that log is the deliverable of this step.
- **Timing.** Record steady-state ms/image for provider `cpu` (4 threads) and `coreml` (`MLComputeUnits` = `ALL`,
  then `CPUAndGPU`, then `CPUAndNeuralEngine`), fp32 export. Budget: embeds happen on the mapping thread per keyframe
  (~1–5 Hz) and on the tracking thread for relocalisation queries; anything under ~150 ms is fine for the first, under
  ~50 ms desirable for the second. TensorRT fp16 on the RTX 4000 Ada is ~5 ms for scale.
- **Parity.** `megaloc_test --reference` on the Mac: ≥ 0.999 on CPU is expected (same fp32 maths); CoreML with the
  Neural Engine runs fp16 internally, so the same ~0.98–0.995 band as TensorRT uniform fp16 is possible — then
  `MLComputeUnits=CPUAndGPU` (fp32 on the GPU) is the fallback to measure. Decision 4's levers (fp16 weights,
  smaller export) only if the fp32 numbers force it.
- **Memory.** 912 MB file + ORT arena + CoreML's compiled copy; watch `MIMALLOC_SHOW_STATS`/Activity Monitor on the
  first run; `ModelCacheDirectory` avoids recompiling on every start.

## 7b. Speed levers, in the order to pull them (phase 6 material, written down now)

Resolution is the worst lever because it is the only one that changes what the descriptor sees. Before it:

1. **The accelerator, at 322** (CUDA EP on Linux; CoreML EP + Neural Engine on the Mac with the fp16 twin). T3 decides.
2. **Scheduling.** Embeds happen per keyframe (1–5 Hz) on the mapping thread already; the only tracking-thread case is
   the relocalisation query while lost, which can run every N-th frame. No quality cost.
3. **int8 dynamic quantisation** of the exported file (`onnxruntime.quantization.quantize_dynamic`, weights int8,
   activations quantised on the fly; Apple Silicon and x86 have integer dot-product units): typically 1.5–2.5× on the
   CPU for a ViT; the retrieval cost is measured with the same self-check / reference. The CPU-only lever.
4. **Resolution and aspect**, each a new model whose quality must be re-measured end to end (loop closures), since the
   README reference exists only for the 322×322 squash.

**Aspect-ratio family (phase-6 experiment, gated).** VSLAM-LAB sequences are mostly ~640×480 pixels but of varying
aspect, and MegaLoc squashes everything to a square. A family of static exports at the 322×322 token budget would
let the host pick by the camera's aspect (the sidecar carries the input size; the C++ side needs no change beyond
the selection rule, and the download task would fetch per file). Sides are multiples of 14; cost relative to
322×322 (529 tokens: tokens drive the linear layers, their square the attention):

| Aspect | Typical source | Export (W×H) | Grid | Tokens | Tokens vs ref | Attention vs ref |
|---|---|---|---|---|---|---|
| 1:1 | the published setting | 322×322 | 23×23 | 529 | 1.00× | 1.00× |
| 4:3 | 640×480 (ETH, TUM, most RGB-D) | 364×266 | 26×19 | 494 | 0.93× | 0.87× |
| 3:2 | 720×480, 1440×960 | 406×266 | 29×19 | 551 | 1.04× | 1.08× |
| 16:9 | 1280×720, phones | 434×238 | 31×17 | 527 | 1.00× | 0.99× |
| 3.3:1 | KITTI 1226×370 | 588×182 | 42×13 | 546 | 1.03× | 1.07× |

Gate: MegaLoc was trained on squashed squares, so whether aspect-true input retrieves better is unknown. The first
phase-6 experiment is one extra export, `--resolution 266 364`, and a `table_3` loop-closure comparison against
322×322 with the ORT backend on both machines. The family is built only if 4:3 wins; otherwise one file stays. A
single dynamic-shape export is not the alternative: it would put the positional-embedding interpolation back into the
graph as a runtime Resize and give up the static shapes CoreML compiles best. Nothing of this is phase-2 work.

## 8. Tests (what "done" means)

| # | Test | Where | Pass |
|---|---|---|---|
| T1 | `megaloc_test <onnx> <4 table_3 frames> --reference …` on the ORT backend, provider cuda | Linux | cos(ref) ≥ 0.995 and cos(ORT-CPU fp32) ≥ 0.9999 each; ms/image recorded next to the TensorRT numbers |
| T2 | same, provider cpu, fp32 and fp16 files | Linux and Mac | cos(ref) ≥ 0.995; fp16 within 1e-3 of fp32; ms/image recorded (CPU numbers on both machines) |
| T3 | same, provider coreml, fp16 file, the three `MLComputeUnits` values, `ProfileComputePlan=1` once | Mac | cos(ORT-CPU fp32) ≥ 0.999 (else fall back to CPUAndGPU); ms/image; the dispatch log saved to the note |
| T4 | `megaloc_embedder_smoke` | both | passes |
| T5 | `table_3`, `features: [orb32]`, `vpr: megaloc`, loop closes at the end: ORT-CUDA vs TensorRT on Linux in one gym entry (`docs/gym/`) | Linux | ATE and keyframe count within the board's range; same loop detected |
| T6 | `table_3` with `vpr: megaloc` on the Mac (provider from T3's winner) | Mac | runs end to end, loop closes, embed latency fits the keyframe rate |
| T7 | phase-1 identity: `vpr: none` runs unchanged (the ORT backend is not even constructed) | Linux | byte-identical to the phase-1 references |

T1–T4 are mine to run (Linux, under swap monitoring) and the user's on the Mac; T5–T6 the user runs through the
framework; T7 mine.

**Status 2026-10-13 (steps 4–5 done on Linux, placecell branch `onnxruntime`).** Backend `onnxruntime_megaloc.{hpp,cpp}`
+ shared `megaloc_common.{hpp,cpp}`, `MegaLocEmbedder::Settings` (precision, provider, threads, CoreML units),
`PLACECELL_MEGALOC_BACKEND`, env `megaloc-onnx` (osx-arm64 added to placecell's platforms), `megaloc_test` on the
public API with `--provider/--threads/--dump`. T1, T2 (Linux) and T4 pass — numbers in placecell
`docs/notes/2026-10-13_megaloc_onnxruntime_backend.md`: CUDA fp32 11.4 ms, CUDA fp16 7.6 ms, CPU fp32 (4 threads)
208 ms, CPU fp16 230 ms (half precision is emulated on x86: the fp16 file is for the accelerators); every run
≥ 0.995 vs the reference, fp32 providers agree to 0.999998, fp16 to ≥ 0.9998. Two environment facts: the `cuda129`
ORT build needs `cuda-cudart` added by hand; the fp16 converter left the output float16 (fixed in the export
script, `ensure_float32_io`). The TensorRT backend was moved onto the shared code but not compiled here (its env is
not installed on this machine); AllFeature's own `build` env compiles it in step 6. Also fixed on the way: the model
download task used to overwrite a local export (`tools/download_models.py`, per file, never overwrites).

**Status 2026-10-13 (step 6 done on Linux, AllFeature branch `megaloc-onnx`).** `ALLFEATURE_WITH_MEGALOC` (ON
everywhere) + `ALLFEATURE_MEGALOC_BACKEND` replace the TensorRT coupling; `onnxruntime-cpp` 1.31 in every
environment (`cuda129` build on linux-64, `cpu` on osx-arm64 and in the `cpu` env); `System.cc` builds the embedder
from `MegaLocEmbedder::Settings` with the keys `megaloc_provider`, `megaloc_threads`, `megaloc_coreml_units`
(`megaloc_onnx` names the file, `megaloc_precision` is TensorRT-only) and logs backend/provider/precision; the
`KeyframeInformationMegaLoc` guard follows; docs regenerated. The full build (all backends ON) and the GPU-free
configure both report `MegaLoc=ON (onnxruntime)`. **T7 passes**: the `vpr: none` run is byte-identical to the
pre-phase-1 reference (trajectories, point set, stripped log). **Host wiring check**: 150 frames of `table_3` with
`vpr: megaloc` on `megaloc_322x322_fp16.onnx`, provider auto → `onnxruntime/cuda, fp16 weights`, runs end to end.
T5 (full `table_3`, ORT-CUDA vs TensorRT, gym entry) is the user's run through the framework.

**Status 2026-10-13 (T2/T3 on the Mac, first numbers).** AllFeature builds and links on the Mac with
`MegaLoc=ON (onnxruntime)` (one phase-1 leftover fixed: `PlaceRecognitionMegaLoc.cc` was still under the TensorRT
option); placecell's `megaloc-onnx` env needed `eigen = "3.4.*"` (the Mac solve took Eigen 5, which placecell's
`find_package(Eigen3 3.3)` rejects). Measured on the M1 Pro (placecell note): CPU fp32 611 / 328 / 229 ms at 2 / 4 / 6
threads, cosines identical to Linux; CoreML 41–48 ms for every configuration, but **the Neural Engine loses
accuracy** (fp16 file on `ALL` or `CPUAndNeuralEngine`: 0.982 vs the 0.9955 floor) while the GPU path is exact
(fp16 + `CPUAndGPU` 46 ms at 0.9955; fp32 + `ALL` 48 ms at 0.9953). **Mac provider decision: CoreML on the GPU**, i.e.
the defaults (fp32 file, `ALL`) or the fp16 file with `megaloc_coreml_units: CPUAndGPU`. 17 of 545 nodes stay on
the CPU provider in 13 partitions. CoreML session creation: 90–175 s on a first compile, **36 s** on a repeat (the compile is cached per model
file under `megaloc_models/coreml_cache/`; the remaining 36 s are ORT's graph conversion and Core ML's load) —
acceptable for now, to be reduced in phase 6 by cutting the 13 partitions.

**Status 2026-10-13 (T6 on the Mac).** `table_3` through the framework on the M1 Pro, `orb32`, `vpr: megaloc`,
fp16 file, provider auto → `onnxruntime/coreml`, `CPUAndGPU`: runs end to end, no tracking loss, global BA, median
tracking 37 ms/frame; placecell's profile shows `add_image` at **60 ms median** per keyframe (347 keyframes
embedded, 60 alive at the end — the covisibility-kernel runaway of the shipped YAML again, unrelated to the port) on
the mapping thread, i.e. the 46 ms CoreML embed plus preprocessing and the store update. No `[LoopClosing]` line in
the run.

**Status 2026-10-13 (T5 on Linux; T5–T7 in `docs/notes/2026-10-13_megaloc_onnxruntime_host.md`).** `table_3` through
the framework on CUDA (`configs/test_exp_megaloc_onnx.yaml`, `orb32+aliked128`): `add_image` 11.9 ms median per
keyframe, ATE 4.39 mm (board range 4.8–6.5), no losses, **no loop on Linux either** — so the missing loop is the
shipped YAML's keyframe policy (covisibility-kernel runaway, 350–460 keyframes inserted), not the Core ML
descriptors. The ORT-vs-TensorRT loop comparison (decision 3) moves to a gym entry with `heuristic` culling, three
runs per backend, before the TensorRT deletion of step 9.

## 9. After parity: TensorRT leaves (decision 3) — **new**, last step

Delete `tensorrt_megaloc.{hpp,cpp}`, the `tensorrt` CMake branch, the `megaloc` pixi feature's TensorRT packages, the
`.engine` mentions in docs; AllFeature's `ALLFEATURE_WITH_TENSORRT` keeps guarding SuperPoint + segmentation until
phases 4–5. The ONNX export's TensorRT-specific rewrites (attention slicing) stay: they are harmless and ORT's CUDA EP
benefits from the same clean pattern. Issue #41 is closed as "won't do" at that point.

## 10. The write-up (the learning deliverable)

placecell's `docs/megaloc.md` is a placeholder. It becomes the document version of steps 0–7 as they are done: how
the model is obtained, why each export rewrite exists, what the sidecar is, the ORT session recipe with the provider
table, the measured numbers on both machines (one table: backend × provider × precision → cos, ms/image, first-start
cost), and the commands to regenerate everything. Written step by step, with the user running the export and the
test programs; a dated `docs/notes/` entry in placecell holds the raw measurements.

## Order of work and commits

1. placecell `onnxruntime`: `megaloc_preprocess` split out + `onnxruntime_megaloc` + CMake backend switch + pixi
   feature (one commit); `megaloc_test`/smoke adapted (one commit); T1/T2 on Linux, numbers into the note.
2. AllFeature `megaloc-onnx`: pixi + CMake (`ALLFEATURE_WITH_MEGALOC`), host wiring + settings keys, docs, submodule
   pointer (one commit); T7; user runs T5.
3. Mac: user builds, runs T2/T3/T4/T6 (I read the logs); CoreML tuning if needed (commits in placecell).
4. TensorRT removal in both repos (one commit each), docs, `docs/megaloc.md` final, gym entry.

## Decisions (2026-10-13)

**Q1 (a)** `megaloc_provider: auto|cuda|coreml|cpu` as its own key, `megaloc_precision` becomes a provider hint.
**Q2 (a)** the user runs the export exercise on Linux. **Q3 (a)** ORT `1.30 cuda129` on Linux (a `1.31 cuda129`
build is taken if it exists at solve time). **Q4 (c)** `megaloc_threads` setting, default 4, tuned from T2/T3.
**Q5 (b)** the fp16-weights ONNX is exported and published now: `export_megaloc.py --fp16` converts the fp32 export
with `onnxconverter_common.float16.convert_float_to_float16(keep_io_types=True)` (weights and internal tensors fp16,
input/output still fp32 so the C++ side is unchanged; the Sinkhorn/aggregator nodes are kept fp32 through the
converter's block list, mirroring the TensorRT mixed-precision policy), writes `megaloc_322x322_fp16.onnx` + sidecar
with `precision: fp16`, and the self-check adds a fourth column, cos(onnx fp16, reference). `megaloc_precision: fp16`
**Amended 2026-10-13 after the user's review:** no implicit file substitution — `megaloc_onnx` names the file that is loaded
(the fp32 export, or `megaloc_322x322_fp16.onnx` for the CUDA / CoreML providers), its sidecar says what it is, and
`megaloc_precision` is a TensorRT-only key (engine precision) that ONNX Runtime ignores. One key, one file, no magic.

**Exporter (2026-10-13, user's call: aim for the best version, not the legacy one).** The export uses PyTorch's
`torch.export` + onnxscript exporter (`dynamo=True`, the default since torch 2.9; the TorchScript tracer is
deprecated), opset 18, `external_data=False` (one self-contained file). The `export-megaloc` env moves to the **CPU**
build of torch ≥ 2.9 (resolved: 2.10.0, onnxscript 0.7.2): the export and the self-check run on the CPU anyway, and
the GPU build of 2.9 needs `cuda-version >= 12.9`, which the TensorRT 10.3 packages of the shared `megaloc` solve
group forbid. `--exporter torchscript|both` keeps the tracer for the side-by-side comparison that goes into the
write-up (node count, operator histogram, cosine against the reference, max |dynamo − torchscript|). Consequence: the
fp16 policy can no longer key on the tracer's `/backbone/` node names; it is now **topological** (the backbone's
final `LayerNormalization` and everything downstream of it is the aggregator and stays fp32), which holds for either
exporter. Step 2's text above describes the tracer; the mechanics of the dynamo path are written up in step 10.
TensorRT (T5) keeps running on its cached engine from the old tracer export; it is deleted in step 9.

**Finding while checking the topological rule on the existing export (2026-10-13).** The tracer export
(`megaloc_322x322.onnx`, 797 nodes, 29 operator types) contains **no** node name with `/backbone/` at all: the
tracer named the nodes `/Add_12`, `/MatMul_3`, … without the module scope. So the TensorRT backend's "mixed
precision" policy (pin every float layer whose name lacks `/backbone/` to fp32) pinned **every** layer, and the
`fp16.engine` that has been running is effectively an fp32 engine with the fp16 flag set, which is why its parity
with the reference was so good and why "uniform fp16" (no pinning) looked so much worse. The topological rule finds
75 aggregator nodes (Sinkhorn `ReduceLogSumExp`×6/`Exp`/`Einsum`, the 4 aggregator convs, `ReduceL2`×4, `Gemm`×3,
the final `LayerNormalization`) and 722 backbone nodes; that split is what the fp16 twin uses. Measuring the real
mixed-precision cost is therefore new information, column (d) of the self-check.

## Questions for the user (answered above; kept for the record)

| # | Question | Options | Recommendation |
|---|---|---|---|
| Q1 | Setting name for the provider | (a) `megaloc_provider: auto\|cuda\|coreml\|cpu` next to `megaloc_precision`; (b) reuse `megaloc_precision` values (`fp16-cuda`, …) | **(a)** — orthogonal things, two keys |
| Q2 | Who runs the export exercise of step 3 | (a) you, on Linux, with 4 `table_3` frames (the learning path; ~5 min incl. the self-check); (b) I run it under swap monitoring and you read the output | **(a)** — the reference file it writes is what every C++ test compares against, worth producing yourself once |
| Q3 | ORT version on Linux | (a) `onnxruntime-cpp 1.31 cuda134`-type builds exist; decision 5 picked `1.30 cuda129` to keep CUDA 12.9 — confirm, or (b) move the whole env to CUDA 13.4 (torch/TensorRT packages would have to follow — not now) | **(a)**, `1.30 cuda129`; if conda-forge has a `1.31 cuda129` build at solve time, take it |
| Q4 | CPU thread count default | (a) 4; (b) 2 (leave more to tracking on the 6+2-core M1 Pro); (c) a setting only, default 4 | **(c)** — measured in T2/T3, then the default is set from the numbers |
| Q5 | Keep an fp16-weights ONNX as a second published file | (a) only if T3 shows CoreML needs it; (b) export it now alongside (half the download, no accuracy change on CPU) | **(a)** — one artefact until a measurement asks for a second |
