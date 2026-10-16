# macOS (Apple Silicon) port — AllFeature-VSLAM without CUDA

- **Author:** Alejandro Fontan (planned with Claude, Fable 5.1)
- **Created:** 2026-10-12
- **Status:** **phase 2 complete 2026-10-13** (AllFeature PR #46, placecell PR #8): MegaLoc runs on ONNX Runtime on
  both machines — CUDA 8–12 ms/keyframe on Linux, Core ML (GPU) 46 ms on the M1 Pro, CPU 120–230 ms — with the
  TensorRT MegaLoc backend deleted; details in `docs/plans/megaloc_onnxruntime.md`, placecell `docs/megaloc.md`.
  **2026-10-15: phase 3 folded into phase 6, next is phase 4** (see the phase 3 note); phase 4 plan drafted 2026-10-16:
  `docs/plans/learned_features_onnxruntime.md` (decision 3b re-opened there as Q5: ONNX `DeformConv` on ORT instead of libtorch CPU/MPS). Phase 1 history: implemented on Linux 2026-10-12 (branch `cpu-only`): GPU backends are CMake options, `build.sh -c` / `pixi run -e cpu ./build.sh -c` build GPU-free, byte-identical to the full build on `table_3` (`docs/notes/2026-10-12_cpu_only_build_identity.md`); macOS viewer-on-main-thread ported untested; first Mac build 2026-10-13 configured correctly (arm64, backends OFF, mimalloc zone override on) and stopped on g2o's `tr1` headers, fixed in the g2o fork; second build stopped on a `std::min(uint64_t, size_t)` in placecell's viz, fixed in placecell; third build 2026-10-13 links every target on the Mac and `table_3` mono runs through the VSLAM-LAB framework there (orb32, vpr none): **phase 1 passes on both machines** and is merged; item 7 done 2026-10-13 (brisk / akaze 1.0.1 published for osx-arm64 and linux-aarch64, now common dependencies); item 5 done 2026-10-13 (mimalloc serves the process on macOS: three 1 GiB arena reservations over a `table_3` run); **phase 1 complete**
- **Pinned to:** AllFeature `2ce9e7f` (`cpu-only` == `dev` == `origin/main`), placecell `71cbaff` (v0.1.0), VSLAM-LAB `dev` 1804957 (+ `origin/stream-iphone` 2c0ff49)
- **Target machine:** MacBook Pro 16" 2021, M1 Pro (6P+2E cores, 14-core GPU/Metal, NEON/FP16/DotProd, no I8MM), 16 GB unified, 1 GB swap, 16 KB pages, macOS 26.6, Xcode 27 / Apple clang 21, pixi 0.81, no Homebrew (`VSLAM-LAB/mac_spec.txt`)
- **Session:** `claude --resume 49010428-7f8f-40f2-9f55-3a1addb2cb77` (Linux); the Mac works from its own clone

## Goal

Run AllFeature-VSLAM on the Mac at interactive speed with the full feature set it has on Linux
today: classical and learned features, MegaLoc place recognition through placecell (keyframe
information policy, loop closing, relocalization) and online segmentation. The Linux/CUDA build
must keep working at every step, and both platforms must end up on **one** inference backend
(ONNX Runtime) instead of TensorRT + libtorch + SiftGPU, so that a model exported once runs on
every machine. The end state is also what the online iPhone demo needs (`stream-iphone --run
allfeature-dev`, parent branch `origin/stream-iphone`).

## What the study found (phase 0, 2026-10-12)

### Where the GPU is today

| Component | Library | Files (host) | Files (vendored) | Mac status |
|---|---|---|---|---|
| ORB/AKAZE/BRISK/KAZE extraction, BF matching, g2o, PoseLib, placecell core/viz, Pangolin viewer | OpenCV, brisk, akaze, Eigen, OpenMP | everything else | `g2o`, `PoseLib`, `placecell`, `mimalloc` | compiles as-is except: `-march=x86-64-v3` default (`CMakeLists.txt:25-33`), `brisk`/`akaze` conda packages exist for linux-64 only (`/home/alejandro/conda-packages`, `skip: win` so osx is not skipped by the recipe), `usleep`/`<unistd.h>` (`System.h:29`, `System.cc:436`, `Viewer.cc:600`, entry points, `DebugKeyStepper.h`), mimalloc interposition on macOS is a zone override, not link order |
| SIFT | SiftGPU (`-cuda`) + `SiftMatchGPU` | `Feature_sift128.{h,cpp}`, `FeatureMatcher.cpp:49-57`, `FeatureMatcher.h:174` | conda `siftgpu` (CUDA 12.6 only) | no SiftGPU on macOS (CUDA or legacy OpenGL) |
| ALIKED + LightGlue | libtorch 2.7 CUDA + 2 custom kernels (`deform_conv2d_kernel.cu` 1370 lines, `get_patches_cuda.cu` 256; `get_patches.cpp` has a CPU twin, `deform_conv2d.cpp` dispatches through the torchvision op registry) | `Feature_aliked128.cpp:9` (`torch::cuda::is_available()`), `FeatureMatcher_lightglue.cpp`, `FeatureMatcher.cpp:88-94` | `Light_Glue_CPP` (`LANGUAGES CUDA CXX`, `find_package(CUDAToolkit REQUIRED)`, `-march=native` hard-coded, CMake ≥ 3.26, C++20) | libtorch 2.14 `cpu_generic` exists for osx-arm64 (MPS build on conda-forge: to verify) |
| SuperPoint + SP-LightGlue | TensorRT 10.3 (`find_package(CUDA)` legacy) | `Feature_superpoint256.*`, `FeatureMatcher_superglue.cpp`, `FeatureMatcher.cpp:59-87` | `SuperPoint-LightGlue-TensorRT` (`tensorrtbuffer`) | ONNX already exported: `superpoint_models/superpoint_v1_fixed.onnx` (5 MB), `superpoint_lightglue_fixed.onnx` (46 MB), fixed 320×240, `config.yaml` names the tensors |
| MegaLoc | TensorRT 10.3 (engine cache `<onnx>.fp16.engine`, fp32 aggregator pinned) | `System.cc:131-160`, `PlaceRecognitionMegaLoc.cc`, `KeyframeInformation.cc` | `placecell/src/megaloc/tensorrt_megaloc.{hpp,cpp}` (only TRT file; `MegaLocEmbedder` is already a pimpl, header has no TRT) | ONNX exists (`megaloc_322x322.onnx`, 912 MB fp32, opset 17, static [1,3,322,322] → [1,8448], `.yaml` sidecar); the export's self-check already runs it on ORT CPU |
| EfficientViT segmentation | TensorRT 10.3 | `SegmentationSettings.h`, the three entry points | `Segmentation-TensorRT` (first-party, not a submodule) | ONNX exists (`efficientvit-seg-l1-ade20k_512x512.onnx`, 162 MB, `.classes.yaml` sidecar) |

### What the toolchain offers (conda-forge, 2026-10-12)

| Package | osx-arm64 | linux-64 |
|---|---|---|
| `onnxruntime-cpp` | 1.31.0 `_cpu` (deps: libcxx only; CoreML EP availability inside this build: **to verify** in phase 2) | 1.31.0 `cuda134`, 1.30.0 `cuda129` (+ cpu builds) |
| `libtorch` / `pytorch-cpu` | 2.14.1 `cpu_generic` | 2.14.1 `cuda129_mkl`, cpu_mkl |
| `llvm-openmp` | 23.1.3 | 23.1.3 |
| `pangolin-opengl` | 0.9.6 | 0.9.6 |
| `libopencv` | 4.13 (ORB-SLAM3 lock) | 4.12 pinned here |
| `mimalloc` | 3.5.4 | 3.5.4 |
| `brisk` / `akaze` / `siftgpu` (vslamlab channel) | **none** | 1.0.0 |

### The recipe that already works on this Mac (ORB-SLAM3-DEV v1.2, ORB-SLAM2-DEV)

- pixi `[package]` (pixi-build-cmake) + `[dev] orbslam3 = { path = "." }`; compiler = conda-forge clang 23 via pixi-build; cmake, ninja; `CMAKE_BUILD_PARALLEL_LEVEL=2-3` (heaviest TU peaks at 4.5 GB).
- No `-march` unless `*_NATIVE=ON` (one flag for every target, Eigen alignment); Eigen 3.4 pinned.
- C++17 headers instead of `tr1`/`stdint-gcc.h`/`usleep` (commit `5a1d8b5`); g2o fork already carries these fixes.
- **Pangolin on the main thread under `__APPLE__`**, SLAM work on an 8 MB-stack pthread (`System::Run`, commit `d2c7c81`: SIGBUS from the 512 KB default secondary-thread stack during monocular init; Shutdown waits for LM/LC/GBA to avoid `mutex lock failed` from static destructors).
- Online entry point `vslamlab_orbslam3_mono_stream` on the ARStream ZeroMQ relay (`libarstream`, header-only, vslamlab channel) — AllFeature's `vslamlab_allfeature_mono_stream.cpp` still speaks the old RTMP URL.
- Parent: `[feature.orbslam3-dev] platforms = ["linux-64", "osx-arm64", "win-64"]`; `allfeature-dev` is `["linux-64"]` and `allfeature` is `["linux-64-cuda129"]` (`VSLAM-LAB/pixi.toml:566`, `:287`).

## Phases

Every phase has: a goal, the tests that define success (Linux first, then Mac), the commits it
produces, and the open decisions. The user pushes; Claude never pushes. Each phase gets its own
branch off `main` (`cpu-only` is phase 1's) and merges into `main` after both machines pass.
Where a phase changes behaviour that the gym measures, it gets a gym entry (`docs/gym/`).

### Phase 1 — CPU-only build (branch `cpu-only`)

**Goal.** One CMake project that configures and builds with **no** CUDA, TensorRT, libtorch or
SiftGPU, on Linux first and then on the Mac, running `features: [orb32]` + `vpr: none` +
`segmentation: none` end to end. GPU paths stay available on Linux behind options, and the Linux
default build is bit-for-bit what it is today.

**Work.**
1. CMake options with the existing defaults on Linux: `ALLFEATURE_WITH_CUDA` (SiftGPU +
   LightGlue CUDA), `ALLFEATURE_WITH_TORCH` (ALIKED/LightGlue), `ALLFEATURE_WITH_TENSORRT`
   (SuperPoint, segmentation, MegaLoc) — later phases rename these into backend choices. Each
   option gates: the `add_subdirectory`, the `find_package`, the source files
   (`Feature_sift128.cpp`, `Feature_aliked128.cpp`, `Feature_superpoint256.cpp`,
   `FeatureMatcher_lightglue.cpp`, `FeatureMatcher_superglue.cpp`, `PlaceRecognitionMegaLoc.cc`?)
   and the registrations in `FeatureFactory.h` / `FeatureMatcher` members / the entry points'
   segmenter. A feature requested in the YAML but compiled out is a clear `AF_ERROR` at startup.
2. Architecture flag: `ALLFEATURE_NATIVE=ON` → `-march=native` (works with clang on arm64);
   OFF → `x86-64-v3` only when `CMAKE_SYSTEM_PROCESSOR` is x86_64, nothing on arm64 (`armv8.5-a`
   or nothing — decide by measuring in phase 6).
3. Portability: `usleep` → `std::this_thread::sleep_for`, drop `<unistd.h>` from `System.h`,
   `DebugKeyStepper.h` select-based key reader kept POSIX (fine on macOS), `__builtin_popcountll`
   fine on clang/arm64, OpenMP via `llvm-openmp`.
4. **Viewer on the main thread under `__APPLE__`** — port ORB-SLAM3's `System::Run(work)` pattern
   (8 MB-stack worker thread, Pangolin on main), keep the current Linux threading.
5. mimalloc: verify the zone override works when linked as a shared lib on macOS (mimalloc docs:
   `-lmimalloc` first + `MIMALLOC_VERBOSE=1` prints when it is active), otherwise make it
   Linux-only.
6. pixi: `platforms = [linux-64 (cuda 12.0), osx-arm64]`; a `cpu` feature/environment on both
   platforms (OpenCV, Eigen 3.4, Pangolin, yaml-cpp, llvm-openmp, cmake, ninja, compilers) and the
   existing CUDA `build` environment restricted to linux-64. Mac parallel level 3.
7. Classical features on the Mac: build `brisk`/`akaze` for osx-arm64 from
   `/home/alejandro/conda-packages` (same recipes, publish to the vslamlab channel) — or fall back
   to OpenCV's `cv::AKAZE`/`cv::BRISK` if upstream BRISK's SSE code does not build on NEON
   (**decision 2**).
8. Parent repo: `allfeature-dev` feature gets `osx-arm64`; `fetch-source` clones the chosen branch.

**Tests.**
- Linux: `pixi run -e cpu ./build.sh -c` configures and builds with no CUDA/TRT/torch package in the env (the `cpu` feature defines no tasks: duplicated task names made the parent's bare `pixi run execute-mono` forward ambiguous on the first Mac run, 2026-10-13)
  (`ldd bin/vslamlab_allfeature_mono` shows none); default `pixi run build` unchanged.
- Linux: gym sequence `eth/table_3` (`exp_gym_mono`, `orb32`, `vpr: none`, `Sequential: 1`,
  `verbose: 0`) is **byte-identical** between the CPU-only build and today's build (issue #16
  determinism makes this a real test).
- Mac: configure + build in the `cpu` env; `table_3` runs mono with `orb32` and `verbose: 1`
  (viewer up, no crash at init/exit), ATE within the Linux range on the gym board; tracking time
  recorded as the Mac baseline number.
- Both: `akaze61`, `brisk48`, `kaze64` run on `table_3`.

**Status 2026-10-13 (first Mac build).** The configure step on the M1 Pro is right (Apple clang 21 through the conda `cc`/`c++` wrappers, `-march=native` on arm64, the three backends OFF, the brisk/akaze warnings, OpenMP 5.1 from `llvm-openmp`, mimalloc with `MI_OSX_ZONE` + `MI_OSX_INTERPOSE` on); the build stopped at the first g2o object: the vendored g2o used `<tr1/unordered_map>` / `<tr1/memory>`, which Apple's libc++ never shipped. Fixed in the g2o fork (`alejandrofontan/g2o`, branch `cxx11-headers`: C++11 headers, `std::` instead of `std::tr1::`, five headers), the same change ORB-SLAM3-DEV made in its copy. On Linux the rebuilt full binary stays byte-identical to the pre-phase-1 references in both culling modes (g2o pours the vertex map into a pointer-ordered set before building the Hessian, so the hash-map implementation does not matter). The fork's GitHub CI (inherited from upstream g2o) fails on every platform in its "Build Introspection" step because it expects `build/g2o/config.h` while this stripped g2o writes `config.h` into the source tree; unrelated to the change, to be fixed or removed in the fork.

**Status 2026-10-13 (item 5, mimalloc on the Mac; phase 1 complete).** `MIMALLOC_VERBOSE=1 MIMALLOC_SHOW_STATS=1` on a `table_3` run (orb32 + akaze61 + brisk48, verbose 1): mimalloc 2.4.5 initialises and reserves a 1 GiB arena at start, a second during initialisation and a third around frame 1163, so the zone override routes the process's allocations through it (an inactive override would never need more than the initial arena). The exit statistics block does not appear in the captured output, but the process exits with status 0 (checked directly with `pixi run execute-mono ... verbose:1; echo $?`), so the viewer-on exit crash of issue #42 is Linux-only; why mimalloc prints no exit statistics on macOS is not known and not worth chasing. Note for anyone repeating the check: `pixi` itself is built on mimalloc and honours `MIMALLOC_*` too, so a statistics block on the terminal after `pixi run vslamlab` (104 s elapsed, 2.5 s CPU, all counters zero) is the outer pixi process, not the SLAM binary, whose output the runner captures into `system_output`. Mac timing for that run: median tracking 73 ms per frame with the three features, OpenCV on GCD with 8 threads, no tracking loss, 473 keyframes inserted / 69 alive (the covisibility-kernel runaway, unrelated to the port).

**Status 2026-10-13 (item 7, BRISK/AKAZE on the Mac).** `brisk` 1.0.1 and `akaze` 1.0.1 are published for linux-64, osx-arm64 and linux-aarch64 (`VSLAM-LAB/conda-recipes`, CI on ubuntu + a `macos-15` runner). brisk's fork: the 2013 sse2neon shim lacked 16 of the 48 SSE intrinsics brisk uses, replaced by sse2neon v1.8.0; the NEON guard was the 32-bit `__ARM_NEON__`, now `defined(__ARM_NEON) || defined(__ARM_NEON__)`; CMake keys on `CMAKE_SYSTEM_PROCESSOR`, no `-mfpu=neon` on AArch64, `-flax-vector-conversions` kept (the NEON code hands `int16x8_t` to sse2neon's `__m128i`). akaze's fork: OpenMP on Apple too (`OpenMP::OpenMP_CXX`, llvm-openmp as host dependency on macOS). Checked on Linux by cross-building both packages for linux-aarch64 before the Mac CI run. `brisk` / `akaze` moved from the linux-64 target to the common dependencies of `pixi.toml` (`>=1.0.1`).

**Status 2026-10-13 (Mac build complete).** Second build: `placecell/src/viz/viz.cpp` called `std::min` on a `std::uint64_t` query index and a `size_t` size; equal types on Linux (`unsigned long`), distinct on macOS (`unsigned long long` vs `unsigned long`), so deduction fails. Fixed in placecell (`std::min<std::size_t>`, branch `macos-build`, merged); every other `std::min`/`max`/`clamp` in the host and placecell was checked. Third build links `libg2o.dylib`, `libAllFeature-VSLAM.dylib`, the three executables and `test_bfmatcher_parity`. Apple clang warnings worth knowing: 84 × `-Wdelete-non-abstract-non-virtual-dtor` because `FeatureExtractor` has virtual methods and a non-virtual destructor (harmless through `shared_ptr` made from the derived type, made virtual on `cpu-only`), unused private fields `PnPsolver::mRansacTh` / `cws_determinant`, an unused `iteration` in `FeatureExtractor.cpp`; and the linker's `-F/opt/conda-sdks/MacOSX11.0.sdk/...` directory warning, a build-time SDK path baked into a conda package's CMake config, harmless.

**Status 2026-10-12 (Linux side done).** Items 1-4, 6 and 8 implemented; the Linux tests pass
(`docs/notes/2026-10-12_cpu_only_build_identity.md`: GPU-free build identical to the full build with
the same toolchain, in both culling modes; the `cpu` environment's different gcc/BLAS/OpenMP/OpenCV
build drifts in the last bits, which is expected). Beyond the plan: `brisk`/`akaze` are optional at
configure time (`ALLFEATURE_WITH_BRISK`/`_AKAZE` when found), so the Mac builds with `orb32`/`kaze64`
before the recipes land; `setup` on macOS skips the model downloads. Item 5 (mimalloc zone override)
and item 7 (recipes) wait for the Mac; the exit-time segfault with the viewer on turned out to be
pre-existing (issue #42). Mac to-do: `pixi install -e build` on the Mac from this branch, `pixi run
build -v` (CPU-only is the Apple default), then `table_3` mono with `verbose:1`; report the configure
log and the first run's `system_output`.

### Phase 2 — MegaLoc on ONNX Runtime (placecell + host), Linux and Mac

**Goal.** `vpr: megaloc` works with ONNX Runtime as the only inference engine in placecell: CUDA
execution provider on Linux, CPU and CoreML on the Mac. TensorRT leaves placecell (and, with
phases 4–5, the whole project; issue #41 closes as "won't do").

**Work (placecell submodule, its own commits + pointer bump here).**
1. `src/megaloc/onnxruntime_megaloc.{hpp,cpp}` with `TensorRTMegaLoc`'s method surface
   (`infer`, `descriptorDim`, `inputWidth/Height`, `enginePath` → the ONNX path,
   `loadedFromCache` → true); reuse `loadSidecarYaml`, the preprocessing and the host-side
   re-normalisation verbatim. Provider list from a setting: `cuda` → `cpu` fallback on Linux,
   `coreml` → `cpu` on macOS; thread counts explicit (`intra_op` = 2–4, never the core count).
2. CMake `PLACECELL_MEGALOC_BACKEND = onnxruntime | tensorrt` (default onnxruntime); the TRT
   backend stays until phase 2 is measured, then is deleted.
3. `precision`: fp16 is meaningless to ORT CPU; keep the key, map it to graph optimisation /
   provider options; on CoreML try `MLComputeUnits=ALL` and fp16 compute.
4. Model size: 912 MB fp32 ONNX is the backbone's weights; an fp16-weight export (half the
   file, same graph) and a smaller export resolution (e.g. 224×224 = 16×16 patches) are the
   levers if CPU latency is too high (**decision 4**). The export script already produces
   `_reference.txt` for the parity test.
5. placecell pixi: add `osx-arm64`, a `megaloc-onnx` feature without CUDA/TRT.
6. Host: `System.cc:131-160` log lines and the `.engine` probe become backend-neutral;
   `megaloc_precision` documented as "provider hint".

**Tests.**
- `megaloc_test --reference` (placecell): cosine(ORT, torch reference) ≥ 0.999 on Linux CUDA EP
  and on Mac CPU/CoreML; steady-state ms/image recorded for each (TRT fp16 was ~5 ms on an RTX
  4000 Ada; expect 50–150 ms CPU on M1, less on CoreML — measured, not assumed).
- `megaloc_embedder_smoke` passes on both machines.
- Linux: `table_3` with `vpr: megaloc` + `features: [orb32]`: loop closes at the end, ATE and
  keyframe count in the gym board range; compare TRT vs ORT-CUDA in one gym entry.
- Mac: same run; embedding latency must fit the keyframe rate (embeds happen off the tracking
  thread in LocalMapping, reloc queries on the tracking thread).

### Phase 3 — placecell keyframe information + loop closing on the Mac

**Folded into phase 6 (2026-10-15).** Everything below is verification, not porting: the placecell core is Eigen-only and already builds and runs on the Mac (phase 1), MegaLoc retrieval runs there on ONNX Runtime (phase 2, T6), and the loop / relocalisation / `megaloc`-kernel checks need the profiling and the Mac settings profile that phase 6 builds anyway. Phase 6 therefore inherits the three tests: a heuristic-culling settings file so `table_3` closes its loop on both machines (the shipped YAML's covisibility-kernel runaway prevents it), a run with `LocalMapping.InformationKernel: megaloc`, and a forced-loss relocalisation; plus the `verbose: 1` PlaceCell window on the Mac's main thread. Known gap until then: loop closing has not been exercised on the Mac; phase 4 changes features and matchers, not loop closing, so a Mac-only regression there stays unlikely.

**Goal.** The information-based keyframe policy (`LocalMapping.KeyframeCullingMethod:
information`, `Tracking.KeyframeMinInformation`, both kernels `megaloc` and `covisibility`),
relocalization and loop closing run on the Mac with `verbose: 1` including the PlaceCell Pangolin
window, at the same keyframe/ATE numbers as Linux.

**Work.** Mostly verification: the placecell core is Eigen-only. Expected Mac-specific items:
the second Pangolin window on the main thread (phase 1's `System::Run` must own both), `viz`
with the headless OpenCV, `PlaceCell.Dump` paths, `std::filesystem`. Thread budget: 6 P-cores
for tracking + local mapping + loop closing + viewer + ORT intra-op threads — set explicitly.

**Tests.** `table_3` with the dev YAML as shipped (`orb32` only, `information` culling,
`covisibility` and `megaloc` kernels): same `PlaceCell.Dump` statistics (keyframes inserted /
culled, loop detected) on both machines within run-to-run variance; `plot_placecell.py` of both
dumps side by side in the gym entry.

### Phase 4 — learned features and matchers without CUDA

**Goal.** `aliked128` + LightGlue and `superpoint256` + SP-LightGlue run on both machines on ONNX
Runtime; libtorch and `Light_Glue_CPP`'s CUDA kernels leave the build (or become an optional Linux
backend, **decision 3**). SIFT gets a CPU extractor.

**Work.**
1. SuperPoint + SP-LightGlue: the ONNX files exist and everything but the TensorRT call is
   CPU code over float buffers (SuperPoint: threshold / borders / top-k / bilinear descriptor
   sampling; LightGlue: four dynamic inputs `keypoints_0/1 [1,N,2]`, `descriptors_0/1 [1,N,256]`
   → `scores [1,N0,N1]`, mutual-NN + `exp(s) > 0.1` on the CPU). Seam: an `InferenceSession`
   (set input shape, run by tensor name) with an ORT implementation; pimpl `super_point.h` /
   `light_glue.h` so `Feature_superpoint256.h` and `FeatureMatcher.h` stop including NvInfer.
   Keep the existing quirk that SP-LightGlue normalises keypoints with the config's 320×240
   rather than the real image size (`light_glue.cpp:391-400`) until a gym entry changes it.
   Replace the `SuperPoint-LightGlue-TensorRT` submodule by an ORT fork (same VSLAM-LAB org).
2. ALIKED + LightGlue (**decision 3b**). The weights are libtorch state-dict pickles
   (`aliked-n16.pt`, `aliked_lightglue.pt`), the code is native libtorch modules, and only two
   things are CUDA-only: `torchvision::deform_conv2d` (registered for the CUDA backend only,
   `deform_conv2d_kernel.cu:1360`; libtorch ships no CPU kernel) used by ALIKED's `block3/4`
   (3×3, no modulation mask), and `get_patches` (CPU twin exists but uses `accessor<long>`,
   which breaks on macOS where `int64_t` is `long long`; MPS tensors would fall into the CUDA
   branch). Everything else (DKD: unfold/topk/nonzero/grid_sample; SDDH: grid_sample; LightGlue:
   einsum attention, flash disabled) runs on CPU and MPS already, and the matcher even has an
   `"mps"` pruning entry.
   - (a) **libtorch CPU/MPS** — make `LANGUAGES CUDA` + the `.cu` files optional in
     `Light_Glue_CPP` (fork, submodule bump), add a device-agnostic `deform_conv2d` (9 sampling
     grids → `grid_sample` → matmul with the `[O, C·9]` weight; exact for the unmasked case) and
     an `unfold`-based `get_patches`; host: `selectTorchDevice()` cuda → mps → cpu in
     `Feature_aliked128.cpp:9` and `FeatureMatcher.cpp:91`, `!is_cpu()` instead of `is_cuda()`
     in `tensorDescToMatCopy`. Two inference runtimes remain (libtorch + ORT). Lower risk.
   - (b) **ONNX for both** — export ALIKED (deform conv decomposed as in (a), traced) and
     ALIKED-LightGlue (as the SuperPoint-LightGlue export does); one runtime, CoreML possible,
     drops the 34 MB libtorch and the C++20 requirement. Riskier: `nonzero`/`topk` dynamic
     shapes in DKD, no existing export for `aliked-n16` with these weights.
   Parity test either way: keypoints/descriptors/matches vs today's CUDA output on 10 `table_3`
   frames (repeatability, descriptor cosine, match count).
3. SIFT: `Feature_sift128` on `cv::SIFT` (RootSIFT as today) + `BF_L2`/LightGlue-SIFT matching
   on every platform; SiftGPU only under `ALLFEATURE_WITH_CUDA` or dropped (**decision 3**).
4. Resolution and keypoint budget per device: the Mac may need `maxNumFeatures`/input size
   reduced for real time — keep the Linux settings, add a Mac profile YAML.

**Tests.** `table_3` with `features: [orb32, aliked128]` (the dev default) and with
`[superpoint256]`: Linux ORT-CUDA ≈ today's torch/TRT numbers (gym entry); Mac runs in real
time or we know by how much it misses (ms per stage from the profiling block). Parity tests above.

### Phase 5 — online segmentation without TensorRT

**Goal.** `segmentation: efficientvit` on ORT (CUDA EP / CoreML), same mask semantics
(`CV_8UC1`, 1 = static), `Segmentation-TensorRT` renamed/replaced by an ORT implementation behind
the `SegmentationSettings.h` interface (the entry points only see `makeSegmenter()`/`inferMask`).

**Tests.** mask IoU between TRT and ORT on 20 frames ≥ 0.98; `table_3` and one dynamic sequence
(BONN-RGBD) with segmentation on, both machines; latency per frame recorded.

### Phase 6 — profile, optimise, tune (Mac first, Linux must not regress)

**Goal.** The best configuration for the M1 Pro: per-stage timing (extraction, matching, pose
optimisation, local BA, embedding, segmentation), thread pinning/budget across the four SLAM
threads and the inference thread pools, `-march`/`-mcpu=apple-m1` and NEON hot spots
(`BruteForceMatcher`, `Feature_orb32` popcount loops, Eigen alignment), ORT graph optimisations
and CoreML coverage per model, fp16 where it is free, memory (16 GB, 1 GB swap: ORT arenas, the
912 MB MegaLoc model), the Viewer's cost on the main thread, and a Mac settings profile. Each
change is a gym entry. Candidate end-to-end demo: `stream-iphone --run allfeature-dev` (port
`vslamlab_allfeature_mono_stream.cpp` to the ARStream relay, as ORB-SLAM3 v1.2 did).

## Decisions

Taken 2026-10-12: **1 (b)** one branch per phase off `main`; **3 (a)** TensorRT, SiftGPU and the CUDA kernels are deleted once ORT parity is measured; **3b (a)** ALIKED + LightGlue on libtorch CPU/MPS first; **6 (a)** Claude runs short `table_3` tests on the Linux machine, always under swap monitoring (builds and runs both fill the swap and freeze the machine). Also taken 2026-10-12: **2 (a)** BRISK/AKAZE built for osx-arm64 from the existing recipes and published to the vslamlab channel (OpenCV fallback only if BRISK's SSE does not port); **4 (a)** measure the 322×322 fp32 MegaLoc export on ORT CPU/CoreML first, fp16 weights or a smaller export only if latency forces it; **5 (a)** ORT 1.30 `cuda129` on Linux (CUDA 12.9 stays), 1.31 `cpu` on the Mac. Phase 1 stays on branch `cpu-only`; later phases use short names.

| # | Question | Options | Recommendation |
|---|---|---|---|
| 1 | Branching | (a) one long `mac` branch with everything; (b) one branch per phase off `main`, PR'd after Linux + Mac pass | **(b)** — Linux stays releasable, each phase is a reviewable diff; `cpu-only` is phase 1 |
| 2 | BRISK/AKAZE on the Mac | (a) build the existing recipes for osx-arm64 and publish; (b) OpenCV's built-in `cv::BRISK`/`cv::AKAZE` | **(a)** first (same descriptors as the paper's numbers), (b) as fallback if upstream BRISK's SSE does not port |
| 3 | TensorRT and SiftGPU after the port | (a) delete them once ORT parity is measured; (b) keep them as optional Linux backends | **(a)** — fewer envs, issue #41 disappears, the conda package shrinks; the old code stays reachable by tag |
| 3b | ALIKED + LightGlue runtime | (a) libtorch CPU/MPS with a device-agnostic deform conv (two runtimes stay: libtorch + ORT); (b) export both to ONNX, ORT only | **(a)** in phase 4 (lowest risk, measured first), (b) as a phase-6 candidate if ORT/CoreML turns out faster than libtorch MPS |
| 4 | MegaLoc model on CPU | (a) 322×322 fp32 as is; (b) fp16 weights; (c) smaller resolution export | measure (a) in phase 2 first; (b) is free; (c) only if latency forces it (retrieval quality changes) |
| 5 | Linux inference | (a) ORT CUDA EP (needs cuDNN, cuda129 build pins ORT 1.30); (b) ORT CPU only | **(a)** — Linux stays the fast reference |
| 6 | Who runs tests on this Linux machine | (a) Claude runs the gym sequence here after each build (overrides the project rule for this effort); (b) the user runs everything | **(a)** for the deterministic `orb32`/`vpr: none` byte-identity check and short `table_3` runs; the user runs anything longer and all Mac runs |

## Risks

- **ALIKED deformable conv on ORT** (phase 4) — the one model without an existing ONNX path.
- **CoreML EP coverage**: a ViT-B with Sinkhorn may fall back to CPU for many ops; the conda
  `onnxruntime-cpp` build may not include the CoreML EP at all (then: build ORT ourselves, or
  CPU only, or `coremltools`).
- **Memory on the Mac**: 912 MB model + ORT arena + OpenCV + Pangolin in 16 GB with 1 GB swap;
  build TUs of ORT-free code are small, so the build itself is fine.
- **Mac viewer threading**: every Pangolin call must be on the main thread; the placecell viz
  window and `FrameDrawer` need the same treatment.
