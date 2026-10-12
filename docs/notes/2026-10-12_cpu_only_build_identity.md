# CPU-only build: byte identity with the GPU build, and what breaks it

- **Date:** 2026-10-12
- **Kind:** note
- **Context:** `docs/plans/mac_port.md` phase 1 (branch `cpu-only`): the GPU backends become CMake options
  (`ALLFEATURE_WITH_SIFTGPU` / `_TORCH` / `_TENSORRT`), `build.sh -c` and the `cpu` pixi environment build
  without CUDA, TensorRT, libtorch or SiftGPU. Phase 1's Linux test is that this build reproduces today's
  build byte for byte on the deterministic configuration of issue #16.

## Setup

`eth/table_3` (1180 frames), mono, `verbose:0`, the dev settings YAML with `features: ["orb32"]`,
`vpr: "none"`, `Tracking.Sequential: 1`, placecell diagnostics off; two culling policies:
`LocalMapping.KeyframeCullingMethod: information` (the shipped dev value, covisibility kernel) and
`heuristic`. Runs launched directly (`bin/vslamlab_allfeature_mono`, not through the VSLAM-LAB runner), each
under a swap monitor (peak used 9.4 GB, swap untouched). Compared: the two `KeyFrameTrajectory` CSVs
(`cmp`), the PLY as a *set* of vertices, and `system_output` with paths, timings and `file:line` log
prefixes stripped.

## Results

| Run | Binary | Toolchain | Culling | Keyframes | vs reference |
|---|---|---|---|---|---|
| A1, A2 | `2ce9e7f` (pre-phase-1), all backends | `build` env: gcc 14.4, MKL BLAS, LLVM OpenMP, OpenCV 4.12 `_611` | information | 43 | A1 == A2: identical (determinism baseline) |
| A3 | same | same | heuristic | 77 | reference |
| G1 | phase-1 code, all backends ON | same | information | 43 | **identical** to A1 (0 log diff lines) |
| B2 | phase-1 code, all backends OFF (`build.sh -c`) | same | information | 43 | **identical** to A1 |
| B4 | same | same | heuristic | 77 | **identical** to A3 |
| B, B3 | phase-1 code, all backends OFF | `cpu` env: gcc 15.3, OpenBLAS, GNU OpenMP, OpenCV 4.12 `_612` | information / heuristic | 39 / 85 | **diverge** after frame ~56: `info=0.235` vs `0.236`, then different keyframe decisions |

So the code changes of phase 1 are behaviour-neutral: with the same compiler and libraries, the GPU-free
configuration is byte-identical to the full build in both culling modes. The divergence of the `cpu`
environment runs is the toolchain (a different gcc major, BLAS and OpenMP runtime, and a different OpenCV
build): floating-point results differ in the last bits and the keyframe policy amplifies them. Identity
across toolchains was never promised by issue #16 (same binary, same machine) and is not a phase-1 goal;
the Mac will differ in the same way.

Two observations recorded for later, not acted on here:

- **Point cloud order is address-dependent.** `SavePointCloudVSLAMLAB` writes the same vertex set in a
  different order on every run (the PLYs of A1 and A2 differ as bytes, agree as sets): the map points are
  iterated in pointer order. Harmless for evaluation; sort by point id for a reproducible file if the
  PLY ever becomes a test artefact.
- **The shipped dev YAML runs away on `table_3` with the covisibility kernel.** `information` culling with
  `KeyframeCullingMaxUnexplained: 0.5` on `InformationKernel: covisibility` inserted 911 keyframes over
  1180 frames and culled 868 of them (41 alive at the end), 852 culled keyframes end above tau, and
  tracking is lost at frame 1142; the trajectory has 43 keyframes. `heuristic` on the same run: 77
  keyframes, no loss. This is the never-run state the covisibility-kernel plan left
  (`docs/plans/covisibility_kernel.md`, step 6) and belongs to that plan's gym entry, not to the port.
- **Viewer on, exit segfault** (pre-existing, issue #42): with `verbose:1` both the pre-phase-1
  and the CPU-only binary save everything and then crash in `pangolin::GlFont::~GlFont` during static
  destruction. Outputs of the two binaries on a 150-frame slice are identical.

## How to reproduce

```bash
pixi run -e build build -v        # or: pixi run -e build build-cpu -v  (same toolchain, GPU-free)
pixi run -e cpu build -v          # GPU-free toolchain (gcc 15, OpenBLAS): expect the last-bit drift
./bin/vslamlab_allfeature_mono sequence_path:<VSLAM-LAB-Benchmark>/ETH/table_3 calibration_yaml:<copy of calibration.yaml> \
  rgb_csv:<copy of rgb.csv> exp_folder:<out> exp_id:0 settings_yaml:<dev yaml with the keys above> verbose:0
cmp <a>/00000_KeyFrameTrajectory.csv <b>/00000_KeyFrameTrajectory.csv
```
