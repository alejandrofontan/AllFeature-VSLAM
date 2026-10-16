# Learned features on ONNX Runtime, steps 1–3: the exports (2026-10-16)

- **Plan:** `docs/plans/learned_features_onnxruntime.md`, steps 1–3. Tool: `tools/export_features.py`, environment
  `export-features` (`pixi run -e export-features export-features --images <4 table_3 frames>`).
- **Pins:** `cvg/LightGlue` `eb42fee` (2026-02-18); weights `superpoint_v1.pth` (Magic Leap, via the LightGlue release),
  `aliked-n16.pth` (`Shiaoming/ALIKED` `683d7c6`), `{superpoint,aliked,sift}_lightglue.pth` (LightGlue release
  `v0.1_arxiv`); sha256 of each in the sidecars. torch 2.10.0 CPU, onnxscript 0.7.2, onnx 1.20, onnxruntime 1.31.
- **Test frames:** `table_3` 739×458, `rgb_0/2989.023949`, `2989.392583`, `2992.710283`, `2993.078917`; pairs (1,2)
  and (3,4), ~0.4 s apart. Keypoint budget 1000 (`FeatureExtractor.maxNumFeatures`) for SuperPoint and SIFT, ALIKED in
  threshold mode (0.2, no cap) as the host runs it.

## What was exported

| File | Inputs → outputs | Opset | Nodes | Size |
|---|---|---|---|---|
| `superpoint_v1.onnx` | `image [1,1,H,W]` → `scores [1,8(H//8),8(W//8)]`, `descriptors [1,256,H//8,W//8]` | 18 | 91 | 5.0 MB |
| `aliked_n16.onnx` | `image [1,3,H,W]` → `keypoints [N,2]`, `descriptors [N,128]`, `scores [N]` (N data-dependent) | 19 | 303 (`DeformConv`×4, `NonZero`×1, `GridSample`×2, `GatherND`×2) | 2.9 MB |
| `lightglue_superpoint.onnx` | `keypoints_0/1 [1,N,2]`, `descriptors_0/1 [1,N,256]` → `log_assignment [1,N0+1,N1+1]` | 18 | 1786 | 45.9 MB |
| `lightglue_aliked.onnx` | same, 128-d | 18 | 1791 | 46.1 MB |
| `lightglue_sift.onnx` | + `scales_0/1 [1,N]`, `oris_0/1 [1,N]` | 18 | 1797 | 46.1 MB |

Every axis named above is symbolic: H and W for the detectors (the SuperPoint output shape is literally
`8*(height//8)`, the floor of the three max-pools), N0 and N1 ≥ 2 for the matchers. fp16 twins (`*_fp16.onnx`) of
all five with float32 I/O; sidecars `<name>.onnx.yaml` with provenance and the preprocessing / post-processing contract;
references `feature_models/reference/` (upstream PyTorch outputs: 4 frames × 3 detectors, 2 pairs × 3 matchers).

## Results, columns (a) upstream PyTorch · (b) wrapper in PyTorch · (c) ONNX fp32 on ORT CPU · (d) ONNX fp16 on ORT CPU

| | (b) vs (a) | (c) vs (a) | (d) vs (a) |
|---|---|---|---|
| SuperPoint (dense graph + Python tail) | same N (705/709/683/628), Δxy = 0, Δscore = 0, cos ≥ 0.9999996 | same N, Δxy = 0, Δscore ≤ 4e-6, cos ≥ 0.9999996 | N ±1–3; 97.6–98.8 % of the keypoints at the same pixel (NMS ties flip for the rest), cos ≥ 0.9996 on those |
| ALIKED (full graph) | same N (440/444/515/441), Δxy ≤ 3e-5 px, Δscore ≤ 4e-6, cos ≥ 0.9999998 | same N, Δxy ≤ 1.2e-4 px, Δscore ≤ 2.2e-5, cos ≥ 0.9999999 | N ±1–2; ≥ 99.8 % within 0.5 px but the sub-pixel refinement moves points by up to 0.47 px, cos ≥ 0.977 (0.994 typical) |
| LightGlue-SuperPoint | 705/705 and 683/683 assignments identical, Δscore = 0 | identical, Δscore ≤ 1.1e-4 | 705/705 and 681/683, Δscore ≤ 0.08 |
| LightGlue-ALIKED | identical, Δscore = 0 | identical, Δscore ≤ 1.3e-4 | 438/440 and 515/515, Δscore ≤ 0.28 |
| LightGlue-SIFT | identical, Δscore = 0 | identical, Δscore ≤ 8.8e-5 | 845/845 and 843/845, Δscore ≤ 0.31 |
| dynamic axes | — | frame 1 resized to 640×480: SuperPoint N = 633, ALIKED N = 332, same exactness as above | — |

Match counts on the 0.4 s pairs, for the record: SuperPoint 478 / 471 of ~700 keypoints, ALIKED 308 / 314 of ~450–515,
SIFT 444 / 413 of 845.

**Conclusion.** The fp32 exports reproduce upstream to float32 precision on ONNX Runtime, including the ONNX
`DeformConv` kernel against torchvision's and the data-dependent keypoint count. The fp16 twins are usable but not
exact: SuperPoint keeps ~98 % of its keypoints (NMS ties flip in half precision; the kept ones sit on the same pixel
with cos ≥ 0.9996); ALIKED keeps the set but its soft-argmax sub-pixel refinement (exp with temperature 0.1) moves
points by up to 0.47 px and descriptors to cos 0.977 worst case; the matchers change 0–2 assignments per pair while
a few match confidences move by up to 0.3 — the log-softmax assignment tail is the sensitive part (MegaLoc's
aggregator again). If a Core ML or CUDA measurement wants fp16, the block-list candidates are LightGlue's assignment
head and ALIKED's DKD refinement. On this CPU fp16 is slower than fp32 (no half
arithmetic, casts everywhere), as in phase 2, so the twins only matter for the GPU providers (step 7).

CPU timings of column (c) on this machine, informational (first call includes warm-up): SuperPoint 48–137 ms,
ALIKED 445–552 ms, LightGlue 160–474 ms per pair. ALIKED is the slow one on the CPU; where its time goes
(`DeformConv`, `NonZero`/gathers, the 5×5 unfold) is a step 7 profiling question, and CUDA numbers come with the C++
side (step 4, T4).

## What the exporter taught (the exercises of the plan, as they happened)

1. **ALIKED patch corners** (step 2, found by (a) vs (b)): upstream truncates the keypoint to an integer *before*
   subtracting half the patch size; truncating after moved half the 3×3 patches by a pixel → descriptor cosine 0.98
   instead of 1. Integer truncation order is exactly what the exactness column exists for.
2. **Slice with symbolic bounds** → unbacked size. `map[..., pt:pt+H, pl:pl+W]` with a modulo in `pt` gave the result
   a size the exporter could not prove equal to H, and the next `max_pool2d` refused it ("could not guard on
   data-dependent expression"). `narrow(dim, start, length)` carries the length through as the known symbol.
3. **No tensors from sizes.** `torch.tensor([w-1, h-1])` would bake the trace size in; `torch._shape_as_tensor` cannot
   be traced ("data is not allocated yet"). Multiply each coordinate column by its symbolic scalar instead: a
   tensor × SymInt is a `Mul` fed by a `Shape` node. Keep upstream's operation order (`/ (w-1) * 2`, not
   `* (2/(w-1))`) or lose 6e-5 px of exactness to float32 rounding.
4. **No integer remainder by a symbol.** `indices % w` → the translation of `aten.remainder.Scalar` builds a constant
   from the divisor and fails on a symbol. `nonzero()` on the 2-D mask gives (row, col) directly, and row/col index
   tensors replace the flat-index arithmetic in the patch gather as well.
5. **Custom translation table works as advertised**: `{torch.ops.torchvision.deform_conv2d.default: fn}` with `fn`
   emitting `opset19.DeformConv(X, W, offset, B, dilations, group, offset_group, pads, strides)`; `torchvision` must be
   imported for the op namespace to exist; the modulation mask is dropped (ALIKED never uses it).
6. **fp16 conversion and casts**: onnxconverter-common re-declares the outputs of `Cast` nodes as float16 but leaves
   the `to` attribute of a cast whose *input* is not float (bool NMS masks, int indices) at float32; ORT refuses the
   file. `fix_cast_targets` makes every cast's `to` agree with the declared output type (2 casts in SuperPoint, 9 in
   ALIKED). `DeformConv` and `GridSample` stay float32 in the twins (ORT's CPU kernels are float-only); shape
   inference on (phase 2 had it off for the 912 MB graph).
8. **No rank-5 tensors with dynamic shapes if Core ML must run the graph** (step 7, found on the Mac): upstream's
   pixel shuffle `reshape(b,h,w,8,8).permute(0,1,3,2,4)` exports as a Reshape whose target comes from the dynamic
   size plus a 5-D Transpose; the Core ML provider cannot rank-infer that Reshape and refuses the session
   ("axis 4 is not in valid range [-4,3]"). `F.pixel_shuffle(scores, 8)` is the same mapping and exports as one
   4-D `DepthToSpace`: 83 nodes instead of 91, bit-identical on CPU and CUDA. The error survived that fix: its
   real source is the exporter's `Shape(start=k, end=k+1)` read of one symbolic size - ORT's Core ML provider
   (MLProgram) validates the exclusive `end` with `HandleNegativeAxis(end, rank)`, so reading the last dimension
   (`end == rank`) throws (`coreml/builders/impl/shape_op_builder.cc`, an ORT bug). `rewrite_shape_slices` turns
   every sliced Shape into Shape + Slice on the 1-D shape tensor (2 nodes in SuperPoint, 3 in ALIKED; the
   matchers have none); same values, exports still exact.
9. **Core ML compiles only bounded shapes, and only its operator subset** (step 7, Mac): with dynamic H, W every
   partition of the detector graphs fell back to the CPU ("unbounded dimension is not supported"), and the matcher
   with dynamic N0/N1 fails session creation outright (now caught: OrtSession falls back to the CPU provider).
   `--static WxH` exports fixed-size detector variants (`<stem>_<W>x<H>.onnx`). Even static, SuperPoint ran in
   8 Core ML partitions at ~118 ms whatever the precision or compute units: upstream's `simple_nms` is max-pools
   interleaved with Equal / Greater / Not / And / Or / Where, none of which the provider runs, so the NMS moved
   to the host tail (cv::dilate is the max-pool; `superpoint_tail` is the specification; still bit-exact, 2.5 ms
   on the CPU, and the CUDA graph got faster: 7.9 instead of 9.3 ms). The static SuperPoint graph is now 33 nodes
   of 11 Core ML-friendly op types.
7. **`torch.export` specialises 0/1 sizes**: the matcher axes are declared `Dim(min=2)`; the C++ side returns no
   matches below two keypoints per side (sidecar `min_keypoints: 2`).

## Open

- fp16 LightGlue: block-list the assignment head (`MatchAssignment` + log-softmax) and re-measure (d), only if a GPU
  provider wants fp16 (step 7).
- ALIKED CPU cost: profile per op (step 7, `ProfileComputePlan` on Core ML / ORT profiling on CPU).
- The SuperPoint tail in C++ (step 4) is specified by `superpoint_tail` in the tool; the SIFT extractor by
  `lightglue/sift.py` (OpenCV backend, RootSIFT, duplicate-orientation filter) — both to be mirrored exactly.
