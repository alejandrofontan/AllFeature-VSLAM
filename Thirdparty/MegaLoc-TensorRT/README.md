# MegaLoc-TensorRT — online visual place recognition for AllFeature-VSLAM

[MegaLoc](https://github.com/gmberton/MegaLoc) (Berton & Masone, CVPRW 2025, MIT) is an
image-retrieval model — DINOv2 ViT-B/14 backbone + optimal-transport feature aggregation —
producing one 8448-d L2-normalised descriptor per image. AllFeature-VSLAM uses it as the
`vpr: megaloc` backend of its `PlaceRecognition` interface: keyframes are embedded once
(off the tracking thread, in `LocalMapping::ProcessNewKeyFrame`), stored in a flat
descriptor database, and relocalization / loop-closure candidates are retrieved by cosine
similarity instead of DBoW2 bag-of-words. The downstream geometric verification (feature
matching, PnP, Sim3) is unchanged and uses the local feature named by `feature_vpr`.

Same structure as `Thirdparty/Segmentation-TensorRT`: a Python export script produces a
self-contained ONNX + sidecar YAML; a small hand-written TensorRT wrapper builds/caches
the engine and runs inference; a standalone harness validates the C++ chain against the
PyTorch reference.

## 1. Model export (reproducible, one-off)

```bash
# from the AllFeature-VSLAM-DEV root, inside the pixi env
pixi run python Thirdparty/MegaLoc-TensorRT/convert2onnx/export_megaloc.py \
    --test_images /path/to/a.png /path/to/b.png ...
```

What it does (all inside the repo, nothing under `~/.cache`):

1. clones `gmberton/MegaLoc` into `convert2onnx/megaloc/` (gitignored), pinned to the
   commit recorded in the script (`MEGALOC_COMMIT`);
2. downloads the released weights (`gberton/MegaLoc` on HF, `model.safetensors`) into
   `megaloc_models/.cache/huggingface/`;
3. exports `megaloc_models/megaloc_322x322.onnx` — input `input` `[1,3,322,322]` float32
   (already normalised), output `descriptor` `[1,8448]` float32 — plus the sidecar
   `megaloc_322x322.onnx.yaml` (input size, mean/std, tensor names, descriptor dim, source
   commit) that the C++ side reads instead of hardcoding anything;
4. with `--test_images`, prints for every image the cosine between the ONNX descriptor and
   the PyTorch reference (README preprocessing), the pairwise similarity matrix, and writes
   `megaloc_models/megaloc_322x322_reference.txt` for `test_megaloc`.

322×322 is MegaLoc's own evaluation preprocessing (ImageNet normalisation + antialiased
resize). The ONNX is traced at a fixed resolution; re-run with `--resolution H W`
(multiples of 14) to produce another file.

Export notes (see `ExportWrapper` in the script): the positional-embedding interpolation is
precomputed for the export resolution and stored as a constant; the Sinkhorn solver is
written with concatenations instead of slice writes into `torch.empty()`; and attention's
q/k/v are three slices of the QKV projection instead of upstream's 5-D
`reshape→permute→index` pattern — **TensorRT 10.3's Myelin fuser miscompiles that pattern**
(block output cos 0.75 vs ONNX Runtime, found by bisecting with `trtexec --loadInputs`),
the sliced form matches to 1e-3. All three are numerically identical to upstream in PyTorch
(the self-check reports the max abs difference, ~1e-6).

The engine cache is keyed by path + precision only; the loader rebuilds when the ONNX is
newer than the cached engine, so re-exporting never serves a stale engine.

## 2. C++ module

- `include/tensorrt_megaloc.hpp` / `src/tensorrt_megaloc.cpp` — `megaloc::TensorRTMegaLoc`:
  ONNX → TensorRT engine, cached next to the ONNX as `<onnx>.<precision>.engine` (built on
  first use, ~25–60 s; loaded in ms afterwards); `infer(cv::Mat BGR) -> std::vector<float>`
  (re-normalised on the host). Preprocessing mirrors `preprocess_cv2()` in the export
  script exactly. `fp16` is **mixed precision**: the ViT backbone runs fp16, every layer
  after it (Sinkhorn, cluster aggregation, the 16640→8448 linear) is pinned to fp32 — a
  uniformly-fp16 engine lost 2–7% cosine against the fp32 reference, the mixed one loses
  nothing measurable.

Measured on an RTX 4000 Ada (ETH `table_3` frames, cosine vs the PyTorch reference):

| engine | ms/image | cos vs reference (worst of 6) |
|---|---|---|
| ONNX Runtime (fp32, sanity) | — | 0.994 |
| TensorRT fp32 | 7.9 | 0.994 |
| TensorRT fp16 (mixed, default) | 4.0 | 0.994 |
| TensorRT fp16 uniform (rejected) | 3.9 | 0.929 |
- `src/test_megaloc.cpp` — standalone harness:

```bash
./bin/test_megaloc megaloc_models/megaloc_322x322.onnx \
    img1.png img2.png ... --reference megaloc_models/megaloc_322x322_reference.txt
```

prints engine build/load time, per-image and steady-state inference time, the pairwise
cosine matrix, and `cosine(TensorRT, PyTorch reference)` per image.

## 3. SLAM integration (settings YAML)

```yaml
vpr: "megaloc"                 # "bow" | "megaloc" | "none"
feature_vpr: "orb32"           # local feature that verifies retrieved candidates (default: first feature)
megaloc_onnx: megaloc_models/megaloc_322x322.onnx   # default
megaloc_precision: "fp16"      # or "fp32"
PlaceRecognition.MegaLocMinSimilarity: 0.55   # cosine floor for a keyframe to be a candidate
PlaceRecognition.MaxCandidates: 10            # cap on candidates returned per query
```

Cost: one inference per keyframe (LocalMapping thread) and one per lost frame during
relocalization (tracking thread); the engine is shared behind a mutex. Retrieval is a
brute-force dot product over all keyframe descriptors (2500 keyframes × 8448 ≈ 21 MFLOP).

The VSLAM-LAB wrapper (`Baselines/baseline_files/baseline_allfeature.py`, dev class)
downloads the exported ONNX + sidecar from the `vslamlab/allfeature-vslamlab` HF repo on
install, like the segmentation model.
