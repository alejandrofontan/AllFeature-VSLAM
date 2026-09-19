# Visual place recognition: `PlaceRecognition` interface + MegaLoc backend

- **Date:** 2026-08-28
- **Kind:** note
- **Provenance:** moved verbatim from `CLAUDE.md` § *Visual Place Recognition: `PlaceRecognition` interface + MegaLoc backend (2026-08-28)* on 2026-09-19 (documentation plan, `docs/plans/documentation_plan.md`). File/line references are as of the original date.


Issue #17 stages 2+3, on branch `cleanup` (uncommitted at the time of writing). Retrieval for
relocalization and loop detection now goes through one backend interface
(`include/PlaceRecognition.h`) selected by the settings key `vpr:`:

| `vpr:` | Class | Descriptor | Retrieval |
|---|---|---|---|
| `bow` (default) | `PlaceRecognitionBoW` | DBoW2 BoW vector of the `feature_vpr` local descriptors | the unchanged `KeyFrameDatabase` inverted file |
| `megaloc` | `PlaceRecognitionMegaLoc` | MegaLoc 8448-d image embedding (TensorRT, via the `Thirdparty/placecell` submodule's `placecell::MegaLocEmbedder`) | brute-force cosine over all keyframe descriptors + the same covisibility accumulation / 0.75-of-best pruning as BoW, capped by `PlaceRecognition.MaxCandidates` and floored by `PlaceRecognition.MegaLocMinSimilarity` |
| `none` | `PlaceRecognitionNone` | — | nothing (no loop closing, no relocalization) |

- `feature_vpr` now means "the local feature that geometrically verifies retrieved candidates"
  (reloc PnP, loop Sim3) for every backend; for `bow` it is still the vocabulary feature. When
  omitted: bow = first feature with a vocabulary, megaloc = first feature.
- Data flow for image-embedding backends (`needs_image()`): `Frame` keeps a shared header of its
  resized image (`Frame::image`, BGR), the `KeyFrame` inherits it, and
  `LocalMapping::ProcessNewKeyFrame` → `KeyFrame::compute_global_descriptor()` embeds it off the
  tracking thread and releases the image. The relocalization query (`Tracking::relocalize`) embeds
  the lost frame on the tracking thread. The engine is shared behind a mutex.
- `KeyFrame::vpr_similarity(other)` exposes the backend similarity (cosine for MegaLoc, NaN when
  VPR is inactive).
- **Keyframe culling (2026-08-30)** — `LocalMapping.KeyframeCullingMethod: heuristic | information`
  (compiled default `heuristic`; the dev YAML sets `information`). `cull_keyframes()` dispatches to
  `cull_keyframes_heuristic()` (ORB-SLAM2-style point redundancy, keys `KeyframeCullingRedundancyRatio`/
  `MinObservations`) or `cull_keyframes_information()` — the joint-information rule from the shelved
  VPR-matrix session (`git stash@{0}`), now on the **online** keyframe kernel: `ProcessNewKeyFrame` →
  `grow_keyframe_vpr_matrix()` appends the new keyframe's MegaLoc descriptor dot products (raw cosine,
  rows never removed — culled keyframes stay as history), and at each call the kernel is optionally
  double-centred over every keyframe inserted so far (`KeyframeCullingCentred`, default on = Pearson
  correlation of mean-centred descriptors, removes MegaLoc's ~0.37 common-mode floor). Keys
  `KeyframeCullingMaxUnexplained` (tau; live Viewer slider "Cull Max Unexplained"), `MinAge`,
  `MinKeyframes`, `Scope: map|local`, `MaxPerCall`. Needs `vpr: megaloc`; with none it falls back to
  heuristic with a one-time warning. The offline `VPRMatrix`/`vpr_matrix:` CLI path is gone for good;
  the maths/design history lives in `/home/alejandro/cull_keyframes_math.pdf` (rev. 5, outside the repo).
  **Offline matrices live in placecell now (2026-09-07):** `placecell::load_npy` +
  `similarity_from_distance(D, DistanceKind::squared_euclidean)` + `PlaceCell::set_kernel` turn
  VSLAM-LAB's `<sequence>/vpr-lab/D.npy` (faiss squared L2 on MegaLoc descriptors, min over four
  image rotations, so asymmetric; `S = 1 − D/2`, rows = `rgb.csv` rows = `frame_id`) into a
  kernel-only store that `cull_keyframes` can run on (`Thirdparty/placecell/examples/kernel_demo.cpp`).
  No `vpr: matrix` backend exists here yet: a kernel-only store cannot take new rows or answer
  insertion queries (placecell issue #2 was deliberately left for later), so wiring one up needs
  that first. placecell's `tools/colmap_information_kernel.py` (2026-09-07) builds the same kind of
  kernel from a COLMAP model in `VSLAM-LAB-Evaluation/<exp>/<dataset>/<sequence>/colmap_<id>/<best_model>`
  (normalised BA mutual information between images; pass the sub-model named in `best_model`).
- Model pipeline (reproducible, lives in the placecell submodule since 2026-09-02): `Thirdparty/placecell/tools/export_megaloc.py`
  clones `gmberton/MegaLoc` (pinned commit), downloads the HF weights into
  `megaloc_models/.cache`, exports `megaloc_models/megaloc_322x322.onnx` + sidecar yaml, and
  self-checks against the PyTorch reference; placecell's `megaloc_test` example validates the TensorRT engine + C++ preprocessing
  against that reference. `megaloc_models/` is gitignored; the VSLAM-LAB wrapper downloads the two
  files from HF `vslamlab/megaloc-models` on install.
- Export fidelity measured on ETH table_3 frames: export wrapper vs upstream forward max|diff| ≈ 2e-7;
  ONNX vs wrapper cos 0.999998; ONNX vs README-preprocessed (torchvision antialiased resize)
  reference cos ≈ 0.995 — the residual is the cv2 `INTER_AREA` vs torchvision resampling filter.
  TensorRT fp16 engine (mixed: fp16 backbone, fp32 aggregator) reproduces the fp32 numbers
  (worst cos 0.994) at 4.0 ms/image.
- **TensorRT 10.3 gotcha (bit us):** the upstream attention pattern
  `qkv.reshape(B,N,3,heads,hd).permute(2,0,3,1,4)[0..2]` is *miscompiled* by TensorRT's Myelin
  fuser — the full-graph engine returned descriptors with cos 0.05 vs ORT while every isolated
  sub-graph looked fine (exposing the intermediates changes the fusion). Diagnosed by bisecting with
  `trtexec --loadInputs=input:x.bin --exportOutput` against ORT stage by stage; fixed at export time
  by slicing q/k/v from the projection output (`ExportWrapper.attention`). Uniform fp16 was a
  separate, smaller loss (cos 0.93) fixed by pinning the aggregator to fp32.
