#!/usr/bin/env python
"""Export the learned feature models to ONNX: SuperPoint, ALIKED and the three LightGlue matchers.

Author: Alejandro Fontan (with Claude, Fable 5.1)
Version: 0.3 (steps 1-3 of docs/plans/learned_features_onnxruntime.md: clone, weights, wrappers, exports, references)
Created: 2026-10-16
Updated: 2026-10-16

The ONNX file is the contract between the Python side (this script, run once per export on Linux) and the C++ side
(ONNX Runtime in AllFeature, on Linux and macOS). Steps 2 and 3 of the plan extend this file with the export wrappers,
the exports, the sidecars and the references; step 1, implemented here, is only "get the models out of the git repo":

  1. clone cvg/LightGlue next to this script (tools/lightglue/, gitignored) and pin it to LIGHTGLUE_COMMIT. That one
     repository holds the matcher and PyTorch re-implementations of the three extractors, with the download URL of
     every weights file; the LightGlue weights were trained on exactly these extractor implementations.
  2. point TORCH_HOME at feature_models/.cache before torch is imported, so torch.hub's downloads land in the repo
     (never in ~/.cache), and resolve each weights file to a path + sha256 for the sidecars.
  3. construct the models in eval mode with every adaptive / device-dependent behaviour off: LightGlue's early exit
     (depth_confidence) and point pruning (width_confidence) are data-dependent control flow that cannot be a graph,
     flash attention and mixed precision are device-specific kernels - the export must be the plain 9-layer graph.

Step 2 (this version) adds the export wrappers: one nn.Module per model that calls the upstream layers unchanged and
rewrites only the glue the exporter or the runtimes cannot take (Python control flow on data, dict I/O, slice
assignment into empty tensors), plus the ONNX translation of torchvision's deformable convolution and the Python
`superpoint_tail` that is the written specification of what the C++ side does after the SuperPoint graph. With
`--images`, every wrapper is checked against the upstream forward in PyTorch (columns (a) vs (b) of the plan's
self-check table) before any ONNX file exists.

Step 3 (this version) exports the five ONNX files with the torch.export-based exporter (dynamic image size for the
detectors, dynamic keypoint counts for the matchers, the DeformConv translation for ALIKED), writes an fp16-weights
twin and a sidecar YAML per file, runs the exports on ONNX Runtime (CPU) as columns (c) fp32 and (d) fp16 of the
self-check, and dumps the upstream outputs on the given frames as the references the C++ parity test (step 4) reads.

Run from the repository root:  pixi run -e export-features export-features [--images f0 f1 f2 f3]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
LIGHTGLUE_DIR = SCRIPT_DIR / "lightglue"
MODELS_DIR = REPO_DIR / "feature_models"

# TORCH_HOME must be set before `import torch`: torch.hub reads it at import time to place its download cache
os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / ".cache"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision  # noqa: E402, F401  (registers torch.ops.torchvision.deform_conv2d, the op the translation below maps)

# --------------------------------------------------------------------------------------
# Sources, pinned
# --------------------------------------------------------------------------------------
LIGHTGLUE_GIT = "https://github.com/cvg/LightGlue.git"
LIGHTGLUE_COMMIT = "eb42fee2d71449efb0aa5c10549752b5d75384d8"  # main, 2026-02-18
LIGHTGLUE_RELEASE = "v0.1_arxiv"  # the GitHub release the upstream code downloads its .pth files from
ALIKED_COMMIT = "683d7c65197395c0b3f01ebe76e1084a27e73a65"  # Shiaoming/ALIKED main, 2024-05-05 (models/*.pth)

# What each model is, where its weights come from, and the licence that governs them (docs/plans/
# learned_features_onnxruntime.md, Q6: the SuperPoint weights are Magic Leap's, non-commercial research use only and
# no redistribution - the SuperPoint export is produced locally on every machine and never uploaded).
# `file` is the name under which the upstream constructor looks the file up in torch.hub's cache (superpoint.py and
# aliked.py: the URL's basename; lightglue.py: f"{weights}_{version with '.'->'-'}.pth") - download under that name
# and the constructors load from disk instead of downloading themselves.
_LG_URL = f"https://github.com/cvg/LightGlue/releases/download/{LIGHTGLUE_RELEASE}"
_LG_SUFFIX = LIGHTGLUE_RELEASE.replace(".", "-")
WEIGHTS = {
    "superpoint": {
        "url": f"{_LG_URL}/superpoint_v1.pth",
        "file": "superpoint_v1.pth",
        "licence": "Magic Leap Software License Agreement (non-commercial research, no redistribution)",
    },
    "aliked-n16": {
        "url": f"https://github.com/Shiaoming/ALIKED/raw/{ALIKED_COMMIT}/models/aliked-n16.pth",
        "file": "aliked-n16.pth",
        "licence": "BSD-3-Clause",
    },
    "lightglue_superpoint": {
        "url": f"{_LG_URL}/superpoint_lightglue.pth",
        "file": f"superpoint_lightglue_{_LG_SUFFIX}.pth",
        "licence": "Apache-2.0",
    },
    "lightglue_aliked": {
        "url": f"{_LG_URL}/aliked_lightglue.pth",
        "file": f"aliked_lightglue_{_LG_SUFFIX}.pth",
        "licence": "Apache-2.0",
    },
    "lightglue_sift": {
        "url": f"{_LG_URL}/sift_lightglue.pth",
        "file": f"sift_lightglue_{_LG_SUFFIX}.pth",
        "licence": "Apache-2.0",
    },
}

# LightGlue's per-extractor configuration (lightglue/lightglue.py `features`): the descriptor size the input
# projection expects, and whether the matcher takes per-keypoint scale + orientation (SIFT does: 4 inputs per side)
LIGHTGLUE_FEATURES = {
    "superpoint": {"input_dim": 256, "add_scale_ori": False},
    "aliked": {"input_dim": 128, "add_scale_ori": False},
    "sift": {"input_dim": 128, "add_scale_ori": True},
}


# --------------------------------------------------------------------------------------
# Step 1: clone, weights, models
# --------------------------------------------------------------------------------------
def ensure_lightglue_clone(commit: str) -> str:
    """Clone cvg/LightGlue next to this script (gitignored) and pin it to `commit`. Returns the checked-out hash."""
    if not (LIGHTGLUE_DIR / "lightglue" / "lightglue.py").exists():
        print(f"cloning {LIGHTGLUE_GIT} -> {LIGHTGLUE_DIR}")
        subprocess.run(["git", "clone", "--quiet", LIGHTGLUE_GIT, str(LIGHTGLUE_DIR)], check=True)
    subprocess.run(["git", "-C", str(LIGHTGLUE_DIR), "checkout", "--quiet", commit], check=True)
    head = subprocess.run(["git", "-C", str(LIGHTGLUE_DIR), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    sys.path.insert(0, str(LIGHTGLUE_DIR))
    return head


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_weights(name: str) -> Path:
    """Fetch one weights file into TORCH_HOME's hub cache (the same place and filename torch.hub uses, so the upstream
    constructors find it and download nothing themselves). Returns the local path."""
    url = WEIGHTS[name]["url"]
    cache = Path(torch.hub.get_dir()) / "checkpoints"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / WEIGHTS[name]["file"]
    if not path.exists():
        print(f"downloading {url}")
        torch.hub.download_url_to_file(url, str(path), progress=False)
    return path


def load_models(max_keypoints: int) -> dict[str, nn.Module]:
    """The five upstream models in eval mode, built for export. Keys: superpoint, aliked, sift (the extractors; sift is
    OpenCV-based and only produces references) and lightglue_{superpoint,aliked,sift} (the matchers).
    `max_keypoints` is the budget the C++ side applies (FeatureExtractor.maxNumFeatures): the upstream SuperPoint
    and SIFT references keep the best `max_keypoints` so that the (a) column matches what C++ will output; ALIKED is
    threshold mode (no cap), as the host runs it today."""
    from lightglue import ALIKED, SIFT, LightGlue, SuperPoint  # noqa: E402  (from the pinned clone)

    # The upstream constructors call torch.hub.load_state_dict_from_url: with the files already in the hub cache under
    # their release filenames they load from disk. The ALIKED port fetches from Shiaoming/ALIKED's main branch; we
    # fetch the pinned-commit file under the same filename so it is the one that gets loaded.
    for name in WEIGHTS:
        download_weights(name)

    models: dict[str, nn.Module] = {}
    # SuperPoint: upstream defaults (nms_radius 4, detection_threshold 0.0005, remove_borders 4); max_num_keypoints
    # left unset - the top-k lives in C++ (plan Q4: the graph ends at the dense maps)
    models["superpoint"] = SuperPoint(max_num_keypoints=max_keypoints).eval()
    # ALIKED n16: threshold mode as the host uses it today (detection_threshold 0.2, no top-k cap)
    models["aliked"] = ALIKED(model_name="aliked-n16", max_num_keypoints=-1, detection_threshold=0.2,
                              nms_radius=2).eval()
    # SIFT: OpenCV backend, RootSIFT - the reference for the C++ cv::SIFT extractor
    models["sift"] = SIFT(backend="opencv", rootsift=True, max_num_keypoints=max_keypoints).eval()
    # LightGlue x3, adaptive behaviour off: the graph must be the full 9 layers with no data-dependent exit or pruning
    for feat in LIGHTGLUE_FEATURES:
        models[f"lightglue_{feat}"] = LightGlue(features=feat, depth_confidence=-1, width_confidence=-1,
                                                flash=False, mp=False).eval()
    return models


def describe_models(models: dict[str, nn.Module]) -> None:
    print("\nmodels (eval mode, parameters, weights file + sha256):")
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        key = {"superpoint": "superpoint", "aliked": "aliked-n16", "sift": None}.get(name, name)
        if key is None:
            print(f"  {name:22s} {n_params:>12,d} params  (OpenCV cv2.SIFT, no weights)")
            continue
        path = download_weights(key)
        print(f"  {name:22s} {n_params:>12,d} params  {path.name} sha256={sha256_of(path)[:16]}...  "
              f"[{WEIGHTS[key]['licence']}]")
    lg = models["lightglue_sift"]
    print(f"\nlightglue_sift config: input_dim={lg.conf.input_dim} add_scale_ori={lg.conf.add_scale_ori} "
          f"n_layers={lg.conf.n_layers} depth_confidence={lg.conf.depth_confidence} "
          f"width_confidence={lg.conf.width_confidence} flash={lg.conf.flash}")


# --------------------------------------------------------------------------------------
# Step 2: export wrappers - the same maths as the upstream forwards, in a form the exporter and the runtimes take
# --------------------------------------------------------------------------------------
def border_mask(scores: torch.Tensor, pad: int) -> torch.Tensor:
    """Boolean [..., H, W] mask that is False within `pad` pixels of the border - the functional form of upstream's
    `scores[:, :pad] = -1` slice assignments, which trace into scatter nodes (the phase-2 Sinkhorn lesson)."""
    inner = torch.ones_like(scores[..., pad:-pad, pad:-pad], dtype=torch.bool)
    return F.pad(inner, (pad, pad, pad, pad), value=False)


class SuperPointDense(nn.Module):
    """SuperPoint up to the dense maps (plan Q4): the exported graph ends at the raw score map (softmax + pixel
    shuffle) and the L2-normalised descriptor map; NMS, border masking, threshold, top-k and descriptor sampling are
    `superpoint_tail` below in Python and the C++ tail in AllFeature. The NMS left the graph in step 7: upstream's
    `simple_nms` is max-pools interleaved with Equal / Greater / Not / And / Or / Where, which the Core ML provider
    does not run, so it cut the graph into 8 partitions (120 ms on the M1 Pro GPU regardless of precision); the
    trunk alone is one partition. Every learned layer is upstream's; the three max-pools floor the size, so a
    739x458 image gives 736x456 maps with no explicit crop (the TensorRT path behaves the same today).
    Input: [1,1,H,W] grey in [0,1]. Outputs: scores [1,H8,W8], descriptors [1,256,H8/8,W8/8]."""

    def __init__(self, sp: nn.Module):
        super().__init__()
        self.sp = sp

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sp = self.sp
        x = sp.relu(sp.conv1a(image))
        x = sp.relu(sp.conv1b(x))
        x = sp.pool(x)
        x = sp.relu(sp.conv2a(x))
        x = sp.relu(sp.conv2b(x))
        x = sp.pool(x)
        x = sp.relu(sp.conv3a(x))
        x = sp.relu(sp.conv3b(x))
        x = sp.pool(x)
        x = sp.relu(sp.conv4a(x))
        x = sp.relu(sp.conv4b(x))

        scores = sp.convPb(sp.relu(sp.convPa(x)))
        scores = F.softmax(scores, 1)[:, :-1]                       # drop the "no keypoint" dustbin channel
        # 64 channels -> the 8x8 pixels of each cell. Upstream writes this as reshape(b,h,w,8,8).permute(0,1,3,2,4):
        # rank-5 tensors whose Reshape target is computed from the dynamic size, which the Core ML provider cannot
        # rank-infer (it rejected the permutation of axis 4 at session creation). pixel_shuffle is the same mapping
        # (channel k -> row k // 8, col k % 8, one output channel) and exports as one 4-D DepthToSpace node.
        scores = F.pixel_shuffle(scores, 8)[:, 0]                   # [b, 1, 8h, 8w] -> [b, 8h, 8w]

        descriptors = sp.convDb(sp.relu(sp.convDa(x)))
        descriptors = F.normalize(descriptors, p=2, dim=1)
        return scores, descriptors


def superpoint_tail(scores: torch.Tensor, descriptors: torch.Tensor, threshold: float, max_keypoints: int,
                    nms_radius: int = 4, remove_borders: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """What runs after the SuperPoint graph, in the order upstream does it: NMS (`simple_nms`, radius 4: a pixel
    survives if nothing within the radius beats it, two suppression passes), border masking (-1 within
    `remove_borders` px), threshold, best `max_keypoints` by score, bilinear descriptor sampling (upstream
    `sample_descriptors`, s=8, align_corners=True) and L2. This is the specification the C++ tail (step 4) is written
    against (cv::dilate is the max-pool). Returns keypoints [N,2] (x,y), scores [N], descriptors [N,256]."""
    from lightglue.superpoint import sample_descriptors, simple_nms, top_k_keypoints

    scores = simple_nms(scores, nms_radius)
    scores = torch.where(border_mask(scores, remove_borders), scores, torch.full_like(scores, -1.0))
    best = torch.where(scores[0] > threshold)
    kp_scores = scores[0][best]
    keypoints = torch.stack(best, dim=-1)                          # (row, col)
    keypoints, kp_scores = top_k_keypoints(keypoints, kp_scores, max_keypoints)
    keypoints = torch.flip(keypoints, [1]).float()                 # -> (x, y)
    desc = sample_descriptors(keypoints[None], descriptors, 8)[0].t()
    return keypoints, kp_scores, desc


def log_assignment(head: nn.Module, desc0: torch.Tensor, desc1: torch.Tensor) -> torch.Tensor:
    """Upstream MatchAssignment.forward + sigmoid_log_double_softmax with the [b,m+1,n+1] matrix assembled by
    torch.cat instead of slice assignment into new_full() (exported as is: ScatterND). Same numbers."""
    mdesc0, mdesc1 = head.final_proj(desc0), head.final_proj(desc1)
    d = mdesc0.shape[-1]
    mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
    sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
    z0, z1 = head.matchability(desc0), head.matchability(desc1)     # [b,m,1], [b,n,1]
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    inner = scores0 + scores1 + certainties                        # [b,m,n]
    col = F.logsigmoid(-z0)                                        # [b,m,1]  "unmatched in image 1"
    row = F.logsigmoid(-z1).transpose(1, 2)                        # [b,1,n]  "unmatched in image 0"
    corner = torch.zeros_like(inner[:, :1, :1])
    return torch.cat([torch.cat([inner, col], 2), torch.cat([row, corner], 2)], 1)


class LightGlueExport(nn.Module):
    """LightGlue for one pair, fixed depth: input projection, positional encoding of the already-normalised keypoints
    (the C++ side normalises with the real image size: centre and half the long side), the 9 transformer layers
    without masks, and the last layer's assignment head. Output: the [1,N0+1,N1+1] log-assignment with the dustbin
    row/column; mutual-NN + exp(score) > filter_threshold (upstream `filter_matches`) is the C++ side's job.
    Dropped on purpose: early exit and point pruning (data-dependent Python control flow), the empty-set branch (C++
    guards N >= 2), flash/mixed precision (device kernels). With add_scale_ori the positional input is
    [x, y, scale, orientation] (SIFT), hence `LightGlueScaleOriExport` with four inputs per side."""

    def __init__(self, lg: nn.Module):
        super().__init__()
        self.lg = lg

    def match(self, pos0: torch.Tensor, desc0: torch.Tensor, pos1: torch.Tensor, desc1: torch.Tensor) -> torch.Tensor:
        lg = self.lg
        desc0, desc1 = lg.input_proj(desc0), lg.input_proj(desc1)
        enc0, enc1 = lg.posenc(pos0), lg.posenc(pos1)
        for layer in lg.transformers:
            desc0, desc1 = layer(desc0, desc1, enc0, enc1)
        return log_assignment(lg.log_assignment[-1], desc0, desc1)

    def forward(self, keypoints0: torch.Tensor, descriptors0: torch.Tensor,
                keypoints1: torch.Tensor, descriptors1: torch.Tensor) -> torch.Tensor:
        return self.match(keypoints0, descriptors0, keypoints1, descriptors1)


class LightGlueScaleOriExport(LightGlueExport):
    def forward(self, keypoints0: torch.Tensor, scales0: torch.Tensor, oris0: torch.Tensor,  # type: ignore[override]
                descriptors0: torch.Tensor, keypoints1: torch.Tensor, scales1: torch.Tensor,
                oris1: torch.Tensor, descriptors1: torch.Tensor) -> torch.Tensor:
        pos0 = torch.cat([keypoints0, scales0.unsqueeze(-1), oris0.unsqueeze(-1)], -1)
        pos1 = torch.cat([keypoints1, scales1.unsqueeze(-1), oris1.unsqueeze(-1)], -1)
        return self.match(pos0, descriptors0, pos1, descriptors1)


def deform_conv2d_onnx(input, weight, offset, mask, bias, stride_h: int, stride_w: int, pad_h: int, pad_w: int,
                       dilation_h: int, dilation_w: int, groups: int, offset_groups: int, use_mask: bool):
    """ONNX translation of torchvision::deform_conv2d (the op the exporter has no rule for): one DeformConv node
    (ai.onnx opset 19), implemented natively by ONNX Runtime's CPU and CUDA providers. ALIKED never uses the
    modulation mask (use_mask False, `mask` is torchvision's empty placeholder), so it is not forwarded; the
    kernel_shape attribute is left for the runtime to read from W."""
    from onnxscript import opset19 as op

    return op.DeformConv(input, weight, offset, bias, dilations=[dilation_h, dilation_w], group=groups,
                         offset_group=offset_groups, pads=[pad_h, pad_w, pad_h, pad_w], strides=[stride_h, stride_w])


DEFORM_CONV_TRANSLATION = {torch.ops.torchvision.deform_conv2d.default: deform_conv2d_onnx}


def scale_xy(t: torch.Tensor, sx, sy) -> torch.Tensor:
    """[..., 2] (x, y) tensor with x scaled by `sx` and y by `sy`, the scalars being Python numbers or symbolic sizes
    (W - 1, 2 / (H - 1), ...). The export-friendly way to apply image-size factors: a tensor times a symbolic scalar is
    a Mul fed by a Shape node, so the graph stays valid for any image size. torch.tensor([w, h]) would bake the trace
    size in, and torch._shape_as_tensor cannot be traced by torch.export."""
    return torch.stack([t[..., 0] * sx, t[..., 1] * sy], -1)


def div_xy(t: torch.Tensor, dx, dy) -> torch.Tensor:
    """scale_xy's division twin, so upstream's `xy / wh * 2 - 1` keeps its operation order (bit-exact in float32;
    a single multiply by 2 / (W - 1) differs by ~6e-5 px)."""
    return torch.stack([t[..., 0] / dx, t[..., 1] / dy], -1)


def get_patches(x: torch.Tensor, centers_wh: torch.Tensor, ps: int) -> torch.Tensor:
    """Upstream aliked.get_patches (the ps x ps feature patch around each keypoint, corners clamped into the map)
    with row/col index tensors instead of slice assignment. patch[n, :, a, b] = x[:, cy+a, cx+b].
    `centers_wh` is integer (upstream truncates the keypoint BEFORE the - ps/2 + 1 shift; truncating after would move
    half the patches by one pixel - caught by the (a) vs (b) check: descriptor cosine 0.98 instead of 1)."""
    c, h, w = x.shape
    corner = (centers_wh.to(x.dtype) - ps / 2 + 1).long()
    cx = corner[:, 0].clamp(min=0, max=w - 1 - ps)
    cy = corner[:, 1].clamp(min=0, max=h - 1 - ps)
    offs = torch.arange(ps, device=x.device)
    rows = cy[:, None] + offs[None, :]                             # [N, ps]
    cols = cx[:, None] + offs[None, :]
    sampled = x[:, rows[:, :, None], cols[:, None, :]]             # [c, N, ps, ps]
    return sampled.permute(1, 0, 2, 3)                             # [N, c, ps, ps]


def dkd_threshold(dkd: nn.Module, scores_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream DKD.forward in threshold mode with sub-pixel refinement, for one image, without the two
    data-dependent branches: the "nothing above threshold -> use the mean" fallback (a Python `if` on a tensor sum)
    and the n_limit cap (an `if` on the keypoint count; 20000 is never reached after NMS at SLAM resolutions, and the
    C++ side has its own budget). Returns keypoints in [-1,1] (x,y) [N,2] and their bilinear scores [N]."""
    from lightglue.aliked import simple_nms

    _, _, h, w = scores_map.shape
    r = dkd.radius
    nms = simple_nms(scores_map, r)
    nms = torch.where(border_mask(nms, r), nms, torch.zeros_like(nms))
    # (row, col) of every maximum above the threshold, row-major like upstream's flat nonzero. 2-D so that no
    # `index % w` / `index // w` on the symbolic width is needed (the ONNX translation of remainder-by-a-symbol fails).
    rc = (nms[0, 0] > dkd.scores_th).nonzero()                     # [N, 2]
    rows, cols = rc[:, 0], rc[:, 1]
    patch = dkd.unfold(scores_map)[0].reshape(-1, h, w)            # [k*k, H, W]
    patch_scores = patch[:, rows, cols].t()                        # [N, k*k]
    kp_nms = torch.stack([cols, rows], 1)                          # (x, y)
    max_v = patch_scores.max(dim=1).values[:, None]
    x_exp = ((patch_scores - max_v) / dkd.temperature).exp()
    hw_grid = dkd.hw_grid.to(scores_map)
    xy_residual = x_exp @ hw_grid / x_exp.sum(dim=1)[:, None]      # soft-argmax within the k x k patch
    kp = div_xy(kp_nms + xy_residual, w - 1, h - 1) * 2 - 1       # pixels -> [-1, 1], upstream's operation order
    kp_scores = F.grid_sample(scores_map, kp.view(1, 1, -1, 2), mode="bilinear", align_corners=True)[0, 0, 0, :]
    return kp, kp_scores


def sddh(head: nn.Module, x: torch.Tensor, kp: torch.Tensor) -> torch.Tensor:
    """Upstream SDDH.forward for one image (kernel_size 3, n_pos 16, no mask, einsum aggregation): learned sampling
    offsets from the 3x3 patch around each keypoint, bilinear sampling of the feature map at the 16 positions,
    1x1 conv + SELU, weighted aggregation, L2. Returns descriptors [N, dim]."""
    c, h, w = x.shape
    max_offset = torch.sym_max(h, w) / 4.0
    kp_wh = scale_xy(kp / 2 + 0.5, w - 1, h - 1)                   # [-1, 1] -> pixels
    patch = get_patches(x, kp_wh.long(), head.kernel_size)         # [N, c, k, k]
    offset = head.offset_conv(patch).clamp(-max_offset, max_offset)  # [N, 2*n_pos, 1, 1]
    offset = offset[:, :, 0, 0].reshape(-1, 2, head.n_pos).permute(0, 2, 1)  # [N, n_pos, 2]
    pos = div_xy(2.0 * (kp_wh.unsqueeze(1) + offset), w - 1, h - 1) - 1
    feats = F.grid_sample(x.unsqueeze(0), pos.reshape(1, -1, 1, 2), mode="bilinear", align_corners=True)
    feats = feats.reshape(c, -1, head.n_pos, 1).permute(1, 0, 2, 3)  # [N, c, n_pos, 1]
    feats = F.selu(head.sf_conv(feats)).squeeze(-1)                # [N, c, n_pos]
    descs = torch.einsum("ncp,pcd->nd", feats, head.agg_weights)
    return F.normalize(descs, p=2.0, dim=1)


def crop(x: torch.Tensor, top, height, left, width) -> torch.Tensor:
    """x[..., top:top+height, left:left+width] as two narrows (export-friendly: the result size is `height` x `width`)."""
    return x.narrow(-2, top, height).narrow(-1, left, width)


class AlikedExport(nn.Module):
    """ALIKED end to end for one RGB image in [0,1] of any size: replicate-pad to a multiple of 32 (upstream's
    InputPadder, split between both sides, computed from the input shape so it exports as shape arithmetic), the
    dense trunk with its two deformable-convolution blocks, crop the maps back, DKD detection, SDDH description.
    Outputs: keypoints [N,2] in pixels of the input frame, descriptors [N,128], scores [N]; N is data-dependent."""

    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m = self.m
        _, _, H, W = image.shape
        div = 32
        ph = (((H // div) + 1) * div - H) % div
        pw = (((W // div) + 1) * div - W) % div
        pl, pt = pw // 2, ph // 2
        x = F.pad(image, [pl, pw - pl, pt, ph - pt], mode="replicate")

        x1 = m.block1(x)
        x2 = m.block2(m.pool2(x1))
        x3 = m.block3(m.pool4(x2))                                 # deformable
        x4 = m.block4(m.pool4(x3))                                 # deformable
        x1 = m.gate(m.conv1(x1))
        x2 = m.gate(m.conv2(x2))
        x3 = m.gate(m.conv3(x3))
        x4 = m.gate(m.conv4(x4))
        x1234 = torch.cat([x1, m.upsample2(x2), m.upsample8(x3), m.upsample32(x4)], dim=1)
        # Crop the maps back with narrow(start, length): a slice `pt:pt+H` with the symbolic pad in both bounds gives the
        # exporter a result size it cannot prove equal to H (an unbacked symbol the following max-pool rejects);
        # narrow carries the length H through as is.
        score_map = crop(torch.sigmoid(m.score_head(x1234)), pt, H, pl, W)
        feature_map = crop(F.normalize(x1234, p=2, dim=1), pt, H, pl, W)

        kp, kp_scores = dkd_threshold(m.dkd, score_map)
        desc = sddh(m.desc_head, feature_map[0], kp)
        return scale_xy(kp + 1, W - 1, H - 1) / 2.0, desc, kp_scores


# --------------------------------------------------------------------------------------
# Step 2 check: wrapper (b) against upstream (a), in PyTorch, on real frames
# --------------------------------------------------------------------------------------
def load_image(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """One frame as the C++ side will feed it: grey [1,1,H,W] from cv::cvtColor(BGR2GRAY) / 255 and RGB [1,3,H,W]
    from the BGR file / 255. Both sides of every comparison get these same tensors, so the (a) vs (b) columns
    measure the wrappers, not the preprocessing."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    to_t = lambda a: torch.from_numpy(a.astype(np.float32) / 255.0)  # noqa: E731
    return to_t(grey)[None, None], to_t(rgb).permute(2, 0, 1)[None]


def compare_detections(label: str, a: dict, b: dict) -> None:
    """Same keypoint set? Equal counts (the exact columns): max |dxy| px, max |dscore|, min descriptor cosine over the
    rows in order. Different counts (fp16, later CoreML): nearest-neighbour matching of b's keypoints to a's - the
    fraction within 0.5 px and the descriptor cosine over those pairs (the T2 criterion of the plan)."""
    ka, kb = a["keypoints"], b["keypoints"]
    if ka.shape != kb.shape:
        d = torch.cdist(kb, ka)                                    # [Nb, Na]
        dmin, idx = d.min(dim=1)
        hit = dmin < 0.5
        cos = F.cosine_similarity(b["descriptors"][hit], a["descriptors"][idx[hit]], dim=1).min().item() if hit.any() else 0.0
        print(f"  {label:34s} N={kb.shape[0]:5d} vs {ka.shape[0]:5d}  within 0.5px: {int(hit.sum())}/{kb.shape[0]}"
              f"  max|dxy| matched={dmin[hit].max().item():.2e} px  min cos matched={cos:.7f}")
        return
    dxy = (ka - kb).abs().max().item() if ka.numel() else 0.0
    ds = (a["scores"] - b["scores"]).abs().max().item() if ka.numel() else 0.0
    cos = F.cosine_similarity(a["descriptors"], b["descriptors"], dim=1).min().item() if ka.numel() else 1.0
    print(f"  {label:34s} N={ka.shape[0]:5d}  max|dxy|={dxy:.2e} px  max|dscore|={ds:.2e}  min cos={cos:.7f}")


def upstream_detect(models: dict[str, nn.Module], name: str, grey: torch.Tensor, rgb: torch.Tensor) -> dict:
    """Column (a): the upstream forward, batch removed, keys keypoints [N,2] / scores [N] / descriptors [N,D]
    (+ scales / oris for SIFT)."""
    out = models[name]({"image": grey if name in ("superpoint", "sift") else rgb})
    res = {"keypoints": out["keypoints"][0], "descriptors": out["descriptors"][0],
           "scores": out["keypoint_scores"][0]}
    if name == "sift":
        res["scales"], res["oris"] = out["scales"][0], out["oris"][0]
    return res


def upstream_match(lg: nn.Module, f0: dict, f1: dict, size_wh: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    size = torch.tensor([size_wh], dtype=torch.float32)
    d0 = {"keypoints": f0["keypoints"][None], "descriptors": f0["descriptors"][None], "image_size": size}
    d1 = {"keypoints": f1["keypoints"][None], "descriptors": f1["descriptors"][None], "image_size": size}
    if lg.conf.add_scale_ori:
        for d, f in ((d0, f0), (d1, f1)):
            d["scales"], d["oris"] = f["scales"][None], f["oris"][None]
    out = lg({"image0": d0, "image1": d1})
    return out["matches0"][0], out["matching_scores0"][0]


def wrapper_match(lg: nn.Module, f0: dict, f1: dict, size_wh: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Column (b): normalise the keypoints as the C++ side will, run the wrapper, apply upstream filter_matches."""
    from lightglue.lightglue import filter_matches, normalize_keypoints

    size = torch.tensor([size_wh], dtype=torch.float32)
    k0 = normalize_keypoints(f0["keypoints"][None], size)
    k1 = normalize_keypoints(f1["keypoints"][None], size)
    if lg.conf.add_scale_ori:
        scores = LightGlueScaleOriExport(lg)(k0, f0["scales"][None], f0["oris"][None], f0["descriptors"][None],
                                             k1, f1["scales"][None], f1["oris"][None], f1["descriptors"][None])
    else:
        scores = LightGlueExport(lg)(k0, f0["descriptors"][None], k1, f1["descriptors"][None])
    m0, _, ms0, _ = filter_matches(scores, lg.conf.filter_threshold)
    return m0[0], ms0[0]


def check_wrappers(models: dict[str, nn.Module], images: list[Path], max_keypoints: int) -> None:
    """(a) vs (b) for every wrapper: the detectors on each frame, the matchers on consecutive pairs (0,1), (2,3), ..."""
    print(f"\n(a) upstream vs (b) export wrapper, in PyTorch, {len(images)} frames, top-k {max_keypoints}:")
    sp_dense, al_export = SuperPointDense(models["superpoint"]), AlikedExport(models["aliked"])
    feats: list[dict[str, dict]] = []
    for path in images:
        grey, rgb = load_image(path)
        H, W = grey.shape[-2:]
        print(f" {path.name} ({W}x{H})")
        fa = {name: upstream_detect(models, name, grey, rgb) for name in ("superpoint", "aliked", "sift")}
        scores, desc = sp_dense(grey)
        kp, s, d = superpoint_tail(scores, desc, models["superpoint"].conf.detection_threshold, max_keypoints)
        compare_detections("superpoint: dense graph + tail", fa["superpoint"], {"keypoints": kp, "scores": s, "descriptors": d})
        kp, d, s = al_export(rgb)
        compare_detections("aliked: full graph", fa["aliked"], {"keypoints": kp, "scores": s, "descriptors": d})
        print(f"  {'sift: reference only':34s} N={fa['sift']['keypoints'].shape[0]:5d}  (cv2.SIFT + RootSIFT, no wrapper)")
        feats.append(fa)
    for i in range(0, len(images) - 1, 2):
        grey, _ = load_image(images[i])
        H, W = grey.shape[-2:]
        print(f" pair {images[i].name} - {images[i + 1].name}")
        for name in ("superpoint", "aliked", "sift"):
            lg = models[f"lightglue_{name}"]
            m_a, s_a = upstream_match(lg, feats[i][name], feats[i + 1][name], (W, H))
            m_b, s_b = wrapper_match(lg, feats[i][name], feats[i + 1][name], (W, H))
            n_a = int((m_a >= 0).sum())
            same = int(((m_a == m_b)).sum())
            print(f"  {'lightglue_' + name:34s} matches={n_a:4d}  identical assignments={same}/{m_a.numel()}  "
                  f"max|dscore|={(s_a - s_b).abs().max().item():.2e}")


# --------------------------------------------------------------------------------------
# Step 3: export, fp16 twins, sidecars
# --------------------------------------------------------------------------------------
def print_graph_stats(model) -> None:
    """One line of graph anatomy: node count, initialisers, operator histogram - what the runtimes must support."""
    from collections import Counter

    ops = Counter(n.op_type for n in model.graph.node)
    print(f"  graph: {len(model.graph.node)} nodes, {len(model.graph.initializer)} initializers, {len(ops)} op types")
    print("  ops:   " + ", ".join(f"{k}x{v}" for k, v in sorted(ops.items(), key=lambda kv: -kv[1])))


def rewrite_shape_slices(model) -> int:
    """Shape(start=k, end=k+1) -> Shape() + Slice([k:k+1]) on the 1-D shape tensor. The exporter reads one symbolic
    size as a sliced Shape node; ONNX Runtime's Core ML provider (MLProgram) validates `end` with HandleNegativeAxis
    against the tensor rank, and `end` is exclusive, so reading the last dimension (end == rank) throws at session
    creation ("axis 4 is not in valid range [-4,3]", found on the Mac in step 7). The plain-Shape path has no such
    check; the values are identical. Returns the number of nodes rewritten."""
    from onnx import helper, numpy_helper

    count = 0
    new_nodes = []
    for node in model.graph.node:
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type != "Shape" or ("end" not in attrs and attrs.get("start", 0) == 0):
            new_nodes.append(node)
            continue
        start = int(attrs.get("start", 0))
        end = attrs.get("end", None)
        full = node.output[0] + "_full"
        new_nodes.append(helper.make_node("Shape", [node.input[0]], [full], name=node.name + "_full"))
        starts = numpy_helper.from_array(np.array([start], dtype=np.int64), node.output[0] + "_starts")
        ends = numpy_helper.from_array(np.array([end if end is not None else np.iinfo(np.int64).max], dtype=np.int64),
                                       node.output[0] + "_ends")
        model.graph.initializer.extend([starts, ends])
        new_nodes.append(helper.make_node("Slice", [full, starts.name, ends.name], [node.output[0]],
                                          name=node.name + "_slice"))
        count += 1
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return count   # the sliced outputs keep their names and their 1-D value_info


def export_onnx(wrapper: nn.Module, path: Path, sample: tuple, input_names: list[str], output_names: list[str],
                dynamic_shapes: tuple, opset: int, translation: dict | None = None):
    """Capture a wrapper with PyTorch's torch.export-based exporter (dynamo=True): torch.export records the program as
    an FX graph with symbolic sizes for the axes named in `dynamic_shapes`, onnxscript translates and optimises it.
    `translation` maps ops the exporter has no rule for (torchvision's deform_conv2d) to our onnxscript functions.
    external_data=False keeps the weights inside the single .onnx."""
    import onnx

    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper.eval()   # the wrapped upstream models already are; this is the wrapper module's own flag
    with torch.no_grad():
        torch.onnx.export(wrapper, sample, str(path), opset_version=opset, input_names=input_names,
                          output_names=output_names, dynamic_shapes=dynamic_shapes, dynamo=True,
                          external_data=False, optimize=True, custom_translation_table=translation)
    model = onnx.load(str(path), load_external_data=False)
    n_shape = rewrite_shape_slices(model)
    if n_shape:
        onnx.save(model, str(path))
    onnx.checker.check_model(str(path))

    def shp(v):
        return [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]

    print(f"exported: {path.name} ({path.stat().st_size / 2**20:.1f} MB, opset {opset}"
          f"{f', {n_shape} sliced Shape nodes rewritten' if n_shape else ''})")
    print(f"  inputs  {[(i.name, shp(i)) for i in model.graph.input]}")
    print(f"  outputs {[(o.name, shp(o)) for o in model.graph.output]}")
    print_graph_stats(model)
    return model


def ensure_float32_io(model) -> None:
    """keep_io_types=True keeps the inputs float32 but, when a graph output comes from a node left in fp32 (block
    list), onnxconverter-common can still declare the output float16 behind a Cast. The C++ contract is float32 I/O:
    re-point such casts (or add one) and fix the declared type. Phase 2's fix, unchanged."""
    import onnx
    from onnx import TensorProto

    producers = {out: n for n in model.graph.node for out in n.output}
    for output in model.graph.output:
        if output.type.tensor_type.elem_type != TensorProto.FLOAT16:
            continue
        node = producers.get(output.name)
        if node is not None and node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to":
                    attr.i = TensorProto.FLOAT
        else:
            half_name = output.name + "_fp16"
            if node is not None:
                node.output[list(node.output).index(output.name)] = half_name
            model.graph.node.append(onnx.helper.make_node(
                "Cast", [half_name], [output.name], name=output.name + "_to_fp32", to=TensorProto.FLOAT))
        output.type.tensor_type.elem_type = TensorProto.FLOAT
        print(f"  output '{output.name}' restored to float32")


def fix_cast_targets(model) -> int:
    """onnxconverter-common re-declares the outputs of the Cast nodes it meets as float16 but leaves the `to`
    attribute of a Cast whose INPUT is not float (bool masks -> float in the NMS) at float32; ORT then refuses the
    file ("Type (tensor(float16)) of output arg ... does not match expected type (tensor(float))"). Make every
    Cast's `to` agree with the type the graph declares for its output. Returns the number of casts fixed."""
    from onnx import TensorProto

    declared = {v.name: v.type.tensor_type.elem_type for v in list(model.graph.value_info) + list(model.graph.output)}
    fixed = 0
    for node in model.graph.node:
        if node.op_type != "Cast" or node.output[0] not in declared:
            continue
        want = declared[node.output[0]]
        if want not in (TensorProto.FLOAT, TensorProto.FLOAT16):
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i in (TensorProto.FLOAT, TensorProto.FLOAT16) and attr.i != want:
                attr.i = want
                fixed += 1
    return fixed


# Ops kept in float32 in the fp16 twins on top of onnxconverter-common's defaults (Resize, NonZero, TopK, ...):
# ONNX Runtime's DeformConv and GridSample kernels are float32-only on the CPU provider.
FP16_OP_BLOCK_EXTRA = ["DeformConv", "GridSample"]


def convert_fp16(fp32_path: Path, fp16_path: Path) -> None:
    """fp16-weights twin (onnxconverter-common): every float tensor and node to half except the blocked ops and the
    graph I/O, which stay float32 (the converter inserts the casts). Whether a model tolerates it is column (d)."""
    import warnings

    import onnx
    import onnxruntime as ort
    from onnxconverter_common import float16

    warnings.filterwarnings("ignore", message="the float32 number .* will be truncated")   # one line per tiny constant
    model = onnx.load(str(fp32_path))
    # shape inference ON (phase 2 disabled it for the 912 MB MegaLoc graph): the converter needs the intermediate
    # value types to keep the Cast nodes (bool masks, int indices) consistent, else ORT refuses the file
    model16 = float16.convert_float_to_float16(
        model, keep_io_types=True, op_block_list=float16.DEFAULT_OP_BLOCK_LIST + FP16_OP_BLOCK_EXTRA)
    ensure_float32_io(model16)
    n_cast = fix_cast_targets(model16)
    onnx.save(model16, str(fp16_path))
    onnx.checker.check_model(str(fp16_path))
    so = ort.SessionOptions()
    so.log_severity_level = 3
    ort.InferenceSession(str(fp16_path), so, providers=["CPUExecutionProvider"])   # loads, or raises with the name
    print(f"fp16 twin: {fp16_path.name} ({fp16_path.stat().st_size / 2**20:.1f} MB"
          f"{f', {n_cast} cast targets fixed' if n_cast else ''})")


def write_sidecar(onnx_path: Path, model_key: str, weights_key: str | None, commit: str, opset: int,
                  precision: str, contract: list[str]) -> None:
    """Hand-written YAML next to the .onnx (<name>.onnx.yaml, read by the C++ side via yaml-cpp): provenance (repo,
    commit, weights URL + sha256, licence), exporter, precision, and the preprocessing / post-processing contract."""
    lines = [
        f"# Sidecar for {onnx_path.name} - generated by tools/export_features.py",
        f"model: {model_key}",
        f"source: {LIGHTGLUE_GIT} @ {commit}   # lightglue/ package: the matcher and the extractor re-implementations",
    ]
    if weights_key is not None:
        w = WEIGHTS[weights_key]
        lines += [f"weights: {w['url']}",
                  f"weights_sha256: {sha256_of(download_weights(weights_key))}",
                  f"licence: {w['licence']}"]
    lines += [
        f"exporter: torch.export   # torch {torch.__version__}, opset {opset}",
        f"precision: {precision}   # weights/internal tensors; inputs and outputs are float32 either way",
    ]
    onnx_path.with_name(onnx_path.name + ".yaml").write_text("\n".join(lines + contract) + "\n")


def superpoint_contract(sp: nn.Module) -> list[str]:
    c = sp.conf
    return [
        "# Input: one grey image, [1,1,H,W] float32 in [0,1] (cv::cvtColor BGR2GRAY / 255), any H and W; the three",
        "# max-pools floor the size, so the maps cover the top-left (W//8*8) x (H//8*8) pixels of the image.",
        "input_tensor: image",
        "color: gray",
        "# Outputs: the dense maps. scores [1,H8,W8] after softmax and pixel shuffle (raw, no NMS: its boolean ops would",
        "# split the Core ML graph into 8 partitions); descriptors [1,256,H8/8,W8/8] L2-normalised. The graph ends here.",
        "output_tensors: [scores, descriptors]",
        "# Tail (C++, tools/export_features.py superpoint_tail is the specification): NMS with nms_radius (upstream",
        "# simple_nms: 9x9 max-pool == scores, two suppression passes), border pixels within remove_borders set to -1,",
        "# keep scores > detection_threshold, best max_keypoints (FeatureExtractor.maxNumFeatures) by score, sample the",
        "# descriptor map bilinearly at ((x,y) - 4 + 0.5) / (W8 - 4 - 0.5, H8 - 4 - 0.5) * 2 - 1 with align_corners=true,",
        "# then L2-normalise.",
        f"nms_radius: {c.nms_radius}   # host tail",
        f"remove_borders: {c.remove_borders}   # host tail",
        f"detection_threshold: {c.detection_threshold}",
        "descriptor_dim: 256",
        "descriptor_sampling: bilinear_align_corners_s8",
    ]


def aliked_contract(m: nn.Module) -> list[str]:
    c = m.conf
    return [
        "# Input: one RGB image, [1,3,H,W] float32 in [0,1] (cv::cvtColor BGR2RGB / 255), any H and W: the graph",
        "# replicate-pads to a multiple of 32 and crops the maps back, keypoints come back in the input frame.",
        "input_tensor: image",
        "color: rgb",
        "# Outputs: keypoints [N,2] (x,y) pixels, descriptors [N,128] L2-normalised, scores [N]; N data-dependent",
        "# (threshold mode: every NMS maximum above detection_threshold, no cap - the host applies its own budget).",
        "output_tensors: [keypoints, descriptors, scores]",
        f"model_name: {c.model_name}",
        f"detection_threshold: {c.detection_threshold}   # inside the graph",
        f"nms_radius: {c.nms_radius}   # inside the graph",
        "descriptor_dim: 128",
        "# Uses ai.onnx DeformConv (opset 19) in two blocks: native on ORT's CPU and CUDA providers, CPU fallback on CoreML.",
    ]


def lightglue_contract(lg: nn.Module, feat: str) -> list[str]:
    c = lg.conf
    sides = ["keypoints_0", "scales_0", "oris_0", "descriptors_0", "keypoints_1", "scales_1", "oris_1", "descriptors_1"] \
        if c.add_scale_ori else ["keypoints_0", "descriptors_0", "keypoints_1", "descriptors_1"]
    return [
        f"features: {feat}",
        f"input_dim: {c.input_dim}   # descriptor size the input projection expects",
        f"add_scale_ori: {'true' if c.add_scale_ori else 'false'}   # scales [1,N] (px) and oris [1,N] (rad) inputs per side",
        "# Inputs per side: keypoints [1,N,2] float32 NORMALISED by the host: (xy - size/2) / (max(W,H)/2) with the",
        "# real image size; descriptors [1,N,input_dim] float32 (L2-normalised as the extractor outputs them).",
        f"input_tensors: [{', '.join(sides)}]",
        "min_keypoints: 2   # per side; the host returns no matches below that (the graph has no empty-set branch)",
        "# Output: log_assignment [1,N0+1,N1+1] float32 (last row/column = unmatched dustbins). Matches (host, upstream",
        "# filter_matches): mutual row/column argmax over the [N0,N1] block, score = exp(value) > filter_threshold.",
        "output_tensor: log_assignment",
        f"filter_threshold: {c.filter_threshold}",
        f"n_layers: {c.n_layers}   # fixed depth: no early exit, no point pruning in the graph",
    ]


ALL_EXPORTS = ("superpoint", "aliked", "lightglue_superpoint", "lightglue_aliked", "lightglue_sift")


def export_all(models: dict[str, nn.Module], commit: str, only: set[str] | None = None,
               static_size: tuple[int, int] | None = None) -> dict[str, Path]:
    """The fp32 exports + fp16 twins + sidecars into feature_models/ (`only`: a subset of ALL_EXPORTS, e.g. the
    `export-superpoint` pixi task - SuperPoint is the one model every machine exports itself, plan Q6).
    `static_size` (W, H): export the two detectors with that fixed input size as '<stem>_<W>x<H>.onnx' instead of
    the dynamic-axis files - Core ML compiles only bounded shapes (step 7: with dynamic H, W every partition fell
    back to the CPU). Returns {key: fp32 path}."""
    from torch.export import Dim

    wanted = set(ALL_EXPORTS) if not only else only
    paths: dict[str, Path] = {}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if static_size:
        W, H = static_size
        suffix, dyn, size_lines = f"_{W}x{H}", None, [f"input_width: {W}", f"input_height: {H}",
                                                        "# static export: this size only (Core ML compiles only bounded shapes)"]
    else:
        W, H = 739, 458   # the sample is a table_3-sized frame; its values shape the trace, not the graph
        suffix, size_lines = "", []
        dyn = ({2: Dim("height", min=64), 3: Dim("width", min=64)},)

    # SuperPoint: dynamic H, W unless static_size
    if "superpoint" in wanted:
        path = MODELS_DIR / f"superpoint_v1{suffix}.onnx"
        export_onnx(SuperPointDense(models["superpoint"]), path, (torch.rand(1, 1, H, W),), ["image"],
                    ["scores", "descriptors"], dyn, opset=18)
        write_sidecar(path, "superpoint", "superpoint", commit, 18, "fp32",
                      superpoint_contract(models["superpoint"]) + size_lines)
        paths["superpoint"] = path

    # ALIKED: dynamic H, W unless static_size; opset 19 for DeformConv, reached through the custom translation table
    if "aliked" in wanted:
        path = MODELS_DIR / f"aliked_n16{suffix}.onnx"
        export_onnx(AlikedExport(models["aliked"]), path, (torch.rand(1, 3, H, W),), ["image"],
                    ["keypoints", "descriptors", "scores"], dyn, opset=19, translation=DEFORM_CONV_TRANSLATION)
        write_sidecar(path, "aliked", "aliked-n16", commit, 19, "fp32", aliked_contract(models["aliked"]) + size_lines)
        paths["aliked"] = path

    # LightGlue x3: dynamic N0, N1 (>= 2: the graph has no empty/single-keypoint branch and 0/1 sizes specialise)
    n0, n1 = Dim("n0", min=2), Dim("n1", min=2)
    for feat, cfg in LIGHTGLUE_FEATURES.items():
        if f"lightglue_{feat}" not in wanted or static_size:   # the matchers have no image-size axis
            continue
        lg = models[f"lightglue_{feat}"]
        path = MODELS_DIR / f"lightglue_{feat}.onnx"
        d = cfg["input_dim"]
        k0, k1 = torch.rand(1, 512, 2) * 2 - 1, torch.rand(1, 384, 2) * 2 - 1
        d0, d1 = F.normalize(torch.randn(1, 512, d), dim=-1), F.normalize(torch.randn(1, 384, d), dim=-1)
        if cfg["add_scale_ori"]:
            s0, o0 = torch.rand(1, 512) * 10, torch.rand(1, 512) * 6.28 - 3.14
            s1, o1 = torch.rand(1, 384) * 10, torch.rand(1, 384) * 6.28 - 3.14
            export_onnx(LightGlueScaleOriExport(lg), path, (k0, s0, o0, d0, k1, s1, o1, d1),
                        ["keypoints_0", "scales_0", "oris_0", "descriptors_0",
                         "keypoints_1", "scales_1", "oris_1", "descriptors_1"], ["log_assignment"],
                        ({1: n0}, {1: n0}, {1: n0}, {1: n0}, {1: n1}, {1: n1}, {1: n1}, {1: n1}), opset=18)
        else:
            export_onnx(LightGlueExport(lg), path, (k0, d0, k1, d1),
                        ["keypoints_0", "descriptors_0", "keypoints_1", "descriptors_1"], ["log_assignment"],
                        ({1: n0}, {1: n0}, {1: n1}, {1: n1}), opset=18)
        write_sidecar(path, "lightglue", f"lightglue_{feat}", commit, 18, "fp32", lightglue_contract(lg, feat))
        paths[f"lightglue_{feat}"] = path

    for key, path in paths.items():
        fp16 = path.with_name(path.stem + "_fp16.onnx")
        convert_fp16(path, fp16)
        opset = 19 if key == "aliked" else 18
        contract = {"superpoint": lambda: superpoint_contract(models["superpoint"]) + size_lines,
                    "aliked": lambda: aliked_contract(models["aliked"]) + size_lines}.get(
            key, lambda: lightglue_contract(models[key], key.split("_", 1)[1]))()
        weights_key = {"superpoint": "superpoint", "aliked": "aliked-n16"}.get(key, key)
        write_sidecar(fp16, key.split("_")[0], weights_key, commit, opset, "fp16", contract)
    return paths


# --------------------------------------------------------------------------------------
# Step 3 check: the exports on ONNX Runtime (CPU), columns (c) fp32 and (d) fp16, and the references
# --------------------------------------------------------------------------------------
class OrtModel:
    """One ONNX Runtime session on the CPU provider; run(**named numpy/torch inputs) -> list of torch tensors."""

    def __init__(self, path: Path):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.ms = 0.0

    def run(self, **inputs: torch.Tensor) -> list[torch.Tensor]:
        import time

        feed = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in inputs.items()}
        t0 = time.perf_counter()
        outs = self.session.run(None, feed)
        self.ms = (time.perf_counter() - t0) * 1e3
        return [torch.from_numpy(o) for o in outs]


def ort_detect(key: str, sess: OrtModel, grey: torch.Tensor, rgb: torch.Tensor, sp: nn.Module,
               max_keypoints: int) -> dict:
    if key == "superpoint":
        scores, desc = sess.run(image=grey)
        kp, s, d = superpoint_tail(scores, desc, sp.conf.detection_threshold, max_keypoints)
        return {"keypoints": kp, "scores": s, "descriptors": d}
    kp, d, s = sess.run(image=rgb)
    return {"keypoints": kp, "scores": s, "descriptors": d}


def ort_match(lg: nn.Module, sess: OrtModel, f0: dict, f1: dict, size_wh: tuple[int, int]):
    from lightglue.lightglue import filter_matches, normalize_keypoints

    size = torch.tensor([size_wh], dtype=torch.float32)
    k0 = normalize_keypoints(f0["keypoints"][None], size)
    k1 = normalize_keypoints(f1["keypoints"][None], size)
    feed = {"keypoints_0": k0, "descriptors_0": f0["descriptors"][None],
            "keypoints_1": k1, "descriptors_1": f1["descriptors"][None]}
    if lg.conf.add_scale_ori:
        feed.update(scales_0=f0["scales"][None], oris_0=f0["oris"][None],
                    scales_1=f1["scales"][None], oris_1=f1["oris"][None])
    (scores,) = sess.run(**feed)
    m0, _, ms0, _ = filter_matches(scores, lg.conf.filter_threshold)
    return m0[0], ms0[0]


def compare_matches(label: str, a: tuple, b: tuple, ms: float) -> None:
    m_a, s_a = a
    m_b, s_b = b
    print(f"  {label:34s} matches={int((m_a >= 0).sum()):4d}  identical assignments={int((m_a == m_b).sum())}/"
          f"{m_a.numel()}  max|dscore|={(s_a - s_b).abs().max().item():.2e}  {ms:7.1f} ms")


def check_ort(models: dict[str, nn.Module], paths: dict[str, Path], images: list[Path], max_keypoints: int) -> None:
    """(c) fp32 and (d) fp16 exports on ORT CPU against upstream (a); plus the fp32 detectors on a 640x480 resize of
    the first frame against upstream on the same resize (the dynamic axes at work)."""
    print(f"\n(a) upstream vs (c) ONNX fp32 / (d) ONNX fp16 on ONNX Runtime CPU, {len(images)} frames:")
    sess = {col: {k: OrtModel(p if col == "c" else p.with_name(p.stem + "_fp16.onnx")) for k, p in paths.items()}
            for col in ("c", "d")}
    sp = models["superpoint"]
    feats: list[dict[str, dict]] = []
    for path in images:
        grey, rgb = load_image(path)
        H, W = grey.shape[-2:]
        print(f" {path.name} ({W}x{H})")
        fa = {name: upstream_detect(models, name, grey, rgb) for name in ("superpoint", "aliked", "sift")}
        for key in ("superpoint", "aliked"):
            for col in ("c", "d"):
                fb = ort_detect(key, sess[col][key], grey, rgb, sp, max_keypoints)
                compare_detections(f"({col}) {key}", fa[key], fb)
                print(f"  {'':34s} {sess[col][key].ms:7.1f} ms")
        feats.append(fa)
    for i in range(0, len(images) - 1, 2):
        grey, _ = load_image(images[i])
        H, W = grey.shape[-2:]
        print(f" pair {images[i].name} - {images[i + 1].name}")
        for name in ("superpoint", "aliked", "sift"):
            lg, key = models[f"lightglue_{name}"], f"lightglue_{name}"
            ref = upstream_match(lg, feats[i][name], feats[i + 1][name], (W, H))
            for col in ("c", "d"):
                got = ort_match(lg, sess[col][key], feats[i][name], feats[i + 1][name], (W, H))
                compare_matches(f"({col}) {key}", ref, got, sess[col][key].ms)
    # dynamic axes: another size
    grey, rgb = load_image(images[0])
    g2 = F.interpolate(grey, size=(480, 640), mode="area")
    r2 = F.interpolate(rgb, size=(480, 640), mode="area")
    print(f" {images[0].name} resized to 640x480 (dynamic H, W):")
    for key in ("superpoint", "aliked"):
        fa = upstream_detect(models, key, g2, r2)
        fb = ort_detect(key, sess["c"][key], g2, r2, sp, max_keypoints)
        compare_detections(f"(c) {key}", fa, fb)


def write_references(models: dict[str, nn.Module], images: list[Path], commit: str) -> None:
    """Column (a), upstream PyTorch, dumped as text for the C++ parity test: per frame and detector the keypoints,
    scores and descriptors (SIFT also scale and orientation); per consecutive pair and matcher the match list."""
    ref_dir = MODELS_DIR / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    feats: list[dict[str, dict]] = []
    for path in images:
        grey, rgb = load_image(path)
        H, W = grey.shape[-2:]
        fa = {name: upstream_detect(models, name, grey, rgb) for name in ("superpoint", "aliked", "sift")}
        feats.append(fa)
        for name, f in fa.items():
            cols = ["x", "y", "score"] + (["scale", "ori"] if name == "sift" else []) + \
                   [f"d{i}" for i in range(f["descriptors"].shape[1])]
            extra = [f["scales"][:, None], f["oris"][:, None]] if name == "sift" else []
            rows = torch.cat([f["keypoints"], f["scores"][:, None], *extra, f["descriptors"]], 1).numpy()
            out = ref_dir / f"{path.stem}.{name}.txt"
            np.savetxt(out, rows, fmt="%.7g", header=f"model={name} source={LIGHTGLUE_COMMIT[:12]} image={path.name} "
                       f"size={W}x{H} N={rows.shape[0]} columns: {' '.join(cols)}")
    for i in range(0, len(images) - 1, 2):
        grey, _ = load_image(images[i])
        H, W = grey.shape[-2:]
        for name in ("superpoint", "aliked", "sift"):
            m0, ms0 = upstream_match(models[f"lightglue_{name}"], feats[i][name], feats[i + 1][name], (W, H))
            idx = torch.where(m0 >= 0)[0]
            rows = torch.stack([idx.float(), m0[idx].float(), ms0[idx]], 1).numpy()
            out = ref_dir / f"{images[i].stem}__{images[i + 1].stem}.lightglue_{name}.txt"
            np.savetxt(out, rows, fmt=["%d", "%d", "%.7g"], header=f"model=lightglue_{name} "
                       f"N0={feats[i][name]['keypoints'].shape[0]} N1={feats[i + 1][name]['keypoints'].shape[0]} "
                       f"size={W}x{H} matches={rows.shape[0]} columns: i0 i1 score")
    print(f"\nreferences: {len(list(ref_dir.glob('*.txt')))} files in {ref_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", nargs="*", type=Path, default=[],
                        help="frames for the (a) vs (b) check (an even number; consecutive ones form the match pairs)")
    parser.add_argument("--max-keypoints", type=int, default=1000,
                        help="keypoint budget of the SuperPoint / SIFT references (FeatureExtractor.maxNumFeatures)")
    parser.add_argument("--only", nargs="+", choices=ALL_EXPORTS, default=None,
                        help="export only these models (default: all five)")
    parser.add_argument("--static", metavar="WxH", default=None,
                        help="export the detectors with this fixed input size as <stem>_<W>x<H>.onnx (Core ML)")
    args = parser.parse_args()
    static_size = None
    if args.static:
        w, h = args.static.lower().split("x")
        static_size = (int(w), int(h))

    print(f"torch {torch.__version__} (CPU), TORCH_HOME={os.environ['TORCH_HOME']}")
    commit = ensure_lightglue_clone(LIGHTGLUE_COMMIT)
    print(f"cvg/LightGlue at {commit}")
    with torch.no_grad():
        models = load_models(args.max_keypoints)
        describe_models(models)
        if args.images:
            check_wrappers(models, args.images, args.max_keypoints)
        print()
        paths = export_all(models, commit, set(args.only) if args.only else None, static_size)
        if args.images and not args.only and not static_size:
            check_ort(models, paths, args.images, args.max_keypoints)
            write_references(models, args.images, commit)


if __name__ == "__main__":
    main()
