#!/usr/bin/env python
"""Export the learned feature models to ONNX: SuperPoint, ALIKED and the three LightGlue matchers.

Author: Alejandro Fontan (with Claude, Fable 5.1)
Version: 0.2 (steps 1-2 of docs/plans/learned_features_onnxruntime.md: clone, weights, models, export wrappers)
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
    """SuperPoint up to the dense maps (plan Q4): the exported graph ends at the NMS-ed, border-masked score map and
    the L2-normalised descriptor map; threshold, top-k and descriptor sampling are `superpoint_tail` below in Python
    and the C++ tail in AllFeature. Every learned layer is upstream's; the three max-pools floor the size, so a
    739x458 image gives 736x456 maps with no explicit crop (the TensorRT path behaves the same today).
    Input: [1,1,H,W] grey in [0,1]. Outputs: scores [1,H8,W8] (border pixels -1), descriptors [1,256,H8/8,W8/8]."""

    def __init__(self, sp: nn.Module):
        super().__init__()
        self.sp = sp

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from lightglue.superpoint import simple_nms

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
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)  # 64 channels -> 8x8 cell pixels
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores = simple_nms(scores, sp.conf.nms_radius)
        scores = torch.where(border_mask(scores, sp.conf.remove_borders), scores, torch.full_like(scores, -1.0))

        descriptors = sp.convDb(sp.relu(sp.convDa(x)))
        descriptors = F.normalize(descriptors, p=2, dim=1)
        return scores, descriptors


def superpoint_tail(scores: torch.Tensor, descriptors: torch.Tensor, threshold: float,
                    max_keypoints: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """What runs after the SuperPoint graph, in the order upstream does it: threshold, best `max_keypoints` by score,
    bilinear descriptor sampling (upstream `sample_descriptors`, s=8, align_corners=True) and L2. This is the
    specification the C++ tail (step 4) is written against. Returns keypoints [N,2] (x,y), scores [N], descriptors
    [N,256]."""
    from lightglue.superpoint import sample_descriptors, top_k_keypoints

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


def get_patches(x: torch.Tensor, centers_wh: torch.Tensor, ps: int) -> torch.Tensor:
    """Upstream aliked.get_patches (the ps x ps feature patch around each keypoint, corners clamped into the map)
    with one flat Gather instead of slice assignment + 2-D advanced indexing. patch[n, :, a, b] = x[:, cy+a, cx+b].
    `centers_wh` is integer (upstream truncates the keypoint BEFORE the - ps/2 + 1 shift; truncating after would move
    half the patches by one pixel - caught by the (a) vs (b) check: descriptor cosine 0.98 instead of 1)."""
    c, h, w = x.shape
    corner = (centers_wh.to(x.dtype) - ps / 2 + 1).long()
    cx = corner[:, 0].clamp(min=0, max=w - 1 - ps)
    cy = corner[:, 1].clamp(min=0, max=h - 1 - ps)
    offs = torch.arange(ps, device=x.device)
    rows = cy[:, None] + offs[None, :]                             # [N, ps]
    cols = cx[:, None] + offs[None, :]
    flat = (rows[:, :, None] * w + cols[:, None, :]).reshape(-1)   # [N*ps*ps], row-major patches
    sampled = x.reshape(c, h * w)[:, flat]                         # [c, N*ps*ps]
    return sampled.reshape(c, -1, ps, ps).permute(1, 0, 2, 3)      # [N, c, ps, ps]


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
    indices = (nms.reshape(-1) > dkd.scores_th).nonzero()[:, 0]  # [N], row-major
    patch = dkd.unfold(scores_map)[0].t()                          # [H*W, k*k]
    patch_scores = patch[indices]                                  # [N, k*k]
    kp_nms = torch.stack([indices % w, torch.div(indices, w, rounding_mode="trunc")], 1)
    max_v = patch_scores.max(dim=1).values[:, None]
    x_exp = ((patch_scores - max_v) / dkd.temperature).exp()
    hw_grid = dkd.hw_grid.to(scores_map)
    xy_residual = x_exp @ hw_grid / x_exp.sum(dim=1)[:, None]      # soft-argmax within the k x k patch
    wh = torch.tensor([w - 1, h - 1], device=scores_map.device, dtype=scores_map.dtype)
    kp = (kp_nms + xy_residual) / wh * 2 - 1
    kp_scores = F.grid_sample(scores_map, kp.view(1, 1, -1, 2), mode="bilinear", align_corners=True)[0, 0, 0, :]
    return kp, kp_scores


def sddh(head: nn.Module, x: torch.Tensor, kp: torch.Tensor) -> torch.Tensor:
    """Upstream SDDH.forward for one image (kernel_size 3, n_pos 16, no mask, einsum aggregation): learned sampling
    offsets from the 3x3 patch around each keypoint, bilinear sampling of the feature map at the 16 positions,
    1x1 conv + SELU, weighted aggregation, L2. Returns descriptors [N, dim]."""
    c, h, w = x.shape
    wh = torch.tensor([[w - 1, h - 1]], device=x.device, dtype=x.dtype)
    max_offset = max(h, w) / 4.0
    kp_wh = (kp / 2 + 0.5) * wh
    patch = get_patches(x, kp_wh.long(), head.kernel_size)         # [N, c, k, k]
    offset = head.offset_conv(patch).clamp(-max_offset, max_offset)  # [N, 2*n_pos, 1, 1]
    offset = offset[:, :, 0, 0].reshape(-1, 2, head.n_pos).permute(0, 2, 1)  # [N, n_pos, 2]
    pos = (kp_wh.unsqueeze(1) + offset) * 2.0 / wh[None] - 1
    feats = F.grid_sample(x.unsqueeze(0), pos.reshape(1, -1, 1, 2), mode="bilinear", align_corners=True)
    feats = feats.reshape(c, -1, head.n_pos, 1).permute(1, 0, 2, 3)  # [N, c, n_pos, 1]
    feats = F.selu(head.sf_conv(feats)).squeeze(-1)                # [N, c, n_pos]
    descs = torch.einsum("ncp,pcd->nd", feats, head.agg_weights)
    return F.normalize(descs, p=2.0, dim=1)


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
        score_map = torch.sigmoid(m.score_head(x1234))[..., pt:pt + H, pl:pl + W]
        feature_map = F.normalize(x1234, p=2, dim=1)[..., pt:pt + H, pl:pl + W]

        kp, kp_scores = dkd_threshold(m.dkd, score_map)
        desc = sddh(m.desc_head, feature_map[0], kp)
        wh = torch.tensor([W - 1, H - 1], device=image.device, dtype=image.dtype)
        return wh * (kp + 1) / 2.0, desc, kp_scores


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
    """Same keypoint set? Prints count, max |dxy| px, max |dscore|, min descriptor cosine over the rows (in order
    when the counts agree, which they must: (b) re-runs upstream's own ops)."""
    ka, kb = a["keypoints"], b["keypoints"]
    if ka.shape != kb.shape:
        print(f"  {label:34s} COUNT MISMATCH: {tuple(ka.shape)} vs {tuple(kb.shape)}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", nargs="*", type=Path, default=[],
                        help="frames for the (a) vs (b) check (an even number; consecutive ones form the match pairs)")
    parser.add_argument("--max-keypoints", type=int, default=1000,
                        help="keypoint budget of the SuperPoint / SIFT references (FeatureExtractor.maxNumFeatures)")
    args = parser.parse_args()

    print(f"torch {torch.__version__} (CPU), TORCH_HOME={os.environ['TORCH_HOME']}")
    commit = ensure_lightglue_clone(LIGHTGLUE_COMMIT)
    print(f"cvg/LightGlue at {commit}")
    with torch.no_grad():
        models = load_models(args.max_keypoints)
        describe_models(models)
        if args.images:
            check_wrappers(models, args.images, args.max_keypoints)


if __name__ == "__main__":
    main()
