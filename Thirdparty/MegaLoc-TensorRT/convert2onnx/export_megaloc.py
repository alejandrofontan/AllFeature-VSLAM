"""
Module: AllFeature-VSLAM - MegaLoc-TensorRT - export_megaloc.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5)
- Version: 1.0
- Created: 2026-08-28
- Updated: 2026-08-28
- License: GPLv3 License (this script; MegaLoc itself is MIT)

Exports MegaLoc (gmberton/MegaLoc: DINOv2 ViT-B/14 backbone + optimal-transport
feature aggregation, 8448-d L2-normalised global descriptor) to a self-contained ONNX
model for the online visual-place-recognition backend of AllFeature-VSLAM
(`vpr: megaloc`, see Thirdparty/MegaLoc-TensorRT/README.md). Reproducible end to end:

  1. clones the upstream repo as a sibling 'megaloc/' folder, pinned to --commit;
  2. downloads the released weights (HF 'gberton/MegaLoc', model.safetensors) into the
     repo-local cache '<repo>/megaloc_models/.cache' (never touches ~/.cache);
  3. re-expresses three parts of the graph in an equivalent, export/TensorRT-friendly
     form (see ExportWrapper): the positional-embedding interpolation is precomputed for
     the fixed export resolution, the Sinkhorn solver is written with concatenations
     instead of in-place slice writes into torch.empty(), and the attention's q/k/v are
     taken as three slices of the QKV projection instead of the upstream 5-D
     reshape->permute->index pattern, which TensorRT 10.3 (Myelin) miscompiles;
  4. exports [1,3,H,W] float32 (already normalised) -> [1,8448] float32 'descriptor';
  5. writes the sidecar '<export>.yaml' with everything the C++ side must know but
     never hardcode (input size, normalisation, tensor names, descriptor dim, source
     commit);
  6. optionally (--test_images) runs a numerical self-check: original torch model vs
     the export wrapper vs ONNX Runtime on the exported file, and writes the torch
     reference descriptors to '<export_stem>_reference.txt' for test_megaloc (C++).

Resolution lock: the ONNX is traced at one resolution (default 322x322 = MegaLoc's own
evaluation preprocessing, README) and is only valid at that size.

Usage (from the repo root, inside the pixi env):
  pixi run python Thirdparty/MegaLoc-TensorRT/convert2onnx/export_megaloc.py
  pixi run python Thirdparty/MegaLoc-TensorRT/convert2onnx/export_megaloc.py \
      --test_images img1.png img2.png ...
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MEGALOC_DIR = SCRIPT_DIR / "megaloc"
REPO_ROOT = SCRIPT_DIR.parents[2]  # convert2onnx -> MegaLoc-TensorRT -> Thirdparty -> repo
MODELS_DIR = REPO_ROOT / "megaloc_models"

# Keep every download inside the repo (HF_HOME must be set BEFORE huggingface_hub is
# imported, hence the placement before the heavy imports).
os.environ.setdefault("HF_HOME", str(MODELS_DIR / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

MEGALOC_GIT = "https://github.com/gmberton/MegaLoc.git"
MEGALOC_COMMIT = "5fe0dd697c4a70ba3e23607f6716ab3c606b16db"  # main, 2026-08-28
HF_REPO = "gberton/MegaLoc"
HF_WEIGHTS = "model.safetensors"

# ImageNet normalisation, applied OUTSIDE the graph after scaling pixels to [0,1]
# (MegaLoc README: tfm.Normalize(mean, std) then Resize(322, antialias=True)).
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------------------
# Upstream code acquisition
# --------------------------------------------------------------------------------------
def ensure_megaloc_clone(commit: str) -> str:
    """Clone gmberton/MegaLoc next to this script (gitignored) and pin it to `commit`.
    Returns the checked-out commit hash."""
    if not (MEGALOC_DIR / "megaloc_model.py").exists():
        print(f"cloning {MEGALOC_GIT} -> {MEGALOC_DIR}")
        subprocess.run(["git", "clone", "--quiet", MEGALOC_GIT, str(MEGALOC_DIR)], check=True)
    subprocess.run(["git", "-C", str(MEGALOC_DIR), "checkout", "--quiet", commit], check=True)
    head = subprocess.run(["git", "-C", str(MEGALOC_DIR), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    return head


def load_pretrained_megaloc() -> nn.Module:
    """Equivalent of torch.hub.load('gmberton/MegaLoc', 'get_trained_model') without
    torch.hub (which would write to ~/.cache/torch): import the model file from the
    pinned clone and load the released safetensors from the repo-local HF cache."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    sys.path.insert(0, str(MEGALOC_DIR))
    from megaloc_model import MegaLoc  # noqa: E402

    weights = hf_hub_download(repo_id=HF_REPO, filename=HF_WEIGHTS)
    model = MegaLoc()
    model.load_state_dict(load_file(weights))
    model.eval()
    print(f"weights: {weights}")
    return model


# --------------------------------------------------------------------------------------
# Export wrapper: same maths as megaloc_model.py, trace-friendly form
# --------------------------------------------------------------------------------------
class ExportWrapper(nn.Module):
    """Fixed-resolution, ONNX-friendly re-expression of MegaLoc.forward.

    Differences from upstream are purely structural (validated numerically by
    --test_images, expected max |diff| ~1e-6):
      * MegaLoc.forward's "resize if not a multiple of 14" branch is dropped - the
        export resolution is required to be a multiple of 14 (322 = 23 * 14);
      * DINOv2.interpolate_pos_encoding (bicubic Resize of the 37x37 grid to the
        export grid) is evaluated ONCE here with torch's own bicubic kernel and stored
        as a constant, instead of exporting a Resize op whose cubic flavour TensorRT
        would have to reproduce bit-for-bit;
      * FeatureAggregator's Sinkhorn (get_matching_probs / log_otp_solver) is written
        with torch.cat instead of slice-assignment into torch.empty(), and without the
        batch-ambiguous .squeeze();
      * MultiHeadAttention splits q/k/v as three slices of the QKV projection
        ([B,N,3C] -> 3 x [B,N,C] -> [B,heads,N,hd]) instead of upstream's
        reshape(B,N,3,heads,hd).permute(2,0,3,1,4)[0..2]. Same maths, but the upstream
        pattern is MISCOMPILED by TensorRT 10.3's Myelin fuser (block output cos 0.75 vs
        ONNX Runtime; exposing the qkv tensor even crashes the builder with "No Myelin
        Error exists"), while the sliced form matches to 1e-3. Found by bisecting the
        graph stage by stage with trtexec --loadInputs/--exportOutput against ORT.
    """

    def __init__(self, model: nn.Module, height: int, width: int):
        super().__init__()
        if height % 14 or width % 14:
            raise ValueError("export resolution must be a multiple of the 14-px patch size")
        self.backbone = model.backbone
        self.aggregator = model.aggregator  # .agg (FeatureAggregator) + .linear
        self.h, self.w = height, width
        self.grid_h, self.grid_w = height // 14, width // 14

        bb = self.backbone
        with torch.no_grad():
            n_tokens = self.grid_h * self.grid_w + 1
            dummy = torch.zeros(1, n_tokens, bb.embed_dim)
            # Upstream calls interpolate_pos_encoding(x, H, W) with H in the 'w' slot -
            # replicate the exact call so non-square exports match upstream too.
            pos = bb.interpolate_pos_encoding(dummy, height, width)
        self.register_buffer("pos_embed", pos.clone())

        agg = self.aggregator.agg
        m, n = agg.num_clusters, self.grid_h * self.grid_w
        norm = -math.log(n + m)
        log_a = torch.full((m + 1,), norm)
        log_a[-1] = norm + math.log(n - m)
        self.register_buffer("log_a", log_a.unsqueeze(0))               # [1, m+1]
        self.register_buffer("log_b", torch.full((1, n), norm))         # [1, n]
        self.sinkhorn_norm = norm

    def sinkhorn_log_probs(self, S: torch.Tensor) -> torch.Tensor:
        """get_matching_probs(S, dust_bin, num_iters=3, reg=1.0) minus its dustbin row."""
        agg = self.aggregator.agg
        B, m, n = S.shape
        dust = agg.dust_bin.reshape(1, 1, 1).expand(B, 1, n)
        M = torch.cat([S, dust], dim=1)  # reg = 1.0 -> M / reg is M
        log_a = self.log_a.expand(B, -1)
        log_b = self.log_b.expand(B, -1)
        u = torch.zeros_like(log_a)
        v = torch.zeros_like(log_b)
        for _ in range(3):
            u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2)
            v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1)
        log_p = M + u.unsqueeze(2) + v.unsqueeze(1) - self.sinkhorn_norm
        return log_p[:, :-1, :]

    @staticmethod
    def attention(attn: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """MultiHeadAttention.forward with q/k/v sliced from the projection (see class doc)."""
        B, N, C = x.shape
        qkv = attn.qkv(x)  # [B, N, 3C], laid out as (3, heads, head_dim) along the last dim
        q = qkv[..., :C].reshape(B, N, attn.num_heads, attn.head_dim).transpose(1, 2)
        k = qkv[..., C:2 * C].reshape(B, N, attn.num_heads, attn.head_dim).transpose(1, 2)
        v = qkv[..., 2 * C:].reshape(B, N, attn.num_heads, attn.head_dim).transpose(1, 2)
        scores = torch.softmax((q @ k.transpose(-2, -1)) * attn.scale, dim=-1)
        y = (scores @ v).transpose(1, 2).reshape(B, N, C)
        return attn.proj(y)

    def block(self, blk: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """TransformerBlock.forward with the attention above."""
        x = x + blk.ls1(self.attention(blk.attn, blk.norm1(x)))
        x = x + blk.ls2(blk.mlp(blk.norm2(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bb = self.backbone
        B = x.shape[0]
        t = bb.patch_embed(x)
        t = torch.cat((bb.cls_token.expand(B, -1, -1), t), dim=1) + self.pos_embed
        for blk in bb.blocks:
            t = self.block(blk, t)
        t = bb.norm(t)
        cls_token = t[:, 0]
        patches = t[:, 1:].reshape(B, self.grid_h, self.grid_w, bb.embed_dim).permute(0, 3, 1, 2)

        agg = self.aggregator.agg
        f = agg.cluster_features(patches).flatten(2)  # [B, cluster_dim, N]
        p = agg.score(patches).flatten(2)             # [B, num_clusters, N]
        tk = agg.token_features(cls_token)            # [B, token_dim]
        p = torch.exp(self.sinkhorn_log_probs(p))     # [B, num_clusters, N]
        f = torch.cat(
            [F.normalize(tk, p=2, dim=-1),
             F.normalize(torch.einsum("bdn,bkn->bdk", f, p), p=2, dim=1).flatten(1)],
            dim=-1,
        )
        f = F.normalize(f, p=2, dim=-1)
        f = self.aggregator.linear(f)
        return F.normalize(f, p=2, dim=-1)


# --------------------------------------------------------------------------------------
# Preprocessing references
# --------------------------------------------------------------------------------------
def preprocess_cv2(image_bgr: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    """Reference preprocessing for the C++ side - tensorrt_megaloc.cpp must match this
    exactly: BGR->RGB, area-interpolated resize (the closest OpenCV equivalent of
    torchvision's antialiased Resize for downscaling), /255, (x-mean)/std, CHW."""
    h, w = resolution
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    elif image_bgr.shape[2] == 4:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)
    resized = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (rgb - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def preprocess_readme(image_path: Path, resolution: tuple[int, int]) -> torch.Tensor:
    """MegaLoc's published evaluation preprocessing (README): PIL RGB -> ToTensor ->
    Normalize -> antialiased Resize. The descriptor this produces is the 'ground truth'
    every other path is compared against."""
    import torchvision.transforms as tfm
    from PIL import Image

    transform = tfm.Compose([
        tfm.ToTensor(),
        tfm.Normalize(mean=MEAN, std=STD),
        tfm.Resize(size=list(resolution), antialias=True),
    ])
    return transform(Image.open(image_path).convert("RGB")).unsqueeze(0)


# --------------------------------------------------------------------------------------
# Export + sidecar
# --------------------------------------------------------------------------------------
def export(wrapper: nn.Module, export_path: Path, resolution: tuple[int, int], opset: int) -> None:
    import onnx

    export_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.rand((1, 3, *resolution))
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, str(export_path), opset_version=opset,
            input_names=["input"], output_names=["descriptor"],
            do_constant_folding=True, dynamo=False,
        )
    onnx.checker.check_model(str(export_path))
    model = onnx.load(str(export_path), load_external_data=False)
    ins = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in model.graph.input]
    outs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in model.graph.output]
    print(f"exported: {export_path} ({export_path.stat().st_size / 2**20:.1f} MB, opset {opset})")
    print(f"  inputs  {ins}\n  outputs {outs}")


def write_sidecar_yaml(yaml_path: Path, resolution: tuple[int, int], commit: str,
                       descriptor_dim: int) -> None:
    """Hand-written YAML (no pyyaml dependency); read by the C++ side via yaml-cpp."""
    h, w = resolution
    lines = [
        f"# Sidecar for {yaml_path.with_suffix('').name} - generated by export_megaloc.py",
        "model: megaloc",
        f"source: {MEGALOC_GIT} @ {commit}",
        f"weights: {HF_REPO}/{HF_WEIGHTS}",
        f"input_width: {w}",
        f"input_height: {h}",
        "# Preprocessing: BGR->RGB, resize to input size (area interpolation), scale to",
        "# [0,1] (divide by 255), then (x - mean) / std per channel, CHW planar.",
        f"mean: [{', '.join(str(v) for v in MEAN)}]",
        f"std: [{', '.join(str(v) for v in STD)}]",
        "input_tensor: input",
        "output_tensor: descriptor   # [1, descriptor_dim] float32",
        f"descriptor_dim: {descriptor_dim}",
        "# The descriptor is L2-normalised inside the graph: cosine similarity = dot product.",
        "normalized: true",
    ]
    yaml_path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------------------
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def self_check(model: nn.Module, wrapper: nn.Module, export_path: Path,
               image_paths: list[Path], resolution: tuple[int, int]) -> None:
    """Three descriptor sources per image, all compared against the README-faithful
    torch reference: (a) torch original model + README preprocessing [reference],
    (b) export wrapper + cv2 preprocessing (structural rewrite + preprocessing delta),
    (c) ONNX Runtime on the exported file + cv2 preprocessing (what C++ reproduces).
    Writes (a) to '<export_stem>_reference.txt' for Thirdparty/MegaLoc-TensorRT's
    test_megaloc, which prints cosine(TensorRT, reference) per image."""
    import onnxruntime as ort

    providers = ort.get_available_providers()
    chosen = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in providers else []
    chosen.append("CPUExecutionProvider")
    session = ort.InferenceSession(str(export_path), providers=chosen)
    print(f"self-check: ORT providers {session.get_providers()}")

    ref_descs, onnx_descs = [], []
    with torch.no_grad():
        for path in image_paths:
            bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            assert bgr is not None, f"cannot read {path}"
            x_ref = preprocess_readme(path, resolution)
            x_cv = torch.from_numpy(preprocess_cv2(bgr, resolution))

            d_ref = model(x_ref)[0].numpy()             # (a)
            d_wrap_ref = wrapper(x_ref)[0].numpy()      # wrapper on identical input
            d_wrap_cv = wrapper(x_cv)[0].numpy()        # (b)
            (d_onnx,) = session.run(["descriptor"], {"input": x_cv.numpy()})  # (c)
            d_onnx = d_onnx[0]

            print(f"  {path.name}: |ref|={np.linalg.norm(d_ref):.4f}"
                  f"  wrapper-vs-original max|diff|={np.abs(d_wrap_ref - d_ref).max():.2e}"
                  f"  cos(cv2 preprocessing, ref)={cosine(d_wrap_cv, d_ref):.5f}"
                  f"  cos(onnx, ref)={cosine(d_onnx, d_ref):.5f}"
                  f"  cos(onnx, wrapper cv2)={cosine(d_onnx, d_wrap_cv):.6f}")
            ref_descs.append(d_ref)
            onnx_descs.append(d_onnx)

    n = len(image_paths)
    if n > 1:
        S = np.array([[cosine(a, b) for b in onnx_descs] for a in onnx_descs])
        print("self-check: pairwise cosine similarity (ONNX):")
        for i in range(n):
            print("   " + " ".join(f"{S[i, j]:6.3f}" for j in range(n)) + f"   {image_paths[i].name}")

    ref_path = export_path.with_name(export_path.stem + "_reference.txt")
    with ref_path.open("w") as f:
        f.write(f"# torch reference descriptors (README preprocessing), dim {ref_descs[0].size}\n")
        f.write("# <image path> <v0> <v1> ... one image per line\n")
        for path, d in zip(image_paths, ref_descs):
            f.write(str(path.resolve()) + " " + " ".join(f"{v:.7e}" for v in d) + "\n")
    print(f"wrote {ref_path}")


# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--resolution", type=int, nargs=2, default=[322, 322], metavar=("H", "W"),
                        help="export resolution (multiples of 14); the ONNX is locked to it")
    parser.add_argument("--commit", default=MEGALOC_COMMIT, help="gmberton/MegaLoc commit to pin")
    parser.add_argument("--export_path", type=Path, default=None,
                        help="output .onnx (default: <repo>/megaloc_models/megaloc_<HxW>.onnx)")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--threads", type=int, default=4, help="torch CPU threads")
    parser.add_argument("--test_images", type=Path, nargs="*", default=None,
                        help="run the post-export self-check on these images")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    resolution = (args.resolution[0], args.resolution[1])
    export_path = args.export_path or (
        MODELS_DIR / f"megaloc_{resolution[0]}x{resolution[1]}.onnx")

    commit = ensure_megaloc_clone(args.commit)
    print(f"megaloc source: {MEGALOC_DIR} @ {commit}")
    model = load_pretrained_megaloc()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MegaLoc: {n_params / 1e6:.1f}M parameters, descriptor dim {model.feat_dim}")

    wrapper = ExportWrapper(model, *resolution).eval()
    export(wrapper, export_path, resolution, args.opset)

    yaml_path = Path(str(export_path) + ".yaml")
    write_sidecar_yaml(yaml_path, resolution, commit, model.feat_dim)
    print(f"sidecar:  {yaml_path}")

    if args.test_images:
        self_check(model, wrapper, export_path, args.test_images, resolution)


if __name__ == "__main__":
    main()
