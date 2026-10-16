#!/usr/bin/env python
"""Export the learned feature models to ONNX: SuperPoint, ALIKED and the three LightGlue matchers.

Author: Alejandro Fontan (with Claude, Fable 5.1)
Version: 0.1 (step 1 of docs/plans/learned_features_onnxruntime.md: clone, weights, model construction)
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

Run from the repository root:  pixi run -e export-features export-features
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

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

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


def load_models() -> dict[str, nn.Module]:
    """The five upstream models in eval mode, built for export. Keys: superpoint, aliked, sift (the extractors; sift is
    OpenCV-based and only produces references) and lightglue_{superpoint,aliked,sift} (the matchers)."""
    from lightglue import ALIKED, SIFT, LightGlue, SuperPoint  # noqa: E402  (from the pinned clone)

    # The upstream constructors call torch.hub.load_state_dict_from_url: with the files already in the hub cache under
    # their release filenames they load from disk. The ALIKED port fetches from Shiaoming/ALIKED's main branch; we
    # fetch the pinned-commit file under the same filename so it is the one that gets loaded.
    for name in WEIGHTS:
        download_weights(name)

    models: dict[str, nn.Module] = {}
    # SuperPoint: upstream defaults (nms_radius 4, detection_threshold 0.0005, remove_borders 4); max_num_keypoints
    # left unset - the top-k lives in C++ (plan Q4: the graph ends at the dense maps)
    models["superpoint"] = SuperPoint().eval()
    # ALIKED n16: threshold mode as the host uses it today (detection_threshold 0.2, no top-k cap)
    models["aliked"] = ALIKED(model_name="aliked-n16", max_num_keypoints=-1, detection_threshold=0.2,
                              nms_radius=2).eval()
    # SIFT: OpenCV backend, RootSIFT - the reference for the C++ cv::SIFT extractor
    models["sift"] = SIFT(backend="opencv", rootsift=True).eval()
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    print(f"torch {torch.__version__} (CPU), TORCH_HOME={os.environ['TORCH_HOME']}")
    commit = ensure_lightglue_clone(LIGHTGLUE_COMMIT)
    print(f"cvg/LightGlue at {commit}")
    with torch.no_grad():
        models = load_models()
    describe_models(models)


if __name__ == "__main__":
    main()
