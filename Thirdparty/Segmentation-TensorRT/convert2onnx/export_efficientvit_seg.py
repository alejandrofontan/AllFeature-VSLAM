"""
Module: AllFeature-VSLAM - Segmentation-TensorRT - export_efficientvit_seg.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5)
- Version: 1.0
- Created: 2026-08-17
- Updated: 2026-08-17
- License: GPLv3 License (this script; EfficientViT itself is Apache-2.0)

Exports an EfficientViT-Seg checkpoint (mit-han-lab/efficientvit, cloned as a sibling
'efficientvit/' folder) to a self-contained ONNX model for the online static/dynamic
masking stage (see segmentation.md). Two post-processing steps that upstream keeps in
Python are folded INTO the graph so the C++ TensorRT consumer receives a ready class map:

  1. bilinear upsampling of the head's stride-8 logits back to input resolution
     (upsample-then-argmax matches the authors' eval pipeline, so published mIoU applies);
  2. argmax over classes, cast to int32 (TensorRT handles int64 I/O poorly).

Output: [1, H, W] int32 class map, tensor names 'input' / 'class_map'.

A sidecar '<export>.classes.yaml' records everything the C++ side must know but must not
hardcode: input size, normalization (applied OUTSIDE the model - ImageNet mean/std after
/255), and the dynamic class ids derived by name-matching against the canonical label list.

IMPORTANT resolution lock: EfficientViT's linear attention branches on input size
(ops.py: 'if H * W > self.dim'), which tracing freezes as a constant - an exported ONNX is
only valid at the resolution it was exported at. Re-run this script per resolution; never
reuse the file at another size.

Usage (from convert2onnx/, inside the pixi env):
  python export_efficientvit_seg.py                          # L1 ADE20K @ 512x512
  python export_efficientvit_seg.py --test_image path.png    # + self-check PNGs
  python export_efficientvit_seg.py --model efficientvit-seg-l1-cityscapes --resolution 1024 2048
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
EFFICIENTVIT_DIR = SCRIPT_DIR / "efficientvit"
REPO_ROOT = SCRIPT_DIR.parents[2]  # convert2onnx -> Segmentation-TensorRT -> Thirdparty -> repo
sys.path.insert(0, str(EFFICIENTVIT_DIR))

from efficientvit.seg_model_zoo import REGISTERED_EFFICIENTVIT_SEG_MODEL, create_efficientvit_seg_model

# Normalization the model expects, applied outside the graph (after scaling pixels to [0,1]).
# Source: applications/efficientvit_seg/demo_efficientvit_seg_model.py (ImageNet statistics).
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Canonical label lists, index == model output channel. ADE20K follows the standard
# SceneParse150 order (reduce_zero_label convention); Cityscapes the standard 19 trainIds.
# The self-check (--test_image) is the empirical guard against an indexing mismatch.
ADE20K_CLASSES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane",
    "grass", "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant",
    "curtain", "chair", "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror",
    "rug", "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub",
    "railing", "cushion", "base", "box", "column", "signboard", "chest of drawers", "counter",
    "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs",
    "runway", "case", "pool table", "pillow", "screen door", "stairway", "river", "bridge",
    "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench",
    "countertop", "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar",
    "arcade machine", "hovel", "bus", "towel", "light", "truck", "tower", "chandelier",
    "awning", "streetlight", "booth", "television receiver", "airplane", "dirt track",
    "apparel", "pole", "land", "bannister", "escalator", "ottoman", "bottle", "buffet",
    "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer",
    "plaything", "swimming pool", "stool", "barrel", "basket", "waterfall", "tent", "bag",
    "minibike", "cradle", "oven", "ball", "food", "step", "tank", "trade name", "microwave",
    "pot", "animal", "bicycle", "lake", "dishwasher", "screen", "blanket", "sculpture",
    "hood", "sconce", "vase", "traffic light", "tray", "ashcan", "fan", "pier", "crt screen",
    "plate", "monitor", "bulletin board", "shower", "radiator", "glass", "clock", "flag",
]
CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
    "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
]

# Movable ("dynamic") classes, matched by name against the label list above.
DYNAMIC_NAMES = {
    "person", "rider", "car", "bus", "truck", "van", "boat", "ship", "airplane",
    "bicycle", "minibike", "motorcycle", "animal", "train",
}


class SegWrapper(nn.Module):
    """Folds eval-time post-processing (upsample + argmax + int32 cast) into the graph."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)  # [B, n_classes, H/8, W/8]
        logits = F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)
        return logits.argmax(dim=1).to(torch.int32)  # [B, H, W] int32


def class_list_for(model_name: str) -> list[str]:
    if model_name.endswith("-ade20k"):
        return ADE20K_CLASSES
    if model_name.endswith("-cityscapes"):
        return CITYSCAPES_CLASSES
    raise ValueError(f"Cannot infer label list from model name '{model_name}'")


def dynamic_ids(classes: list[str]) -> list[tuple[int, str]]:
    return [(i, name) for i, name in enumerate(classes) if name in DYNAMIC_NAMES]


def export(model: nn.Module, export_path: Path, resolution: tuple[int, int], opset: int) -> None:
    """torch.onnx.export with explicit tensor names (the C++ side looks tensors up by name),
    then onnxsim simplification - same post-pass as upstream's export_onnx helper."""
    import onnx
    from onnxsim import simplify as simplify_func

    wrapper = SegWrapper(model).eval()
    dummy = torch.rand((1, 3, *resolution))
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, str(export_path), opset_version=opset,
            input_names=["input"], output_names=["class_map"],
        )
    onnx_model, ok = simplify_func(onnx.load_model(str(export_path)))
    assert ok, "onnxsim simplification failed"
    onnx.save(onnx_model, str(export_path))


def write_classes_yaml(yaml_path: Path, model_name: str, resolution: tuple[int, int],
                       classes: list[str]) -> None:
    """Hand-written YAML (no pyyaml dependency); read by the C++ side via yaml-cpp."""
    h, w = resolution
    lines = [
        f"# Sidecar for {yaml_path.with_suffix('').name} - generated by export_efficientvit_seg.py",
        f"model: {model_name}",
        f"input_width: {w}",
        f"input_height: {h}",
        "# Preprocessing: BGR->RGB, scale to [0,1] (divide by 255), then (x - mean) / std per channel.",
        f"mean: [{', '.join(str(v) for v in MEAN)}]",
        f"std: [{', '.join(str(v) for v in STD)}]",
        "input_tensor: input",
        "output_tensor: class_map   # [1, H, W] int32 class ids",
        "# Class ids treated as dynamic (mask=0). All other ids are static (mask=1).",
        "dynamic_class_ids:",
    ]
    lines += [f"  - {i}   # {name}" for i, name in dynamic_ids(classes)]
    yaml_path.write_text("\n".join(lines) + "\n")


def preprocess(image_bgr: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    """Reference preprocessing - the C++ implementation must match this exactly."""
    h, w = resolution
    rgb = cv2.cvtColor(cv2.resize(image_bgr, (w, h)), cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = (x - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return x.transpose(2, 0, 1)[None]  # HWC -> [1, 3, H, W]


def self_check(export_path: Path, image_path: Path, resolution: tuple[int, int],
               classes: list[str]) -> None:
    """Run the exported ONNX on a real image; print observed classes and save mask/overlay
    PNGs. This is the reference implementation the C++ port is later diffed against."""
    import onnxruntime as ort

    image = cv2.imread(str(image_path))
    assert image is not None, f"Cannot read {image_path}"
    session = ort.InferenceSession(str(export_path), providers=["CPUExecutionProvider"])
    (class_map,) = session.run(["class_map"], {"input": preprocess(image, resolution)})
    class_map = class_map[0]  # [H, W] int32

    ids, counts = np.unique(class_map, return_counts=True)
    print(f"self-check on {image_path.name} - observed classes:")
    for i, c in sorted(zip(ids, counts), key=lambda t: -t[1]):
        name = classes[i] if 0 <= i < len(classes) else "??"
        tag = " [DYNAMIC]" if name in DYNAMIC_NAMES else ""
        print(f"  {i:3d} {name:<20s} {100.0 * c / class_map.size:5.1f}%{tag}")

    dyn_lut = np.zeros(len(classes), dtype=np.uint8)
    for i, _ in dynamic_ids(classes):
        dyn_lut[i] = 1
    mask = 1 - dyn_lut[class_map]  # 1 = static, 0 = dynamic (repo-wide mask convention)
    mask_full = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    overlay = image.copy()
    overlay[mask_full == 0] = (0.4 * overlay[mask_full == 0] + np.array([0, 0, 153])).astype(np.uint8)
    mask_png = export_path.with_name(export_path.stem + "_selfcheck_mask.png")
    overlay_png = export_path.with_name(export_path.stem + "_selfcheck_overlay.png")
    cv2.imwrite(str(mask_png), mask_full * 255)  # x255 so the mask is visible in a viewer
    cv2.imwrite(str(overlay_png), overlay)
    print(f"wrote {mask_png}\n      {overlay_png} (dynamic region tinted red)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--model", default="efficientvit-seg-l1-ade20k",
                        choices=sorted(REGISTERED_EFFICIENTVIT_SEG_MODEL.keys()))
    parser.add_argument("--resolution", type=int, nargs=2, default=[512, 512],
                        metavar=("H", "W"), help="export resolution; the ONNX is locked to it")
    parser.add_argument("--weight", default=None,
                        help="checkpoint path (default: the registered assets/checkpoints/... "
                             "path inside the efficientvit clone)")
    parser.add_argument("--export_path", type=Path, default=None,
                        help="output .onnx (default: <repo>/segmentation_models/<model>_<HxW>.onnx)")
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument("--test_image", type=Path, default=None,
                        help="run a post-export self-check on this image")
    args = parser.parse_args()

    resolution = (args.resolution[0], args.resolution[1])
    export_path = args.export_path or (
        REPO_ROOT / "segmentation_models" / f"{args.model}_{resolution[0]}x{resolution[1]}.onnx"
    )
    # The registry stores the checkpoint path relative to the efficientvit clone; resolve it
    # explicitly so the script works from any working directory.
    weight = args.weight or str(EFFICIENTVIT_DIR / REGISTERED_EFFICIENTVIT_SEG_MODEL[args.model][2])

    model = create_efficientvit_seg_model(name=args.model, pretrained=True, weight_url=weight)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.model}: {n_params / 1e6:.1f}M parameters, weights: {weight}")

    export(model, export_path, resolution, args.opset)
    print(f"exported: {export_path}")

    classes = class_list_for(args.model)
    yaml_path = Path(str(export_path) + ".classes.yaml")
    write_classes_yaml(yaml_path, args.model, resolution, classes)
    print(f"sidecar:  {yaml_path} ({len(dynamic_ids(classes))} dynamic classes)")

    if args.test_image is not None:
        self_check(export_path, args.test_image, resolution, classes)


if __name__ == "__main__":
    main()
