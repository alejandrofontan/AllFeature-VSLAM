#!/usr/bin/env python
"""Time an ONNX file on an ONNX Runtime provider with random inputs of its declared shapes.

Author: Alejandro Fontan (with Claude, Fable 5.1)
Version: 0.1
Created: 2026-10-16

The measurement tool of the Core ML partitioning work (docs/plans/learned_features_onnxruntime.md, step 7): graph
variants are compared by wall time without touching the C++ side. Static shapes only (a symbolic dimension is
replaced by the --dim value). Prints the provider's partition line (log level 2) and the median of --iters runs.

  pixi run -e export-features python tools/time_onnx.py feature_models/aliked_n16_739x458_trunk.onnx \\
      --provider coreml --units CPUAndGPU --iters 10
"""
import argparse
import time

import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("--provider", default="cpu", choices=["cpu", "cuda", "coreml"])
    parser.add_argument("--units", default="CPUAndGPU", help="Core ML compute units")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--dim", type=int, default=512, help="value for symbolic dimensions")
    args = parser.parse_args()

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.log_severity_level = 2
    providers = {"cpu": ["CPUExecutionProvider"],
                 "cuda": [("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"],
                 "coreml": [("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": args.units,
                                                         "RequireStaticInputShapes": "0"}), "CPUExecutionProvider"]}
    session = ort.InferenceSession(args.model, so, providers=providers[args.provider])
    feed = {}
    for inp in session.get_inputs():
        shape = [d if isinstance(d, int) else args.dim for d in inp.shape]
        feed[inp.name] = np.random.rand(*shape).astype(np.float32)
        print(f"input {inp.name} {shape}")
    session.run(None, feed)   # warm-up / compile
    times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        session.run(None, feed)
        times.append((time.perf_counter() - t0) * 1e3)
    print(f"{args.model} on {args.provider}: median {np.median(times):.1f} ms over {args.iters} runs "
          f"(min {min(times):.1f}, max {max(times):.1f})")


if __name__ == "__main__":
    main()
