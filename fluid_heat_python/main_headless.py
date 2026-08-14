#!/usr/bin/env python3
"""Batch-render a mesh sequence from a mono WAV without opening a window.

Useful for driving 100+ meshes from a recorded track and importing the
sequence into Blender / Houdini as a shape-key animation.

    python3 main_headless.py --wav track.wav --out ./meshes --fps 30 --fmt stl
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from fh.synth import Synth, SynthConfig


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="mono/stereo WAV file to consume")
    ap.add_argument("--out", default="./meshes_batch")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--fmt", nargs="+", default=["stl"],
                    choices=("obj", "stl", "ply"))
    return ap.parse_args()


def main():
    args = parse_args()
    sr, data = wavfile.read(args.wav)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    max_abs = np.abs(data).max()
    if max_abs > 1.0:
        data /= max(np.iinfo(data.dtype).max if hasattr(data.dtype, "kind")
                    and data.dtype.kind == "i" else max_abs, 1.0)

    cfg = SynthConfig(voxel_dim=(args.dim, args.dim, args.dim),
                      threshold=args.threshold)
    synth = Synth(cfg)
    paths = synth.batch_from_audio(
        data, sr, frame_hz=args.fps,
        out_dir=Path(args.out), formats=tuple(args.fmt))
    print(f"wrote {len(paths)} files under {args.out}")


if __name__ == "__main__":
    main()
