#!/usr/bin/env python3
"""Run the 3D modeling synthesizer.

Windowed live session: audio input drives a voxel field, marching cubes
extracts an isosurface each frame, moderngl renders it with heat-palette
shading. Keyboard bindings:

    SPACE   freeze / unfreeze
    E       export current mesh (OBJ + STL) to ~/ExternalRadio/fh_meshes/
    UP/DOWN threshold up/down
    LEFT/RT decay -/+
    1/2/3   cycle waveshape on voices 1..3
    C       clear voxel field
    ESC     quit

Options:
    --device <int|name>   sounddevice input to use
    --dim <int>           voxel resolution (default 48)
    --no-audio            run silent (voxel decays quietly, useful for testing)
"""
from __future__ import annotations

import argparse
import sys

from fh.audio import AudioConfig, list_devices
from fh.synth import Synth, SynthConfig
from fh.voxel import default_voices


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None,
                    help="sounddevice input device (int index or name substring)")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--dim", type=int, default=48,
                    help="voxel resolution (cube edge length)")
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--decay", type=float, default=0.94)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--export-dir", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.list_devices:
        print(list_devices())
        return

    device = args.device
    if device is not None and device.isdigit():
        device = int(device)

    cfg = SynthConfig(
        voxel_dim=(args.dim, args.dim, args.dim),
        threshold=args.threshold,
        decay=args.decay,
        audio=AudioConfig(device=device),
    )
    if args.export_dir:
        from pathlib import Path
        cfg.export_dir = Path(args.export_dir).expanduser()

    synth = Synth(cfg)
    if args.no_audio:
        # skip audio setup - synth.step_frame will feed zero bins each frame
        synth._audio = None
        synth.start_audio = lambda: None
        synth.stop_audio = lambda: None
    synth.run()


if __name__ == "__main__":
    main()
