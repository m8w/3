#!/usr/bin/env python3
"""Run the audio-reactive instrument.

Two instruments share one audio analyser:

    mesh    voxel field -> marching cubes -> shaded, exportable geometry
    fluid   Navier-Stokes + heat + reaction-diffusion -> heat palette
    both    fluid as a volumetric backdrop, mesh drawn into it

Keys:
    M          cycle mode          SPACE  freeze mesh     E  export OBJ+STL
    UP/DOWN    mesh threshold      LEFT/RIGHT  mesh decay
    1/2/3      cycle voice shape   C  clear fields
    [ / ]      fluid buoyancy      - / =  fluid vorticity
    V          volumetric lift     R  reseed reaction-diffusion
    ESC        quit

Examples:
    python3 main.py --list-devices
    python3 main.py --mode fluid --device 1
    python3 main.py --mode both --archive-db videos.sqlite --use-resolver
    python3 main.py --mode fluid --asemic assets/asemic.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

from fh.audio import AudioConfig, list_devices
from fh.synth import Synth, SynthConfig, MODES


def parse_args():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--mode", default="both", choices=MODES)
    ap.add_argument("--device", default=None,
                    help="sounddevice input (index or name substring)")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--no-audio", action="store_true",
                    help="run without an input device (fields stay quiet)")
    # mesh
    ap.add_argument("--dim", type=int, default=48, help="voxel cube edge")
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--decay", type=float, default=0.94)
    ap.add_argument("--export-dir", default=None)
    # fluid
    ap.add_argument("--fluid-width", type=int, default=512)
    ap.add_argument("--fluid-height", type=int, default=288)
    ap.add_argument("--jacobi", type=int, default=40,
                    help="pressure solve iterations (>=10 recommended)")
    ap.add_argument("--asemic", default=None,
                    help="image layer carried by the flow")
    # archive
    ap.add_argument("--archive-db", default=None,
                    help="videos.sqlite from scripts/archive_indexer.py")
    ap.add_argument("--use-resolver", action="store_true",
                    help="resolve remote rows via scripts/archive_resolver.py")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.list_devices:
        print(list_devices())
        return

    device = args.device
    if device is not None and str(device).isdigit():
        device = int(device)

    cfg = SynthConfig(
        mode=args.mode,
        voxel_dim=(args.dim, args.dim, args.dim),
        threshold=args.threshold,
        decay=args.decay,
        fluid_size=(args.fluid_width, args.fluid_height),
        audio=AudioConfig(device=device),
        archive_db=Path(args.archive_db).expanduser() if args.archive_db else None,
        use_resolver=args.use_resolver,
        asemic_image=Path(args.asemic).expanduser() if args.asemic else None,
    )
    cfg.fluid.jacobi_iters = args.jacobi
    if args.export_dir:
        cfg.export_dir = Path(args.export_dir).expanduser()

    synth = Synth(cfg)
    if args.no_audio:
        synth.start_audio = lambda: None
        synth.stop_audio = lambda: None
    synth.run()


if __name__ == "__main__":
    main()
