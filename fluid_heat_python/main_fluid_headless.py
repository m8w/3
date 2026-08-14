#!/usr/bin/env python3
"""Render the fluid instrument offscreen to a PNG sequence - no window needed.

Drives the solver from a WAV file (or a synthetic test signal) and writes one
frame per step. Useful on a render box, over SSH, or for making a clip at a
resolution the display can't show.

    python3 main_fluid_headless.py --wav track.wav --out frames/ --fps 30
    python3 main_fluid_headless.py --frames 300 --out frames/   # test signal

Needs a GL context. On a headless Linux box either EGL or xvfb works:

    xvfb-run -a python3 main_fluid_headless.py ...

Encode the result with:

    ffmpeg -framerate 30 -i frames/fluid_%05d.png -c:v libx264 -pix_fmt yuv420p out.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from fh.audio import AudioAnalyzer, AudioConfig
from fh.fluid import FluidSim


def parse_args():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--wav", default=None, help="drive the sim from this file")
    ap.add_argument("--out", default="./fluid_frames")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--frames", type=int, default=0,
                    help="frame count (0 = length of the wav)")
    ap.add_argument("--warmup", type=int, default=30,
                    help="frames to run before writing, so the field is alive")
    ap.add_argument("--jacobi", type=int, default=40)
    ap.add_argument("--no-volume", action="store_true")
    return ap.parse_args()


def load_bins(wav: str | None, fps: float, n_frames: int):
    """Yield an 8-bin array per frame, from a WAV or a synthetic pattern."""
    if wav:
        from scipy.io import wavfile
        sr, data = wavfile.read(wav)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        peak = float(np.abs(data).max()) or 1.0
        data /= peak
        step = max(1, int(sr / fps))
        analyzer = AudioAnalyzer(AudioConfig(samplerate=sr, blocksize=step))
        total = (len(data) - step) // step
        limit = n_frames or total
        for i in range(min(limit, total)):
            analyzer._process_block(data[i * step:(i + 1) * step], step / sr)
            yield analyzer.snapshot()[0]
    else:
        limit = n_frames or 300
        for i in range(limit):
            t = i / fps
            yield np.array([
                0.9 + 0.6 * np.sin(t * 1.1),
                1.2 + 0.5 * np.sin(t * 0.7 + 1.0),
                0.8 + 0.4 * np.sin(t * 1.7 + 2.0),
                0.5 + 0.4 * np.sin(t * 2.3),
                0.5 + 0.4 * np.cos(t * 1.9),
                0.4 + 0.3 * np.sin(t * 3.1),
                0.4 + 0.3 * np.cos(t * 2.7),
                0.7 + 0.3 * np.sin(t * 0.4),
            ], dtype=np.float32).clip(0.0, 4.0)


def main():
    args = parse_args()
    try:
        import moderngl
        from PIL import Image
    except ImportError as e:
        sys.exit(f"missing dependency: {e}  (pip install moderngl pillow)")

    try:
        ctx = moderngl.create_context(standalone=True, require=330)
    except Exception:
        try:
            ctx = moderngl.create_context(standalone=True, backend="egl",
                                          require=330)
        except Exception as e:
            sys.exit(f"no GL context available ({e}).\n"
                     f"On a headless Linux box try:  xvfb-run -a {sys.argv[0]} ...")

    size = (args.width, args.height)
    sim = FluidSim(ctx, size)
    sim.p.jacobi_iters = args.jacobi
    if args.no_volume:
        sim.p.volume_enabled = False

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    dt = 1.0 / args.fps

    # let the field build up before the first written frame
    warm = np.array([1.0, 1.3, 0.9, 0.5, 0.5, 0.4, 0.4, 0.7], dtype=np.float32)
    for i in range(args.warmup):
        sim.step(warm, dt=dt, time=i * dt)

    n = 0
    for i, bins in enumerate(load_bins(args.wav, args.fps, args.frames)):
        sim.step(bins, dt=dt, time=(args.warmup + i) * dt)
        Image.fromarray(sim.read_rgb()).save(out_dir / f"fluid_{i:05d}.png")
        n += 1
        if n % 30 == 0:
            print(f"  {n} frames", flush=True)

    print(f"wrote {n} frames to {out_dir}")
    print(f"ffmpeg -framerate {args.fps:g} -i {out_dir}/fluid_%05d.png "
          f"-c:v libx264 -pix_fmt yuv420p out.mp4")


if __name__ == "__main__":
    main()
