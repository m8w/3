#!/usr/bin/env python3
"""
youtube_to_csv.py -- shell out to yt-dlp to dump a YouTube channel/playlist
into the CSV format expected by archive_indexer.py --urls.

Output columns:
    url, youtube_id, duration, width, height, fps, organic, energy, viscosity, tags

Usage:
    # Whole channel (flat = fast metadata, no per-video probe):
    python3 youtube_to_csv.py --flat \
        https://www.youtube.com/@yourChannel/videos > channel_A.csv

    # Per-video probe (slow but populates duration/fps):
    python3 youtube_to_csv.py \
        https://www.youtube.com/playlist?list=PLxxxx > playlist.csv

    # With explicit role hint (passed straight through):
    python3 youtube_to_csv.py --role velocity \
        https://www.youtube.com/@nerves > nerves.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ORGANIC_KEYS  = ("asemic", "ink", "brush", "moss", "leaf", "vein",
                 "cloud", "smoke", "flame", "water", "root",
                 "organic", "flow", "ritual", "sigil", "petal")
SYNTHETIC_KEYS = ("ui", "screen", "logo", "title", "hud", "tutorial")
FAST_KEYS = ("fast", "glitch", "strobe", "flash", "pulse", "beat", "attack")
SLOW_KEYS = ("slow", "drift", "wash", "fade", "ritual", "ambient", "breath")


def score(title: str, fps: float, duration: float | None) -> tuple[float, float, float]:
    t = (title or "").lower()
    organic = 0.5
    for k in ORGANIC_KEYS:
        if k in t:
            organic += 0.12
    for k in SYNTHETIC_KEYS:
        if k in t:
            organic -= 0.18
    if duration and duration >= 30:
        organic += 0.05
    organic = max(0.0, min(1.0, organic))

    energy = 0.5 + 0.02 * max(0.0, fps - 24.0)
    for k in FAST_KEYS:
        if k in t:
            energy += 0.15
    for k in SLOW_KEYS:
        if k in t:
            energy -= 0.15
    energy = max(0.0, min(1.0, energy))

    viscosity = max(0.0, min(1.0, 1.0 - energy * 0.7 + (1.0 - organic) * 0.3))
    return organic, energy, viscosity


def run_ytdlp_flat(url: str) -> list[dict]:
    bin_ = shutil.which("yt-dlp") or "yt-dlp"
    out = subprocess.check_output(
        [bin_, "--flat-playlist", "-J", url],
        stderr=subprocess.DEVNULL)
    blob = json.loads(out)
    entries = blob.get("entries") or []
    rows = []
    for e in entries:
        if not e or not e.get("id"):
            continue
        rows.append({
            "url":        e.get("url") or f"https://www.youtube.com/watch?v={e['id']}",
            "youtube_id": e["id"],
            "duration":   e.get("duration") or "",
            "width":      e.get("width") or "",
            "height":     e.get("height") or "",
            "fps":        e.get("fps") or "",
            "title":      e.get("title") or "",
        })
    return rows


def run_ytdlp_full(url: str) -> list[dict]:
    bin_ = shutil.which("yt-dlp") or "yt-dlp"
    proc = subprocess.Popen(
        [bin_, "-J", "--no-warnings", url],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    blob = json.loads(proc.stdout.read())
    proc.wait()
    if "entries" in blob and isinstance(blob["entries"], list):
        items = blob["entries"]
    else:
        items = [blob]
    rows = []
    for e in items:
        if not e or not e.get("id"):
            continue
        rows.append({
            "url":        e.get("webpage_url") or f"https://www.youtube.com/watch?v={e['id']}",
            "youtube_id": e["id"],
            "duration":   e.get("duration") or "",
            "width":      e.get("width") or "",
            "height":     e.get("height") or "",
            "fps":        e.get("fps") or "",
            "title":      e.get("title") or "",
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urls", nargs="+",
                    help="One or more channel/playlist/video URLs")
    ap.add_argument("--flat", action="store_true",
                    help="Use --flat-playlist (fast metadata; missing fields will be blank)")
    ap.add_argument("--role", default="",
                    help="Optional role hint to write into each row (texture / velocity / both)")
    ap.add_argument("--channel", default="",
                    help="Optional channel name to write into each row")
    ap.add_argument("--out", default="-",
                    help="Output CSV path (- = stdout)")
    args = ap.parse_args()

    rows = []
    for url in args.urls:
        try:
            if args.flat:
                rows.extend(run_ytdlp_flat(url))
            else:
                rows.extend(run_ytdlp_full(url))
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"warn: could not fetch {url}: {e}", file=sys.stderr)

    fout = sys.stdout if args.out == "-" else open(args.out, "w", newline="", encoding="utf-8")
    cols = ["url", "youtube_id", "duration", "width", "height", "fps",
            "organic", "energy", "viscosity", "tags", "channel", "role"]
    writer = csv.DictWriter(fout, fieldnames=cols)
    writer.writeheader()
    for r in rows:
        try:
            dur = float(r.get("duration") or 0) or None
        except (TypeError, ValueError):
            dur = None
        try:
            fps = float(r.get("fps") or 0)
        except (TypeError, ValueError):
            fps = 0.0
        organic, energy, viscosity = score(r.get("title", ""), fps, dur)
        writer.writerow({
            "url":        r["url"],
            "youtube_id": r["youtube_id"],
            "duration":   dur or "",
            "width":      r.get("width") or "",
            "height":     r.get("height") or "",
            "fps":        fps or "",
            "organic":    f"{organic:.3f}",
            "energy":     f"{energy:.3f}",
            "viscosity":  f"{viscosity:.3f}",
            "tags":       (r.get("title") or "")[:120],
            "channel":    args.channel,
            "role":       args.role,
        })
    if args.out != "-":
        fout.close()
    print(f"wrote {len(rows)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
