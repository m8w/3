#!/usr/bin/env python3
"""
archive_indexer.py -- scan directories for video files, extract metadata,
and populate a SQLite database consumed by fh.archive_fetcher in Max.

Usage:
    # Local files: Channel A (53k primary, "skin/texture")
    python3 archive_indexer.py --roots /Volumes/channelA \
        --db videos.sqlite --channel A --role texture --workers 6

    # Local files: Channel B (10k secondary, "nerves/velocity")
    python3 archive_indexer.py --roots /Volumes/channelB /Volumes/channelC \
        --db videos.sqlite --channel B --role velocity --workers 6

    # Remote URLs (no local copy required - resolver streams/caches at runtime).
    # CSV must have at minimum a `url` column; other columns are optional.
    python3 archive_indexer.py --urls youtube_channel_A.csv \
        --db videos.sqlite --channel A --role texture

    # YouTube ingest one-liner (export 53k channel handle to CSV first):
    #   yt-dlp --flat-playlist --print "url=%(url)s,youtube_id=%(id)s,duration=%(duration)s" \
    #          https://www.youtube.com/@yourChannel/videos > channel_A.csv

Schema:
    videos(id INTEGER PRIMARY KEY,
           path TEXT UNIQUE NOT NULL,
           size INTEGER, duration REAL,
           width INTEGER, height INTEGER, fps REAL,
           codec TEXT, ctime REAL, mtime REAL,
           tags TEXT, organic REAL, heat_bucket INTEGER,
           brightness REAL, motion REAL)

The `organic` and `heat_bucket` columns let the Max patch select clips
by fluid-heat threshold. `brightness` / `motion` can be filled later by
a second pass (e.g. ffmpeg signalstats).
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi",
             ".mpg", ".mpeg", ".wmv", ".flv", ".hevc", ".hap"}

DDL = """
CREATE TABLE IF NOT EXISTS videos (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT    UNIQUE NOT NULL,
    size        INTEGER,
    duration    REAL,
    width       INTEGER,
    height      INTEGER,
    fps         REAL,
    codec       TEXT,
    ctime       REAL,
    mtime       REAL,
    tags        TEXT,
    organic     REAL DEFAULT 0.5,
    heat_bucket INTEGER DEFAULT 0,
    brightness  REAL,
    motion      REAL,
    channel     TEXT    DEFAULT '',
    role        TEXT    DEFAULT 'texture',
    energy      REAL    DEFAULT 0.5,
    viscosity   REAL    DEFAULT 0.5,
    remote      INTEGER DEFAULT 0,
    youtube_id  TEXT    DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_videos_heat ON videos(heat_bucket);
CREATE INDEX IF NOT EXISTS idx_videos_organic ON videos(organic);
CREATE INDEX IF NOT EXISTS idx_videos_duration ON videos(duration);
CREATE INDEX IF NOT EXISTS idx_videos_role ON videos(role);
CREATE INDEX IF NOT EXISTS idx_videos_channel ON videos(channel);
CREATE INDEX IF NOT EXISTS idx_videos_energy ON videos(energy);
CREATE INDEX IF NOT EXISTS idx_videos_viscosity ON videos(viscosity);
CREATE INDEX IF NOT EXISTS idx_videos_remote ON videos(remote);
"""

# idempotent migration for upgrading an older db
MIGRATE_COLS = [
    ("channel",    "TEXT DEFAULT ''"),
    ("role",       "TEXT DEFAULT 'texture'"),
    ("energy",     "REAL DEFAULT 0.5"),
    ("viscosity",  "REAL DEFAULT 0.5"),
    ("remote",     "INTEGER DEFAULT 0"),
    ("youtube_id", "TEXT DEFAULT ''"),
]


def probe(path: Path, probe_bin: str) -> dict | None:
    try:
        out = subprocess.check_output(
            [probe_bin, "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            stderr=subprocess.DEVNULL, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    try:
        d = json.loads(out)
    except json.JSONDecodeError:
        return None
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), None)
    if not v:
        return None
    fps_str = v.get("avg_frame_rate", "0/1")
    try:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 0.0
    except ValueError:
        fps = 0.0
    return {
        "duration": float(d.get("format", {}).get("duration", 0.0)),
        "width":    int(v.get("width") or 0),
        "height":   int(v.get("height") or 0),
        "fps":      fps,
        "codec":    v.get("codec_name", ""),
    }


def infer_organic(path: Path, width: int, height: int, duration: float) -> float:
    """Heuristic 0..1 "organic-ness" score from filename + shape.

    Looks for known motifs ('asemic', 'ink', 'brush', 'moss', 'flow' ...),
    skews toward longer clips and natural aspect ratios.
    """
    name = path.stem.lower()
    organic_keys = ("asemic", "ink", "brush", "moss", "leaf", "vein",
                    "cloud", "smoke", "flame", "water", "root",
                    "organic", "flow", "ritual", "sigil", "petal")
    synthetic_keys = ("ui", "screen", "logo", "title", "hud")
    score = 0.5
    for k in organic_keys:
        if k in name:
            score += 0.12
    for k in synthetic_keys:
        if k in name:
            score -= 0.20
    if duration >= 30:
        score += 0.05
    if width and height:
        ratio = max(width, height) / min(width, height)
        if ratio < 2.2:
            score += 0.05
    return max(0.0, min(1.0, score))


def infer_energy_viscosity(path: Path, organic: float, fps: float) -> tuple[float, float]:
    """Rough heuristic for audio-matching columns.

    energy     = how "active" the clip should be - maps to audio loudness
    viscosity  = how "slow / syrupy" it reads - inverse of energy with bias
    These should be overwritten later by a second pass (e.g. optical flow).
    """
    name = path.stem.lower()
    fast = ("fast", "glitch", "strobe", "flash", "pulse", "beat", "attack")
    slow = ("slow", "drift", "wash", "fade", "ritual", "ambient", "breath")
    e = 0.5 + (0.02 * max(0.0, fps - 24.0))   # higher fps = more energy
    for k in fast:
        if k in name:
            e += 0.15
    for k in slow:
        if k in name:
            e -= 0.15
    e = max(0.0, min(1.0, e))
    v = max(0.0, min(1.0, 1.0 - e * 0.7 + (1.0 - organic) * 0.3))
    return e, v


def heat_bucket_of(score: float) -> int:
    """Quantize 0..1 organic into 5 buckets (matching blackbody LUT stops)."""
    return max(0, min(4, int(score * 5)))


def walk_videos(roots: list[Path]):
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix.lower() in VIDEO_EXT:
                    yield p


def process(p: Path, probe_bin: str, channel: str, role: str) -> tuple[str, dict] | None:
    try:
        st = p.stat()
    except OSError:
        return None
    meta = probe(p, probe_bin) or {}
    organic = infer_organic(p, meta.get("width", 0), meta.get("height", 0),
                            meta.get("duration", 0.0))
    energy, viscosity = infer_energy_viscosity(p, organic, meta.get("fps", 0.0))
    row = {
        "path":     str(p),
        "size":     st.st_size,
        "duration": meta.get("duration"),
        "width":    meta.get("width"),
        "height":   meta.get("height"),
        "fps":      meta.get("fps"),
        "codec":    meta.get("codec"),
        "ctime":    st.st_ctime,
        "mtime":    st.st_mtime,
        "tags":     "",
        "organic":  organic,
        "heat_bucket": heat_bucket_of(organic),
        "channel":  channel,
        "role":     role,
        "energy":   energy,
        "viscosity": viscosity,
    }
    return str(p), row


def ingest_url_csv(csv_path: Path, channel: str, role: str,
                   conn: sqlite3.Connection) -> int:
    """Ingest remote URLs from a CSV. Header columns recognised:
        url             (required)            -> stored in `path`, remote=1
        youtube_id      (optional)            -> stored in `youtube_id`
        duration        (optional, seconds)
        width, height, fps, codec             (optional)
        organic         (optional 0..1)
        energy, viscosity (optional 0..1)
        tags            (optional)
        channel, role   (optional, override CLI)
    """
    import csv
    n = 0
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            url = (r.get("url") or "").strip()
            if not url:
                continue

            def fnum(k, default=None):
                v = r.get(k)
                if v is None or v == "":
                    return default
                try:
                    return float(v)
                except ValueError:
                    return default

            organic = fnum("organic", 0.5)
            row = {
                "path":     url,
                "size":     None,
                "duration": fnum("duration"),
                "width":    int(fnum("width", 0) or 0) or None,
                "height":   int(fnum("height", 0) or 0) or None,
                "fps":      fnum("fps"),
                "codec":    r.get("codec", "") or "",
                "ctime":    None,
                "mtime":    None,
                "tags":     r.get("tags", "") or "",
                "organic":  organic,
                "heat_bucket": heat_bucket_of(organic),
                "channel":  (r.get("channel") or channel),
                "role":     (r.get("role") or role),
                "energy":   fnum("energy", 0.5),
                "viscosity":fnum("viscosity", 0.5),
                "remote":   1,
                "youtube_id": r.get("youtube_id", "") or "",
            }
            try:
                conn.execute("""INSERT OR REPLACE INTO videos
                    (path, size, duration, width, height, fps, codec,
                     ctime, mtime, tags, organic, heat_bucket,
                     channel, role, energy, viscosity, remote, youtube_id)
                    VALUES (:path, :size, :duration, :width, :height, :fps,
                            :codec, :ctime, :mtime, :tags, :organic, :heat_bucket,
                            :channel, :role, :energy, :viscosity, :remote, :youtube_id)""",
                    row)
                n += 1
            except sqlite3.DatabaseError as e:
                print(f"db error on {url}: {e}", file=sys.stderr)
            if n % 500 == 0:
                conn.commit()
    conn.commit()
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", default=[],
                    help="Directories to scan recursively (local files)")
    ap.add_argument("--urls", default="",
                    help="CSV file of remote URLs (use instead of --roots, "
                         "or in addition to mix local + remote in one db)")
    ap.add_argument("--db", default="videos.sqlite", help="SQLite output path")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--probe", default=shutil.which("ffprobe") or "ffprobe")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--channel", default="",
                    help="Channel name tag (e.g. 'A', 'primary53k', 'secondaryB').")
    ap.add_argument("--role", default="texture",
                    choices=("texture", "velocity", "both"),
                    help="'texture' -> skin/density source (53k channel). "
                         "'velocity' -> flow/nerves source (10k channel). "
                         "'both' -> available for either.")
    args = ap.parse_args()

    if not args.roots and not args.urls:
        sys.exit("provide --roots and/or --urls")

    roots = [Path(r).expanduser().resolve() for r in args.roots]
    for r in roots:
        if not r.is_dir():
            sys.exit(f"not a directory: {r}")

    conn = sqlite3.connect(args.db)
    conn.executescript(DDL)
    # migrate older dbs missing the channel/role columns
    existing = {row[1] for row in conn.execute("PRAGMA table_info(videos)")}
    for col, decl in MIGRATE_COLS:
        if col not in existing:
            conn.execute(f"ALTER TABLE videos ADD COLUMN {col} {decl}")
    conn.commit()

    if args.urls:
        csv_path = Path(args.urls).expanduser()
        if not csv_path.is_file():
            sys.exit(f"--urls file not found: {csv_path}")
        n = ingest_url_csv(csv_path, args.channel, args.role, conn)
        print(f"ingested {n} remote rows from {csv_path}", file=sys.stderr)
        if not roots:
            conn.close()
            return

    t0 = time.time()
    count = 0
    inserted = 0

    paths = list(walk_videos(roots))
    if args.limit:
        paths = paths[:args.limit]
    print(f"discovered {len(paths)} candidate files in {time.time()-t0:.1f}s",
          file=sys.stderr)

    t1 = time.time()
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, p, args.probe, args.channel, args.role): p
                   for p in paths}
        for fut in cf.as_completed(futures):
            row = fut.result()
            count += 1
            if not row:
                continue
            _, r = row
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO videos
                       (path, size, duration, width, height, fps, codec,
                        ctime, mtime, tags, organic, heat_bucket,
                        channel, role, energy, viscosity, remote, youtube_id)
                       VALUES (:path, :size, :duration, :width, :height, :fps,
                               :codec, :ctime, :mtime, :tags, :organic, :heat_bucket,
                               :channel, :role, :energy, :viscosity, 0, '')""",
                    r)
                inserted += 1
            except sqlite3.DatabaseError as e:
                print(f"db error on {r['path']}: {e}", file=sys.stderr)
            if inserted % 200 == 0:
                conn.commit()
                elapsed = time.time() - t1
                print(f"{inserted}/{count}  ({elapsed:.1f}s, "
                      f"{inserted/max(elapsed,0.01):.1f}/s)", file=sys.stderr)

    conn.commit()
    conn.close()
    print(f"done: {inserted} rows in {time.time()-t1:.1f}s -> {args.db}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
