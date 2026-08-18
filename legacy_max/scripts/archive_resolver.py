#!/usr/bin/env python3
"""
archive_resolver.py -- OSC sidecar for the fluid_heat_audio archive system.

Resolves YouTube/remote video IDs to playable local files via yt-dlp,
maintains an LRU disk cache, and answers Max via OSC. Lets a 53k+ remote
archive look "local" to jit.movie even though only a rolling subset is
ever on disk.

Listens on :7401 (default), replies to localhost:7402.

OSC commands:
    /resolve   "<url-or-id>"            -> /path "<local>" or /stream "<url>" or /error "..."
    /resolve_stream  "<url-or-id>"      -> /stream "<direct-url>"  (no download)
    /prefetch  "<url1>" "<url2>" ...    -> background download; /prefetched <n>
    /thumb     "<url-or-id>"            -> /thumb_path "<jpg-path>" (instant fallback)
    /status                             -> /status n_cached, bytes_cached, jobs_pending
    /evict                              -> force eviction sweep
    /size_limit_gb <float>              -> change cache budget at runtime

Dependencies (pip):
    pip install yt-dlp python-osc
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import queue
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

def _require_osc():
    try:
        from pythonosc import dispatcher, osc_server, udp_client  # noqa: F401
        return dispatcher, osc_server, udp_client
    except ImportError:
        sys.stderr.write("missing dep: pip install python-osc\n")
        sys.exit(1)


def yt_dlp_bin() -> str:
    return shutil.which("yt-dlp") or "yt-dlp"


class LRUCache:
    """Disk cache keyed by URL/id, evicted oldest-access-first when over budget."""

    def __init__(self, root: Path, max_bytes: int):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.db = sqlite3.connect(str(root / "_cache.db"), check_same_thread=False)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key   TEXT PRIMARY KEY,
                path  TEXT,
                size  INTEGER,
                atime REAL
            )
        """)
        self.db.commit()
        self.lock = threading.Lock()

    def _key_path(self, key: str) -> Path:
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        return self.root / f"{h}.mp4"

    def get(self, key: str) -> str | None:
        with self.lock:
            row = self.db.execute(
                "SELECT path FROM cache WHERE key=?", (key,)).fetchone()
            if row and Path(row[0]).exists():
                self.db.execute(
                    "UPDATE cache SET atime=? WHERE key=?",
                    (time.time(), key))
                self.db.commit()
                return row[0]
            if row:
                self.db.execute("DELETE FROM cache WHERE key=?", (key,))
                self.db.commit()
            return None

    def put(self, key: str, src: Path) -> str:
        dst = self._key_path(key)
        if src.resolve() != dst.resolve():
            shutil.move(str(src), str(dst))
        size = dst.stat().st_size
        with self.lock:
            self.db.execute(
                "INSERT OR REPLACE INTO cache (key, path, size, atime) VALUES (?,?,?,?)",
                (key, str(dst), size, time.time()))
            self.db.commit()
            self._evict_locked()
        return str(dst)

    def stats(self) -> tuple[int, int]:
        with self.lock:
            row = self.db.execute("SELECT COUNT(*), COALESCE(SUM(size),0) FROM cache").fetchone()
            return int(row[0]), int(row[1])

    def evict(self) -> int:
        with self.lock:
            return self._evict_locked()

    def _evict_locked(self) -> int:
        rows = self.db.execute(
            "SELECT key, path, size FROM cache ORDER BY atime ASC").fetchall()
        total = sum(r[2] for r in rows)
        evicted = 0
        i = 0
        while total > self.max_bytes and i < len(rows):
            k, p, s = rows[i]
            try:
                Path(p).unlink()
            except OSError:
                pass
            self.db.execute("DELETE FROM cache WHERE key=?", (k,))
            total -= s
            evicted += 1
            i += 1
        if evicted:
            self.db.commit()
        return evicted


class Resolver:
    """yt-dlp wrapper -- supports per-key in-flight deduplication."""

    def __init__(self, cache: LRUCache, height_max: int = 720,
                 thumb_dir: Path | None = None,
                 cookies_file: str = "",
                 cookies_browser: str = ""):
        self.cache = cache
        self.height_max = height_max
        self.thumb_dir = thumb_dir
        if self.thumb_dir is not None:
            self.thumb_dir.mkdir(parents=True, exist_ok=True)
        self.cookies_file = cookies_file
        self.cookies_browser = cookies_browser
        self.inflight: dict[str, threading.Event] = {}
        self.inflight_lock = threading.Lock()

    def _cookie_args(self) -> list[str]:
        if self.cookies_file and Path(self.cookies_file).expanduser().exists():
            return ["--cookies", str(Path(self.cookies_file).expanduser())]
        if self.cookies_browser:
            return ["--cookies-from-browser", self.cookies_browser]
        return []

    def _begin(self, key: str) -> tuple[bool, threading.Event]:
        with self.inflight_lock:
            ev = self.inflight.get(key)
            if ev is not None:
                return False, ev
            ev = threading.Event()
            self.inflight[key] = ev
            return True, ev

    def _end(self, key: str):
        with self.inflight_lock:
            ev = self.inflight.pop(key, None)
            if ev:
                ev.set()

    def stream_url(self, key: str) -> str | None:
        try:
            out = subprocess.check_output(
                [yt_dlp_bin(), "-g",
                 "-f", f"best[height<={self.height_max}][ext=mp4]/best[ext=mp4]/best",
                 *self._cookie_args(),
                 key],
                stderr=subprocess.DEVNULL, timeout=30).decode().splitlines()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError):
            return None
        return out[0] if out else None

    def download(self, key: str) -> str | None:
        cached = self.cache.get(key)
        if cached:
            return cached
        owner, ev = self._begin(key)
        if not owner:
            ev.wait(timeout=300)
            return self.cache.get(key)
        try:
            tmp = self.cache.root / "tmp"
            tmp.mkdir(exist_ok=True)
            tmp_template = str(tmp / f"%(id)s_{int(time.time()*1000)}.%(ext)s")
            try:
                subprocess.check_call(
                    [yt_dlp_bin(),
                     "-f", f"best[height<={self.height_max}][ext=mp4]/best[ext=mp4]/best",
                     "--no-progress",
                     *self._cookie_args(),
                     "-o", tmp_template, key],
                    stderr=subprocess.DEVNULL, timeout=600)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                    FileNotFoundError):
                return None
            files = sorted(tmp.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
            if not files:
                return None
            return self.cache.put(key, files[-1])
        finally:
            self._end(key)

    def thumb(self, key: str) -> str | None:
        if self.thumb_dir is None:
            return None
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        target = self.thumb_dir / f"{h}.jpg"
        if target.exists():
            return str(target)
        try:
            subprocess.check_call(
                [yt_dlp_bin(), "--skip-download", "--write-thumbnail",
                 "--convert-thumbnails", "jpg",
                 *self._cookie_args(),
                 "-o", str(self.thumb_dir / f"{h}.%(ext)s"),
                 key],
                stderr=subprocess.DEVNULL, timeout=60)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError):
            return None
        if target.exists():
            return str(target)
        # yt-dlp may have written .webp despite --convert-thumbnails
        for cand in self.thumb_dir.glob(f"{h}.*"):
            if cand.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                return str(cand)
        return None


class ResolverServer:
    def __init__(self, args, cache: LRUCache, resolver: Resolver):
        _, _, udp_client = _require_osc()
        self.args = args
        self.cache = cache
        self.resolver = resolver
        self.client = udp_client.SimpleUDPClient(args.reply_host, args.reply_port)
        self.jobs: queue.Queue = queue.Queue()
        self.workers = [threading.Thread(target=self._worker, daemon=True)
                        for _ in range(args.workers)]
        for w in self.workers:
            w.start()
        self.stopping = False

    def _send(self, addr: str, *args):
        try:
            self.client.send_message(addr, list(args) if len(args) > 1 else args[0] if args else "")
        except Exception as e:
            print(f"OSC reply failed: {e}", file=sys.stderr)

    def _worker(self):
        while not self.stopping:
            try:
                job = self.jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            kind, payload = job
            try:
                if kind == "resolve":
                    key = payload
                    # answer immediately with thumb if available, then download
                    if self.resolver.thumb_dir is not None:
                        t = self.resolver.thumb(key)
                        if t:
                            self._send("/thumb_path", t)
                    p = self.resolver.download(key)
                    if p:
                        self._send("/path", p)
                    else:
                        url = self.resolver.stream_url(key)
                        if url:
                            self._send("/stream", url)
                        else:
                            self._send("/error", f"resolve failed: {key}")
                elif kind == "stream_only":
                    url = self.resolver.stream_url(payload)
                    if url:
                        self._send("/stream", url)
                    else:
                        self._send("/error", f"stream resolve failed: {payload}")
                elif kind == "prefetch":
                    self.resolver.download(payload)
                elif kind == "thumb":
                    t = self.resolver.thumb(payload)
                    if t:
                        self._send("/thumb_path", t)
                    else:
                        self._send("/error", f"no thumb for {payload}")
            except Exception as e:
                self._send("/error", f"{kind}: {e}")
            finally:
                self.jobs.task_done()

    def on_resolve(self, _, key):
        self.jobs.put(("resolve", str(key)))

    def on_resolve_stream(self, _, key):
        self.jobs.put(("stream_only", str(key)))

    def on_prefetch(self, _, *keys):
        n = 0
        for k in keys:
            self.jobs.put(("prefetch", str(k)))
            n += 1
        self._send("/prefetched", n)

    def on_thumb(self, _, key):
        self.jobs.put(("thumb", str(key)))

    def on_status(self, _):
        n, b = self.cache.stats()
        self._send("/status", n, b, self.jobs.qsize())

    def on_evict(self, _):
        n = self.cache.evict()
        self._send("/evicted", n)

    def on_size_limit(self, _, gb):
        try:
            self.cache.max_bytes = int(float(gb) * (1 << 30))
            n = self.cache.evict()
            self._send("/size_limit_set", float(gb), n)
        except Exception as e:
            self._send("/error", f"size_limit: {e}")

    def serve_forever(self):
        dispatcher, osc_server, _ = _require_osc()
        d = dispatcher.Dispatcher()
        d.map("/resolve", self.on_resolve)
        d.map("/resolve_stream", self.on_resolve_stream)
        d.map("/prefetch", self.on_prefetch)
        d.map("/thumb", self.on_thumb)
        d.map("/status", self.on_status)
        d.map("/evict", self.on_evict)
        d.map("/size_limit_gb", self.on_size_limit)
        srv = osc_server.ThreadingOSCUDPServer(
            (self.args.listen_host, self.args.listen_port), d)
        print(f"archive_resolver: listening on {self.args.listen_host}:"
              f"{self.args.listen_port}, replying to "
              f"{self.args.reply_host}:{self.args.reply_port}", file=sys.stderr)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            self.stopping = True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=str(Path.home() / ".fh_archive_cache"),
                    help="Where to store the LRU video cache")
    ap.add_argument("--cache-gb", type=float, default=50.0,
                    help="Maximum cache size in GB")
    ap.add_argument("--thumb-dir", default="",
                    help="Directory for one-frame thumbnails (empty = disabled)")
    ap.add_argument("--height-max", type=int, default=720,
                    help="Max video height passed to yt-dlp format selector")
    ap.add_argument("--workers", type=int, default=2,
                    help="Concurrent yt-dlp downloads")
    ap.add_argument("--cookies-file",
                    default=str(Path.home() / "ExternalRadio" / "youtube_cookies.txt"),
                    help="Netscape-format cookies file (matches "
                         "external_radio.py default). Empty disables.")
    ap.add_argument("--cookies-browser", default="",
                    help="Alternative: --cookies-from-browser value "
                         "('safari', 'chrome', 'firefox'). Used if "
                         "--cookies-file is missing.")
    ap.add_argument("--listen-host", default="127.0.0.1")
    ap.add_argument("--listen-port", type=int, default=7401)
    ap.add_argument("--reply-host", default="127.0.0.1")
    ap.add_argument("--reply-port", type=int, default=7402)
    args = ap.parse_args()

    cache = LRUCache(Path(args.cache_dir).expanduser(),
                     int(args.cache_gb * (1 << 30)))
    thumb_dir = Path(args.thumb_dir).expanduser() if args.thumb_dir else None
    resolver = Resolver(cache, height_max=args.height_max, thumb_dir=thumb_dir,
                        cookies_file=args.cookies_file,
                        cookies_browser=args.cookies_browser)
    server = ResolverServer(args, cache, resolver)
    server.serve_forever()


if __name__ == "__main__":
    main()
