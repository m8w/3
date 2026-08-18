"""Archive video -> GL textures, with A/B crossfade.

Decodes frames on background threads and uploads them to moderngl textures so
the fluid solver can use your archive as a skin layer (Channel A) and a
velocity field (Channel B).

Decoder backends, in preference order:
    1. PyAV            (``pip install av``) - fastest, in-process
    2. ffmpeg subprocess pipe - no extra Python deps, needs ffmpeg on PATH
Both yield contiguous uint8 RGB frames at a requested size.

Clip selection reads the same ``videos.sqlite`` that archive_indexer.py
builds. Remote rows (``remote=1``) are handed to archive_resolver.py over
OSC, which returns a locally cached path.
"""
from __future__ import annotations

import queue
import shutil
import sqlite3
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import av  # type: ignore
    _HAS_AV = True
except ImportError:
    _HAS_AV = False

try:
    import moderngl
except ImportError:  # pragma: no cover
    moderngl = None


# --------------------------------------------------------------------- decode
class FrameSource:
    """Background decoder producing RGB uint8 frames at a fixed size."""

    def __init__(self, path: str, size: tuple[int, int],
                 *, loop: bool = True, max_queue: int = 4):
        self.path = str(path)
        self.size = size
        self.loop = loop
        self.q: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            if _HAS_AV:
                self._run_av()
            else:
                self._run_ffmpeg()
        except Exception:
            # A dead clip must never take the render loop with it; the channel
            # simply keeps showing its previous frame.
            pass

    def _run_av(self):
        w, h = self.size
        while not self._stop.is_set():
            with av.open(self.path) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                for frame in container.decode(stream):
                    if self._stop.is_set():
                        return
                    img = frame.to_ndarray(format="rgb24", width=w, height=h)
                    self._put(img)
            if not self.loop:
                return

    def _run_ffmpeg(self):
        w, h = self.size
        exe = shutil.which("ffmpeg") or "ffmpeg"
        nbytes = w * h * 3
        while not self._stop.is_set():
            cmd = [
                exe, "-hide_banner", "-loglevel", "error",
                "-i", self.path,
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-vf", f"scale={w}:{h}",
                "-",
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL)
            try:
                while not self._stop.is_set():
                    raw = proc.stdout.read(nbytes)
                    if not raw or len(raw) < nbytes:
                        break
                    self._put(np.frombuffer(raw, dtype=np.uint8)
                                .reshape(h, w, 3))
            finally:
                proc.kill()
                proc.wait()
            if not self.loop:
                return

    def _put(self, img: np.ndarray):
        # Drop frames rather than block: the renderer sets the pace, not the
        # decoder. A stalled clip should never stall the fluid.
        try:
            self.q.put_nowait(img)
        except queue.Full:
            try:
                self.q.get_nowait()
                self.q.put_nowait(img)
            except queue.Empty:
                pass

    def latest(self) -> np.ndarray | None:
        img = None
        while True:
            try:
                img = self.q.get_nowait()
            except queue.Empty:
                return img

    def stop(self):
        self._stop.set()


# ------------------------------------------------------------------- selection
@dataclass
class ClipQuery:
    role: str = ""          # 'texture' | 'velocity' | '' (any)
    channel: str = ""       # 'A' | 'B' | '' (any)
    min_duration: float = 2.0
    allow_remote: bool = True


class ArchiveDB:
    """Read-only view over videos.sqlite with the same match ranking the
    Max JS used (weighted L2 across organic / energy / viscosity)."""

    def __init__(self, db_path: str | Path):
        self.path = str(Path(db_path).expanduser())
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(videos)")}
        self._has = cols

    def _where(self, q: ClipQuery) -> tuple[str, list]:
        clauses, args = ["(duration IS NULL OR duration >= ?)"], [q.min_duration]
        if q.role and "role" in self._has:
            clauses.append("(role = ? OR role = 'both')")
            args.append(q.role)
        if q.channel and "channel" in self._has:
            clauses.append("channel = ?")
            args.append(q.channel)
        if not q.allow_remote and "remote" in self._has:
            clauses.append("remote = 0")
        return "WHERE " + " AND ".join(clauses), args

    def match(self, q: ClipQuery, heat: float, energy: float,
              viscosity: float) -> sqlite3.Row | None:
        where, args = self._where(q)
        if {"organic", "energy", "viscosity"} <= self._has:
            rank = ("ABS(organic - ?) * 1.0 + ABS(energy - ?) * 0.9 "
                    "+ ABS(viscosity - ?) * 0.7 + (ABS(RANDOM() % 100) / 1000.0)")
            sql = f"SELECT * FROM videos {where} ORDER BY {rank} ASC LIMIT 1"
            args = args + [heat, energy, viscosity]
        else:
            sql = f"SELECT * FROM videos {where} ORDER BY RANDOM() LIMIT 1"
        row = self._conn.execute(sql, args).fetchone()
        return row

    def random(self, q: ClipQuery) -> sqlite3.Row | None:
        where, args = self._where(q)
        return self._conn.execute(
            f"SELECT * FROM videos {where} ORDER BY RANDOM() LIMIT 1",
            args).fetchone()

    def count(self, q: ClipQuery) -> int:
        where, args = self._where(q)
        return int(self._conn.execute(
            f"SELECT COUNT(*) FROM videos {where}", args).fetchone()[0])


class RemoteResolver:
    """Thin OSC client for scripts/archive_resolver.py.

    Fire-and-forget: ``request()`` sends /resolve and returns immediately.
    Resolved paths arrive on the reply port and are collected in ``ready``.
    """

    def __init__(self, host="127.0.0.1", send_port=7401, recv_port=7402):
        from pythonosc import dispatcher, osc_server, udp_client  # noqa
        self.client = udp_client.SimpleUDPClient(host, send_port)
        self.ready: queue.Queue[str] = queue.Queue()
        disp = dispatcher.Dispatcher()
        disp.map("/path", lambda addr, p: self.ready.put(str(p)))
        disp.map("/thumb_path", lambda addr, p: None)
        disp.map("/error", lambda addr, *a: None)
        self._srv = osc_server.ThreadingOSCUDPServer((host, recv_port), disp)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    def request(self, url: str):
        self.client.send_message("/resolve", url)

    def poll(self) -> str | None:
        try:
            return self.ready.get_nowait()
        except queue.Empty:
            return None


# --------------------------------------------------------------------- channel
class VideoChannel:
    """One archive channel: A/B clip slots, crossfade, GL texture output."""

    def __init__(self, ctx, size=(512, 288), *, db: ArchiveDB | None = None,
                 query: ClipQuery | None = None,
                 resolver: RemoteResolver | None = None,
                 crossfade_seconds: float = 1.5,
                 min_clip_seconds: float = 6.0):
        self.ctx = ctx
        self.size = size
        self.db = db
        self.query = query or ClipQuery()
        self.resolver = resolver
        self.crossfade_seconds = crossfade_seconds
        self.min_clip_seconds = min_clip_seconds

        black = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        self.tex_a = ctx.texture(size, 3, black.tobytes())
        self.tex_b = ctx.texture(size, 3, black.tobytes())
        for t in (self.tex_a, self.tex_b):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = False
            t.repeat_y = False

        self.src_a: FrameSource | None = None
        self.src_b: FrameSource | None = None
        self._active_is_a = True
        self._fade_t = 1.0            # 1.0 = fully on the active slot
        self._last_swap = 0.0
        self._pending_remote = False

    # ------------------------------------------------------------ clip loading
    def load_path(self, path: str):
        """Start a new clip in the inactive slot and begin crossfading."""
        src = FrameSource(path, self.size, loop=True)
        if self._active_is_a:
            if self.src_b:
                self.src_b.stop()
            self.src_b = src
        else:
            if self.src_a:
                self.src_a.stop()
            self.src_a = src
        self._active_is_a = not self._active_is_a
        self._fade_t = 0.0
        self._last_swap = time.time()

    def request_next(self, heat: float, energy: float, viscosity: float):
        """Pick a clip matching the current audio descriptors and load it."""
        if self.db is None:
            return
        row = self.db.match(self.query, heat, energy, viscosity)
        if row is None:
            return
        path = row["path"]
        is_remote = ("remote" in row.keys() and row["remote"] == 1)
        if is_remote:
            if self.resolver is not None:
                self.resolver.request(path)
                self._pending_remote = True
            return
        self.load_path(path)

    # ------------------------------------------------------------------ update
    def update(self, heat=0.5, energy=0.5, viscosity=0.5, *, auto=True):
        """Pump decoded frames into the textures and advance the crossfade."""
        now = time.time()

        if self._pending_remote and self.resolver is not None:
            p = self.resolver.poll()
            if p:
                self._pending_remote = False
                self.load_path(p)

        if auto and self.db is not None and \
                now - self._last_swap > self.min_clip_seconds:
            self.request_next(heat, energy, viscosity)

        for src, tex in ((self.src_a, self.tex_a), (self.src_b, self.tex_b)):
            if src is None:
                continue
            img = src.latest()
            if img is not None:
                tex.write(np.ascontiguousarray(img).tobytes())

        if self._fade_t < 1.0 and self.crossfade_seconds > 0:
            self._fade_t = min(1.0, self._fade_t +
                               (now - self._last_swap) / self.crossfade_seconds)

    # ------------------------------------------------------------------ output
    @property
    def fade(self) -> float:
        """0 = fully the outgoing slot, 1 = fully the incoming slot."""
        return self._fade_t

    @property
    def outgoing(self):
        return self.tex_a if self._active_is_a else self.tex_b

    @property
    def incoming(self):
        return self.tex_b if self._active_is_a else self.tex_a

    def release(self):
        for s in (self.src_a, self.src_b):
            if s:
                s.stop()
        self.tex_a.release()
        self.tex_b.release()


def open_archive(db_path: str | Path, *, use_resolver: bool = False
                 ) -> tuple[ArchiveDB | None, RemoteResolver | None]:
    """Convenience: open the SQLite archive and (optionally) the OSC resolver."""
    db = None
    try:
        db = ArchiveDB(db_path)
    except sqlite3.Error:
        return None, None
    resolver = None
    if use_resolver:
        try:
            resolver = RemoteResolver()
        except Exception:
            resolver = None
    return db, resolver
