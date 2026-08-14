"""Live audio -> 8 log-spaced frequency bins with per-band ADSR envelopes
and slow Brownian drift so silence still breathes."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

try:
    import sounddevice as sd
except ImportError as e:  # pragma: no cover
    sd = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


@dataclass
class AudioConfig:
    samplerate: int = 44100
    blocksize: int = 1024
    channels: int = 1
    n_bands: int = 8
    fmin: float = 60.0
    fmax: float = 18000.0
    gain: float = 1.5
    # ADSR (seconds); one set applied to every band
    attack: float = 0.010
    release: float = 0.400
    sustain: float = 0.7
    # Brownian drift amplitude (fraction of the envelope range)
    drift: float = 0.15
    drift_rate: float = 0.37
    device: int | str | None = None       # sounddevice input index or name


class AudioAnalyzer:
    """Runs on a sounddevice input stream; emits 8 floats each block.

    Read the latest values via ``analyzer.bins`` (numpy float32 array, length 8).
    Values are envelope-smoothed and drift-modulated, roughly 0..3 in normal
    listening; clip to 0..4 in downstream code.

    ``analyzer.peak`` is a scalar peak-amplitude estimate (0..1+).
    ``analyzer.centroid`` is a normalised spectral centroid (0..1).
    """

    def __init__(self, cfg: AudioConfig | None = None):
        if sd is None:
            raise RuntimeError(f"sounddevice import failed: {_IMPORT_ERR}")
        self.cfg = cfg or AudioConfig()
        self._bin_edges = np.geomspace(
            self.cfg.fmin, self.cfg.fmax, self.cfg.n_bands + 1)
        self._window = np.hanning(self.cfg.blocksize).astype(np.float32)

        self.bins = np.zeros(self.cfg.n_bands, dtype=np.float32)
        self._env = np.zeros(self.cfg.n_bands, dtype=np.float32)
        self.peak = 0.0
        self.centroid = 0.5

        # Two independent random walks per band -> low-freq drift
        rng = np.random.default_rng(0xF1D)
        self._drift_a = rng.uniform(0, 2 * np.pi, self.cfg.n_bands)
        self._drift_b = rng.uniform(0, 2 * np.pi, self.cfg.n_bands)
        self._drift_t = 0.0

        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None

    # ------------------------------------------------------------------ core
    def _process_block(self, block: np.ndarray, dt: float):
        """Called on the audio thread."""
        mono = block.mean(axis=1) if block.ndim > 1 else block
        n = mono.shape[0]
        if n != self.cfg.blocksize:
            window = np.hanning(n).astype(np.float32)
        else:
            window = self._window

        spec = np.fft.rfft(mono * window)
        mag = np.abs(spec) / n
        freqs = np.fft.rfftfreq(n, 1.0 / self.cfg.samplerate)

        raw = np.zeros(self.cfg.n_bands, dtype=np.float32)
        for i in range(self.cfg.n_bands):
            lo, hi = self._bin_edges[i], self._bin_edges[i + 1]
            mask = (freqs >= lo) & (freqs < hi)
            if mask.any():
                # log-scale magnitude -> perceptually flat response
                raw[i] = np.sqrt(np.mean(mag[mask] ** 2)) * 40.0

        raw = np.clip(raw * self.cfg.gain, 0.0, 8.0)

        # per-band ADSR-ish envelope: fast attack, slow release
        att = 1.0 - np.exp(-dt / max(self.cfg.attack, 1e-4))
        rel = 1.0 - np.exp(-dt / max(self.cfg.release, 1e-4))
        rising = raw > self._env
        self._env[rising] += (raw[rising] - self._env[rising]) * att
        self._env[~rising] += (raw[~rising] - self._env[~rising]) * rel

        # sustain floor (bins never drop below sustain * env_peak until quiet)
        env = np.maximum(self._env, self._env.max() * self.cfg.sustain * 0.05)

        # Brownian drift (deterministic pseudo-noise, cheap to update)
        self._drift_t += dt * self.cfg.drift_rate
        drift = 0.5 * (
            np.sin(self._drift_a + self._drift_t * 3.1)
            + np.sin(self._drift_b + self._drift_t * 2.3)
        )
        env = np.clip(env + drift * self.cfg.drift * env.max(), 0.0, 8.0)

        peak = float(np.sqrt(np.mean(mono ** 2)) * 4.0)
        centre_freqs = 0.5 * (self._bin_edges[:-1] + self._bin_edges[1:])
        if env.sum() > 1e-4:
            centroid_hz = float(np.sum(centre_freqs * env) / env.sum())
        else:
            centroid_hz = 0.5 * (self.cfg.fmin + self.cfg.fmax)
        centroid = np.log(max(centroid_hz, 1.0) / self.cfg.fmin) / \
                   np.log(self.cfg.fmax / self.cfg.fmin)

        with self._lock:
            self.bins = env.astype(np.float32)
            self.peak = peak
            self.centroid = float(np.clip(centroid, 0.0, 1.0))

    # -------------------------------------------------------------- lifecycle
    def start(self):
        if self._stream is not None:
            return

        def cb(indata, frames, time_info, status):
            dt = frames / self.cfg.samplerate
            self._process_block(indata, dt)

        self._stream = sd.InputStream(
            samplerate=self.cfg.samplerate,
            blocksize=self.cfg.blocksize,
            channels=self.cfg.channels,
            device=self.cfg.device,
            callback=cb,
            dtype="float32",
        )
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def snapshot(self) -> tuple[np.ndarray, float, float]:
        """Thread-safe snapshot of (bins, peak, centroid) for the main loop."""
        with self._lock:
            return self.bins.copy(), self.peak, self.centroid

    # helpful for headless / offline use
    def feed_offline(self, mono: np.ndarray, block: int | None = None,
                     samplerate: int | None = None):
        """Push a mono waveform through the analyser without a live device.

        Useful for batch mesh rendering from a recorded file.
        """
        if samplerate:
            self.cfg.samplerate = samplerate
        block = block or self.cfg.blocksize
        for start in range(0, len(mono) - block + 1, block):
            self._process_block(mono[start:start + block], block / self.cfg.samplerate)


def list_devices() -> str:  # pragma: no cover - CLI helper
    if sd is None:
        return f"sounddevice unavailable: {_IMPORT_ERR}"
    return str(sd.query_devices())
