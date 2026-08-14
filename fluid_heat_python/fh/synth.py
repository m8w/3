"""Top-level orchestrator: wires audio -> voxel -> marching cubes -> renderer."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import moderngl
    import moderngl_window as mglw
except ImportError as e:  # pragma: no cover
    moderngl = None
    mglw = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None

from .audio import AudioAnalyzer, AudioConfig
from .voxel import VoxelField, Voice, default_voices, SHAPE_NAMES
from .mc import extract
from .mesh import Mesh
from .render import MeshRenderer, RenderState


@dataclass
class SynthConfig:
    voxel_dim: tuple[int, int, int] = (48, 48, 48)
    threshold: float = 0.35
    decay: float = 0.94
    blur: float = 0.25
    frozen: bool = False
    export_dir: Path = field(default_factory=lambda:
        Path("~/ExternalRadio/fh_meshes").expanduser())
    audio: AudioConfig = field(default_factory=AudioConfig)
    voices: list[Voice] = field(default_factory=default_voices)
    auto_rotate: float = 0.35   # rad / sec


class Synth:
    """The instrument. Call .run() for a windowed session or step manually
    (feed_bins + step + extract_mesh) in headless mode."""

    def __init__(self, cfg: SynthConfig | None = None):
        self.cfg = cfg or SynthConfig()
        self.voxel = VoxelField(self.cfg.voxel_dim)
        self.mesh: Mesh | None = None
        self._state = RenderState(displace=0.12, noise_scale=3.0,
                                  organic_bias=0.35, rim=0.35)
        self._audio: AudioAnalyzer | None = None
        self._t0 = time.time()

    # ---------------------------------------------------------- audio wiring
    def start_audio(self):
        if self._audio is None:
            self._audio = AudioAnalyzer(self.cfg.audio)
        self._audio.start()

    def stop_audio(self):
        if self._audio:
            self._audio.stop()
            self._audio = None

    # ------------------------------------------------------------- frame step
    def step_frame(self):
        """Advance one simulation frame. Returns (bins, peak)."""
        if self._audio:
            bins, peak, _ = self._audio.snapshot()
        else:
            bins = np.zeros(len(self.cfg.voices), dtype=np.float32)
            peak = 0.0

        if not self.cfg.frozen:
            self.voxel.step(bins, self.cfg.voices,
                            decay=self.cfg.decay, blur=self.cfg.blur)
            iso = extract(self.voxel.field, self.cfg.threshold)
            self.mesh = Mesh.from_iso(iso)

        self._state.pulse = min(1.0, float(peak))
        self._state.time = time.time() - self._t0
        self._state.rotation = self._state.time * self.cfg.auto_rotate
        return bins, peak

    # ---------------------------------------------------------------- export
    def export_current(self, formats=("obj", "stl")) -> list[Path]:
        if self.mesh is None:
            return []
        return self.mesh.export(self.cfg.export_dir, formats=formats)

    # ------------------------------------------------------- headless render
    def batch_from_audio(self, mono: np.ndarray, samplerate: int,
                         *, frame_hz: float = 30.0,
                         out_dir: Path | None = None,
                         formats=("stl",)) -> list[Path]:
        """Consume a mono waveform and export one mesh per frame period."""
        cfg = self.cfg
        analyzer = AudioAnalyzer(AudioConfig(samplerate=samplerate,
                                             blocksize=int(samplerate / frame_hz),
                                             n_bands=len(cfg.voices)))
        step_samples = int(samplerate / frame_hz)
        out_dir = out_dir or (cfg.export_dir / "batch")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for i, start in enumerate(range(0, len(mono) - step_samples + 1, step_samples)):
            analyzer._process_block(mono[start:start + step_samples],
                                    step_samples / samplerate)
            bins, peak, _ = analyzer.snapshot()
            self.voxel.step(bins, cfg.voices, decay=cfg.decay, blur=cfg.blur)
            iso = extract(self.voxel.field, cfg.threshold)
            m = Mesh.from_iso(iso)
            if m is None:
                continue
            for p in m.export(out_dir, prefix=f"fh_batch_{i:05d}",
                              formats=formats):
                written.append(p)
        return written

    # -------------------------------------------------------------- windowed
    def run(self):  # pragma: no cover - interactive
        if mglw is None:
            raise RuntimeError(f"moderngl-window import failed: {_IMPORT_ERR}")
        SynthApp.synth_instance = self
        SynthApp.window_size = (1280, 800)
        mglw.run_window_config(SynthApp)


class SynthApp(mglw.WindowConfig if mglw else object):  # pragma: no cover
    """moderngl-window entry point. `Synth.run()` sets `synth_instance`
    before invoking mglw.run_window_config."""
    gl_version = (3, 3)
    title = "fluid_heat_python - 3D modeling synthesizer"
    resource_dir = str(Path(__file__).resolve().parent.parent / "shaders")
    synth_instance: Synth | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.synth = self.__class__.synth_instance
        self.synth.start_audio()
        self.renderer = MeshRenderer(self.ctx)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.front_face = "ccw"

    def on_close(self):
        self.synth.stop_audio()

    def on_render(self, time_delta, frame_time):
        self.synth.step_frame()
        self.renderer.upload_mesh(self.synth.mesh)
        self.ctx.clear(0.02, 0.02, 0.03)
        aspect = self.wnd.aspect_ratio or (self.wnd.size[0] / self.wnd.size[1])
        self.renderer.draw(self.synth._state, aspect)

    def on_key_event(self, key, action, modifiers):
        keys = self.wnd.keys
        if action != keys.ACTION_PRESS:
            return
        if key == keys.SPACE:
            self.synth.cfg.frozen = not self.synth.cfg.frozen
            print(f"freeze = {self.synth.cfg.frozen}")
        elif key == keys.E:
            paths = self.synth.export_current(("obj", "stl"))
            for p in paths:
                print(f"exported: {p}")
        elif key == keys.UP:
            self.synth.cfg.threshold = min(1.5, self.synth.cfg.threshold + 0.02)
            print(f"threshold = {self.synth.cfg.threshold:.2f}")
        elif key == keys.DOWN:
            self.synth.cfg.threshold = max(0.05, self.synth.cfg.threshold - 0.02)
            print(f"threshold = {self.synth.cfg.threshold:.2f}")
        elif key == keys.LEFT:
            self.synth.cfg.decay = max(0.5, self.synth.cfg.decay - 0.005)
            print(f"decay = {self.synth.cfg.decay:.3f}")
        elif key == keys.RIGHT:
            self.synth.cfg.decay = min(0.999, self.synth.cfg.decay + 0.005)
            print(f"decay = {self.synth.cfg.decay:.3f}")
        elif key == keys.NUMBER_1:
            self._cycle_shape(0)
        elif key == keys.NUMBER_2:
            self._cycle_shape(1)
        elif key == keys.NUMBER_3:
            self._cycle_shape(2)
        elif key == keys.C:
            self.synth.voxel.clear()
            print("cleared voxel field")

    def _cycle_shape(self, voice_i: int):
        v = self.synth.cfg.voices[voice_i]
        v.shape = (v.shape + 1) % len(SHAPE_NAMES)
        print(f"voice {voice_i} shape -> {SHAPE_NAMES[v.shape]}")
