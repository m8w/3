"""Top-level orchestrator.

Two instruments share one audio analyser:

    MESH   audio -> voxel field -> marching cubes -> shaded mesh (exportable)
    FLUID  audio -> Navier-Stokes + heat + reaction-diffusion -> heat palette

``mode`` selects one or both. In BOTH the fluid renders first as a volumetric
backdrop and the mesh is drawn into it with depth testing.

CPU-side state (audio, voxel, mesh) lives on ``Synth``. GPU-side state
(renderer, fluid, video channels) is created by ``SynthApp`` once a GL
context exists.
"""
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
from .fluid import FluidSim, FluidParams

MODES = ("mesh", "fluid", "both")


@dataclass
class SynthConfig:
    mode: str = "both"
    # mesh instrument
    voxel_dim: tuple[int, int, int] = (48, 48, 48)
    threshold: float = 0.35
    decay: float = 0.94
    blur: float = 0.25
    frozen: bool = False
    voices: list[Voice] = field(default_factory=default_voices)
    auto_rotate: float = 0.35
    # fluid instrument
    fluid_size: tuple[int, int] = (512, 288)
    fluid: FluidParams = field(default_factory=FluidParams)
    # archive (optional)
    archive_db: Path | None = None
    use_resolver: bool = False
    asemic_image: Path | None = None
    # shared
    export_dir: Path = field(default_factory=lambda:
        Path("~/ExternalRadio/fh_meshes").expanduser())
    audio: AudioConfig = field(default_factory=AudioConfig)


class Synth:
    def __init__(self, cfg: SynthConfig | None = None):
        self.cfg = cfg or SynthConfig()
        if self.cfg.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.cfg.mode!r}")
        self.voxel = VoxelField(self.cfg.voxel_dim)
        self.mesh: Mesh | None = None
        self._state = RenderState()
        self._audio: AudioAnalyzer | None = None
        self._t0 = time.time()
        # latest audio descriptors, shared by both instruments
        self.bins = np.zeros(8, dtype=np.float32)
        self.peak = 0.0
        self.centroid = 0.5

    # ---------------------------------------------------------------- audio
    def start_audio(self):
        if self._audio is None:
            self._audio = AudioAnalyzer(self.cfg.audio)
        self._audio.start()

    def stop_audio(self):
        if self._audio:
            self._audio.stop()
            self._audio = None

    def poll_audio(self):
        if self._audio:
            self.bins, self.peak, self.centroid = self._audio.snapshot()
        return self.bins, self.peak, self.centroid

    @property
    def time(self) -> float:
        return time.time() - self._t0

    # --------------------------------------------------------- mesh instrument
    def step_mesh(self):
        if self.cfg.frozen:
            return
        self.voxel.step(self.bins, self.cfg.voices,
                        decay=self.cfg.decay, blur=self.cfg.blur)
        self.mesh = Mesh.from_iso(extract(self.voxel.field, self.cfg.threshold))

    def update_render_state(self):
        self._state.pulse = min(1.0, float(self.peak))
        self._state.time = self.time
        self._state.rotation = self.time * self.cfg.auto_rotate
        return self._state

    # ---------------------------------------------------------------- export
    def export_current(self, formats=("obj", "stl")) -> list[Path]:
        if self.mesh is None:
            return []
        return self.mesh.export(self.cfg.export_dir, formats=formats)

    # ------------------------------------------------------- headless batches
    def batch_from_audio(self, mono: np.ndarray, samplerate: int,
                         *, frame_hz: float = 30.0,
                         out_dir: Path | None = None,
                         formats=("stl",)) -> list[Path]:
        """Consume a mono waveform and export one mesh per frame period."""
        cfg = self.cfg
        analyzer = AudioAnalyzer(AudioConfig(samplerate=samplerate,
                                             blocksize=int(samplerate / frame_hz),
                                             n_bands=len(cfg.voices)))
        step = int(samplerate / frame_hz)
        out_dir = Path(out_dir or (cfg.export_dir / "batch"))
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for i, start in enumerate(range(0, len(mono) - step + 1, step)):
            analyzer._process_block(mono[start:start + step], step / samplerate)
            bins, _, _ = analyzer.snapshot()
            self.voxel.step(bins, cfg.voices, decay=cfg.decay, blur=cfg.blur)
            m = Mesh.from_iso(extract(self.voxel.field, cfg.threshold))
            if m is None:
                continue
            written.extend(m.export(out_dir, prefix=f"fh_batch_{i:05d}",
                                    formats=formats))
        return written

    # -------------------------------------------------------------- windowed
    def run(self):  # pragma: no cover - interactive
        if mglw is None:
            raise RuntimeError(f"moderngl-window import failed: {_IMPORT_ERR}")
        SynthApp.synth_instance = self
        mglw.run_window_config(SynthApp)


class SynthApp(mglw.WindowConfig if mglw else object):  # pragma: no cover
    gl_version = (3, 3)
    title = "fluid_heat_python"
    window_size = (1280, 720)
    resource_dir = str(Path(__file__).resolve().parent.parent / "shaders")
    synth_instance: Synth | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.synth = self.__class__.synth_instance
        cfg = self.synth.cfg
        self.synth.start_audio()

        self.renderer = MeshRenderer(self.ctx)
        self.fluid: FluidSim | None = None
        self.chan_a = None
        self.chan_b = None
        self.asemic_tex = None

        if cfg.mode in ("fluid", "both"):
            self.fluid = FluidSim(self.ctx, cfg.fluid_size, cfg.fluid)
            self._setup_archive()
            self._load_asemic()

        self.ctx.enable(moderngl.DEPTH_TEST)
        self._print_help()

    # ------------------------------------------------------------------ setup
    def _setup_archive(self):
        cfg = self.synth.cfg
        if cfg.archive_db is None:
            return
        try:
            from .video import open_archive, VideoChannel, ClipQuery
        except ImportError as e:
            print(f"archive disabled ({e})")
            return
        db, resolver = open_archive(cfg.archive_db, use_resolver=cfg.use_resolver)
        if db is None:
            print(f"archive disabled: cannot open {cfg.archive_db}")
            return
        qa = ClipQuery(role="texture", channel="A")
        qb = ClipQuery(role="velocity", channel="B")
        n_a, n_b = db.count(qa), db.count(qb)
        # Fall back to an unfiltered pool if the db was never split by role
        if n_a == 0 and n_b == 0:
            qa = qb = ClipQuery()
            n_a = n_b = db.count(qa)
        print(f"archive: {n_a} skin clips (A), {n_b} vector clips (B)")
        if n_a:
            self.chan_a = VideoChannel(self.ctx, cfg.fluid_size, db=db,
                                       query=qa, resolver=resolver)
        if n_b:
            self.chan_b = VideoChannel(self.ctx, cfg.fluid_size, db=db,
                                       query=qb, resolver=resolver)

    def _load_asemic(self):
        p = self.synth.cfg.asemic_image
        if p is None or not Path(p).is_file():
            return
        try:
            from PIL import Image
            img = Image.open(p).convert("RGBA").resize(self.synth.cfg.fluid_size)
            self.asemic_tex = self.ctx.texture(
                self.synth.cfg.fluid_size, 4, np.asarray(img).tobytes())
            self.asemic_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            print(f"asemic layer: {p}")
        except Exception as e:
            print(f"asemic layer failed: {e}")

    def _print_help(self):
        print(f"""
fluid_heat_python  [mode: {self.synth.cfg.mode}]
  M          cycle mode (mesh / fluid / both)
  SPACE      freeze mesh          E  export mesh (OBJ+STL)
  UP/DOWN    mesh threshold       LEFT/RIGHT  mesh decay
  1/2/3      cycle voice shape    C  clear fields
  [ / ]      fluid buoyancy       - / =  fluid vorticity
  V          toggle volumetric lift
  R          reseed reaction-diffusion
  ESC        quit
""".rstrip())

    # ----------------------------------------------------------------- render
    def on_render(self, t, frame_time):
        synth = self.synth
        cfg = synth.cfg
        dt = min(max(frame_time, 1e-4), 1 / 20.0)   # clamp against hitches
        synth.poll_audio()

        self.ctx.clear(0.02, 0.02, 0.03)

        if self.fluid is not None and cfg.mode in ("fluid", "both"):
            skin = vector = None
            if self.chan_a is not None:
                self.chan_a.update(synth.centroid, min(1.0, synth.peak),
                                   1.0 - min(1.0, synth.peak))
                skin = self.chan_a.incoming
            if self.chan_b is not None:
                self.chan_b.update(1.0 - synth.centroid, min(1.0, synth.peak),
                                   1.0 - min(1.0, synth.peak))
                vector = self.chan_b.incoming
            self.fluid.step(synth.bins, dt=dt, time=synth.time,
                            asemic_tex=self.asemic_tex,
                            skin_tex=skin, vector_tex=vector)
            # fluid is a flat backdrop: no depth writes, mesh draws over it
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.fluid.present(self.wnd.fbo)
            self.ctx.enable(moderngl.DEPTH_TEST)

        if cfg.mode in ("mesh", "both"):
            synth.step_mesh()
            self.renderer.upload_mesh(synth.mesh)
            self.wnd.fbo.use()
            aspect = self.wnd.aspect_ratio or (self.wnd.size[0] / self.wnd.size[1])
            self.renderer.draw(synth.update_render_state(), aspect)

    def on_close(self):
        self.synth.stop_audio()
        for ch in (self.chan_a, self.chan_b):
            if ch:
                ch.release()

    # ------------------------------------------------------------------- keys
    def on_key_event(self, key, action, modifiers):
        keys = self.wnd.keys
        if action != keys.ACTION_PRESS:
            return
        cfg = self.synth.cfg
        p = cfg.fluid

        if key == keys.M:
            i = (MODES.index(cfg.mode) + 1) % len(MODES)
            cfg.mode = MODES[i]
            if cfg.mode in ("fluid", "both") and self.fluid is None:
                self.fluid = FluidSim(self.ctx, cfg.fluid_size, cfg.fluid)
                self._setup_archive()
                self._load_asemic()
            print(f"mode = {cfg.mode}")
        elif key == keys.SPACE:
            cfg.frozen = not cfg.frozen
            print(f"freeze = {cfg.frozen}")
        elif key == keys.E:
            for path in self.synth.export_current(("obj", "stl")):
                print(f"exported: {path}")
        elif key == keys.UP:
            cfg.threshold = min(1.5, cfg.threshold + 0.02)
            print(f"threshold = {cfg.threshold:.2f}")
        elif key == keys.DOWN:
            cfg.threshold = max(0.05, cfg.threshold - 0.02)
            print(f"threshold = {cfg.threshold:.2f}")
        elif key == keys.LEFT:
            cfg.decay = max(0.5, cfg.decay - 0.005)
            print(f"decay = {cfg.decay:.3f}")
        elif key == keys.RIGHT:
            cfg.decay = min(0.999, cfg.decay + 0.005)
            print(f"decay = {cfg.decay:.3f}")
        elif key in (keys.NUMBER_1, keys.NUMBER_2, keys.NUMBER_3):
            i = {keys.NUMBER_1: 0, keys.NUMBER_2: 1, keys.NUMBER_3: 2}[key]
            v = cfg.voices[i]
            v.shape = (v.shape + 1) % len(SHAPE_NAMES)
            print(f"voice {i} shape -> {SHAPE_NAMES[v.shape]}")
        elif key == keys.C:
            self.synth.voxel.clear()
            if self.fluid:
                self.fluid.clear()
            print("cleared")
        elif key == keys.LEFT_BRACKET:
            p.alpha = max(0.0, p.alpha - 0.1)
            print(f"buoyancy alpha = {p.alpha:.2f}")
        elif key == keys.RIGHT_BRACKET:
            p.alpha = min(6.0, p.alpha + 0.1)
            print(f"buoyancy alpha = {p.alpha:.2f}")
        elif key == keys.MINUS:
            p.epsilon = max(0.0, p.epsilon - 0.05)
            print(f"vorticity = {p.epsilon:.2f}")
        elif key == keys.EQUAL:
            p.epsilon = min(3.0, p.epsilon + 0.05)
            print(f"vorticity = {p.epsilon:.2f}")
        elif key == keys.V:
            p.volume_enabled = not p.volume_enabled
            print(f"volumetric lift = {p.volume_enabled}")
        elif key == keys.R:
            if self.fluid:
                self.fluid.seed_reaction()
                print("reseeded reaction-diffusion")
