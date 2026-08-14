"""GPU Stable Fluids solver on moderngl framebuffer ping-pong.

Port of the Max/Jitter jit.gl.slab chain. One RGBA32F texture pair carries
the whole simulation state:

    R = velocity x      G = velocity y      B = temperature T   A = density D

Per-frame pass order (matching the original patch):

    [video_vector]  archive Channel B -> velocity steering   (optional)
    inject          audio bins -> 8 drifting jets
    buoyancy        Boussinesq lift
    advect          semi-Lagrangian back-trace
    diffuse         viscosity + thermal + molecular diffusion
    viscosity       heat-modulated thinning / thickening
    vorticity       curl confinement
    divergence      div(u) -> scratch
    jacobi x N      pressure solve
    gradient        subtract grad(p) -> divergence-free
    reaction        Gray-Scott fed by heat (separate RG target)
    organic_lut     heat -> colour, veins, asemic/skin composite
    [video_skin]    archive Channel A -> skin overlay          (optional)
    volume          pseudo-3D raymarch lift
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import moderngl
except ImportError as e:  # pragma: no cover
    moderngl = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None

SHADER_DIR = Path(__file__).resolve().parent.parent / "shaders" / "fluid"


@dataclass
class FluidParams:
    # injection.  Units are *normalized screen space per second*: a velocity
    # of 1.0 crosses the whole frame in one second. T and D inject toward
    # saturation (see inject.frag) so these are rates, not accumulations.
    gain: float = 1.0
    heat_amt: float = 2.2
    force_amt: float = 0.9
    density_amt: float = 1.6
    max_vel: float = 1.2           # CFL clamp
    jitter: float = 0.04
    swarm_rate: float = 0.37
    # buoyancy
    alpha: float = 1.8
    beta: float = 0.25
    T_amb: float = 0.0
    up: tuple[float, float] = (0.0, 1.0)
    # advection / dissipation (per frame at 60 fps; ~0.5-1 s half-life)
    diss_v: float = 0.985
    diss_T: float = 0.985
    diss_D: float = 0.990
    # diffusion
    k_vel: float = 0.10
    k_T: float = 0.22
    k_D: float = 0.08
    # heat-modulated viscosity
    visc_cold: float = 0.93
    visc_hot: float = 1.015
    T_knee: float = 0.15
    # vorticity confinement
    epsilon: float = 0.35
    # Pressure projection. Measured on the field interior, 20 iterations
    # removes ~95% of divergence and 40 removes ~96%; past that it's flat.
    #
    # Warm-starting from the previous frame's pressure is free accuracy, BUT
    # only once there are enough iterations to correct the stale field: at
    # <10 iterations the leftover gradient is subtracted before it has been
    # relaxed and the projection *injects* divergence (measured 13x worse
    # than no projection at all). WARM_START_MIN_ITERS enforces that.
    jacobi_iters: int = 40
    jacobi_warm_start: bool = True
    grad_scale: float = 1.0
    # reaction-diffusion
    rd_enabled: bool = True
    Du: float = 0.16
    Dv: float = 0.08
    F: float = 0.035
    k: float = 0.065
    heat_feed: float = 0.04
    rd_dt: float = 1.0
    # colour
    heat_gain: float = 1.0
    density_gain: float = 1.2
    asemic_mix: float = 0.65
    asemic_flow: float = 0.055
    glow: float = 0.35
    exposure: float = 1.1
    organic_bias: float = 0.55
    vein_gain: float = 1.2
    decay_mix: float = 0.8
    # archive channels
    video_vector_force: float = 0.12
    video_vector_damp: float = 0.6
    video_vector_curl: float = 0.7
    skin_mix: float = 0.55
    skin_warp: float = 0.045
    skin_tint: float = 0.45
    skin_contrast: float = 1.15
    skin_heat_mask: float = 1.0
    # Volumetric lift. Every sample is the same 2D image at a rotated/sheared
    # offset, so this is a directional blur as much as a depth cue - the Max
    # defaults (depth 1.2 / swirl 0.6 / shear 0.35) smear the plumes into
    # mush at this resolution. These are tuned to read as parallax while
    # keeping the flow structure legible.
    volume_enabled: bool = True
    volume_steps: int = 32
    volume_depth: float = 0.5
    volume_shear: float = 0.15
    volume_swirl: float = 0.30


class _Target:
    """A single colour framebuffer + its texture."""

    def __init__(self, ctx, size, components=4, dtype="f4", filter_linear=True):
        self.tex = ctx.texture(size, components, dtype=dtype)
        self.tex.repeat_x = False
        self.tex.repeat_y = False
        if filter_linear:
            self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        else:
            self.tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.fbo = ctx.framebuffer(color_attachments=[self.tex])

    def clear(self, *rgba):
        self.fbo.use()
        self.fbo.clear(*(rgba or (0.0, 0.0, 0.0, 0.0)))

    def release(self):
        self.fbo.release()
        self.tex.release()


class _PingPong:
    def __init__(self, ctx, size, components=4, dtype="f4"):
        self.a = _Target(ctx, size, components, dtype)
        self.b = _Target(ctx, size, components, dtype)

    @property
    def src(self) -> _Target:
        return self.a

    @property
    def dst(self) -> _Target:
        return self.b

    def swap(self):
        self.a, self.b = self.b, self.a

    def clear(self, *rgba):
        self.a.clear(*rgba)
        self.b.clear(*rgba)

    def release(self):
        self.a.release()
        self.b.release()


class FluidSim:
    """Owns every framebuffer and program for the fluid chain."""

    PASSES = (
        "inject", "advect", "buoyancy", "diffuse", "viscosity", "vorticity",
        "divergence", "jacobi", "gradient", "reaction", "organic_lut",
        "volume", "video_vector", "video_skin", "crossfade", "blit",
    )

    #: Below this iteration count a warm-started solve makes divergence worse
    #: than doing no projection at all, so we force a cold start instead.
    WARM_START_MIN_ITERS = 10

    def __init__(self, ctx: "moderngl.Context", size=(512, 288),
                 params: FluidParams | None = None):
        if moderngl is None:
            raise RuntimeError(f"moderngl import failed: {_IMPORT_ERR}")
        self.ctx = ctx
        self.size = size
        self.p = params or FluidParams()
        self.texel = (1.0 / size[0], 1.0 / size[1])

        vert_src = (SHADER_DIR / "fullscreen.vert").read_text(encoding="utf-8")
        self.progs: dict[str, moderngl.Program] = {}
        for name in self.PASSES:
            frag = SHADER_DIR / f"{name}.frag"
            if not frag.is_file():
                continue
            self.progs[name] = ctx.program(
                vertex_shader=vert_src,
                fragment_shader=frag.read_text(encoding="utf-8"),
            )

        # fullscreen quad shared by every pass
        quad = np.array([
            # x,  y,   u, v
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 1.0,
        ], dtype="f4")
        self._quad_vbo = ctx.buffer(quad.tobytes())
        self._vaos = {
            name: ctx.vertex_array(
                prog, [(self._quad_vbo, "2f 2f", "in_position", "in_texcoord")])
            for name, prog in self.progs.items()
        }

        # simulation targets
        self.state = _PingPong(ctx, size, 4, "f4")     # u v T D
        self.pressure = _PingPong(ctx, size, 1, "f4")
        self.divergence = _Target(ctx, size, 1, "f4")
        self.rd = _PingPong(ctx, size, 2, "f4")        # U V
        self.color = _Target(ctx, size, 4, "f2")
        self.final = _Target(ctx, size, 4, "f2")

        # 1x1 transparent stand-in used when no asemic/archive texture is bound
        self._null_tex = ctx.texture((1, 1), 4, data=b"\x00\x00\x00\x00")

        #: Texture that the last step() actually produced. Which target holds
        #: the result depends on whether the skin and volume passes ran, so
        #: present()/read_rgb() follow this rather than assuming `final`.
        self._presented = self.color

        self.clear()
        self.seed_reaction()

    # --------------------------------------------------------------- helpers
    def clear(self):
        self.state.clear(0.0, 0.0, 0.0, 0.0)
        self.pressure.clear(0.0, 0.0, 0.0, 0.0)
        self.divergence.clear(0.0, 0.0, 0.0, 0.0)
        self.color.clear(0.0, 0.0, 0.0, 0.0)
        self.final.clear(0.0, 0.0, 0.0, 0.0)

    def seed_reaction(self):
        """Gray-Scott needs U=1 everywhere plus a few V seeds to get going.

        Seeds are soft radial blobs - square seeds stay square for hundreds of
        frames and read as digital artifacts rather than growth.
        """
        w, h = self.size
        data = np.zeros((h, w, 2), dtype="f4")
        data[..., 0] = 1.0
        rng = np.random.default_rng(0xF1D)
        yy, xx = np.mgrid[0:h, 0:w].astype("f4")
        for _ in range(18):
            cx = float(rng.integers(10, w - 10))
            cy = float(rng.integers(10, h - 10))
            r = float(rng.uniform(2.5, 5.0))
            blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * r * r)))
            data[..., 1] = np.maximum(data[..., 1], blob * 0.55)
            data[..., 0] = np.minimum(data[..., 0], 1.0 - blob * 0.75)
        self.rd.a.tex.write(data.tobytes())
        self.rd.b.tex.write(data.tobytes())

    def _run(self, name: str, target: _Target, textures=(), uniforms=None):
        prog = self.progs.get(name)
        if prog is None:
            return
        for unit, tex in enumerate(textures):
            tex.use(unit)
            key = f"u_tex{unit}"
            if key in prog:
                prog[key].value = unit
        if "u_texel" in prog:
            prog["u_texel"].value = self.texel
        for key, val in (uniforms or {}).items():
            if key in prog:
                prog[key].value = val
        target.fbo.use()
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        self._vaos[name].render(moderngl.TRIANGLE_STRIP)

    # ------------------------------------------------------------------ step
    def step(self, bins: np.ndarray, *, dt: float = 1.0 / 60.0,
             time: float = 0.0,
             asemic_tex=None, skin_tex=None, vector_tex=None):
        """Advance the simulation one frame. Returns the presented texture."""
        p = self.p
        prog_inject = self.progs.get("inject")

        # 0. archive Channel B steers the velocity field before anything else
        if vector_tex is not None and "video_vector" in self.progs:
            self._run("video_vector", self.state.dst,
                      (self.state.src.tex, vector_tex),
                      {"u_force": p.video_vector_force,
                       "u_damp": p.video_vector_damp,
                       "u_curl": p.video_vector_curl})
            self.state.swap()

        # 1. inject audio jets
        if prog_inject is not None:
            padded = np.zeros(8, dtype="f4")
            n = min(8, len(bins))
            padded[:n] = np.asarray(bins, dtype="f4")[:n]
            if "u_bins" in prog_inject:
                prog_inject["u_bins"].write(padded.tobytes())
            self._run("inject", self.state.dst, (self.state.src.tex,),
                      {"u_gain": p.gain,
                       "u_heat_amt": p.heat_amt * dt,
                       "u_force_amt": p.force_amt * dt,
                       "u_density_amt": p.density_amt * dt,
                       "u_max_vel": p.max_vel,
                       "u_time": time,
                       "u_jitter": p.jitter,
                       "u_swarm_rate": p.swarm_rate})
            self.state.swap()

        # 2. buoyancy
        self._run("buoyancy", self.state.dst, (self.state.src.tex,),
                  {"u_alpha": p.alpha, "u_beta": p.beta, "u_dt": dt,
                   "u_T_amb": p.T_amb, "u_up": p.up})
        self.state.swap()

        # 3. advection (self-advecting: same texture as quantity and velocity)
        self._run("advect", self.state.dst,
                  (self.state.src.tex, self.state.src.tex),
                  {"u_dt": dt, "u_diss_v": p.diss_v,
                   "u_diss_T": p.diss_T, "u_diss_D": p.diss_D})
        self.state.swap()

        # 4. diffusion
        self._run("diffuse", self.state.dst, (self.state.src.tex,),
                  {"u_k_vel": p.k_vel, "u_k_T": p.k_T, "u_k_D": p.k_D})
        self.state.swap()

        # 5. heat-modulated viscosity
        self._run("viscosity", self.state.dst, (self.state.src.tex,),
                  {"u_visc_cold": p.visc_cold, "u_visc_hot": p.visc_hot,
                   "u_T_knee": p.T_knee})
        self.state.swap()

        # 6. vorticity confinement
        self._run("vorticity", self.state.dst, (self.state.src.tex,),
                  {"u_epsilon": p.epsilon, "u_dt": dt})
        self.state.swap()

        # 7. divergence
        self._run("divergence", self.divergence, (self.state.src.tex,))

        # 8. Jacobi pressure solve. Warm start only when there are enough
        #    iterations to relax the previous frame's field (see FluidParams).
        iters = max(1, p.jacobi_iters)
        if not p.jacobi_warm_start or iters < self.WARM_START_MIN_ITERS:
            self.pressure.clear(0.0, 0.0, 0.0, 0.0)
        for _ in range(iters):
            self._run("jacobi", self.pressure.dst,
                      (self.pressure.src.tex, self.divergence.tex),
                      {"u_alpha": -1.0, "u_rbeta": 0.25})
            self.pressure.swap()

        # 9. subtract pressure gradient -> divergence-free
        self._run("gradient", self.state.dst,
                  (self.state.src.tex, self.pressure.src.tex),
                  {"u_scale": p.grad_scale})
        self.state.swap()

        # 10. reaction-diffusion fed by the fluid's heat
        if p.rd_enabled:
            self._run("reaction", self.rd.dst,
                      (self.rd.src.tex, self.state.src.tex),
                      {"u_Du": p.Du, "u_Dv": p.Dv, "u_F": p.F,
                       "u_k": p.k, "u_heat_feed": p.heat_feed,
                       "u_dt": p.rd_dt})
            self.rd.swap()

        # 11. heat -> colour
        self._run("organic_lut", self.color,
                  (self.state.src.tex,
                   asemic_tex if asemic_tex is not None else self._null_tex,
                   self.rd.src.tex),
                  {"u_heat_gain": p.heat_gain,
                   "u_density_gain": p.density_gain,
                   "u_asemic_mix": p.asemic_mix,
                   "u_asemic_flow": p.asemic_flow,
                   "u_glow": p.glow,
                   "u_exposure": p.exposure,
                   "u_organic_bias": p.organic_bias,
                   "u_vein_gain": p.vein_gain,
                   "u_decay_mix": p.decay_mix})

        presented = self.color

        # 12. archive Channel A skin overlay
        if skin_tex is not None and "video_skin" in self.progs:
            self._run("video_skin", self.final,
                      (self.state.src.tex, self.color.tex, skin_tex),
                      {"u_skin_mix": p.skin_mix, "u_warp": p.skin_warp,
                       "u_tint": p.skin_tint, "u_contrast": p.skin_contrast,
                       "u_heat_mask": p.skin_heat_mask})
            presented = self.final
            # volume reads from `color`, so copy the composite back
            self._run("blit", self.color, (self.final.tex,), {"u_gain": 1.0})

        # 13. pseudo-3D raymarch lift
        if p.volume_enabled:
            self._run("volume", self.final, (self.color.tex,),
                      {"u_time": time, "u_steps": p.volume_steps,
                       "u_depth": p.volume_depth, "u_shear": p.volume_shear,
                       "u_swirl": p.volume_swirl})
            presented = self.final

        self._presented = presented
        return presented.tex

    # -------------------------------------------------------------- present
    def present(self, screen_fbo, gain: float = 1.0, tex=None):
        """Blit the most recent result to a screen framebuffer."""
        self._run_to_screen(screen_fbo, tex or self._presented.tex, gain)

    def _run_to_screen(self, screen_fbo, tex, gain):
        prog = self.progs["blit"]
        tex.use(0)
        if "u_tex0" in prog:
            prog["u_tex0"].value = 0
        if "u_gain" in prog:
            prog["u_gain"].value = gain
        screen_fbo.use()
        self.ctx.disable(moderngl.DEPTH_TEST)
        self._vaos["blit"].render(moderngl.TRIANGLE_STRIP)

    def read_rgb(self, tex=None) -> np.ndarray:
        """Read the most recent result back as uint8 RGB (headless export)."""
        src = tex or self._presented.tex
        tmp = _Target(self.ctx, self.size, 4, "f1", filter_linear=False)
        try:
            self._run_to_screen(tmp.fbo, src, 1.0)
            raw = tmp.fbo.read(components=3, dtype="f1")
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(
                self.size[1], self.size[0], 3)
            return np.flipud(arr).copy()
        finally:
            tmp.release()

    def release(self):
        self.state.release()
        self.pressure.release()
        self.divergence.release()
        self.rd.release()
        self.color.release()
        self.final.release()
        self._null_tex.release()
        self._quad_vbo.release()
