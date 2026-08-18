"""moderngl-window renderer for the mesh synthesizer.

The Renderer manages GL context + shader program + a dynamic VBO/EBO that
is re-uploaded each time the mesh changes. Uniforms are updated from the
synth every frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import moderngl
    import moderngl_window as mglw
    from moderngl_window import geometry
    from pyrr import Matrix44
except ImportError as e:  # pragma: no cover
    moderngl = None
    mglw = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None

from .mesh import Mesh
from .shaders import load as load_shader


@dataclass
class RenderState:
    pulse: float = 0.0
    displace: float = 0.12
    noise_scale: float = 3.0
    heat_gain: float = 1.0
    organic_bias: float = 0.35
    rim: float = 0.35
    ambient: float = 0.18
    time: float = 0.0
    rotation: float = 0.0     # radians, y-axis auto-rotate


class MeshRenderer:
    """Holds a moderngl program + dynamic buffers. Update mesh each frame."""

    def __init__(self, ctx: "moderngl.Context"):
        if moderngl is None:
            raise RuntimeError(f"moderngl import failed: {_IMPORT_ERR}")
        self.ctx = ctx
        v, f = load_shader("mesh_shade")
        self.prog = ctx.program(vertex_shader=v, fragment_shader=f)

        self._u = {
            k: self.prog.get(k, None) for k in (
                "u_mvp", "u_mv", "u_normal_mat",
                "u_pulse", "u_displace", "u_noise_scale",
                "u_heat_gain", "u_organic_bias", "u_rim",
                "u_ambient", "u_pulse", "u_time",
            )
        }

        # Start with a tiny placeholder buffer; upload_mesh grows it.
        self._vbo = ctx.buffer(reserve=1024, dynamic=True)
        self._ebo = ctx.buffer(reserve=512, dynamic=True)
        self._vao = ctx.vertex_array(
            self.prog,
            [(self._vbo, "3f 3f", "in_position", "in_normal")],
            index_buffer=self._ebo,
            index_element_size=4,
        )
        self._n_indices = 0

    def upload_mesh(self, mesh: Mesh | None):
        if mesh is None or mesh.n_tris == 0:
            self._n_indices = 0
            return
        # interleave position + normal
        interleaved = np.hstack([mesh.verts, mesh.normals]).astype(np.float32)
        raw = interleaved.tobytes()
        if len(raw) > self._vbo.size:
            self._vbo.orphan(len(raw) * 2)
        self._vbo.write(raw)
        idx = mesh.faces.astype(np.int32).tobytes()
        if len(idx) > self._ebo.size:
            self._ebo.orphan(len(idx) * 2)
        self._ebo.write(idx)
        self._n_indices = mesh.faces.size

    def draw(self, state: RenderState, aspect: float):
        if self._n_indices == 0:
            return
        proj = Matrix44.perspective_projection(45.0, aspect, 0.1, 20.0, dtype="f4")
        eye = np.array([0.0, 0.4, 3.2], dtype="f4")
        view = Matrix44.look_at(eye, np.array([0.0, 0.0, 0.0], "f4"),
                                np.array([0.0, 1.0, 0.0], "f4"), dtype="f4")
        model = Matrix44.from_y_rotation(state.rotation, dtype="f4")
        mv = view * model
        mvp = proj * mv
        # normal matrix = transpose(inverse(upper-left 3x3 of mv))
        m3 = np.asarray(mv, dtype=np.float32)[:3, :3]
        normal_mat = np.linalg.inv(m3).T.astype("f4")

        def _set(name, value):
            u = self._u.get(name)
            if u is None:
                return
            u.write(value.tobytes() if hasattr(value, "tobytes") else
                    np.asarray(value, dtype="f4").tobytes())

        _set("u_mvp", mvp)
        _set("u_mv", mv)
        _set("u_normal_mat", normal_mat.T)   # column-major upload for mat3
        for k, v in (
            ("u_pulse", state.pulse),
            ("u_displace", state.displace),
            ("u_noise_scale", state.noise_scale),
            ("u_heat_gain", state.heat_gain),
            ("u_organic_bias", state.organic_bias),
            ("u_rim", state.rim),
            ("u_ambient", state.ambient),
            ("u_time", state.time),
        ):
            u = self._u.get(k)
            if u is not None:
                u.value = float(v)

        self._vao.render(mode=moderngl.TRIANGLES, vertices=self._n_indices)


class SynthWindow(mglw.WindowConfig if mglw else object):
    """Overridden by main.py; kept here as a subclass template if the caller
    wants a minimal window with keyboard bindings but no synth object.
    """
    gl_version = (3, 3)
    title = "fluid_heat_python"
    window_size = (1280, 800)
    aspect_ratio = None
    resource_dir = str((__file__.rsplit("/", 2)[0] + "/shaders"))
