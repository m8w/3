"""3D scalar field with audio-driven SDF splats + decay + smoothing.

Port of the Max abstraction fh.voxel_field + voxel_splat.js. All work is
vectorised numpy; a 48^3 field with 8 voices runs comfortably at 60 fps on
a laptop CPU. Move to numba/cython later if needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Shape codes (match the Max version so presets are portable)
SPHERE, BOX, TORUS, OCTA, CAPSULE, GYROID = 0, 1, 2, 3, 4, 5
SHAPE_NAMES = ["sphere", "box", "torus", "octahedron", "capsule", "gyroid"]


@dataclass
class Voice:
    x: float = 0.5
    y: float = 0.5
    z: float = 0.5
    shape: int = SPHERE
    radius: float = 0.18


def default_voices() -> list[Voice]:
    """Same layout as the Max preset in fh.mesh_synth.maxpat."""
    return [
        Voice(0.30, 0.20, 0.30, SPHERE,  0.18),   # sub-bass  bottom-left
        Voice(0.50, 0.20, 0.50, SPHERE,  0.20),   # bass      bottom-centre
        Voice(0.70, 0.20, 0.70, SPHERE,  0.18),   # low-mid   bottom-right
        Voice(0.20, 0.50, 0.50, TORUS,   0.16),   # mid       left wall
        Voice(0.80, 0.50, 0.50, TORUS,   0.16),   # upper-mid right wall
        Voice(0.30, 0.80, 0.30, BOX,     0.14),   # presence  top-left
        Voice(0.70, 0.80, 0.70, BOX,     0.14),   # brilliance top-right
        Voice(0.50, 0.55, 0.50, GYROID,  0.28),   # air       centre
    ]


class VoxelField:
    """Additive float32 volume evolved per frame."""

    def __init__(self, dim: tuple[int, int, int] = (48, 48, 48)):
        self.field = np.zeros(dim, dtype=np.float32)
        self.dim = np.array(dim, dtype=np.int32)
        # Kernel for the smoothing step (26-connected Gaussian-ish)
        k = np.array([[[0.02, 0.05, 0.02],
                       [0.05, 0.10, 0.05],
                       [0.02, 0.05, 0.02]],
                      [[0.05, 0.10, 0.05],
                       [0.10, 0.20, 0.10],
                       [0.05, 0.10, 0.05]],
                      [[0.02, 0.05, 0.02],
                       [0.05, 0.10, 0.05],
                       [0.02, 0.05, 0.02]]], dtype=np.float32)
        self._kernel = k / k.sum()

    # ------------------------------------------------------------ evolution
    def step(self, bins: np.ndarray, voices: list[Voice],
             *, decay: float = 0.94, blur: float = 0.25):
        """Advance one frame: decay -> splat -> optional blur -> clamp."""
        # 1. decay
        self.field *= float(decay)

        # 2. splat
        for i, v in enumerate(voices):
            amp = float(bins[i]) if i < len(bins) else 0.0
            if amp <= 1e-3:
                continue
            self._splat(v, amp)

        # 3. smoothing (mix)
        if blur > 1e-3:
            blurred = _convolve3d_same(self.field, self._kernel)
            b = float(np.clip(blur, 0.0, 1.0))
            self.field = self.field * (1.0 - b) + blurred * b

        # 4. clamp
        np.maximum(self.field, 0.0, out=self.field)
        np.minimum(self.field, 1.5, out=self.field)

    def clear(self):
        self.field[...] = 0.0

    # ---------------------------------------------------------------- splat
    def _splat(self, v: Voice, amp: float):
        W, H, D = self.dim
        # voxel-space centre + radius
        cx, cy, cz = v.x * W, v.y * H, v.z * D
        vr = v.radius * float(min(W, H, D))
        vri = int(np.ceil(vr)) + 1

        x0 = max(0, int(cx - vri)); x1 = min(W, int(cx + vri) + 1)
        y0 = max(0, int(cy - vri)); y1 = min(H, int(cy + vri) + 1)
        z0 = max(0, int(cz - vri)); z1 = min(D, int(cz + vri) + 1)
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            return

        zz, yy, xx = np.meshgrid(
            np.arange(z0, z1) - cz,
            np.arange(y0, y1) - cy,
            np.arange(x0, x1) - cx,
            indexing="ij",
        )
        vals = _sdf_evaluate(v.shape, xx, yy, zz, vr)
        if vals is None:
            return
        contribution = vals * (amp * 0.9)
        target = self.field[z0:z1, y0:y1, x0:x1]
        np.add(target, contribution.astype(np.float32), out=target)


# ---------------------------------------------------------------- primitives
def _sdf_evaluate(shape: int, dx, dy, dz, r):
    if shape == SPHERE:
        d2 = dx * dx + dy * dy + dz * dz
        r2 = r * r
        t = np.maximum(0.0, 1.0 - d2 / r2)
        return t * t
    if shape == BOX:
        m = np.maximum(np.abs(dx), np.maximum(np.abs(dy), np.abs(dz)))
        t = np.maximum(0.0, 1.0 - m / r)
        return t * t
    if shape == TORUS:
        R = r * 0.7
        mr = r * 0.35
        q = np.sqrt(dx * dx + dz * dz) - R
        d = np.sqrt(q * q + dy * dy)
        t = np.maximum(0.0, 1.0 - d / mr)
        return t * t
    if shape == OCTA:
        s = np.abs(dx) + np.abs(dy) + np.abs(dz)
        t = np.maximum(0.0, 1.0 - s / r)
        return t * t
    if shape == CAPSULE:
        h = r * 1.4
        cy = np.where(dy > h, dy - h, np.where(dy < -h, dy + h, 0.0))
        d2 = dx * dx + cy * cy + dz * dz
        r_local = r * 0.6
        t = np.maximum(0.0, 1.0 - d2 / (r_local * r_local))
        return t * t
    if shape == GYROID:
        d2 = dx * dx + dy * dy + dz * dz
        r2 = r * r
        s = 6.0
        g = (np.sin(dx * s) * np.cos(dy * s)
             + np.sin(dy * s) * np.cos(dz * s)
             + np.sin(dz * s) * np.cos(dx * s))
        v = np.maximum(0.0, 1.0 - np.abs(g) * 0.5)
        falloff = np.maximum(0.0, 1.0 - np.sqrt(d2) / r)
        mass = v * falloff * falloff
        mass[d2 >= r2] = 0.0
        return mass
    return None


# ------------------------------------------------------------ tiny 3D conv
def _convolve3d_same(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Same-shape 3D convolution via FFT.

    scipy.ndimage.convolve would be faster for small kernels but we avoid
    the dep for the core module - fallback works with only numpy.
    """
    try:
        from scipy.ndimage import convolve
        return convolve(a, k, mode="nearest")
    except ImportError:
        # numpy fallback (slow, but correct)
        from numpy.fft import rfftn, irfftn
        s = tuple(a.shape[i] + k.shape[i] - 1 for i in range(3))
        A = rfftn(a, s)
        K = rfftn(k, s)
        c = irfftn(A * K, s).real
        # crop back to a.shape with the kernel centred
        off = [(k.shape[i] - 1) // 2 for i in range(3)]
        return c[off[0]:off[0] + a.shape[0],
                 off[1]:off[1] + a.shape[1],
                 off[2]:off[2] + a.shape[2]]
