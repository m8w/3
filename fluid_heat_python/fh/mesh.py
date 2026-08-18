"""Mesh state + OBJ/STL/PLY exporters. Pure-Python, no external deps."""
from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mc import Iso


@dataclass
class Mesh:
    verts: np.ndarray       # (N, 3) float32
    faces: np.ndarray       # (M, 3) int32
    normals: np.ndarray     # (N, 3) float32

    @classmethod
    def from_iso(cls, iso: Iso | None) -> "Mesh | None":
        if iso is None:
            return None
        return cls(iso.verts, iso.faces, iso.normals)

    @property
    def n_tris(self) -> int:
        return len(self.faces)

    # ------------------------------------------------------------- exporters
    def write_obj(self, path: str | Path, *, include_normals: bool = True) -> Path:
        path = Path(path)
        lines: list[str] = [
            "# fluid_heat_python - fh.mesh_synth export",
            f"# {self.n_tris} triangles, {len(self.verts)} vertices",
        ]
        for v in self.verts:
            lines.append(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}")
        if include_normals:
            for n in self.normals:
                lines.append(f"vn {n[0]:.5f} {n[1]:.5f} {n[2]:.5f}")
        for f in self.faces:
            a, b, c = int(f[0]) + 1, int(f[1]) + 1, int(f[2]) + 1
            if include_normals:
                lines.append(f"f {a}//{a} {b}//{b} {c}//{c}")
            else:
                lines.append(f"f {a} {b} {c}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def write_ply(self, path: str | Path, *, ascii: bool = True) -> Path:
        path = Path(path)
        if not ascii:
            raise NotImplementedError("binary PLY not implemented yet")
        with path.open("w", encoding="utf-8") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write("comment fluid_heat_python export\n")
            f.write(f"element vertex {len(self.verts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property float nx\nproperty float ny\nproperty float nz\n")
            f.write(f"element face {self.n_tris}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            for v, n in zip(self.verts, self.normals):
                f.write(f"{v[0]:.5f} {v[1]:.5f} {v[2]:.5f} "
                        f"{n[0]:.5f} {n[1]:.5f} {n[2]:.5f}\n")
            for face in self.faces:
                f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")
        return path

    def write_stl(self, path: str | Path) -> Path:
        """Binary STL. Portable to Blender / Cinema 4D / Houdini / Unreal."""
        path = Path(path)
        tri_verts = self.verts[self.faces]                   # (M, 3, 3)
        # face normal from vertex normals (mean) fallback to computed
        if len(self.normals) == len(self.verts):
            face_normals = self.normals[self.faces].mean(axis=1)
        else:
            e1 = tri_verts[:, 1] - tri_verts[:, 0]
            e2 = tri_verts[:, 2] - tri_verts[:, 0]
            face_normals = np.cross(e1, e2)
        norm = np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8
        face_normals = face_normals / norm

        with path.open("wb") as f:
            header = b"fluid_heat_python export".ljust(80, b" ")
            f.write(header)
            f.write(struct.pack("<I", self.n_tris))
            for i in range(self.n_tris):
                fn = face_normals[i]
                tv = tri_verts[i]
                f.write(struct.pack("<fff", *fn.astype(np.float32)))
                for v in tv:
                    f.write(struct.pack("<fff", *v.astype(np.float32)))
                f.write(b"\x00\x00")
        return path

    # ------------------------------------------------------------- exporter dispatch
    def export(self, out_dir: str | Path, *,
               prefix: str = "fh_mesh",
               formats: tuple[str, ...] = ("obj", "stl")) -> list[Path]:
        out_dir = Path(out_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        written: list[Path] = []
        for fmt in formats:
            fname = out_dir / f"{prefix}_{stamp}.{fmt}"
            if fmt == "obj":
                written.append(self.write_obj(fname))
            elif fmt == "stl":
                written.append(self.write_stl(fname))
            elif fmt == "ply":
                written.append(self.write_ply(fname))
            else:
                raise ValueError(f"unknown format: {fmt}")
        return written
