"""Tiny GLSL loader. Given a name, returns (vertex_src, fragment_src)."""
from __future__ import annotations

from pathlib import Path

SHADER_DIR = Path(__file__).resolve().parent.parent / "shaders"


def load(name: str) -> tuple[str, str]:
    """Load `<name>.vert` and `<name>.frag` from the shaders directory."""
    vert = SHADER_DIR / f"{name}.vert"
    frag = SHADER_DIR / f"{name}.frag"
    if not vert.is_file() or not frag.is_file():
        raise FileNotFoundError(f"shader pair missing: {vert}, {frag}")
    return vert.read_text(encoding="utf-8"), frag.read_text(encoding="utf-8")
