"""fluid_heat_audio (Python) - audio-reactive 3D modeling synthesizer.

Package layout:
    fh.audio    - live input, 8-band FFT, ADSR + Brownian modulation
    fh.voxel    - numpy 3D scalar field with SDF splats + decay + blur
    fh.mc       - marching-cubes wrapper (scikit-image, numpy fallback)
    fh.mesh     - mesh state + OBJ/STL/PLY export
    fh.shaders  - GLSL loader / uniform helpers
    fh.render   - moderngl-window renderer for the mesh
    fh.synth    - top-level orchestrator (Synth class)

Public entry points:
    from fh.synth import Synth
    Synth().run()
"""
from importlib.metadata import PackageNotFoundError, version as _v
try:
    __version__ = _v("fluid_heat_audio")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

__all__ = ["audio", "voxel", "mc", "mesh", "shaders", "render", "synth"]
