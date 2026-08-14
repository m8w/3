# fluid_heat_python

Audio-reactive 3D modelling synthesizer. Pure Python + moderngl + numpy.
Formerly a Max/Jitter patch (see `../legacy_max/`); the Python
implementation is the active path.

## Instrument model

    audio bins        →  oscillators (voices)
    ADSR envelope     →  per-band attack / release
    waveshape         →  SDF primitive (sphere / box / torus / octa / capsule / gyroid)
    filter cutoff     →  marching-cubes threshold
    resonance         →  voxel decay
    LFO               →  vertex-normal noise displacement (biological pulse)
    key / position    →  xyz voice location in the unit cube
    volume            →  radius × amp × gain
    record            →  freeze + export OBJ / STL / PLY

## Signal flow

    audio in (sounddevice)
        │
        ▼
    fh.audio.AudioAnalyzer          ← FFT + 8 log bands + ADSR + Brownian drift
        │  (8 floats)
        ▼
    fh.voxel.VoxelField.step()      ← decay + SDF splats + 3x3x3 blur
        │  (48^3 float32 numpy)
        ▼
    fh.mc.extract()                 ← marching cubes (scikit-image or numpy fallback)
        │  (Iso: verts + faces + normals)
        ▼
    fh.mesh.Mesh                    ← state + OBJ/STL/PLY writers
        │
        ├──► fh.render.MeshRenderer  ← moderngl VBO/EBO + heat-palette shader
        │       │
        │       ▼
        │    window (moderngl-window / glfw)
        │
        └──► on E key: exports to ~/ExternalRadio/fh_meshes/

## Install

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Required: `numpy`, `moderngl`, `moderngl-window`, `sounddevice`, `scipy`, `pyrr`.
Recommended: `scikit-image` (C-backed MC; the numpy fallback is ~5-10× slower).

## Run

Live windowed session:

    python3 main.py                     # default input device
    python3 main.py --list-devices      # pick one
    python3 main.py --device 1 --dim 48
    python3 main.py --no-audio          # silent test (mesh stays still)

Batch: consume a mono/stereo WAV and export one mesh per frame:

    python3 main_headless.py --wav track.wav --out ./meshes --fps 30 --fmt stl

## Keyboard bindings (live window)

| key           | action                                          |
|---------------|-------------------------------------------------|
| `SPACE`       | freeze / unfreeze mesh                          |
| `E`           | export current mesh (OBJ + STL)                 |
| `↑` / `↓`     | threshold up / down (filter cutoff)             |
| `←` / `→`     | decay -/+ (resonance)                           |
| `1` / `2` / `3` | cycle waveshape on voices 1..3                |
| `C`           | clear voxel field                               |
| `ESC`         | quit                                            |

## Voices

Eight voices, one per audio band. Configure by editing
`SynthConfig.voices` or by calling `Voice(x, y, z, shape, radius)` directly:

```python
from fh.voxel import Voice, TORUS, GYROID
from fh.synth import Synth, SynthConfig

cfg = SynthConfig()
cfg.voices[3] = Voice(0.15, 0.5, 0.5, TORUS, 0.20)
cfg.voices[7] = Voice(0.5, 0.5, 0.5, GYROID, 0.32)
Synth(cfg).run()
```

Shape codes: `0` sphere · `1` box · `2` torus · `3` octahedron ·
`4` capsule (Y-axis) · `5` gyroid (thickened iso).

## Layout

    fluid_heat_python/
    ├── requirements.txt
    ├── main.py                          live windowed session
    ├── main_headless.py                 batch WAV -> mesh sequence
    ├── fh/
    │   ├── __init__.py
    │   ├── audio.py                     sounddevice + FFT + ADSR + Brownian
    │   ├── voxel.py                     3D field + SDF splats + decay + blur
    │   ├── mc.py                        marching cubes (skimage + numpy fallback)
    │   ├── mesh.py                      mesh state + OBJ/STL/PLY exporters
    │   ├── shaders.py                   GLSL file loader
    │   ├── render.py                    moderngl VBO/EBO + program
    │   └── synth.py                     top-level orchestrator + window app
    ├── shaders/
    │   ├── mesh_shade.vert              vertex noise displacement
    │   └── mesh_shade.frag              heat palette + rim + Reinhard tonemap
    ├── scripts/
    │   ├── archive_indexer.py           SQLite indexer for local + remote videos
    │   ├── archive_resolver.py          OSC yt-dlp sidecar + LRU cache
    │   └── youtube_to_csv.py            dump YouTube channel to CSV
    ├── exports/                         default STL/OBJ destination
    └── docs/
        ├── SYNTH.md                     instrument playbook (Python edition)
        ├── ARCHIVE.md                   video archive integration
        ├── STREAMING.md                 yt-dlp + LRU cache workflow
        └── EXTERNAL_RADIO.md            bridging with external_radio.py

## Loading exports elsewhere

- **Blender**: File → Import → Wavefront (.obj) / STL / PLY
- **Cinema 4D**: File → Merge (binary STL is most reliable at 60k+ tris)
- **Houdini**: File node → OBJ/PLY. STL if you need welded topology
- **Unreal / Unity**: STL via converter, or FBX after Blender pass

## Escape hatches

- **Faster inner loops** (32³ → 64³ at 60 fps): add `numba` and `@njit`
  the splat + convolve. The pure-numpy version handles 48³ comfortably.
- **True 2D fluid** (Navier-Stokes + heat): the original Max GLSL is
  under `../legacy_max/shaders/*.jxs` — the shader bodies port straight
  to `moderngl` framebuffer ping-pong. Not yet re-implemented in Python.
- **Web target**: because the shaders are plain GLSL 3.30, the same
  pipeline works with pyodide + WebGL2 or a Rust/wgpu port when the
  time comes.

See `../legacy_max/docs/` for the historical Max patches — kept for
reference and because the archive-management docs still describe the
Python scripts under `scripts/`.
