# fh.synth — instrument playbook (Python)

The Python port keeps the same mapping the Max version had, minus the
Max-specific plumbing. `fh.synth.Synth` is the object. `main.py` runs it.

## Playing it

1. `python3 main.py --list-devices` — pick your input.
2. `python3 main.py --device 1` — open the window; audio starts.
3. Bass hits swell the lower voices (spheres near the floor); mids ring
   the tori on the walls; highs sparkle top corners; the gyroid pad
   in the centre grows with the "air" band.
4. Move the **threshold** (`↑ ↓`) — high threshold = only strongest
   voices survive (topological low-pass); low threshold = everything
   fuses into one membrane.
5. Move **decay** (`← →`) — high decay = long lingering tails; low
   decay = punchy short-lived hits.
6. `SPACE` freezes the mesh so you can sit with it. `E` writes the
   current geometry to `~/ExternalRadio/fh_meshes/`.

## Timing

Default config runs the whole loop on the main thread:

    audio callback (sounddevice, ~23 ms for a 1024-frame block)
        → analyzer state (fast, ~1 ms)
    render callback (moderngl-window, called each vsync)
        → voxel.step(bins, voices)      ~ 3-8 ms at 48³ with 8 voices
        → mc.extract(field, threshold)  ~ 4-15 ms depending on triangle count
        → renderer.upload_mesh(mesh)    < 1 ms
        → renderer.draw(state)          < 1 ms
    total per frame: ~ 10-25 ms → comfortable 60 fps at 48³

If you push to 64³, add `numba` and JIT the splat + blur; or drop MC to
skimage's already-C-backed path (default when installed).

## Advanced: driving from an offline audio file

```python
import numpy as np
from scipy.io import wavfile
from fh.synth import Synth, SynthConfig

sr, data = wavfile.read("session.wav")
mono = data.mean(axis=1).astype(np.float32) if data.ndim > 1 else data.astype(np.float32)
Synth(SynthConfig()).batch_from_audio(mono, sr, frame_hz=24, formats=("stl",))
```

Produces one STL per frame period into
`~/ExternalRadio/fh_meshes/batch/fh_batch_00000.stl` etc. Blender's
"Import Sequence" then plays them as a shape-key animation.

## Wiring your own audio source

`fh.audio.AudioAnalyzer.feed_offline()` pushes samples through the
analyzer without a sounddevice input. Pipe FFT bins directly with:

```python
from fh.synth import Synth
synth = Synth()
# bypass the AudioAnalyzer entirely:
synth._audio = None
import numpy as np
custom_bins = np.array([0.9, 0.4, ...], dtype=np.float32)
synth.voxel.step(custom_bins, synth.cfg.voices,
                 decay=synth.cfg.decay, blur=synth.cfg.blur)
```

## Extending

- **New primitive** — add a shape constant to `fh.voxel` and a case to
  `_sdf_evaluate`. Vectorised numpy on a `(dz, dy, dx)` meshgrid is the
  natural style.
- **Different palette** — the two 5-stop LUTs live in
  `shaders/mesh_shade.frag`. Editing GLSL doesn't require recompiling
  Python; just restart the window.
- **Per-voice modulation** — replace the AudioAnalyzer output list with
  a numpy array of shape `(n_voices,)` that comes from anywhere (OSC,
  MIDI, network, your own DSP).

## What's missing vs. the Max version

- The 2D Navier-Stokes fluid slab chain (GLSL shaders exist under
  `legacy_max/shaders/*.jxs` — they port to moderngl framebuffer
  ping-pong, but that's not built yet).
- The dual-channel video-archive skin/nerves compositing (the archive
  fetch scripts are still Python and reusable, but the render side is
  Max-only right now).

Both are on the roadmap; the mesh synthesizer is the first slice
because it's where the Max version had the most gravity and it's the
piece you asked for.
