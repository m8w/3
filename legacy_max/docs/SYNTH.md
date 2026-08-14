# fh.mesh_synth — the 3D Modeling Synthesizer

The fluid layer treats sound as a physical *impulse*. The synth layer
treats it as *notes* — each 8-bin audio band drives a **voice** that
plays a solid shape into a 3D voxel field. Marching cubes then extracts
the surface. The result is a mesh you can watch, freeze, and export as
OBJ / STL / PLY for Blender, Cinema 4D, Houdini, Unreal — anything.

## Instrument metaphor

    audio bins        →  oscillators (voices)
    ADSR envelope     →  per-bin attack / release (fh.organic_mod)
    waveshape         →  SDF primitive (sphere / box / torus / octahedron / capsule / gyroid)
    filter cutoff     →  marching-cubes threshold
    resonance         →  voxel decay (how long the shape lingers)
    LFO               →  biological-pulse displacement + rotation
    key / position    →  xyz voice location in the unit cube
    volume            →  radius × amp × gain
    record            →  freeze + export to OBJ / STL / PLY

## Signal flow

    audio in ──┐
               ├─→ fh.audio_bins ──→ fh.organic_mod ──┐
               │                                       │
               │                                       ▼
               └───────────────────────────────→ fh.voxel_field  (48³ float32 3D matrix)
                                                       │
                                                       │ jit_matrix
                                                       ▼
                                                marching_cubes.js
                                                       │
                                                       │ pos + nrm matrices
                                                       ▼
                                          jit.gl.mesh + fh.mesh_shade.jxs
                                                       │
                                                       ▼
                                                jit.world display
                                                       ▲
                                                       │
                                             mesh_export.js  (obj / stl / ply)

## Voice format

`voices <x y z shape r> × 8` — 40 floats total. Send the message once
(loadbang) or re-send to reshape the instrument:

    voices  0.30 0.20 0.30 0 0.18   // voice 1: sphere at bottom-left
            0.50 0.20 0.50 0 0.20   // voice 2: sphere at bottom-centre
            0.70 0.20 0.70 0 0.18   // voice 3: sphere at bottom-right
            0.20 0.50 0.50 2 0.16   // voice 4: torus, left wall
            0.80 0.50 0.50 2 0.16   // voice 5: torus, right wall
            0.30 0.80 0.30 1 0.14   // voice 6: box, top-left
            0.70 0.80 0.70 1 0.14   // voice 7: box, top-right
            0.50 0.55 0.50 5 0.28   // voice 8: gyroid, centre (air band)

Or update one at a time:

    voice 3 0.5 0.5 0.5 3 0.22       // voice 4: octahedron, centred, r=0.22
    shape 7 5                        // voice 8: change to gyroid

## Shape codes

| id | primitive                          | good for                        |
|----|------------------------------------|---------------------------------|
| 0  | sphere                             | drums / kick / low frequencies  |
| 1  | box                                | punchy transients               |
| 2  | torus                              | ringing sustained mids          |
| 3  | octahedron                         | metallic / bell-like            |
| 4  | capsule (Y-axis)                   | vocals / body-shape sustains    |
| 5  | gyroid (thickened isosurface)      | pads / ambient / textural       |

## Parameters (live.sliders on the front panel)

| slider       | range        | maps to                              |
|--------------|--------------|--------------------------------------|
| decay        | 0.5 .. 0.999 | how long voxels linger between hits  |
| blur         | 0 .. 1       | smoothness of the field (soft/hard)  |
| threshold    | 0.05 .. 1.5  | isosurface cut - the "filter cutoff" |
| displace     | 0 .. 0.6    | vertex-normal noise displacement      |
| noise_scale  | 0.5 .. 12    | granularity of the noise             |
| organic_bias | 0 .. 1       | palette weight (hot vs moss)         |
| rim          | 0 .. 2       | rim-light strength                   |

## Freeze + export

Flip `freeze-tgl` on to hold the current mesh (voxel field keeps
evolving, but MC stops re-running). Click **obj**, **stl**, **ply**, or
**all**. Files are written to `~/ExternalRadio/fh_meshes/` by default
(matches the ExternalRadio directory convention). Filenames include a
timestamp so nothing overwrites.

Change the export directory with the `dir` message:

    dir /Volumes/work/synth_exports

### Loading exports elsewhere

- **Blender** : File → Import → Wavefront (.obj) / STL / PLY. Normals
  are per-vertex, no material.
- **Cinema 4D** : File → Merge (STL is most reliable at 60k+ tris).
- **Houdini** : File node → OBJ/PLY. STL if you need welded topology.
- **Unreal / Unity** : STL via a converter, or FBX after passing
  through Blender.

## Performance notes

- 48³ voxel field × 8 voices × 60 fps runs comfortably on M1/M2 CPUs
  (the splat loop is JS and touches roughly voice_radius³ cells per
  frame). Drop to 32³ if you need headroom, raise to 64³ for detail.
- Marching cubes is CPU-side JS. Typical output is 2k-8k triangles at
  threshold 0.4; jit.gl.mesh renders this comfortably. Very low
  thresholds (< 0.1) can produce 20k+ triangles.
- The exported STL is binary; a 6k-triangle mesh is ~300 KB.

## Playing it like an instrument

- Bass frequencies feed the low voices (0..2) — they seed spheres near
  the floor. Loud bass = solid base.
- Mids feed voice 4-5 — they seed rings on the walls (organ pipes).
- Highs feed 6-7 — small boxes at the top like sparkles.
- Sub band (voice 8) feeds a big gyroid centred in the volume — the
  "harmonic pad" that fills empty air.

Raise the **threshold** and only the strongest voices survive (like a
low-pass filter, but for topology). Drop it and the field bleeds into
one continuous membrane.

Increase **decay** for long tails; drop it and each note is punchy and
short-lived.

Turn up **displace** for fleshy / molten surfaces that breathe with the
biological pulse; turn it off for hard sculpted geometry.

## Wiring into the main patch

Drop `fh.mesh_synth` next to the existing `jit.world` in
`fluid_heat_audio.maxpat`:

    fh.audio_bins  →  fh.mesh_synth (in-bins)
    qmetro         →  fh.mesh_synth (in-frame)
    pulse output   →  fh.mesh_synth (in-pulse)

The `jit.gl.mesh` inside the synth adds itself to the render context
automatically (same `@context fh` used by the fluid). The heat-colour
mesh and the fluid volume raymarch coexist in one frame.
