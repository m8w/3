# fluid_heat_audio

3D fluid dynamics + heat transfer visualizer that reacts to sound. Built for
Max 9.1 / Jitter with GLSL fragment shaders running on the GPU.

## Mathematical model

The system couples two PDEs on a ping-pong texture grid.

**Navier-Stokes (velocity field `u = (u, v)`)**

    du/dt + (u . grad) u = -grad p / rho + nu * laplacian(u) + f

**Convection-Diffusion (temperature `T`)**

    dT/dt + (u . grad) T = kappa * laplacian(T) + S

**Boussinesq coupling** - heat drives upward force on the fluid:

    f_buoy = (alpha * (T - T_amb) - beta * D) * up

Where `S` (heat source) and `f` (impulse) are **driven by audio**. Loud = hot,
frequency bin = spatial jet.

## State encoding

One RGBA32F texture holds the full simulation state:

| plane | meaning                          |
|-------|----------------------------------|
| R     | velocity x (u)                   |
| G     | velocity y (v)                   |
| B     | temperature T                    |
| A     | density / "ink" D                |

## Pipeline (per frame)

Each frame `qmetro 16` (~60 fps) bangs the solver chain:

1. **inject**     - 8 audio bins become 8 spatial jets (`fh.inject.jxs`)
2. **buoyancy**   - Boussinesq lift from heat (`fh.buoyancy.jxs`)
3. **advect**     - semi-Lagrangian back-trace (`fh.advect.jxs`)
4. **diffuse**    - per-channel viscosity + thermal diffusion (`fh.diffuse.jxs`)
5. **vorticity**  - curl confinement for expressive swirl (`fh.vorticity.jxs`)
6. **divergence** - d(u)/dx + d(v)/dy       (`fh.divergence.jxs`)
7. **jacobi x20** - iterative pressure solve (`fh.jacobi.jxs`)
8. **gradient**   - subtract grad(p) -> incompressible flow (`fh.gradient.jxs`)
9. **blackbody**  - heat -> color via 5-stop LUT, distorts asemic layer (`fh.blackbody.jxs`)
10. **volume**    - pseudo-3D raymarch for depth (`fh.volume.jxs`)

## Audio mapping

`fh.audio_bins.maxpat` takes mono audio in, runs `zsa.bands~` (or `analyzer~`),
and emits an 8-element list of 0..4 floats at 40 Hz:

| bin | spatial jet              | direction |
|-----|--------------------------|-----------|
| 0   | sub-bass, bottom-left    | up        |
| 1   | bass, bottom-center      | up        |
| 2   | low-mid, bottom-right    | up        |
| 3   | mid, left wall           | right     |
| 4   | upper-mid, right wall    | left      |
| 5   | presence, top-left       | down-right |
| 6   | brilliance, top-right    | down-left |
| 7   | air, center              | up        |

Each bin injects velocity impulse (`force_amt`), heat (`heat_amt`), and
density (`density_amt`) within a Gaussian falloff.

### Using external JSON

To bypass the built-in analyzer and drive from JSON/OSC, send a list of
eight floats (your own FFT) into inlet 0 of the `fh.inject` slab, prepended
with `bins`:

    bins 0.4 0.9 0.1 0.0 0.2 0.6 0.8 0.3

## Heat -> color LUT

Five stops, interpolated in linear RGB:

    0.00  #000000  black        (cold)
    0.25  #1a0033  deep purple  (cool)
    0.50  #e63e00  burning      (warm)
    0.75  #ffcc00  yellow       (hot)
    1.00  #ffffff  white        (peak)

Tone-mapped with Reinhard and gained by `exposure`. Density modulates
brightness. Neighborhood tap adds soft glow.

## Asemic layer

Drop `assets/asemic.png` next to the patch. It is sampled with UVs warped
by the local velocity field (scale `asemic_flow`) and tinted by the local
heat color before being additively mixed.

## Parameters (live sliders)

| param        | range    | effect                          |
|--------------|----------|---------------------------------|
| heat_gain    | 0 .. 4   | overall heat -> color intensity |
| alpha        | 0 .. 5   | buoyancy strength               |
| beta         | 0 .. 2   | gravity on density              |
| epsilon      | 0 .. 2   | vorticity confinement           |
| asemic_mix   | 0 .. 1   | asemic layer blend              |
| exposure     | 0.1 .. 3 | final tonemap gain              |

## File layout

    fluid_heat_audio/
    |-- fluid_heat_audio.maxpat       main patch
    |-- abstractions/
    |   |-- fh.audio_bins.maxpat      audio -> 8 bins
    |   |-- fh.organic_mod.maxpat     ASR + Brownian modulation (nervous system)
    |   |-- fh.archive_fetcher.maxpat SQLite + jit.movie A/B crossfader
    |   |-- fh.archive_pair.maxpat    dual-channel fetcher (53k skin + 10k nerves)
    |   |-- fh.resolver_bridge.maxpat OSC bridge to archive_resolver.py
    |-- shaders/
    |   |-- fh.inject.jxs             audio-driven source term + drift
    |   |-- fh.video_displace.jxs     archive video as vector field (legacy single-archive)
    |   |-- fh.video_vector.jxs       Channel B -> velocity field only (nerves, 10k)
    |   |-- fh.video_skin.jxs         Channel A -> skin/density overlay (texture, 53k)
    |   |-- fh.advect.jxs             semi-Lagrangian
    |   |-- fh.buoyancy.jxs           Boussinesq
    |   |-- fh.diffuse.jxs            viscosity + thermal
    |   |-- fh.viscosity.jxs          heat-modulated viscosity
    |   |-- fh.vorticity.jxs          curl confinement
    |   |-- fh.reaction.jxs           Gray-Scott reaction-diffusion
    |   |-- fh.divergence.jxs         div(u)
    |   |-- fh.jacobi.jxs             pressure iteration
    |   |-- fh.gradient.jxs           subtract grad(p)
    |   |-- fh.blackbody.jxs          heat -> color + asemic
    |   |-- fh.organic_lut.jxs        biological palette (incandescent -> moss)
    |   |-- fh.volume.jxs             raymarched "3D" lift
    |   |-- fh.crossfade.jxs          flow-warped A/B clip crossfade
    |-- scripts/
    |   |-- archive_indexer.py        ffprobe + SQLite indexer (local + remote URLs)
    |   |-- archive_fetcher.js        Max JS heat-aware picker
    |   |-- archive_resolver.py       OSC sidecar: yt-dlp + LRU cache (no full local copy needed)
    |   |-- youtube_to_csv.py         dump a YouTube channel/playlist to CSV for indexer
    |-- docs/
        |-- README.md                 this file
        |-- PATCHING.md               open & run checklist
        |-- ORGANIC.md                living-system layer (organic palette, RD, ASR)
        |-- ARCHIVE.md                50k-video archive integration

## Dependencies

- Max 9.1 (Jitter) with OpenGL 2.1+ capable GPU
- One of: `zsa.descriptors` (zsa.bands~) or `analyzer~` (from IRCAM / jean-francois charles)
- Optional: any PNG/JPG at `assets/asemic.png`

## Organic / living extensions

See [ORGANIC.md](ORGANIC.md) for the biological modulation layer: ASR
envelopes, Brownian drift on injection sites, heat-modulated viscosity,
Gray-Scott reaction-diffusion, and the incandescent->moss palette shift.

See [ARCHIVE.md](ARCHIVE.md) for wiring a 50k+ video SQLite archive into
the solver as a heat-aware vector field with flow-warped crossfades.

See [STREAMING.md](STREAMING.md) when the archive lives on YouTube and
can't be fully copied to disk - a resolver sidecar uses yt-dlp + an LRU
disk cache so only the working set ever lands locally.

## Notes

The `jit.gl.slab` chain uses `@file` shader paths relative to the patch.
Keep the `shaders/` folder next to `fluid_heat_audio.maxpat`. Texture
storage is `float32` for stable simulation; the final display quad uses
standard 8-bit output.
