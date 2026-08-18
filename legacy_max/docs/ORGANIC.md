# Living Audio: the organic layer

Everything in this folder extends the fluid-heat solver from pure Navier-Stokes
math into a *breathing* system. The rule: audio is the stimulus, not the output.

## Design

| biological metaphor | implementation                                  |
|---------------------|-------------------------------------------------|
| nervous system      | Brownian drift on injection sites + ASR envelopes |
| metabolism          | heat-modulated viscosity (hot = thin, cold = thick) |
| chemistry           | Gray-Scott reaction-diffusion fed by heat       |
| decay / aging       | palette crossfade incandescent -> organic       |
| memory              | archive video as spatial vector field           |
| dreaming            | flow-warped ghost crossfade between clips       |

## Component files

### `abstractions/fh.organic_mod.maxpat`
Turns the raw 8-bin FFT list into something alive. For each bin:
- threshold gate triggers an `adsr~` envelope (per-bin attack/release)
- a `jit.bfg` simplex-noise generator adds a low-amplitude drift
- the two are summed; silence still breathes.

Wire it between `fh.audio_bins` and the inject slab:

    fh.audio_bins  -->  fh.organic_mod  -->  bins list  -->  fh.inject

### `shaders/fh.inject.jxs`  (updated)
Adds `time`, `jitter`, `swarm_rate` uniforms. Each of the 8 injection
centres now wobbles through a pseudo-Perlin noise field; jet direction
is also modulated, so the "source of heat" appears to swim around the grid
instead of sitting at fixed pixels.

### `shaders/fh.viscosity.jxs`
Runs between vorticity and divergence. Local `vel` is multiplied by a heat-
dependent factor (0.93 .. 1.02), so cold regions settle into a syrupy drag
and hot regions accelerate.

### `shaders/fh.reaction.jxs`
Gray-Scott reaction-diffusion on its own RG texture. Heat (fluid B plane)
raises the feed rate; density (fluid A) seeds V. Produces Turing veins that
grow where the fluid has been recently active. Feeds `fh.organic_lut.jxs`.

### `shaders/fh.organic_lut.jxs`
Replaces `fh.blackbody.jxs`. Cross-fades between an **incandescent** palette
(black -> #1a0033 -> #e63e00 -> #ffcc00 -> white) for fresh heat, and an
**organic** palette (forest -> vein purple -> moss -> lichen -> bone) for
decay. The weighting is `decay = D - T` with additional bias so lingering
density with low temperature reads as cooling/biological.

### `shaders/fh.video_displace.jxs`
Pre-inject stage. Samples an archive video as:
- luminance -> local heat + density injection
- luminance gradient (Sobel) -> directional velocity force
- dark areas -> velocity damping ("thick ink")

Your old brush-and-ink footage literally becomes a flow field that today's
audio is forced to move through.

## Recommended pipeline order

For a fully "living" per-frame chain:

    tex_state_a
      -> [fh.video_displace]   (optional: archive vector field)
      -> fh.inject             (audio jets with drift + ASR)
      -> fh.buoyancy
      -> fh.advect
      -> fh.diffuse
      -> fh.viscosity          (heat -> sluggishness)
      -> fh.vorticity
      -> fh.divergence -> jacobi x20 -> fh.gradient
      -> tex_state_a  (ping-pong)
                                |
                                +-> fh.reaction (rd texture) -> rd_state
                                |
    (tex_state_a, asemic, rd_state) -> fh.organic_lut -> fh.volume -> display

## Tuning

| param         | file                    | feel                           |
|---------------|-------------------------|--------------------------------|
| `jitter`      | fh.inject.jxs           | 0 = robotic, 0.2 = wandering   |
| `swarm_rate`  | fh.inject.jxs           | drift speed                    |
| `visc_cold`   | fh.viscosity.jxs        | how "thick" cold ink gets      |
| `F`, `kk`     | fh.reaction.jxs         | RD regime - try 0.035/0.065 (spots), 0.026/0.055 (maze) |
| `heat_feed`   | fh.reaction.jxs         | how strongly audio feeds RD    |
| `organic_bias`| fh.organic_lut.jxs      | bias toward moss palette       |
| `decay_mix`   | fh.organic_lut.jxs      | how quickly "aging" kicks in   |
| `vein_gain`   | fh.organic_lut.jxs      | RD veins visibility            |
