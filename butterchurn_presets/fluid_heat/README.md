# fluid_heat — a Butterchurn preset family

Six presets that port the fluid/heat/reaction-diffusion solver from
`fluid_heat_python/` into Milkdrop's feedback architecture.

| file | what it does |
|------|--------------|
| `fh_01_boussinesq_plume.milk` | heat rises from bass jets; blackbody palette |
| `fh_02_vorticity_confinement.milk` | curl-dominated swirl, two orbiting vortices |
| `fh_03_gray_scott_veins.milk` | real Gray-Scott reaction-diffusion in the buffer |
| `fh_04_organic_decay.milk` | incandescent → moss palette crossfade as heat drains |
| `fh_05_volumetric_lift.milk` | 6-slice raymarch with front-to-back alpha |
| `fh_06_isosurface_contours.milk` | marching-squares iso bands (the mesh-synth cousin) |

## Why this port is natural

Milkdrop's **warp shader is already a semi-Lagrangian advection step**. It
samples the previous frame at a displaced UV and writes the result — which is
exactly `advect.frag` in the Python solver:

    back = uv - velocity * dt;  ret = texture(state, back);

So the framebuffer feedback *is* the fluid integrator. Everything else is
choosing what the displacement means:

| solver concept | Milkdrop mechanism |
|---|---|
| semi-Lagrangian advection | warp shader UV displacement + frame feedback |
| velocity field `u` | the per-pixel `dx`/`dy`/`rot`/`zoom` + shader UV offset |
| temperature `T`, density `D` | separate colour channels of the feedback buffer |
| Boussinesq buoyancy | `uv.y += T²·α` — hot pixels sample from below, so they rise |
| vorticity confinement | curl of the scalar field, rotated 90°, added to the UV |
| diffusion `κ∇²T` | `GetBlur1()` / `GetBlur2()` taps blended into the advected value |
| dissipation | `fDecay` + an explicit multiply in the warp shader |
| audio source term `S`, `f` | Gaussian falloffs around jet sites × `bass`/`mid`/`treb` |
| Gray-Scott reaction | 9-tap Laplacian + the U/V update, in the warp shader |
| blackbody LUT | inlined `lerp` chain in the comp shader |
| isosurface extraction | contour bands: `abs(f - level) / |∇f|` |
| volumetric raymarch | unrolled slice taps with front-to-back alpha |

## What is *not* portable

Two things from the solver have no Milkdrop equivalent and are deliberately
absent:

- **Pressure projection.** Jacobi needs 20–40 full-field passes per frame;
  Milkdrop gives you one warp pass. These presets are therefore *not*
  incompressible — they are advection + buoyancy + vorticity without the
  divergence-free constraint. Visually this reads as slightly smokier and
  more dissipative, which is fine here; it is not a CFD solver.
- **Marching cubes / mesh export.** No geometry stage. `fh_06` substitutes
  2D contour bands, which is the same threshold operation one dimension down.

## Two corrections applied during the port

Values from the original Max patch do **not** transfer directly, and both
of these produce a broken-looking preset if carried over verbatim:

1. **Saturating injection.** Additive injection (`T += amp·k`) integrates
   without bound against a 0.99 decay and flat-clips to white within a few
   seconds. All six presets inject *toward* saturation instead:

       adv += src * saturate(1.0 - adv);

   which is the same `x += a·(1-x)` form the Python solver uses.

2. **Gentler volumetric parameters.** Every raymarch sample is the *same*
   2D image at an offset, so the pass is as much a directional blur as a
   depth cue. `fh_05` uses depth 0.24–0.40 and shear 0.07–0.12; the Max
   defaults (1.2 / 0.35) smear the plumes into mush.

## Installing

These are canonical `.milk` files, so they drop into any Milkdrop-compatible
preset directory. For Butterchurn specifically:

**If your repo loads `.milk` at runtime** — copy `presets/*.milk` into your
preset folder and they will be picked up by whatever index you already use.

**If your repo ships converted JSON** — convert with the official tool:

```bash
npm i -g milkdrop-preset-converter-aws
for f in presets/*.milk; do
  milkdrop-preset-converter "$f" > "converted/$(basename "${f%.milk}").json"
done
```

or in-process:

```js
import { convertPreset } from 'milkdrop-preset-converter-aws';
import fs from 'fs';

const milk = fs.readFileSync('fh_01_boussinesq_plume.milk', 'utf8');
const preset = convertPreset(milk);          // -> butterchurn preset object
visualizer.loadPreset(preset, 2.0);          // 2.0s blend
```

## Validating

```bash
python3 tools/validate_milk.py
```

Checks numbering gaps, missing backticks on shader lines, unbalanced
braces/parens, `ret` never assigned, `q` variables read but never set, and
calls to user-defined functions (which Milkdrop pixel shaders do not
support — everything must be inlined). All six pass.

This is a **structural** check only. It does not compile the HLSL, and I had
no Butterchurn runtime available to render them — see the caveat below.

## Tuning

Each preset drives its main parameters from `q1..q8` in `per_frame`, so you
can retune without touching shader code:

| preset | knobs |
|---|---|
| 01 | `q1` heat, `q2` momentum, `q3` diffusion, `decay` tail length |
| 02 | `q4` vorticity ε, `q5..q8` vortex orbit centres |
| 03 | `q4` feed F, `q5` kill k — the pair selects the Turing regime |
| 04 | `q4` organic_bias, `q5` decay_mix — how fast heat turns to moss |
| 05 | `q4` yaw, `q5` slab depth, `q6` shear/parallax |
| 06 | `q4` iso threshold, `q5` band width, `q6` shell spacing |

Gray-Scott regimes worth trying in `fh_03` (`q4`/`q5`):

    F .035  k .065   spots        F .026  k .055   maze
    F .014  k .045   solitons     F .062  k .061   coral

## Sign convention caveat

`fh_01`, `fh_04`, `fh_05` and `fh_06` assume **v increases downward** in
warp-shader UVs (standard D3D/Milkdrop), so `uv.y += heat` samples from
below and makes heat rise. If plumes sink instead of rising in your build,
flip the sign on that one line in the affected preset — it is commented in
each file.

## Untested

I wrote and structurally validated these but could not run them: no
Butterchurn/WebGL runtime was available in the session where they were
authored. The physics and the palette maths are ported from code that *was*
verified numerically (see `fluid_heat_python/docs/FLUID.md`), but the HLSL
has not been compiled and no preset has been seen on screen. Expect to
adjust injection gains and decay rates on first run.
