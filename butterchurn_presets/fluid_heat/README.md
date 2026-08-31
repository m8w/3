# fluid_heat — a Butterchurn preset family

Six presets that port the fluid/heat/reaction-diffusion solver from
`fluid_heat_python/` into Milkdrop's feedback architecture.

| file | what it does |
|------|--------------|
| `fh_01_boussinesq_plume` | heat rises from bass jets; blackbody palette |
| `fh_02_vorticity_confinement` | curl-dominated swirl, two orbiting vortices |
| `fh_03_gray_scott_veins` | real Gray-Scott reaction-diffusion in the buffer |
| `fh_04_organic_decay` | incandescent → moss palette crossfade as heat drains |
| `fh_05_volumetric_lift` | 6-slice raymarch with front-to-back alpha |
| `fh_06_isosurface_contours` | marching-squares iso bands (the mesh-synth cousin) |

    presets/          .milk sources — readable, commented, the thing you edit
    butterchurn_json/ built Butterchurn JSON — the thing you install
    tools/            build + validation

## Installing into ButterchurnVisualizer

**Install the JSON, not the .milk.** See "Why not .milk" below.

```bash
cp butterchurn_json/*.json \
   Sources/ButterchurnVisualizer/Resources/presets/
./scripts/build-app.sh
```

`PresetLoader` scans `Resources/` recursively and passes `.json` through
untouched, so no code changes are needed.

Keep the `.milk` sources **outside** `Resources/` (they are, here at the repo
root). If they end up inside, the loader will pick them up as a second,
broken copy of every preset.

## Why not .milk, in this app

`PresetLoader.swift` does advertise `.milk` support, but that path runs
`MilkPresetConverter.swift`, whose `toButterchurnDict` returns exactly:

```swift
["baseVals", "initEQs", "perFrameEQs", "perPixelEQs", "waves", "shapes"]
```

There is no `warp` and no `comp`. **The Swift `.milk` path silently drops
pixel shaders.** These presets are ~70% shader code — the fluid, the
palettes, the reaction-diffusion all live there — so loaded as `.milk` they
would render as bare warp-mesh motion and look broken. The JSON is built
with the real converter, which keeps both shaders.

(That is a general limitation, not specific to this family: any `.milk`
preset with custom shaders loses them on that path.)

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
| velocity field `u` | per-pixel `dx`/`dy`/`rot`/`zoom` + shader UV offset |
| temperature `T`, density `D` | separate colour channels of the feedback buffer |
| Boussinesq buoyancy | `uv.y += T²·α` — hot pixels sample from below, so they rise |
| vorticity confinement | curl of the scalar field, rotated 90°, added to the UV |
| diffusion `κ∇²T` | explicit 5-tap neighbour average |
| dissipation | `fDecay` + an explicit multiply in the warp shader |
| audio source term | Gaussian falloffs at jet sites × `bass`/`mid`/`treb` |
| Gray-Scott reaction | 9-tap Laplacian + the U/V update, in the warp shader |
| blackbody LUT | inlined `lerp` chain in the comp shader |
| isosurface extraction | contour bands: `abs(f - level) / |∇f|` |
| volumetric raymarch | unrolled slice taps with front-to-back alpha |

## What is *not* portable

- **Pressure projection.** Jacobi needs 20–40 full-field passes per frame;
  Milkdrop gives you one warp pass. These presets are therefore *not*
  incompressible — advection + buoyancy + vorticity without the
  divergence-free constraint. Reads smokier. It is not a CFD solver.
- **Marching cubes / mesh export.** No geometry stage. `fh_06` substitutes
  2D contour bands — the same threshold operation one dimension down.

## Building from source

```bash
npm i milkdrop-preset-converter          # into e.g. /tmp/conv
python3 tools/build_presets.py --npm-dir /tmp/conv
```

The build does three things, each of which fixes real breakage:

**1. Full parenthesisation.** `hlslparser-js` 0.1.1 — pinned by
`milkdrop-preset-converter` 0.1.2, the newest of both — mistranslates any
binary expression with three or more operands at one nesting level. The top
operator becomes `&&` and both sides get `bool()` casts:

| source | translated | |
|---|---|---|
| `a * b * c` | `bool(a*b) && bool(c)` | ✗ |
| `a + b + c + d` | `bool(a+b) && bool(c+d)` | ✗ |
| `a * b + c * d` | `bool(a*b) && bool(c*d)` | ✗ (even mixed precedence) |
| `(a * b) * c` | `(a*b)*c` | ✓ |

The arithmetic silently becomes boolean logic. The shader still compiles and
still renders — just not what was written. `tools/reparen.py` is a
precedence-climbing parser that rewrites every expression into strictly
two-operand form before conversion.

**2. No `GetBlur`.** The converter expands `GetBlur1(uv)` internally to
`tex2D(...).xyz * scaleN + biasN` — a three-operand chain — so the bug fires
*inside its own macro expansion*, where source-level parenthesisation cannot
reach it. All diffusion uses explicit neighbour taps instead, which is closer
to the solver's 5-tap `diffuse.frag` anyway.

**3. A gate that actually catches it.** These shaders contain no boolean
logic at all, so any `&&`, `||`, `bool(` or `bvec(` in the translated GLSL is
proof the bug was hit. `build_presets.py` fails the build on it. Without the
gate the presets convert "successfully" and render nonsense — which is
exactly what happened on the first attempt here.

## Validating

```bash
python3 tools/validate_milk.py     # structure of the .milk sources
python3 tools/normalize_milk.py    # fix comment placement / equation-line comments
python3 tools/build_presets.py     # convert + gate
```

`validate_milk.py` checks numbering gaps, missing shader backticks,
unbalanced braces/parens, unassigned `ret`, `q` variables read but never set,
and calls to user-defined functions (Milkdrop pixel shaders have none —
everything must be inlined). It was itself verified by injecting each of
those defect classes and confirming it catches them.

### Two .milk gotchas this project's parsers impose

- Comments must use `;` and sit **above** `[preset00]`.
  `PresetParser.swift` skips only `;` and `#`, then throws
  `malformedLine` on any in-section line without an `=`.
- A `per_frame_N=// comment` line is **destructive**.
  `MilkPresetConverter` joins equations with a single space, so one `//`
  comments out every equation after it on the joined line. The preset loads,
  runs, and does almost nothing. `normalize_milk.py` hoists these out.

## Tuning

Each preset drives its parameters from `q1..q8` in `per_frame`, so you can
retune without touching shader code:

| preset | knobs |
|---|---|
| 01 | `q1` heat, `q2` momentum, `q3` diffusion, `decay` tail length |
| 02 | `q4` vorticity ε, `q5..q8` vortex orbit centres |
| 03 | `q4` feed F, `q5` kill k — the pair selects the Turing regime |
| 04 | `q4` organic_bias, `q5` decay_mix — how fast heat turns to moss |
| 05 | `q4` yaw, `q5` slab depth, `q6` shear/parallax |
| 06 | `q4` iso threshold, `q5` band width, `q6` shell spacing |

Gray-Scott regimes for `fh_03` (`q4`/`q5`):

    F .035  k .065   spots        F .026  k .055   maze
    F .014  k .045   solitons     F .062  k .061   coral

## Sign convention caveat

`fh_01`, `fh_04`, `fh_05`, `fh_06` assume **v increases downward** in
warp-shader UVs (standard Milkdrop), so `uv.y += heat` samples from below and
heat rises. If plumes sink instead, flip that one sign — it is commented in
each source.

## Untested on screen

The shaders now translate to verified-correct GLSL — the Gray-Scott update,
the 9-tap Laplacian and the palette LUTs were each read back from the
converted output and checked against the source maths. But no
Butterchurn/WebGL runtime was available where these were authored, so nothing
has actually been rendered. Expect to adjust injection gains and decay rates
on first run.
