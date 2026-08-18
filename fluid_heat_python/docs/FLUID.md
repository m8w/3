# The fluid instrument

GPU Stable Fluids (Stam) coupled to a heat field via the Boussinesq
approximation, with Gray-Scott reaction-diffusion riding on top. Audio is
the source term for both momentum and thermal energy.

## Equations

**Momentum** (Navier-Stokes, incompressible):

    du/dt + (u . grad)u = -grad p / rho + nu * lap(u) + f

**Heat** (convection-diffusion):

    dT/dt + (u . grad)T = kappa * lap(T) + S

**Coupling** (Boussinesq) - heat lifts, density sinks:

    f_buoy = (alpha * (T - T_amb) - beta * D) * up

`S` and `f` come from the audio: 8 frequency bands become 8 spatial jets.

## State packing

One RGBA32F texture pair carries everything:

| channel | quantity          |
|---------|-------------------|
| R       | velocity x        |
| G       | velocity y        |
| B       | temperature T     |
| A       | density D ("ink") |

Units are **normalized screen space per second** — a velocity of 1.0 crosses
the frame in one second. This matters: the Max original worked in pixel
units, and porting those numbers directly makes the field integrate to
T≈5, D≈3 and blow the palette out to solid white.

## Pass order (per frame)

    video_vector   archive Channel B luma gradient -> velocity   (optional)
    inject         8 audio jets, Brownian-drifting positions
    buoyancy       Boussinesq lift
    advect         semi-Lagrangian back-trace
    diffuse        viscosity + thermal + molecular diffusion
    viscosity      heat-modulated thinning / thickening
    vorticity      curl confinement
    divergence     div(u)
    jacobi x N     pressure Poisson solve
    gradient       subtract grad(p)  ->  divergence-free
    reaction       Gray-Scott fed by heat
    organic_lut    heat -> colour, veins, asemic/skin composite
    video_skin     archive Channel A -> skin overlay             (optional)
    volume         pseudo-3D raymarch lift

## Bounded injection

Temperature and density inject *toward saturation* rather than accumulating:

    T += amp * heat_amt * falloff * (1 - T)

so a sustained loud signal settles at 1.0 instead of integrating without
bound. Velocity stays additive (it has to be able to cancel itself out) but
is clamped to `max_vel` as a CFL guard on the semi-Lagrangian back-trace.

## Pressure projection — measured behaviour

Divergence removed, measured on the field interior (a 4-texel boundary band
is excluded; the textures are clamp-to-edge, so a wrapping measurement
reports a large fake discontinuity there and hides the real result):

| iterations | divergence removed |
|-----------:|-------------------:|
| 2          | 51%                |
| 5          | 66%                |
| 10         | 91%                |
| 20         | 95%                |
| 40         | 96%                |
| 80         | 97%                |

**Warm start caveat.** Reusing the previous frame's pressure is free
accuracy once there are enough iterations to relax it — but below ~10 the
stale gradient is subtracted before it has been corrected and the
projection *injects* divergence (measured 13× worse than doing no
projection at all). `FluidSim.WARM_START_MIN_ITERS` forces a cold start
below that threshold, so low iteration counts degrade gracefully instead
of exploding.

The divergence/gradient pair uses backward/forward differences, which
compose to exactly the compact 5-point Laplacian that Jacobi inverts. This
is the textbook-consistent discretization and one texture fetch cheaper
than central differences — but measured convergence is the same either
way, so it is a tidiness/cost choice, not a fix.

## Parameters

Everything lives on `FluidParams` (`fh/fluid.py`). The ones worth playing:

| param            | default | effect                                    |
|------------------|--------:|-------------------------------------------|
| `heat_amt`       | 2.2     | how hot a loud band makes its jet         |
| `force_amt`      | 0.9     | how hard a band pushes the fluid          |
| `max_vel`        | 1.2     | CFL clamp (raise for wilder motion)       |
| `alpha`          | 1.8     | buoyancy — heat's upward pull             |
| `beta`           | 0.25    | gravity on density                        |
| `epsilon`        | 0.35    | vorticity confinement (curl/swirl)        |
| `diss_T` `diss_D`| 0.985   | per-frame decay (~0.75 s half-life)       |
| `visc_cold/hot`  | .93/1.02| cold = syrupy, hot = thin and fast        |
| `jacobi_iters`   | 40      | pressure solve quality                    |
| `F`, `k`         | .035/.065 | Gray-Scott regime (spots / maze / coral) |
| `organic_bias`   | 0.55    | palette weight, incandescent vs moss      |
| `exposure`       | 1.1     | final tonemap gain                        |

## Volumetric lift

The `volume` pass raymarches the 2D colour field at rotated/sheared slice
offsets. It is as much a directional blur as a depth cue — the Max defaults
(`depth 1.2`, `swirl 0.6`, `shear 0.35`) smear the plumes into mush at these
resolutions. Defaults here are tuned down (`0.5 / 0.30 / 0.15`) to read as
parallax while keeping flow structure legible. `V` toggles it live.

## Running it

    python3 main.py --mode fluid
    python3 main.py --mode both                      # fluid backdrop + mesh
    python3 main.py --mode fluid --asemic art.png    # image carried by the flow
    python3 main.py --mode fluid --fluid-width 1024 --fluid-height 576

Offscreen, no window (works over SSH):

    xvfb-run -a python3 main_fluid_headless.py --wav track.wav --out frames/
    ffmpeg -framerate 30 -i frames/fluid_%05d.png -c:v libx264 -pix_fmt yuv420p out.mp4

## Archive integration

With a `videos.sqlite` built by `scripts/archive_indexer.py`:

    python3 main.py --mode fluid --archive-db videos.sqlite --use-resolver

- **Channel A** (`role='texture'`, your 53k) becomes the *skin*: UV-warped
  by local velocity, tinted by heat colour, mixed by density × heat.
- **Channel B** (`role='velocity'`, your 10k) becomes the *nerves*: its luma
  gradient is a pure directional force. Its heat query is inverted, so when
  the audio heats Channel A the nerves pull cool clips and you get
  counter-flow rather than two layers doing the same thing.

Clip choice ranks rows by weighted distance across `organic`, `energy` and
`viscosity` against the live audio descriptors. Remote rows (`remote=1`)
are handed to `scripts/archive_resolver.py` over OSC, which returns a
locally cached path — the render loop never blocks on a download.

If the database was never split by role, both channels fall back to the
full pool automatically.

## Performance

Measured on software GL (llvmpipe) at 256×144: ~17 fps for the full chain
with 40 Jacobi iterations. On real GPU hardware this is not close to a
bottleneck — the pass count is ~55 fullscreen draws per frame, which a
2015-era discrete GPU handles at 1080p without effort. If you do need
headroom, `jacobi_iters` is the biggest single lever.
