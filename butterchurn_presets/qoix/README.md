# qoix — a Butterchurn preset family from the QOIX synthesizer

Six presets adapting the QOIX software synthesizer's **visual language** and
**sound architecture** into Butterchurn. Cool and electric, deliberately the
counterpart to `../fluid_heat/`, which is hot and incandescent — the two
families are designed to sit in the same rotation without looking related.

| file | the synth concept it draws |
|------|----------------------------|
| `qoix_01_fm_operator_lattice` | 4-operator FM: carriers glow, modulators bend |
| `qoix_02_wavetable_scan` | wavetable scan head + 16 Oxford harmonics |
| `qoix_03_modmatrix_web` | SN2 mod matrix: bipolar source→destination links |
| `qoix_04_microtonal_shear` | EDO bins q1–q32; adjacent-bin differential shear |
| `qoix_05_spectral_bloom` | the spectrum display's `hsl(h,70%,25+v*45%)` bars |
| `qoix_06_rave_strobe` | beat-gated strobe (⚠ see photosensitivity note) |

## Install

```bash
cp butterchurn_json/*.json Sources/ButterchurnVisualizer/Resources/presets/
./scripts/build-app.sh
```

Install the **JSON**, and keep the `.milk` sources outside `Resources/`. The
Swift `.milk` path drops pixel shaders — see `../fluid_heat/README.md`.

## The palette

Lifted from QOIX's `styles.css` custom properties and the canvas drawing in
`ui.js`:

| role | hex | used for |
|------|-----|----------|
| panel black | `#09090d` | background floor |
| inset | `#090e0c` / `#141f1b` | vignette, panel body |
| border | `#2a4038` | low-end of the ramp |
| accent | `#5b67d8` | indigo — the primary signal colour |
| glow | `#7b86f5` | the `shadowBlur` bloom |
| green | `#4fc97e` | mint — carriers, sources, "active" |
| yellow | `#e6c84a` | gold — peaks and destination nodes |
| accent2 | `#e05c5c` | coral — negative/bipolar |
| bright | `#eef6f3` | white-point |

QOIX's scopes get their look from `ctx.shadowColor` + `ctx.shadowBlur` on
neon strokes over near-black. There is no blur primitive available here (see
below), so every preset fakes it with a 4-tap ring sampled at 3–9 texels and
added back **in the accent colour**, not in white — that is what keeps it
reading as neon rather than as glare.

## Sound → visual mapping

- **FM** (`fm.js`): operators are placed on a square. Modulators are not
  drawn at all; they displace the sampling position of the operator they
  feed, which is what an FM modulator does to a carrier's phase. Raising
  bass raises the index, and the carrier rings visibly break into sidebands.
- **Wavetable** (`wavetable.js`): the screen *is* the table. A scan head
  sweeps at the morph rate; harmonic *n* stripes at spatial frequency *n*
  with a 1/n rolloff that treble tilts — the Oxford harmonic bank collapsed
  to one control.
- **Mod matrix** (`modmatrix.js`): sources left, destinations right, links
  between. Amount is brightness; **sign is hue** — indigo positive, coral
  negative, matching how the QOIX UI draws bipolar controls.
- **Microtonal** (`microtonal.js`): uses q1–q32 if your `MicrotonalFFT.swift`
  is injecting them, and falls back to bass/mid/treb otherwise, so it is not
  dead weight in a stock host. The payload is adjacent-bin *differences* —
  a note sitting between two scale degrees fires both bins with one winning,
  and that asymmetry tilts the field. A 12-TET preset cannot show this.
- **Beat detection** (`qoix_06`): peak tracker with a decaying threshold,
  the same shape as the envelope followers in the synth.

## ⚠ Photosensitivity

`qoix_06_rave_strobe` flashes at beat rate — 2–4 Hz for most dance material.
That is at the low edge of the 3–60 Hz band associated with photosensitive
seizures, not safely clear of it. **Do not put it in an unattended public
rotation.** To soften it: raise `fDecay` toward 0.99 and lower `q6` (strobe
depth). The other five presets are continuous-motion and carry no such
warning.

## Building

```bash
npm i milkdrop-preset-converter          # into e.g. /tmp/conv
python3 tools/validate_milk.py
python3 tools/build_presets.py --npm-dir /tmp/conv
```

Same pipeline as `fluid_heat` — full expression parenthesisation to dodge
the `hlslparser-js` operand-chain bug, then a gate that rejects any `&&`,
`||`, `bool(` or `bvec(` in the translated GLSL. These shaders contain no
boolean logic, so any of those is proof the bug fired. All 12 presets across
both families currently build with **zero** artifacts.

### Toolchain constraints, all confirmed against the real converter

Every one of these fails the **entire preset**, and most fail *silently*:

| you write | where | what happens | write instead |
|---|---|---|---|
| `mod(a, b)` | per_frame / per_pixel | `No function matching: mod` — whole preset fails | `a % b` |
| `mod(a, b)` | warp / comp | shader comes back **empty** | `fmod(a, b)` |
| `distance(a, b)` | warp / comp | shader comes back **empty** | `length(a - b)` |
| `_name` | per_frame | parse error, per_frame block lost | `name` |
| `GetBlur1(uv)` | warp / comp | arithmetic silently becomes boolean | explicit taps |
| `a * b * c` | warp / comp | becomes `bool(a*b) && bool(c)` | `(a * b) * c` |
| `// comment` | per_frame line | comments out **every equation after it** | hoist to header |

`tools/validate_milk.py` checks all of these. It is worth running against
your existing 1716 presets — the `mod()` and `//`-on-an-equation-line cases
in particular produce presets that load, run, and quietly do almost nothing.

## Tuning

| preset | knobs |
|---|---|
| 01 | `q4` FM index, `q5`/`q6` operator ratios, `alg` algorithm walk rate |
| 02 | `q4` scan position, `q5` harmonic tilt, `q6` head width |
| 03 | `q4`–`q6` link amounts, `q7`/`q8` bipolar signs |
| 04 | `hasmt` fallback switch, `shear*` differentials, `cent` centroid hue |
| 05 | `q4` saturation, `q5` outward rate, `q7` bloom width |
| 06 | `q6` strobe depth, `0.16` in `isbeat` = minimum beat spacing |

## Untested on screen

Same caveat as `fluid_heat`: the GLSL is verified correct by reading it back
out of the converter, but no WebGL runtime was available here, so none of
these has actually been rendered. Expect to adjust deposit gains and decay
on first run.
