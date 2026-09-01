# Butterchurn preset families

Two families, designed to contrast in the same rotation.

| family | register | source of the design |
|--------|----------|----------------------|
| [`fluid_heat/`](fluid_heat/) | hot, incandescent — black → purple → orange → white | the Navier-Stokes + heat + reaction-diffusion solver in `fluid_heat_python/` |
| [`qoix/`](qoix/) | cool, electric — near-black, neon indigo, mint, gold | the QOIX software synthesizer's UI and sound architecture |

Twelve presets total, all built through the real `milkdrop-preset-converter`
and gated for the toolchain bugs documented in each family's README.

## Install

```bash
cp */butterchurn_json/*.json Sources/ButterchurnVisualizer/Resources/presets/
./scripts/build-app.sh
```

Install the **JSON**. Keep the `.milk` sources outside `Resources/` — the
Swift `.milk` loader path drops pixel shaders entirely
(`MilkPresetConverter.toButterchurnDict` emits no `warp`/`comp`), which
would render these as bare mesh motion.

## Before you write another .milk

`fluid_heat/tools/validate_milk.py` (identical copy in `qoix/tools/`) checks
the failure modes that silently break presets in this project. Several were
found the hard way here, and at least two look likely to be present in the
existing library:

- `mod()` in per_frame — EEL has no such function, use `a % b`. Fails the
  whole preset.
- `// comment` on a `per_frame_N=` line — equations are joined with a single
  space, so it comments out every equation after it. The preset loads and
  quietly does almost nothing.
- Identifiers starting with `_` — rejected by milkdrop-eel-parser.
  `Resources/microtonal_warp.milk` has this (`_denom`) and cannot convert.

```bash
python3 fluid_heat/tools/validate_milk.py path/to/*.milk
```
