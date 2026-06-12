# Curate Mode — culling a big `.milk` MegaPack

Curate mode converts a folder of MilkDrop `.milk` presets to Butterchurn JSON,
renders each one with synthetic beat-driven audio, scores it for **colourfulness
+ spatial contrast + motion + temporal colour change**, and keeps only the
high-dynamic, colourful ones. Dull / static / solid-black / solid-white presets
are dropped.

It runs **inside the app** (real WKWebView + real GPU), so shader presets convert
and render exactly as they will in normal use — no software-renderer false
positives.

## Where to put the presets

You do **not** need to move the MegaPack into the repo. Leave it in Downloads and
point the app at it:

```bash
# from the repo root (so output lands in the right place)
CURATE_INPUT="$HOME/Downloads/MilkDrop 130k+ Presets MegaPack 2026 2" swift run
```

If your folder is already at that exact Downloads path you can just use:

```bash
CURATE=1 swift run
```

A window opens showing presets flashing by with a live progress HUD
(`processed / total · kept · failed`). Keepers are written as `.json` into:

```
Sources/ButterchurnVisualizer/Resources/presets/curated/
```

When it finishes, run the app normally (`swift run`) and the curated presets are
bundled and played automatically.

## Resumable

Every preset's result is appended to
`Resources/presets/curated/_manifest.tsv` (content-hash, keep, score, file,
name). Re-launching curate mode **skips anything already processed** and skips
duplicate `.milk` contents — so you can stop with ⌘Q and pick up later, or run it
over several nights. Expect roughly ~1 second per unique preset.

## Tuning (all optional env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `CURATE_INPUT` | `~/Downloads/MilkDrop 130k+ Presets MegaPack 2026 2` | source folder (recursive) |
| `CURATE_OUTPUT` | `<repo>/Sources/.../presets/curated` | where keepers are written |
| `CURATE_LIMIT` | _(all)_ | process only the first N (quick test) |
| `CURATE_FRAMES` | `30` | frames rendered per preset |
| `CURATE_FPS` | `30` | render pacing (real time advances motion) |
| `CURATE_WARM` | `6` | warm-up frames before measuring |
| `CURATE_TIMEOUT` | `20` | per-preset seconds before skip+reload |
| `CURATE_MIN_SCORE` | `22` | overall keep threshold (0–100) |
| `CURATE_MIN_MOTION` | `1.2` | min frame-to-frame motion |
| `CURATE_MIN_COLOR` | `8` | min colourfulness |
| `CURATE_MIN_LUMASTD` | `5` | min spatial contrast (OR tcc) |
| `CURATE_MIN_TCC` | `6` | min temporal colour change (OR lumaStd) |

**Want a stricter "best of"?** Raise the bars, e.g.:

```bash
CURATE_MIN_SCORE=35 CURATE_MIN_COLOR=14 CURATE_MIN_MOTION=3 \
  CURATE_INPUT="$HOME/Downloads/MilkDrop 130k+ Presets MegaPack 2026 2" swift run
```

## Quick test before the full run

Do a 300-preset dry run first to sanity-check thresholds:

```bash
CURATE_LIMIT=300 CURATE_INPUT="$HOME/Downloads/MilkDrop 130k+ Presets MegaPack 2026 2" swift run
```

Look at `_manifest.tsv` — the `score` column shows how presets are landing.
Adjust the `CURATE_MIN_*` vars, delete the manifest + curated `.json`s, and rerun
if you want a different cut.
