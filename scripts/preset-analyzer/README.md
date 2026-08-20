# Preset Analyzer

Headless WebGL2 harness (Playwright + butterchurn) that loads each preset,
feeds it synthetic audio, renders ~1.5s, and measures the resulting canvas
for color variance (`std2`) and frame-to-frame motion (`motion`).

## Usage

```bash
# Run the full analysis (or pass a LIMIT as first arg for a quick test)
node scripts/preset-analyzer/analyze.js [limit] [frames] /tmp/preset_results.json

# Classify results into good / dull / dead / errored
node scripts/preset-analyzer/classify.js /tmp/preset_results.json
```

Outputs `presets_dead.txt`, `presets_dull.txt`, `presets_good.txt`,
`presets_errored.txt` next to the results file.

## Caveat: shader presets

This runs on the `swiftshader` software GL renderer, which often fails to
compile advanced custom warp/comp shaders used by "fractal"/"shader" style
presets. Those presets can show up as false-positive "dead" here even though
they render fine on a real GPU (e.g. the M2 Mac). When reviewing flagged
presets, check whether `warp`/`comp` fields contain non-trivial custom
shader code before deleting — those need visual confirmation on real
hardware instead.

## 2026-06-11 run (1754 presets)

- Good: 1617
- Dull: 55 (38 have custom shaders — needs GPU re-check)
- Dead: 82 (60 have custom shaders — needs GPU re-check)
- Errors: 0

The 39 non-shader dead/dull presets were moved to
`Sources/ButterchurnVisualizer/Resources/presets_review/` (excluded from the
app bundle) for manual review before permanent deletion. Reports for the full
run are in `reports/`.
