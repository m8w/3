#!/bin/bash
# fresh-visuals.sh — swap the whole rotation for a brand-new, all-different set.
#
# Archives the current curated presets (SAVED, never deleted), then curates a
# fresh batch from the MegaPack that EXCLUDES everything already seen, so the new
# set has zero overlap with the old one. Finally rebuilds the app so the new set
# is what plays.
#
# USAGE:
#   ./scripts/fresh-visuals.sh
#   ./scripts/fresh-visuals.sh "/path/to/megapack/presets"
#
# Tunables (env):
#   FRESH_COUNT=5000    how many new keepers to collect before it stops
#   CURATE_FRAMES=20    render frames scored per preset (fewer = faster, rougher)
#   CURATE_KEEP_ALL=1   skip scoring entirely — keep everything, fastest, unfiltered
#
# The old set lands in ~/butterchurn_preset_archive/curated_<timestamp>/ with its
# manifest intact. To go back to it later, run:  ./scripts/restore-visuals.sh
set -euo pipefail
cd "$(dirname "$0")/.."

INPUT="${1:-$HOME/Downloads/MilkDrop 135k+ Presets MegaPack 2026/presets}"
KEEP="${FRESH_COUNT:-5000}"
FRAMES="${CURATE_FRAMES:-20}"
CURATED_DIR="Sources/ButterchurnVisualizer/Resources/presets/curated"
ARCHIVE_ROOT="$HOME/butterchurn_preset_archive"

if [ ! -d "$INPUT" ]; then
  echo "Input folder not found: $INPUT"
  echo "Pass your MegaPack presets folder as the first argument."
  exit 1
fi

stamp=$(date +%Y%m%d-%H%M%S)
archive="$ARCHIVE_ROOT/curated_$stamp"
mkdir -p "$ARCHIVE_ROOT"

count=$(find "$CURATED_DIR" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
if [ "${count:-0}" -gt 0 ]; then
  mv "$CURATED_DIR" "$archive"
  mkdir -p "$CURATED_DIR"
  echo "Saved $count current preset(s) to:"
  echo "    $archive"
else
  echo "No current curated presets found to archive; curating fresh."
  archive=""
fi

exclude=""
if [ -n "$archive" ] && [ -f "$archive/_manifest.tsv" ]; then
  exclude="$archive"
  echo "New batch will EXCLUDE everything already seen (zero overlap)."
elif [ -n "$archive" ]; then
  echo "Archived set has no manifest; some presets may repeat."
fi

echo ""
echo "Curating up to $KEEP brand-new keeper(s) from:"
echo "    $INPUT"
echo ""

CURATE_EXCLUDE="$exclude" CURATE_KEEP_TARGET="$KEEP" CURATE_SHUFFLE=1 \
  CURATE_AUTOQUIT=1 CURATE_FRAMES="$FRAMES" CURATE_INPUT="$INPUT" swift run

echo ""
echo "Curation finished — bundling the new set into the app…"
./scripts/build-app.sh

after=$(find "$CURATED_DIR" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "Done. $after brand-new presets are now the rotation."
echo "Old set saved at: ${archive:-<none — nothing was archived>}"
echo "Launch:  MIDI=1 ./ButterchurnVisualizer.app/Contents/MacOS/ButterchurnVisualizer"
