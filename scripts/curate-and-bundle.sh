#!/bin/bash
# curate-and-bundle.sh — hands-off: curate the WHOLE megapack, then bundle every
# keeper into the app. Run this on a night you're NOT broadcasting (it wants the
# whole machine, and it takes many hours for a full pack).
#
# It's resumable: if you ⌘Q or it's interrupted, just run it again — it picks up
# where it left off, finishes the pack, then auto-quits and builds the app.
#
# USAGE:
#   ./scripts/curate-and-bundle.sh
#   ./scripts/curate-and-bundle.sh "/path/to/megapack/presets"   # custom input
#
# Tunables (env):
#   CURATE_FRAMES=12   # fewer render frames per preset = faster, slightly rougher
set -euo pipefail
cd "$(dirname "$0")/.."

INPUT="${1:-$HOME/Downloads/MilkDrop 135k+ Presets MegaPack 2026/presets}"
FRAMES="${CURATE_FRAMES:-20}"
CURATED_DIR="Sources/ButterchurnVisualizer/Resources/presets/curated"

if [ ! -d "$INPUT" ]; then
  echo "✗ Input folder not found: $INPUT"
  echo "  Pass the megapack's presets folder as the first argument."
  exit 1
fi

before=$(ls "$CURATED_DIR"/*.json 2>/dev/null | wc -l | tr -d ' ')
echo "▶ Curating everything under:"
echo "    $INPUT"
echo "  Starting from $before keeper(s) already on disk."
echo "  (resumes if interrupted · auto-quits when the whole pack is done)"
echo ""

# Curate. CURATE_AUTOQUIT makes the app terminate itself when the pack is fully
# processed, so this script can continue to the build step unattended.
CURATE_AUTOQUIT=1 CURATE_FRAMES="$FRAMES" CURATE_INPUT="$INPUT" swift run

echo ""
echo "▶ Curation finished — bundling all keepers into the app…"
./scripts/build-app.sh

after=$(ls "$CURATED_DIR"/*.json 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "✔ Done. $after curated presets bundled (was $before)."
echo "  Launch:  ./ButterchurnVisualizer.app/Contents/MacOS/ButterchurnVisualizer"
