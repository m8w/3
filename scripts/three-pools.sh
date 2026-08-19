#!/bin/bash
# three-pools.sh — build three SEPARATE preset pools, one per visualizer screen.
#
# Each screen gets its own all-different 5,000-preset folder (15,000 total, zero
# overlap between screens), curated from the MegaPack. Screen 1's pool is built
# first; screen 2 excludes everything in screen 1; screen 3 excludes both. The
# app then draws each screen only from its own pool, so the three screens can
# never show the same preset — far less repetition.
#
# The previous shared set (presets/curated) is archived first, never deleted.
#
# USAGE:
#   ./scripts/three-pools.sh
#   ./scripts/three-pools.sh "/path/to/megapack/presets"
#
# Tunables (env):
#   POOL_SIZE=5000      keepers per screen
#   CURATE_FRAMES=20    render frames scored per preset (fewer = faster, rougher)
#   CURATE_KEEP_ALL=1   skip scoring — keep everything, fastest, unfiltered
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

INPUT="${1:-$HOME/Downloads/MilkDrop 135k+ Presets MegaPack 2026/presets}"
SIZE="${POOL_SIZE:-5000}"
FRAMES="${CURATE_FRAMES:-20}"
PRESETS="$ROOT/Sources/ButterchurnVisualizer/Resources/presets"
ARCHIVE_ROOT="$HOME/butterchurn_preset_archive"

if [ ! -d "$INPUT" ]; then
  echo "Input folder not found: $INPUT"
  echo "Pass your MegaPack presets folder as the first argument."
  exit 1
fi

stamp=$(date +%Y%m%d-%H%M%S)
mkdir -p "$ARCHIVE_ROOT"

if [ -d "$PRESETS/curated" ] && [ -n "$(find "$PRESETS/curated" -maxdepth 1 -name '*.json' 2>/dev/null)" ]; then
  mv "$PRESETS/curated" "$ARCHIVE_ROOT/curated_$stamp"
  echo "Archived the old shared set to $ARCHIVE_ROOT/curated_$stamp"
fi

s1="$PRESETS/screen1"; s2="$PRESETS/screen2"; s3="$PRESETS/screen3"
mkdir -p "$s1" "$s2" "$s3"

curate() {
  out="$1"; exclude="$2"; label="$3"
  echo ""
  echo "Curating $SIZE preset(s) for $label"
  echo "    into:     $out"
  echo "    excludes: ${exclude:-none}"
  CURATE_EXCLUDE="$exclude" CURATE_OUTPUT="$out" CURATE_KEEP_TARGET="$SIZE" \
    CURATE_SHUFFLE=1 CURATE_AUTOQUIT=1 CURATE_FRAMES="$FRAMES" CURATE_INPUT="$INPUT" swift run
}

curate "$s1" ""          "screen 1"
curate "$s2" "$s1"       "screen 2"
curate "$s3" "$s1:$s2"   "screen 3"

echo ""
echo "Bundling the three pools into the app…"
./scripts/build-app.sh

c1=$(find "$s1" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
c2=$(find "$s2" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
c3=$(find "$s3" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "Done. Screen pools (all different): $c1 / $c2 / $c3"
echo "Launch:  MIDI=1 ./ButterchurnVisualizer.app/Contents/MacOS/ButterchurnVisualizer"
