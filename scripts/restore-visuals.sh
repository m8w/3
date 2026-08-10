#!/bin/bash
# restore-visuals.sh — bring a saved preset set back into the rotation.
#
# Whatever is currently loaded is archived first (so nothing is ever lost), then
# a saved batch from ~/butterchurn_preset_archive/ is restored and the app is
# rebuilt.
#
# USAGE:
#   ./scripts/restore-visuals.sh              restore the most recent saved set
#   ./scripts/restore-visuals.sh --list       list every saved set
#   ./scripts/restore-visuals.sh curated_20260810-231455   restore a specific one
set -euo pipefail
cd "$(dirname "$0")/.."

CURATED_DIR="Sources/ButterchurnVisualizer/Resources/presets/curated"
ARCHIVE_ROOT="$HOME/butterchurn_preset_archive"

if [ ! -d "$ARCHIVE_ROOT" ]; then
  echo "No saved sets yet at $ARCHIVE_ROOT (run fresh-visuals.sh first)."
  exit 1
fi

if [ "${1:-}" = "--list" ]; then
  echo "Saved sets in $ARCHIVE_ROOT:"
  for d in "$ARCHIVE_ROOT"/*/; do
    [ -d "$d" ] || continue
    n=$(find "$d" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
    echo "    $(basename "$d")  ($n presets)"
  done
  exit 0
fi

if [ -n "${1:-}" ]; then
  target="$ARCHIVE_ROOT/$1"
else
  target="$(ls -1dt "$ARCHIVE_ROOT"/*/ 2>/dev/null | head -1)"
fi
target="${target%/}"

if [ -z "$target" ] || [ ! -d "$target" ]; then
  echo "Saved set not found: ${1:-<latest>}"
  echo "Run:  ./scripts/restore-visuals.sh --list"
  exit 1
fi

stamp=$(date +%Y%m%d-%H%M%S)
count=$(find "$CURATED_DIR" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
if [ "${count:-0}" -gt 0 ]; then
  mv "$CURATED_DIR" "$ARCHIVE_ROOT/curated_$stamp"
  echo "Saved the current $count preset(s) to $ARCHIVE_ROOT/curated_$stamp"
fi
rm -rf "$CURATED_DIR"
cp -R "$target" "$CURATED_DIR"

restored=$(find "$CURATED_DIR" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
echo "Restored $restored preset(s) from $(basename "$target")."

./scripts/build-app.sh
echo ""
echo "Done. Launch:  MIDI=1 ./ButterchurnVisualizer.app/Contents/MacOS/ButterchurnVisualizer"
