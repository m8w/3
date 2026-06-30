#!/bin/bash
# broadcast.sh — stream the screen + BlackHole audio to an RTMP target.
#
# This is the proven, reliable path: ffmpeg run directly from your Terminal,
# which already has Screen-Recording permission. Fullscreen the visualizer
# first (press F) so the broadcast is just the mix.
#
# USAGE:
#   ./scripts/broadcast.sh "rtmp://live.restream.io/live/YOUR_RESTREAM_KEY"
#   ./scripts/broadcast.sh "rtmp://a.rtmp.youtube.com/live2/YOUR_YOUTUBE_KEY"
#
# Stop the broadcast with  q  (or Ctrl+C).
set -euo pipefail

TARGET="${1:-}"
if [ -z "$TARGET" ]; then
  echo "Usage: $0 \"rtmp://host/app/STREAMKEY\""
  echo "  Restream: rtmp://live.restream.io/live/<key>"
  echo "  YouTube:  rtmp://a.rtmp.youtube.com/live2/<key>"
  exit 1
fi

# Auto-detect the avfoundation device indices for the screen + BlackHole 2ch
# (they shift with however many cameras/audio devices are attached).
DEVICES="$(ffmpeg -hide_banner -f avfoundation -list_devices true -i "" 2>&1 || true)"
VIDEO="$(printf '%s\n' "$DEVICES" | grep -i 'Capture screen' | head -1 | sed -E 's/.*\] \[([0-9]+)\] .*/\1/')"
AUDIO="$(printf '%s\n' "$DEVICES" | grep 'BlackHole 2ch'  | head -1 | sed -E 's/.*\] \[([0-9]+)\] .*/\1/')"
VIDEO="${VIDEO:-3}"
AUDIO="${AUDIO:-0}"

echo "▶ capturing  video[$VIDEO] + audio[$AUDIO]  →  $TARGET"
echo "  (fullscreen the visualizer with F · press q to stop)"

exec ffmpeg -hide_banner \
  -f avfoundation -capture_cursor 0 -framerate 60 -i "${VIDEO}:${AUDIO}" \
  -c:v h264_videotoolbox -realtime 1 -b:v 16M -maxrate 16M -bufsize 32M \
  -pix_fmt yuv420p -g 120 \
  -c:a aac -b:a 160k -ar 48000 \
  -f flv "$TARGET"
