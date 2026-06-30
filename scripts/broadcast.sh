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
# Quality is tuned for a smooth stream (1080p30 @ 6 Mbps) — high enough to look
# great, low enough not to outrun your upload (which is what breaks up the audio
# on the receiving end). Override if your connection is fast:
#   FPS=60 VBITRATE=9M ./scripts/broadcast.sh "rtmp://…"
#
# Stop the broadcast with  q  (or Ctrl+C).
set -euo pipefail

FPS="${FPS:-30}"
VBITRATE="${VBITRATE:-6M}"

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

echo "▶ capturing  video[$VIDEO] + audio[$AUDIO]  →  $TARGET   (${FPS}fps ${VBITRATE})"
echo "  (fullscreen the visualizer with F · press q to stop)"

# Capture VIDEO and AUDIO as SEPARATE avfoundation inputs so the heavy screen
# grab can't starve the audio thread (that's what chops the streamed audio).
# Audio gets a large -thread_queue_size buffer; aresample=async absorbs drift.
exec ffmpeg -hide_banner \
  -thread_queue_size 512  -f avfoundation -capture_cursor 0 -framerate "$FPS" -i "${VIDEO}:none" \
  -thread_queue_size 4096 -f avfoundation -i "none:${AUDIO}" \
  -map 0:v:0 -map 1:a:0 \
  -c:v h264_videotoolbox -realtime 1 -b:v "$VBITRATE" -maxrate "$VBITRATE" -bufsize "$VBITRATE" \
  -pix_fmt yuv420p -g $((FPS * 2)) \
  -c:a aac -b:a 160k -ar 48000 -af "aresample=async=1000" \
  -f flv "$TARGET"
