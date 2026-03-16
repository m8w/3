#!/bin/bash
set -e

echo "Installing Python dependencies..."
pip3 install -r requirements.txt

echo ""
if ! command -v ffmpeg &>/dev/null; then
    echo "WARNING: ffmpeg is not installed."
    echo "  Install it with:  sudo apt install ffmpeg"
    echo "  (ffmpeg is required to mux audio into the output video)"
    exit 1
fi

echo "ffmpeg found: $(ffmpeg -version 2>&1 | head -1)"
echo ""
echo "Setup complete. Run the tool with:"
echo "  python3 photos_to_video.py --photos ./my_photos --music ./song.mp3 --output out.mp4"
