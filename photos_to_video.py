#!/usr/bin/env python3
"""
photos_to_video.py — Turn a folder of photos into a video with morphing
transitions and background music.

Usage:
    python3 photos_to_video.py --photos ./my_photos --music ./song.mp3
    python3 photos_to_video.py --photos ./pics --music ./song.mp3 \\
        --output out.mp4 --transition morph --photo-duration 3 \\
        --transition-duration 1.5 --fps 30 --resolution 1920x1080

Transition modes:
    morph       Keypoint-matched Delaunay triangle warp + alpha blend (default)
    crossfade   Simple alpha blend only
    zoom        Zoom-out A while fading to B
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Image loading / resizing
# ---------------------------------------------------------------------------

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def load_photos(directory: str, width: int, height: int) -> list[np.ndarray]:
    """Load all images from *directory*, resize to (width, height), return BGR arrays."""
    paths = sorted(
        p for p in Path(directory).iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    )
    if len(paths) < 2:
        sys.exit(f"ERROR: Need at least 2 photos in '{directory}', found {len(paths)}.")

    frames = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        img = fit_image(img, width, height)
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        frames.append(bgr.astype(np.float32))
        print(f"  Loaded: {p.name}")
    return frames


def fit_image(img: Image.Image, width: int, height: int) -> Image.Image:
    """Resize & crop image to exactly (width, height) using cover mode."""
    img_ratio = img.width / img.height
    target_ratio = width / height
    if img_ratio > target_ratio:
        # Image is wider — fit height, crop width
        new_h = height
        new_w = int(img.width * height / img.height)
    else:
        # Image is taller — fit width, crop height
        new_w = width
        new_h = int(img.height * width / img.width)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    return img.crop((left, top, left + width, top + height))


# ---------------------------------------------------------------------------
# Keypoint detection & matching
# ---------------------------------------------------------------------------

def find_corresponding_points(
    A: np.ndarray, B: np.ndarray, w: int, h: int, n_features: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect ORB keypoints in both images, match them, and return two
    aligned arrays of (x, y) control points — one per image.
    Corner and edge midpoints are always included so the triangulation
    covers the entire frame.
    """
    gray_A = cv2.cvtColor(A.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray_B = cv2.cvtColor(B.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=n_features)
    kp_A, desc_A = orb.detectAndCompute(gray_A, None)
    kp_B, desc_B = orb.detectAndCompute(gray_B, None)

    pts_A: list[tuple[float, float]] = []
    pts_B: list[tuple[float, float]] = []

    if desc_A is not None and desc_B is not None and len(kp_A) >= 4 and len(kp_B) >= 4:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(desc_A, desc_B)
        matches = sorted(matches, key=lambda m: m.distance)[:80]
        for m in matches:
            pts_A.append(kp_A[m.queryIdx].pt)
            pts_B.append(kp_B[m.trainIdx].pt)

    # Always anchor the four corners + edge midpoints
    border_pts = [
        (0.0, 0.0), (float(w - 1), 0.0),
        (0.0, float(h - 1)), (float(w - 1), float(h - 1)),
        (float(w // 2), 0.0), (float(w // 2), float(h - 1)),
        (0.0, float(h // 2)), (float(w - 1), float(h // 2)),
    ]
    pts_A.extend(border_pts)
    pts_B.extend(border_pts)

    return np.array(pts_A, dtype=np.float32), np.array(pts_B, dtype=np.float32)


# ---------------------------------------------------------------------------
# Delaunay triangulation
# ---------------------------------------------------------------------------

def build_triangle_indices(
    pts_A: np.ndarray, pts_B: np.ndarray, w: int, h: int
) -> list[tuple[int, int, int]]:
    """
    Triangulate the *average* of pts_A and pts_B, then map each resulting
    triangle back to point indices so we can reuse the triangulation for
    every interpolation step.
    """
    mid = ((pts_A + pts_B) / 2).astype(np.float32)

    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    for pt in mid:
        x, y = float(pt[0]), float(pt[1])
        # Clamp to avoid Subdiv2D boundary errors
        x = max(0.0, min(float(w - 1), x))
        y = max(0.0, min(float(h - 1), y))
        subdiv.insert((x, y))

    raw_triangles = subdiv.getTriangleList()  # shape (N, 6): x1,y1,x2,y2,x3,y3

    # Build a lookup from (rounded x, y) → index in mid
    lookup: dict[tuple[int, int], int] = {}
    for idx, pt in enumerate(mid):
        key = (round(pt[0]), round(pt[1]))
        lookup[key] = idx

    tri_indices: list[tuple[int, int, int]] = []
    for t in raw_triangles:
        pts_in_rect = all(
            0 <= t[i] <= w and 0 <= t[i + 1] <= h for i in (0, 2, 4)
        )
        if not pts_in_rect:
            continue
        keys = [
            (round(t[0]), round(t[1])),
            (round(t[2]), round(t[3])),
            (round(t[4]), round(t[5])),
        ]
        idxs = [lookup.get(k) for k in keys]
        if None not in idxs:
            tri_indices.append((idxs[0], idxs[1], idxs[2]))

    return tri_indices


# ---------------------------------------------------------------------------
# Triangle warping helper
# ---------------------------------------------------------------------------

def _warp_triangle(
    src: np.ndarray,
    dst: np.ndarray,
    tri_src: list,
    tri_dst: list,
) -> None:
    """
    Affine-warp the triangle *tri_src* from *src* into *tri_dst* in *dst*.
    Both *src* and *dst* are float32 BGR images of the same shape.
    """
    tri_src_f = np.float32(tri_src)
    tri_dst_f = np.float32(tri_dst)

    r_src = cv2.boundingRect(tri_src_f)
    r_dst = cv2.boundingRect(tri_dst_f)

    if r_src[2] == 0 or r_src[3] == 0 or r_dst[2] == 0 or r_dst[3] == 0:
        return

    # Offset triangle vertices to bounding-rect origin
    src_off = tri_src_f - np.float32([r_src[:2]])
    dst_off = tri_dst_f - np.float32([r_dst[:2]])

    M = cv2.getAffineTransform(src_off, dst_off)

    h_src = src.shape[0]
    w_src = src.shape[1]
    x, y, rw, rh = r_src
    x = max(0, x); y = max(0, y)
    rw = min(rw, w_src - x); rh = min(rh, h_src - y)
    if rw <= 0 or rh <= 0:
        return

    patch = cv2.warpAffine(
        src[y: y + rh, x: x + rw],
        M,
        (r_dst[2], r_dst[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Triangle mask in destination bounding rect
    mask = np.zeros((r_dst[3], r_dst[2]), dtype=np.float32)
    cv2.fillConvexPoly(mask, dst_off.astype(np.int32), 1.0)
    mask3 = mask[:, :, np.newaxis]

    h_dst = dst.shape[0]
    w_dst = dst.shape[1]
    xd, yd, rwd, rhd = r_dst
    xd = max(0, xd); yd = max(0, yd)
    rwd = min(rwd, w_dst - xd); rhd = min(rhd, h_dst - yd)
    if rwd <= 0 or rhd <= 0:
        return

    region = dst[yd: yd + rhd, xd: xd + rwd]
    region[:] = (
        region * (1.0 - mask3[:rhd, :rwd])
        + patch[:rhd, :rwd] * mask3[:rhd, :rwd]
    )


# ---------------------------------------------------------------------------
# Transition generators
# ---------------------------------------------------------------------------

def morph_transition(
    A: np.ndarray,
    B: np.ndarray,
    n_frames: int,
    w: int,
    h: int,
) -> list[np.ndarray]:
    """
    Delaunay triangle warp morph.
    For each frame at time t ∈ [0,1]:
      1. Interpolate keypoints: P(t) = (1-t)*pts_A + t*pts_B
      2. Warp A to P(t), warp B to P(t)
      3. Blend: result = (1-t)*warped_A + t*warped_B
    """
    pts_A, pts_B = find_corresponding_points(A, B, w, h)
    tri_indices = build_triangle_indices(pts_A, pts_B, w, h)

    frames = []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        pts_mid = (1.0 - t) * pts_A + t * pts_B

        warped_A = np.zeros_like(A)
        warped_B = np.zeros_like(B)

        for (ia, ib, ic) in tri_indices:
            tri_mid = [pts_mid[ia], pts_mid[ib], pts_mid[ic]]
            _warp_triangle(A, warped_A, [pts_A[ia], pts_A[ib], pts_A[ic]], tri_mid)
            _warp_triangle(B, warped_B, [pts_B[ia], pts_B[ib], pts_B[ic]], tri_mid)

        blended = (1.0 - t) * warped_A + t * warped_B
        frames.append(blended)

    return frames


def crossfade_transition(
    A: np.ndarray,
    B: np.ndarray,
    n_frames: int,
    **_,
) -> list[np.ndarray]:
    """Simple alpha-blend from A to B."""
    frames = []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        frames.append((1.0 - t) * A + t * B)
    return frames


def zoom_transition(
    A: np.ndarray,
    B: np.ndarray,
    n_frames: int,
    w: int,
    h: int,
) -> list[np.ndarray]:
    """
    Zoom into photo A (scale 1→1.15) while cross-fading to photo B.
    Creates a gentle "push into the next scene" feel.
    """
    frames = []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        scale = 1.0 + 0.15 * t
        cx, cy = w / 2, h / 2
        M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
        zoomed = cv2.warpAffine(A.astype(np.float32), M, (w, h))
        blended = (1.0 - t) * zoomed + t * B
        frames.append(blended)
    return frames


TRANSITION_FN = {
    "morph": morph_transition,
    "crossfade": crossfade_transition,
    "zoom": zoom_transition,
}


# ---------------------------------------------------------------------------
# Frame sequence assembly
# ---------------------------------------------------------------------------

def build_frame_sequence(
    photos: list[np.ndarray],
    transition: str,
    hold_frames: int,
    trans_frames: int,
    w: int,
    h: int,
) -> list[np.ndarray]:
    fn = TRANSITION_FN[transition]
    sequence: list[np.ndarray] = []
    total = len(photos)

    for idx, photo in enumerate(photos):
        # Hold current photo
        sequence.extend([photo] * hold_frames)

        if idx < total - 1:
            next_photo = photos[idx + 1]
            print(f"  Computing {transition} transition {idx + 1}/{total - 1}...")
            trans_frames_list = fn(photo, next_photo, trans_frames, w=w, h=h)
            sequence.extend(trans_frames_list)

    return sequence


# ---------------------------------------------------------------------------
# Video writing
# ---------------------------------------------------------------------------

def write_silent_video(
    frames: list[np.ndarray],
    fps: int,
    path: str,
    w: int,
    h: int,
) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))
    writer.release()


def mux_audio(
    video_path: str,
    music_path: str,
    output_path: str,
) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", music_path,
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg error:", result.stderr[-500:])
        sys.exit("ERROR: ffmpeg failed to mux audio.")


# ---------------------------------------------------------------------------
# Audio duration helper
# ---------------------------------------------------------------------------

def get_audio_duration(music_path: str) -> float | None:
    """Return audio duration in seconds using ffprobe, or None on failure."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        music_path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
        return float(out.strip())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a video from photos with morphing transitions and music."
    )
    p.add_argument("--photos", required=True, help="Directory containing photos (JPG/PNG).")
    p.add_argument("--music", required=True, help="Background music file (MP3/WAV/AAC).")
    p.add_argument("--output", default="output.mp4", help="Output video path (default: output.mp4).")
    p.add_argument(
        "--transition",
        choices=list(TRANSITION_FN.keys()),
        default="morph",
        help="Transition style (default: morph).",
    )
    p.add_argument(
        "--photo-duration",
        type=float,
        default=None,
        help="Seconds each photo is displayed. Omit to auto-fit music length.",
    )
    p.add_argument(
        "--transition-duration",
        type=float,
        default=1.5,
        help="Seconds per transition (default: 1.5).",
    )
    p.add_argument("--fps", type=int, default=30, help="Output frame rate (default: 30).")
    p.add_argument(
        "--resolution",
        default="1920x1080",
        help="Output resolution WxH (default: 1920x1080).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Parse resolution
    try:
        w, h = (int(x) for x in args.resolution.lower().split("x"))
    except ValueError:
        sys.exit("ERROR: --resolution must be in WxH format, e.g. 1920x1080")

    # Validate inputs
    if not os.path.isdir(args.photos):
        sys.exit(f"ERROR: --photos '{args.photos}' is not a directory.")
    if not os.path.isfile(args.music):
        sys.exit(f"ERROR: --music '{args.music}' not found.")
    if not any(
        subprocess.run(["which", cmd], capture_output=True).returncode == 0
        for cmd in ["ffmpeg"]
    ):
        sys.exit("ERROR: ffmpeg is not installed. Run: sudo apt install ffmpeg")

    print(f"\nLoading photos from '{args.photos}'...")
    photos = load_photos(args.photos, w, h)
    n = len(photos)
    print(f"  {n} photos loaded at {w}x{h}")

    # Calculate durations
    trans_dur = args.transition_duration
    trans_frames = max(2, int(trans_dur * args.fps))

    if args.photo_duration is not None:
        photo_dur = args.photo_duration
    else:
        audio_dur = get_audio_duration(args.music)
        if audio_dur:
            photo_dur = max(1.0, (audio_dur - (n - 1) * trans_dur) / n)
            print(f"  Auto photo duration: {photo_dur:.2f}s (to match {audio_dur:.1f}s audio)")
        else:
            photo_dur = 3.0
            print("  Could not read audio duration; defaulting to 3.0s per photo.")

    hold_frames = max(1, int(photo_dur * args.fps))
    total_frames = n * hold_frames + (n - 1) * trans_frames
    total_dur = total_frames / args.fps

    print(f"\nSettings:")
    print(f"  Transition:   {args.transition}")
    print(f"  Photo hold:   {photo_dur:.1f}s ({hold_frames} frames)")
    print(f"  Transition:   {trans_dur:.1f}s ({trans_frames} frames)")
    print(f"  Total video:  {total_dur:.1f}s @ {args.fps}fps ({total_frames} frames)")

    # Build frames
    print(f"\nBuilding frame sequence...")
    frames = build_frame_sequence(photos, args.transition, hold_frames, trans_frames, w, h)
    print(f"  {len(frames)} frames generated")

    # Write silent video to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    print(f"\nWriting video frames...")
    write_silent_video(frames, args.fps, tmp_path, w, h)
    print(f"  Silent video written to temp file")

    # Mux audio
    print(f"\nMuxing audio...")
    mux_audio(tmp_path, args.music, args.output)
    os.unlink(tmp_path)

    print(f"\nDone! Output: {args.output}")


if __name__ == "__main__":
    main()
