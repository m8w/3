# Living Archive: using 50k+ videos as DNA

The fluid-heat system can treat your video archive as *material*, not content.
Each clip is a potential vector field, displacement map, or colour source that
the audio can summon.

## One-time: index the archive

    cd scripts
    python3 archive_indexer.py \
        --roots /Volumes/archive1 /Volumes/archive2 ~/Movies/ink \
        --db    ../videos.sqlite \
        --workers 6

This walks the roots, runs `ffprobe`, and writes a SQLite file (~10-40 MB
for 50k rows). Columns:

    path, size, duration, width, height, fps, codec,
    ctime, mtime, tags, organic (0..1), heat_bucket (0..4),
    brightness, motion

`organic` is a heuristic score from filename + shape that you can later
overwrite with real measurements (brightness average, optical flow). Rows
with organic >= 0.8 are your "highest heat" material (white/yellow palette
stop); rows with organic <= 0.2 fall into the "cold" bucket.

## Runtime: heat-aware selection in Max

`abstractions/fh.archive_fetcher.maxpat` wraps `scripts/archive_fetcher.js`:

    [open /path/to/videos.sqlite]    // load once
    [heat 0.72]                      // pick a clip in bucket 3
    --> "path /vid/foo.mov" out
    --> jit.movie loads, jit.gl.slab crossfades
    --> output is a jit_gl_texture

Wire the current `fluid heat` output through the fetcher:

    fh.organic_lut output (peak_amp) -> fh.archive_fetcher:heat inlet

Every time the audio breaches a heat threshold, a new clip is summoned from
the bucket nearest that heat value. An internal A/B `jit.movie` pair
alternates so there's always a crossfade buffer ready.

## How the fluid uses the archive

Two integration points:

1. **Pre-inject vector field** via `shaders/fh.video_displace.jxs`
   The archive clip becomes a spatial modulator that precedes the audio
   jets; the fluid is *already* aware of the old image before today's
   sound touches it.
2. **Background pass** (display only) via the same texture piped into the
   asemic slot of `fh.organic_lut.jxs`. The heat palette then tints your
   old ink videos with live heat colour.

Both wirings are provided via the `in-tex` and `in-pulse` inlets of
`fh.archive_fetcher.maxpat`.

## Performance

- Keep 2 `jit.movie` objects (A/B). Max caches decoded frames, so crossfading
  is cheap; instantiating more than two causes thrashing on mechanical drives.
- Prefer HAP or ProRes LT codecs for 50k-video archives - H.264 won't seek
  quickly enough for reactive crossfades.
- `jit.movie @output_texture 1` keeps decoded frames on the GPU; don't
  `jit.pwindow` them.
- Set `min_duration 2` on the JS to skip clip fragments shorter than 2 s.

## Bridging to external JSON (OSC / FastAPI)

If you prefer to drive selection from a Python server (e.g. a semantic
search over embeddings, or spectral-match to the current bin profile),
just POST a path message into the JS inlet:

    udpreceive 7400  ->  [route path]  ->  fh.archive_fetcher
    (your Python sends: `/path /vol/archive/foo.mov`)

The JS only owns the SQLite read; any external selector can bypass it.

## Extending

To add a column (e.g. embedding distance to the current audio signature):

    ALTER TABLE videos ADD COLUMN embed_dist REAL;

Then write a new query in `archive_fetcher.js`:

    query SELECT path FROM videos ORDER BY embed_dist ASC LIMIT 1

The JS `query` message passes raw SQL through, so you can keep iterating
without patching Max each time.
