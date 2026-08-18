# Living Archive: 63k videos as DNA (53k skin + 10k nerves)

The archive is split by *role* in the solver, not just by source:

| channel | size  | role       | shader binding                |
|---------|-------|------------|-------------------------------|
| A       | 53k   | `texture`  | fh.video_skin.jxs             |
| B       | 10k   | `velocity` | fh.video_vector.jxs           |

Channel A is your "skin" - colour and density that rides on top of the
rendered fluid. Channel B is your "nerves" - its luminance gradient
becomes a direct velocity field that steers the fluid. The audio drives
SQL matching for both so each one responds to a different facet of the
sound.

## Legacy usage (single archive)


The fluid-heat system can treat your video archive as *material*, not content.
Each clip is a potential vector field, displacement map, or colour source that
the audio can summon.

## One-time: index the archive

    cd scripts
    # Channel A: 53k primary -> skin
    python3 archive_indexer.py \
        --roots /Volumes/primaryA \
        --db    ../videos.sqlite \
        --channel A --role texture --workers 6

    # Channel B: 10k secondary -> nerves
    python3 archive_indexer.py \
        --roots /Volumes/secondaryB /Volumes/secondaryC \
        --db    ../videos.sqlite \
        --channel B --role velocity --workers 6

Single-archive (legacy) call:

    python3 archive_indexer.py \
        --roots /Volumes/archive1 /Volumes/archive2 ~/Movies/ink \
        --db    ../videos.sqlite \
        --workers 6

This walks the roots, runs `ffprobe`, and writes a SQLite file (~15-60 MB
for 63k rows). Columns:

    path, size, duration, width, height, fps, codec,
    ctime, mtime, tags, organic (0..1), heat_bucket (0..4),
    brightness, motion,
    channel (A|B|...), role (texture|velocity|both),
    energy (0..1), viscosity (0..1)

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

## Audio -> SQL: "living query"

`archive_fetcher.js` accepts multi-criterion queries that rank rows
against three audio descriptors simultaneously:

    match <heat 0..1> <energy 0..1> <viscosity 0..1>

Internally this becomes:

    ORDER BY ABS(organic - $heat)    * 1.0
           + ABS(energy  - $energy)  * 0.9
           + ABS(viscosity - $visc)  * 0.7
           + jitter
    ASC LIMIT 1

A loud/fast passage selects high-energy low-viscosity clips; a quiet
drone pulls thick slow material. `role <texture|velocity>` filters first
so channel A and channel B pick from their own pools.

## How the fluid uses the archive

Three integration points (any combination):

1. **Velocity field** via `shaders/fh.video_vector.jxs` (channel B)
   Sobel gradient of the 10k "nerves" clip becomes a pure directional
   force. `curl` parameter rotates gradient 0..90 degrees for tangential
   swirl. No heat, no density added - only flow.
2. **Skin overlay** via `shaders/fh.video_skin.jxs` (channel A)
   The 53k "skin" clip is UV-warped by local fluid velocity and tinted
   by the current heat colour, then mixed into the rendered fluid by
   density x heat. The fluid's motion literally pulls your past footage
   through itself.
3. **Legacy displacement** via `shaders/fh.video_displace.jxs`
   A single-channel combination of both; use when you don't have a
   split archive.

## Wiring the dual-channel pair

`abstractions/fh.archive_pair.maxpat` runs two `fh.archive_fetcher`
instances in lockstep:

    audio heat 0..1  ---+
    audio energy 0..1 --+--> fh.archive_pair
    audio visc 0..1  ---+          |
    biological pulse ---+          +-> out 0: channel A skin texture
                                   +-> out 1: channel B velocity texture

Each fetcher is pre-configured on loadbang with:

    role texture       // or 'velocity'
    channel A          // or 'B'
    min_duration 2.0

Channel B receives an *inverted* heat value so when audio heats up
channel A (fresh skin), channel B pulls cool-nerve clips - producing
counter-flow between skin and nerves instead of redundant motion.

The fetcher's internal A/B jit.movie slots still handle smooth
crossfades (via `fh.crossfade.jxs`), so even rapid SQL matches don't
drop frames.

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
