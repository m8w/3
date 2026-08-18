# Archive scripts (Python, unchanged from the Max era)

These three CLI tools have always been Python and are reused as-is:

    scripts/
    ├── archive_indexer.py    build SQLite from local dirs and/or CSV/URL list
    ├── archive_resolver.py   OSC yt-dlp sidecar with LRU disk cache
    └── youtube_to_csv.py     yt-dlp channel/playlist dump

## Typical workflow

    # 1. Dump the channel(s)
    python3 scripts/youtube_to_csv.py --flat --channel A --role texture \
        https://www.youtube.com/@yourPrimary > primary_A.csv

    # 2. Ingest into SQLite (also accepts Title/Video_ID/Duration/YouTube_URL)
    python3 scripts/archive_indexer.py \
        --urls primary_A.csv --db videos.sqlite \
        --channel A --role texture --min-duration 60

    # 3. Start the resolver (yt-dlp + LRU cache; OSC 7401/7402)
    pip install python-osc yt-dlp
    python3 scripts/archive_resolver.py \
        --cookies-file ~/ExternalRadio/youtube_cookies.txt \
        --cache-gb 50 --thumb-dir ~/.fh_archive_cache/thumbs

## Bridging with the Python synth

Right now the video archive is not yet consumed by the Python
`fluid_heat_python` synth (the current renderer only draws the audio-
driven MC mesh). To pull an archive frame into the mesh renderer you
would:

1. Query `videos.sqlite` yourself (`sqlite3` stdlib) or via the
   resolver's OSC `/resolve` message.
2. Decode a frame with `imageio-ffmpeg` or `av`.
3. Upload to a moderngl texture and bind it as an extra sampler in the
   fragment shader (add `uniform sampler2D u_skin;` and sample it based
   on world-space UV or fluid velocity).

The `fh.mesh_shade.frag` shader is 90 lines; adding a skin texture
sample is a few lines. Left out of the initial port to keep scope small.

## Notes

- The three scripts have no `moderngl`/`sounddevice` dependencies -
  they're safe to run headless on a server.
- SQLite schema is unchanged (`videos` table with `path`, `remote`,
  `channel`, `role`, `organic`, `energy`, `viscosity`, `heat_bucket`
  and friends). Anything you've already indexed keeps working.
- See `docs/STREAMING.md` for the full remote/LRU workflow and
  `docs/EXTERNAL_RADIO.md` for the bridge to your `external_radio.py`.
