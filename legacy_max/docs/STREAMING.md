# Streaming Archive (no full local copy needed)

You don't need 53k videos on disk. The system stores **URLs** in SQLite,
and a small Python sidecar (`archive_resolver.py`) resolves them on demand
via `yt-dlp`, downloads into a rolling **LRU cache** (default 50 GB), and
hands the local cache path to Max. Thumbnails are returned instantly so
the visual never goes blank while a clip downloads.

## Pieces

| component                     | role                                            |
|-------------------------------|-------------------------------------------------|
| `scripts/youtube_to_csv.py`   | one-shot: dump YouTube channel/playlist to CSV  |
| `scripts/archive_indexer.py`  | `--urls foo.csv` ingests CSV into SQLite (remote=1) |
| `scripts/archive_resolver.py` | OSC server: yt-dlp + LRU cache + thumb fallback |
| `abstractions/fh.resolver_bridge.maxpat` | OSC bridge inside Max          |
| `abstractions/fh.archive_fetcher.maxpat` | routes `resolve` -> bridge automatically |

## Workflow

### 1. Dump the channel(s) to CSV

    python3 scripts/youtube_to_csv.py --flat \
        --channel A --role texture \
        https://www.youtube.com/@yourPrimary53k \
        > primary_A.csv

    python3 scripts/youtube_to_csv.py --flat \
        --channel B --role velocity \
        https://www.youtube.com/@yourSecondaryB \
        https://www.youtube.com/@yourSecondaryC \
        > secondary_B.csv

`--flat` uses `yt-dlp --flat-playlist` for fast metadata (53k videos in
under an hour for most channels). Drop `--flat` for a slow but complete
per-video probe that fills in duration / fps / dimensions.

### 2. Ingest the CSVs

    python3 scripts/archive_indexer.py \
        --urls primary_A.csv \
        --db   videos.sqlite \
        --channel A --role texture

    python3 scripts/archive_indexer.py \
        --urls secondary_B.csv \
        --db   videos.sqlite \
        --channel B --role velocity

Rows land in the same `videos` table with `remote=1` and the URL stored
in `path`. They mix freely with any local rows you've added.

### 3. Start the resolver sidecar

    pip install yt-dlp python-osc

    python3 scripts/archive_resolver.py \
        --cache-dir ~/.fh_archive_cache \
        --cache-gb  50 \
        --thumb-dir ~/.fh_archive_cache/thumbs \
        --workers   2 \
        --listen-port 7401 --reply-port 7402

The resolver listens for OSC on port 7401 and replies on port 7402
(loopback only). Leave it running while Max is open.

### 4. Run the patch

`fh.archive_fetcher.maxpat` already routes `resolve <url>` from the JS
into `fh.resolver_bridge`, which talks to the sidecar. When the resolver
answers `/path <local-cache>`, the existing A/B `jit.movie` crossfade
loads it - so from Max's point of view nothing changed.

When a row is new and not yet downloaded, the resolver immediately
returns a `/thumb_path <jpg>` so you can show the frame at lower
fidelity while the full clip downloads in the background. Wire the
`out-thumb` outlet of `fh.archive_fetcher` into a `jit.gl.texture
@file <thumbpath>` and crossfade to the full clip when `/path` arrives.

## Tuning

| flag                | default              | effect                          |
|---------------------|----------------------|---------------------------------|
| `--cache-gb`        | 50                   | LRU disk budget                 |
| `--height-max`      | 720                  | yt-dlp format selector cap      |
| `--workers`         | 2                    | concurrent downloads            |
| `--thumb-dir`       | (off)                | enables instant thumb fallback  |

Send `/size_limit_gb 100` over OSC to raise the budget at runtime;
send `/evict` to force a sweep.

## Network discipline

- `archive_resolver.py` is the **only** component that talks to the
  internet (via `yt-dlp` to YouTube). Max <-> resolver is loopback OSC.
- Set `min_duration 2` on the JS (default in `fh.archive_pair`) to skip
  YouTube Shorts and intro stings.
- Use `--workers 1` on slow connections; the queue serialises requests
  so Max never blocks waiting for a download.

## Mixed local + remote

The `videos` table doesn't care - rows can be local (`remote=0`, `path`
is a filesystem path) or remote (`remote=1`, `path` is a URL). The
fetcher emits `path <local>` or `resolve <url>` accordingly, and the
existing patch wiring handles both. To restrict to one or the other:

    [allow_remote 0]   // local only - no resolver needed
    [only_remote  1]   // remote only - exercise the resolver

## Failure modes & fallbacks

1. **yt-dlp fails** (geo-block, takedown) -> `/error` on outlet 2,
   fetcher falls through and the patch keeps showing the previous clip.
2. **Resolver not running** -> `udpsend` packets are dropped silently;
   no `/path` ever arrives. The patch still runs the audio + fluid solver,
   just without archive content.
3. **Out of cache** -> LRU evicts oldest by access time; downloads
   automatically reclaim space.
4. **Slow download** -> thumbnail texture displays immediately; full
   clip crossfades in when ready (frame stays alive).

## Privacy / cost

`yt-dlp` makes HTTPS requests to `googlevideo.com` only when the
resolver is actively fetching. Watch quota at scale: 53k videos at ~30 MB
cached average is 1.5 TB if fully resolved, so keep `--cache-gb` set to
a sane working-set size and let the LRU do its job.
