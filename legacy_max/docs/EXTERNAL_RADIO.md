# Bridging ExternalRadio -> fluid_heat_audio

You already have `external_radio.py` driving audio playback from the same
archive (Archive.org + YouTube + Alonetone + Bandcamp). The CSV at
`~/ExternalRadio/youtube_videos.csv` and the cookies at
`~/ExternalRadio/youtube_cookies.txt` plug directly into this system - no
file conversion, no re-export, no duplicate accounts.

## Shared assets

| ExternalRadio file                                | fluid_heat_audio use                          |
|---------------------------------------------------|-----------------------------------------------|
| `~/ExternalRadio/youtube_videos.csv`              | `archive_indexer.py --urls <csv>`             |
| `~/ExternalRadio/youtube_cookies.txt`             | `archive_resolver.py --cookies-file <path>`   |
| Google Drive file ID (`gdrive_file_id`)           | `archive_indexer.py --gdrive-id <id>`         |

## Ingest the same CSV (one command)

The indexer recognises the ExternalRadio column names directly:

    Title -> title (also used for organic/energy/viscosity heuristic)
    Video_ID -> youtube_id
    Duration -> duration   (supports HH:MM:SS, MM:SS, or seconds)
    YouTube_URL -> url      (stored in path with remote=1)

So this just works:

    python3 scripts/archive_indexer.py \
        --urls ~/ExternalRadio/youtube_videos.csv \
        --db   videos.sqlite \
        --channel A --role texture \
        --min-duration 60

Or pull the same CSV from Google Drive (mirrors `_resolve_csv`):

    python3 scripts/archive_indexer.py \
        --gdrive-id 1xrRHifxPnQH7pZPin8Gazd2p9mZNjTGR \
        --db   videos.sqlite \
        --channel A --role texture

A header from your CSV like `Asemic ink flow #1` auto-scores high organic
(matches the `asemic/ink/flow` motifs); `Glitch strobe pulse` scores high
energy; `Slow ambient drift wash` scores high viscosity. The fluid solver
will pull each clip when the live audio matches that descriptor.

## Run both daemons together

Two processes, both reading the same CSV/cookies, talking over loopback
OSC and the same shared SQLite database:

    # Terminal 1: audio mixer (your existing script)
    python3 ~/ExternalRadio/external_radio.py

    # Terminal 2: video resolver for Max
    python3 scripts/archive_resolver.py \
        --cookies-file ~/ExternalRadio/youtube_cookies.txt \
        --cache-dir ~/.fh_archive_cache \
        --cache-gb 50 \
        --thumb-dir ~/.fh_archive_cache/thumbs

    # Max: open fluid_heat_audio.maxpat - it reads the same videos.sqlite

The resolver and ExternalRadio call yt-dlp independently. No coordination
required - YouTube doesn't care; the cookies file is read-only.

## How the streaming actually works

ExternalRadio (audio): `yt-dlp -o -` writes raw audio to stdout and
ffplay reads stdin. The whole clip is a single pipe; no file ever
touches the disk.

fluid_heat_audio (video) can't pipe the same way because `jit.movie`
needs a seekable file. So the resolver downloads to the LRU disk cache
and hands `jit.movie` a local path. The thumbnail returned via
`/thumb_path` provides an instant texture while the full clip downloads,
so the frame never goes blank.

If you want truly disk-free video the resolver also exposes
`/resolve_stream` which returns the direct googlevideo URL (`yt-dlp -g`).
Some Max versions can read those URLs through `jit.movie` but seek
performance is poor; the cached path is the reliable route.

## Same heuristics, same buckets

The indexer's `infer_organic` and `infer_energy_viscosity` are kept in
sync with `youtube_to_csv.py` and produce the same numeric ranges your
ExternalRadio could score with if you wired it up. Anything you compute
in Python (loudness, optical flow, embedding distance) can be UPDATEd
into the same SQLite columns; the fluid patch picks up the new values
on the next query.

## Net effect

You point Max at `videos.sqlite`, click play, and you're streaming the
same 86k+ catalog you've been auditioning aurally through
ExternalRadio - now used as displacement + skin in the fluid solver.
The web of cookies/CSV/Drive is shared verbatim.
