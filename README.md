# ButterchurnVisualizer

A native macOS Milkdrop/Butterchurn visualizer built for a 24/7 live audiovisual
broadcast. Three independent Butterchurn engines render at once and are blended by
a WebGL2 compositor; the mix reacts to system audio **and** locks to a live MIDI
stream so the visuals breathe on the same song structure as the music.

- **Three-screen mixer** — three Butterchurn visualizers composited live with
  signed weights and **10 blend modes** (add, subtract, lighten, dodge, burn,
  hard light, difference, exclusion, alpha-over, RGB split)
- **Audio-reactive** — captures system audio via **BlackHole 2ch** through a
  **32-bin microtonal FFT**; reacts in mono low/mid/high or **stereo L/mid/R**
- **MIDI 17:17 lock** — subscribes to the `sn2 chaos` virtual port and
  reconstructs the SN2 song oscillators from the channel-16 anchor, so rotation,
  zoom, blend and mix breathe on the 17:17 (1037s) arcs
  (see [scripts/sn2_note_generator.md](scripts/sn2_note_generator.md))
- **Thousands of curated presets** — culled from a 130k+ MilkDrop MegaPack with an
  offline render-scoring pipeline; one-command swap to a whole new set
- **Broadcast-ready** — window capture (not fullscreen), keeps rendering while
  backgrounded, tuned for OBS

Built with Swift + WKWebView + [Butterchurn](https://github.com/jberg/butterchurn).
Repo: **https://github.com/m8w/3**

---

## Requirements

| | |
|---|---|
| macOS | 13 Ventura or later |
| Xcode / Swift toolchain | 15 or later |
| [BlackHole 2ch](https://existential.audio/blackhole/) | Free virtual audio driver |
| (optional) `python-rtmidi` | For the SN2 MIDI companion |

---

## One-time audio setup

BlackHole routes your system audio into the visualizer without a microphone.

1. Install [BlackHole 2ch](https://existential.audio/blackhole/) (free)
2. Open **Audio MIDI Setup** (in /Applications/Utilities)
3. Click **+** → **Create Multi-Output Device**
4. Check both **BlackHole 2ch** and your speakers/headphones
5. **System Settings → Sound → Output** → select that Multi-Output Device

Your audio now plays through your speakers AND into the visualizer at once.

---

## Build & run

```
./scripts/build-app.sh
MIDI=1 ./ButterchurnVisualizer.app/Contents/MacOS/ButterchurnVisualizer
```

`build-app.sh` builds a debug `.app` and ad-hoc signs it. Drop `MIDI=1` to run
without the MIDI lock. For a single full-resolution screen instead of the mixer:

```
SINGLE_SCREEN=1 ./ButterchurnVisualizer.app/Contents/MacOS/ButterchurnVisualizer
```

### Environment knobs

| Var | Effect |
|-----|--------|
| `MIDI=1` | Subscribe to the `sn2 chaos` MIDI stream and lock to the 17:17 arcs |
| `MIDI_PORT=name` | Match a different MIDI source name |
| `MIXER_FPS=30` | Pin the render rate (otherwise 60) |
| `MIXER_SRC_W` / `MIXER_SRC_H` | Per-screen render resolution (default 1280×720) |
| `AUDIO_DEVICE="BlackHole 2ch"` | Capture device name |
| `MONO_AUDIO=1` | Force mono capture (disables the stereo react toggle) |
| `SINGLE_SCREEN=1` | Single-screen host instead of the 3-screen mixer |
| `PRESETS_ONLY=/path` | Load only that folder (skip the bundled set) |

---

## Controls

| Key | Action |
|-----|--------|
| `1` `2` `3` | Select screen 1 / 2 / 3 |
| `N` / `P` or `→` / `←` | Next / previous preset on the selected screen |
| `B` | Hard cut (instant switch) |
| `S` | Toggle the selected screen's polarity (＋ add / − subtract) |
| `[` / `]` | Selected screen weight down / up |
| `M` | Toggle audio-reactive auto-mix |
| `C` | Cycle blend mode |
| `R` | Toggle reactivity: low/mid/high ⇄ left/mid/right (stereo) |
| `G` | Momentary preset mutation (OFF → SUBTLE → WILD on the current look) |
| `X` | Reject the selected screen's preset (culls it for good) |
| `0` | Reset the mix to defaults |
| `F` | Fill screen on the current Space (not ⌃⌘F — that Space freezes under capture) |
| `H` | Hide / show the HUD (graphics-only) |
| `Space` | Pause / resume |

The HUD bar is also fully clickable. Presets auto-cycle independently per screen
(~27s / 33s / 39s), each jumping to a random preset across the whole set.

**Mutation is momentary:** the rotation always loads clean, normal presets — every
change resets to normal. `G` mutates only what's on screen right now and clears
itself the next time each screen cycles.

---

## MIDI: locking visuals to the music

Run the SN2 generator (it opens the `sn2 chaos` virtual port) and launch the app
with `MIDI=1`:

```
python3 scripts/sn2_chaos8_runs.py
MIDI=1 ./ButterchurnVisualizer.app/Contents/MacOS/ButterchurnVisualizer
```

The SN2 rig sends no MIDI clock — its **channel-16 Program Change is the sync
anchor** (t≡0), and every sound-design modulator is a slow sine whose period is a
multiple of the 17:17 song unit (1037s). The app reconstructs that same sine
family from the anchor, so the composite rotation, zoom, gamma, mix weights and
blend all breathe on the identical song arcs, and each new Performance turns the
mix over on the downbeat. Full MIDI details:
[scripts/sn2_note_generator.md](scripts/sn2_note_generator.md).

---

## Presets & curation

Presets live in `Sources/ButterchurnVisualizer/Resources/presets/curated/` and are
bundled into the `.app` at build time. They're culled from a 130k+ MilkDrop
MegaPack by an offline pipeline that converts each `.milk` to Butterchurn JSON,
renders it, and scores it for colourfulness, contrast and motion. See
[scripts/CURATE.md](scripts/CURATE.md).

Swap the whole rotation for a brand-new, all-different set (the current one is
saved, never deleted):

```
./scripts/fresh-visuals.sh
```

Bring a saved set back:

```
./scripts/restore-visuals.sh --list
./scripts/restore-visuals.sh
```

---

## Broadcast

Window capture into OBS (keep OBS foregrounded — macOS throttles background apps,
which starves the encoder and stutters the stream even when the source is smooth).
Details and an ffmpeg alternative: [scripts/BROADCAST.md](scripts/BROADCAST.md).

---

## How it works

```
System audio ── Multi-Output Device ─┬─ speakers/headphones   (you hear it)
                                      └─ BlackHole 2ch          (app reads it)
                                            └─ AudioQueue (CoreAudio)
                                                  └─ MicrotonalFFT → 32 bins
SN2 synth ── sn2 chaos virtual port ── CoreMIDI → MidiEngine → song oscillators
                                      │
   three Butterchurn engines ────────┴──→ WebGL2 compositor → broadcast output
```

- **AudioEngine** reads BlackHole 2ch by device name; stereo de-interleave feeds a
  dual FFT so screens can react to left / center / right.
- **MidiEngine** (CoreMIDI) parses notes, CCs (cutoff 74, reso 71, hardness 83,
  sync 19, mod 1), pitch bend and the channel-16 Program Change anchor.
- **mixer_host.html** hosts the three engines, the compositor shader, the song
  oscillators and the HUD; each screen renders in a crash-guarded loop so a bad
  preset recovers without freezing the others.

---

## Project structure

```
Sources/ButterchurnVisualizer/
├── AudioEngine.swift          — CoreAudio capture of BlackHole 2ch (mono/stereo)
├── MicrotonalFFT.swift        — vDSP FFT, 32 microtonal probes + AdaptiveNormaliser
├── MidiEngine.swift           — CoreMIDI reader (sn2 chaos stream, ch-16 anchor)
├── WebViewContainer.swift     — WKWebView, key handling, audio/MIDI injection
├── PresetLoader.swift         — bundles/streams presets; hot folder; reject
├── Curator.swift              — offline .milk → JSON render-scoring pipeline
├── PresetParser.swift         — parses .milk files
├── MilkPresetConverter.swift  — .milk → Butterchurn JSON
└── Resources/
    ├── mixer_host.html        — 3-screen mixer, compositor, MIDI oscillators, HUD
    ├── butterchurn_host.html  — single-screen host (SINGLE_SCREEN=1)
    ├── curate_host.html       — curation render/score host
    └── presets/curated/       — bundled curated presets
scripts/
├── build-app.sh · broadcast.sh · curate-and-bundle.sh
├── fresh-visuals.sh · restore-visuals.sh
├── sn2_chaos8_runs.py         — the SN2 MIDI generator (see sn2_note_generator.md)
└── BROADCAST.md · CURATE.md · sn2_note_generator.md
```

---

## License

MIT
