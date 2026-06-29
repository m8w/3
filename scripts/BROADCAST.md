# 3-Source Mixer + YouTube/Restream Broadcast

## The mixer (default visual)

The app now runs **three independent Butterchurn visualizers** composited
together. A WebGL2 shader blends them:

```
output = clamp( sign₀·weight₀·screen₀ + sign₁·weight₁·screen₁ + sign₂·weight₂·screen₂ )
```

Each screen cycles its own preset on its own timer, so the mix is always
evolving. Each screen is either **added** (+) or **subtracted** (−) from the
composite, with a weight. Weights are **audio-reactive** (screen 1 follows bass,
2 follows mids, 3 follows treble) and you can nudge everything live.

### Mixer keys

| Key | Action |
|-----|--------|
| `1` `2` `3` | select a screen (for the controls below) |
| `n` / `p` | next / previous preset on the selected screen |
| `b` | hard-cut the selected screen to a new preset |
| `s` | toggle the selected screen **add ⇄ subtract** |
| `[` / `]` | selected screen weight − / + |
| `m` | toggle audio-reactive auto-mix on/off |
| `0` | reset the mix to defaults |
| `space` | pause · `h` HUD · `f` fullscreen |

Fall back to the old single-screen visual with `SINGLE_SCREEN=1 swift run`.

## Broadcasting to YouTube / Restream

Broadcast controls live in the **menu bar** (the antenna icon) so they're never
captured into the stream.

1. In **YouTube Studio → Go Live → Stream settings**, copy your **stream key**.
   (Restream.io works too — use its RTMP ingest URL + key.)
2. Open the visualizer window and press **F** to fullscreen the mix.
3. Click the menu-bar **antenna icon**, paste:
   - **RTMP ingest URL**: `rtmp://a.rtmp.youtube.com/live2`
   - **Stream key**: from step 1
4. Hit **Go Live**. The dot turns red and the status reads `● LIVE`.

Video is captured from the visualizer window (just the mix); audio is the
display's system audio (whatever music is playing). Encoding is H.264 1080p30 /
AAC via HaishinKit.

### Important on-Mac notes

- **Screen Recording permission** is required. macOS prompts on first Go Live.
  ScreenCaptureKit ties this permission to an app bundle, so for a reliable
  experience **run from Xcode** (Open `Package.swift`, ⌘R) or a built `.app`,
  rather than a bare `swift run` binary — a raw SwiftPM binary may not retain the
  permission.
- **HaishinKit version**: pinned to the 1.x line in `Package.swift`. Its
  codec-settings API differs slightly between versions; if the build errors on
  the `stream.videoSettings.*` lines in `Broadcaster.swift → configureEncoder()`,
  comment that method body out (defaults still stream) and report the exact error.
- First `swift package resolve` / build will fetch HaishinKit — needs network.

### Tuning

Bitrate/resolution are in `Broadcaster.swift → configureEncoder()`
(default 6 Mbps, 1920×1080). YouTube recommends 4.5–9 Mbps for 1080p.
