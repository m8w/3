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

### ⚠️ Screen Recording permission — run as a built `.app`, not via Xcode

Broadcasting captures the visualizer window, which needs **Screen Recording**
permission. This permission **does not work when the app is run through the
Xcode debugger or `swift run`** — the debugger becomes the "responsible process"
and a bare SwiftPM binary has no stable identity, so the grant never sticks
(you'll see macOS re-prompt every time and `Go Live` fails with a permission
error even though the toggle looks ON).

**Fix — build and run a real app bundle:**

```bash
cd ~/music/3
./scripts/build-app.sh        # produces ButterchurnVisualizer.app
open ButterchurnVisualizer.app  # launch it THIS way, not through Xcode
```

First launch prompts for **Screen Recording** and **Microphone** — grant both
(Screen Recording may need you to quit & reopen once). After that, **Go Live**
works.

- Ad-hoc signing (the default) means you re-grant Screen Recording after each
  rebuild. To make it **permanent**, sign with your Apple Development identity:
  ```bash
  security find-identity -v -p codesigning      # copy your "Apple Development: …" line
  CODESIGN_IDENTITY="Apple Development: you@example.com (TEAMID)" ./scripts/build-app.sh
  ```
- You can still use Xcode for editing/iterating — just do the actual
  *broadcasting* from the built `.app`.

### Other notes

- **HaishinKit version**: pinned to the 1.x line in `Package.swift`. Its
  codec-settings API differs slightly between versions; if the build errors on
  the `stream.videoSettings.*` lines in `Broadcaster.swift → configureEncoder()`,
  comment that method body out (defaults still stream) and report the exact error.
- **HaishinKit version**: pinned to the 1.x line in `Package.swift`. Its
  codec-settings API differs slightly between versions; if the build errors on
  the `stream.videoSettings.*` lines in `Broadcaster.swift → configureEncoder()`,
  comment that method body out (defaults still stream) and report the exact error.
- First `swift package resolve` / build will fetch HaishinKit — needs network.

### Tuning

Bitrate/resolution are in `Broadcaster.swift → configureEncoder()`
(default 6 Mbps, 1920×1080). YouTube recommends 4.5–9 Mbps for 1080p.
