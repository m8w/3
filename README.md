# ButterchurnVisualizer

A macOS Milkdrop visualizer that reacts to system audio in real time.

- **1,754 Milkdrop presets** bundled — no internet required
- Captures system audio via **BlackHole 2ch** (no microphone)
- **31-EDO microtonal FFT** — 32 frequency probes tuned to microtonal intervals
- Presets cycle every 30 seconds across a ~14-hour rotation
- Built with Swift + WKWebView + [Butterchurn](https://github.com/jberg/butterchurn)

![screenshot placeholder](screenshot.png)

---

## Requirements

| | |
|---|---|
| macOS | 13 Ventura or later |
| Xcode | 15 or later |
| [BlackHole 2ch](https://existential.audio/blackhole/) | Free virtual audio driver |

---

## One-time audio setup

BlackHole routes your system audio into the visualizer without a microphone.

1. Install [BlackHole 2ch](https://existential.audio/blackhole/) (free)
2. Open **Audio MIDI Setup** (in /Applications/Utilities)
3. Click **+** → **Create Multi-Output Device**
4. Check both **BlackHole 2ch** and your speakers/headphones
5. Go to **System Settings → Sound → Output** → select that Multi-Output Device

Your audio now plays through your speakers AND into the visualizer simultaneously.

---

## Build & run

```bash
git clone https://github.com/YOUR_USERNAME/ButterchurnVisualizer.git
cd ButterchurnVisualizer
open Package.swift
```

In Xcode:
1. Select the **ButterchurnVisualizer** scheme, destination **My Mac**
2. First time only: **Signing & Capabilities** → **+ Capability** → **Audio Input**
3. Press **⌘R**

---

## Controls

| Key | Action |
|-----|--------|
| `N` or `→` | Next preset |
| `P` or `←` | Previous preset |
| `B` | Hard cut (instant switch) |
| `Space` | Pause / resume |
| `H` | Show / hide HUD |

---

## How it works

```
System audio
    └── Multi-Output Device (Audio MIDI Setup)
            ├── Your speakers/headphones   (you hear it)
            └── BlackHole 2ch              (app reads it)
                    └── AudioQueue (CoreAudio)
                            └── MicrotonalFFT  →  32 frequency bins
                                    └── viz.render({frequencyData})
                                            └── Butterchurn / WebGL
```

- **AudioQueue** reads directly from BlackHole 2ch by device UID — no system default input change required
- **MicrotonalFFT** targets 32 frequencies spaced in 31-EDO (31 equal divisions of the octave)
- **AdaptiveNormaliser** keeps levels 0–1 regardless of input volume
- The 32 bins are spread across a 1024-bin frequency buffer and passed to `viz.render()` so Butterchurn's internal audio analysis drives all GLSL color and motion shaders

---

## Project structure

```
Sources/ButterchurnVisualizer/
├── AudioEngine.swift          — CoreAudio AudioQueue → BlackHole 2ch
├── MicrotonalFFT.swift        — Accelerate vDSP FFT, 31-EDO probe frequencies
├── AdaptiveNormaliser (in MicrotonalFFT.swift)
├── WebViewContainer.swift     — WKWebView + keyboard forwarding
├── PresetLoader.swift         — Bundles all 1754 JSON presets into the WebView
├── PresetParser.swift         — Parses .milk Milkdrop preset files
├── MilkPresetConverter.swift  — Converts .milk → Butterchurn JSON format
└── Resources/
    ├── butterchurn_host.html  — Butterchurn + render loop + HUD
    ├── microtonal_warp.milk   — Custom microtonal preset
    └── presets/               — 1754 JSON presets from butterchurn-presets
```

---

## License

MIT
