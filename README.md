# ButterchurnVisualizer

A macOS Milkdrop visualizer that reacts to system audio in real time.

- **1,754 Milkdrop presets** bundled — no internet required
- Captures system audio via **BlackHole 2ch** (no microphone)
- **31-EDO microtonal FFT** — 32 frequency probes tuned to microtonal intervals
- Presets cycle every 30 seconds across a ~14-hour rotation
- Built with Swift + WKWebView + [Butterchurn](https://github.com/jberg/butterchurn)

---

## Download

👉 **[Download latest release](https://github.com/m8w/ButterchurnVisualizer/releases/latest)**

Unzip, right-click the app → **Open** → **Open** (first launch only, bypasses Gatekeeper).

---

## Requirements

- macOS 13 Ventura or later
- [BlackHole 2ch](https://existential.audio/blackhole/) — free virtual audio driver

---

## One-time audio setup

BlackHole routes your system audio into the visualizer without a microphone.

1. Install [BlackHole 2ch](https://existential.audio/blackhole/) (free)
2. Open **Audio MIDI Setup** (in /Applications/Utilities)
3. Click **+** → **Create Multi-Output Device**
4. Check both **BlackHole 2ch** and your speakers/headphones
5. Go to **System Settings → Sound → Output** → select that Multi-Output Device

Your audio now plays through your speakers AND feeds the visualizer simultaneously.

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

## Build from source

Requires Xcode 15+ on macOS 13+.

```bash
git clone https://github.com/m8w/ButterchurnVisualizer.git
cd ButterchurnVisualizer
open Package.swift
```

In Xcode: select **ButterchurnVisualizer** scheme → **My Mac** → **⌘R**

First time: **Signing & Capabilities → + Capability → Audio Input**

---

## How it works

```
System audio
    └── Multi-Output Device (Audio MIDI Setup)
            ├── Your speakers/headphones
            └── BlackHole 2ch  ← app reads here
                    └── CoreAudio AudioQueue
                            └── MicrotonalFFT (31-EDO, 32 bins)
                                    └── viz.render({frequencyData})
                                            └── Butterchurn / WebGL
```

---

## License

MIT
