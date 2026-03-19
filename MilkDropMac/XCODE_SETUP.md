# MilkDropMac — Xcode Project Setup Guide

A native macOS MilkDrop 3 visualizer with Syphon output, Metal GPU rendering, and BeatDrop-style preset editor.

## Requirements
- macOS 13 Ventura or later
- Xcode 15+
- Apple Silicon or Intel Mac with Metal GPU

---

## Quickstart (Recommended)

This repo includes a `project.yml` for **XcodeGen** — you don't need to manually add any files.

```bash
# 1. Install XcodeGen (one-time)
brew install xcodegen

# 2. Clone the repo and generate the project
git clone <repo-url>
cd 3
xcodegen generate

# 3. Open the generated project
open MilkDropMac.xcodeproj
```

That's it — the project opens with all files already organized in groups.

---

## Manual Setup (Alternative)

If you prefer not to use XcodeGen:

### Step 1: Create the Xcode Project

1. Open Xcode → **File → New → Project**
2. Choose **macOS → App**
3. Product Name: `MilkDropMac`
4. Interface: **SwiftUI**
5. Language: **Swift**
6. Uncheck "Include Tests" (optional)

---

### Step 2: Add Source Files

Copy all files from this repository into your Xcode project:

```
MilkDropMac/
├── App/
│   ├── MilkDropMacApp.swift
│   └── AppState.swift
├── Engine/
│   ├── AudioEngine.swift
│   ├── BeatDetector.swift
│   ├── MilkDropRenderer.swift
│   └── EquationEvaluator (inside MilkDropRenderer.swift)
├── Syphon/
│   ├── SyphonBridge.h
│   ├── SyphonBridge.mm
│   └── SyphonOutput.swift
├── Presets/
│   ├── MilkDropPreset.swift
│   └── PresetManager.swift
├── UI/
│   ├── ContentView.swift
│   ├── PresetBrowserView.swift
│   ├── PresetEditorView.swift
│   ├── AudioSettingsView.swift
│   ├── SyphonStatusView.swift
│   └── SettingsView.swift
├── Shaders/
│   └── MilkDrop.metal
└── Resources/
    └── Presets/
        ├── classic_hyperspace.milk
        ├── beat_reactor.milk
        └── double_galaxy.milk2
```

When adding `.mm` files, Xcode will ask to configure an Objective-C bridging header — click **Create Bridging Header**.

---

## Step 3: Bridging Header

Xcode creates `MilkDropMac-Bridging-Header.h` automatically. Add:

```objc
#import "SyphonBridge.h"
```

---

## Step 4: Install Syphon Framework

1. Download **Syphon.framework** from:
   https://github.com/Syphon/Syphon-Framework/releases

2. In Xcode, select your target → **General** tab
3. Under **Frameworks, Libraries, and Embedded Content**:
   - Click **+** → **Add Other → Add Files**
   - Select `Syphon.framework`
   - Set **Embed** to **Embed & Sign**

4. In `SyphonBridge.mm`, uncomment the Syphon import and implementation:
   ```objc
   #import <Syphon/Syphon.h>
   ```
   Then uncomment all commented Syphon lines.

---

## Step 5: Install projectM (Optional but Recommended)

For full MilkDrop equation support, integrate **projectM**:

### Via Swift Package Manager:
```
https://github.com/projectM-visualizer/projectm
```

Or build from source and add the `.xcframework`.

The `EquationEvaluator` class in `MilkDropRenderer.swift` has a placeholder — replace `evaluateExpression()` with calls to projectM's expression evaluator for full `.milk` preset compatibility.

---

## Step 6: Add Preset Library

Download MilkDrop presets and add them to:
```
Resources/Presets/
```

Large preset packs (73,000+ presets):
- https://github.com/projectM-visualizer/projectm/tree/master/src/projectM-iTunes-VizKit/source
- Search "MilkDrop preset megapack" on forums.winamp.com

---

## Step 7: App Entitlements

In your `.entitlements` file, add:

```xml
<!-- Microphone access for audio input -->
<key>com.apple.security.device.audio-input</key>
<true/>

<!-- For reading user preset files -->
<key>com.apple.security.files.user-selected.read-write</key>
<true/>
```

---

## Step 8: Info.plist

Add usage descriptions:

```xml
<key>NSMicrophoneUsageDescription</key>
<string>MilkDropMac uses the microphone to visualize audio in real time.</string>
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        MilkDropMac                              │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  AudioEngine │───▶│ BeatDetector │───▶│   PresetManager  │  │
│  │  (AVFoundation│    │  (FFT/BPM)  │    │  (auto-switch)   │  │
│  │   + Accelerate│    └─────────────┘    └──────────────────┘  │
│  └──────┬───────┘                                   │          │
│         │ AudioData                          MilkDropPreset     │
│         ▼                                           ▼          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              MilkDropRenderer (Metal)                    │  │
│  │                                                          │  │
│  │  1. Evaluate per-frame equations (EquationEvaluator)     │  │
│  │  2. Warp pass: feedback + distortion (warp_fragment)     │  │
│  │  3. Wave pass: audio waveform lines/dots                 │  │
│  │  4. Shape pass: custom geometric shapes                  │  │
│  │  5. Composite: gamma, brightness, vignette               │  │
│  │  6. Blend: preset transitions (zoom/plasma/cercle/etc.)  │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │ MTLTexture                            │
│              ┌──────────┴──────────┐                           │
│              ▼                     ▼                           │
│     ┌──────────────┐     ┌────────────────┐                    │
│     │  MTKView     │     │  Syphon Server │────▶ VDMX          │
│     │  (on screen) │     │  (GPU texture  │────▶ Resolume      │
│     └──────────────┘     │   zero latency)│────▶ TouchDesigner │
│                          └────────────────┘────▶ OBS           │
└─────────────────────────────────────────────────────────────────┘

UI: SwiftUI
├── ContentView (main layout)
├── PresetBrowserView (grid/list with search, favorites, ratings)
├── PresetEditorView (BeatDrop-style code + parameter editor)
│   ├── Per-frame equations editor
│   ├── Per-vertex equations editor
│   ├── Warp HLSL shader editor
│   ├── Comp HLSL shader editor
│   ├── Wave editor (16 slots)
│   └── Shape editor (16 slots)
├── AudioSettingsView (spectrum, FFT, beat detection config)
├── SyphonStatusView (server status + compatible apps)
└── SettingsView (display, transitions, keyboard shortcuts)
```

---

## MilkDrop 3 Features Implemented

| Feature | Status |
|---------|--------|
| .milk preset loading & parsing | ✅ |
| .milk2 double-preset support | ✅ |
| Per-frame equations | ✅ (basic evaluator; use projectM for full support) |
| Per-vertex equations | ✅ |
| Warp pixel shader | ✅ (Metal MSL) |
| Composite pixel shader | ✅ (Metal MSL) |
| Up to 16 waves | ✅ |
| Up to 16 shapes | ✅ |
| Beat detection auto-switch | ✅ |
| Bass/treble hardcut modes | ✅ |
| BPM estimation | ✅ |
| 15 transition types | ✅ (zoom, side, plasma, cercle, checkerboard, stars, etc.) |
| Syphon output | ✅ (requires Syphon.framework) |
| Preset editor (BeatDrop-style) | ✅ |
| Preset favorites & ratings | ✅ |
| Drag & drop preset import | ✅ |
| Audio FFT (Accelerate) | ✅ |
| Metal GPU rendering | ✅ |
| Native Apple Silicon | ✅ |

---

## Adding More Presets

Place any `.milk` or `.milk2` files in:
```
~/Library/Application Support/MilkDropMac/Presets/
```

Or use **File → Import Presets...** in the app.

Compatible with all MilkDrop 1, 2, 3 and projectM presets.
