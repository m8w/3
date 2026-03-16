# CLAUDE.md — AI Assistant Guide

## Repository Overview

This repository documents the design and development of **Metal Video Morpher**, a native macOS application for GPU-accelerated video morphing. The `README.md` is a conversation transcript capturing the iterative design decisions, code proposals, and architectural direction for the Xcode project located at:

```
/Users/wvn/Documents/github/wvn/metal_video_morpher/
```

The repo itself is a **documentation/design artifact** — the live Xcode project is external to this repository.

---

## Project: Metal Video Morpher

### What It Does
A macOS desktop app that takes video input and applies GPU-accelerated morphing effects using Apple's Metal framework. The app supports multiple morphing techniques and provides real-time performance metrics while the user adjusts parameters.

### Target Platform
- **OS**: macOS 11.0+
- **Language**: Swift + Metal Shading Language (MSL)
- **UI Framework**: SwiftUI
- **GPU**: Apple Metal (via `MTLDevice`, `MTLCommandQueue`, `CVMetalTextureCache`)
- **Video**: AVFoundation (`AVPlayer`, `AVKit`)
- **Vision**: Apple Vision framework (face landmark detection for feature-based morphing)

---

## Described File Structure (Xcode Project)

```
metal_video_morpher/
├── metal_video_morpher/
│   ├── ContentView.swift         # Main SwiftUI UI — controls, preview, metrics panels
│   ├── VideoProcessor.swift      # Core video processing logic (async/await)
│   ├── MetalUtilities.swift      # Singleton GPU utility: texture cache, batch processing
│   ├── MorphingTechniques.swift  # MorphingProcessor class, technique enum, algorithms
│   ├── PerformanceMonitor.swift  # ObservableObject for real-time GPU/CPU/memory metrics
│   └── Shaders.metal             # Metal compute shaders (morphKernel, Bezier helpers, Delaunay)
```

---

## Core Components

### ContentView.swift
- Root SwiftUI view using `NavigationView` + `HSplitView` layout
- Left panel: video file picker, morphing settings, technique selector, performance estimates
- Right panel: `AVPlayer`-based video preview
- `@StateObject` for `VideoProcessor` and `PerformanceMonitor`
- File import via `.fileImporter` (accepts `.movie` UTType)
- Settings sheet (`SettingsView`) with `@AppStorage` persisted prefs

### VideoProcessor.swift
- `@MainActor`-compatible `ObservableObject`
- Async method: `processVideo(sourceURL:outputURL:morphStrength:keyframeCount:) async throws`
- Publishes `isProcessing: Bool` and `progress: Double`
- Saves output to the user's Downloads directory

### MetalUtilities.swift
- Singleton (`MetalUtilities.shared`)
- Manages `MTLDevice`, `MTLCommandQueue`, `CVMetalTextureCache`
- `createMetalTexture(from:)` → wraps `CVPixelBuffer` as `MTLTexture`
- `processFrames(_:morphStrength:keyframeCount:progressHandler:) async throws` — batch-processes frames in groups of 16

### MorphingTechniques.swift
Four morphing modes defined in `enum MorphingTechnique: Int`:

| Value | Name | Description |
|---|---|---|
| 0 | `featureBased` | Vision framework face landmark detection + warp |
| 1 | `delaunay` | Delaunay triangulation → barycentric interpolation |
| 2 | `bezier` | Cubic Bezier curve interpolation between control points |
| 3 | `crossDissolve` | Mesh warp + alpha cross-dissolve blend |

`MorphingProcessor`:
- Owns `MTLComputePipelineState` loaded from `morphKernel` in `Shaders.metal`
- `detectFeatures(in:) async throws` via `VNDetectFaceLandmarksRequest`
- `computeDelaunayTriangulation(points:)` → GPU-accelerated via Accelerate/Metal
- `createBezierMorph(from:to:t:)` → SIMD3<Float> cubic evaluation

### PerformanceMonitor.swift
- `ObservableObject` with `@Published` properties: `estimatedDuration`, `processingTimePerFrame`, `totalMemoryUsage`, `gpuUtilization`, `fps`
- `calculateEstimates(frameCount:keyframeCount:morphStrength:)` — updates estimates in real time as slider values change
- `startMonitoring()` — fires a 1s repeating timer to sample GPU counter and FPS

### Shaders.metal
- `MorphingTechnique` struct (type, strength, time)
- `bezier_curve(p0:p1:p2:p3:t:) → float3` — cubic Bezier helper
- `morph_delaunay(pos:triangles:triangle_count:alpha:) → float2` — barycentric triangle lookup
- `morphKernel` — main Metal compute kernel

---

## Key Conventions

### Swift Style
- Use `async/await` and `Task {}` for all video processing; never block the main thread
- Prefer `@StateObject` for owned observable objects, `@ObservedObject` for injected ones
- Use `@AppStorage` for persisted user preferences (Metal acceleration toggle, max concurrent frames)
- Follow Apple HIG for macOS: `GroupBox`, `HSplitView`, `.borderedProminent` buttons, toolbar items

### Metal / GPU
- Always guard `MTLCreateSystemDefaultDevice()` and `makeCommandQueue()` — fatal-error on failure is acceptable at app init
- Use `CVMetalTextureCache` for zero-copy `CVPixelBuffer` → `MTLTexture` conversion
- Process frames in batches of 16 to balance throughput vs. memory pressure
- Pipeline state is set up once at init via `device.makeDefaultLibrary()` + `makeComputePipelineState`

### Error Handling
- Propagate errors with `throws` from processing functions; catch in `Task {}` blocks in views
- Print errors to console with `error.localizedDescription`; surface UI alerts for user-facing failures

### Performance
- `PerformanceMonitor.calculateEstimates` must be called on every slider/stepper change (`.onChange` modifier)
- Base estimate: `0.033s/frame × (1 + keyframeCount × morphStrength)`
- Memory estimate: `width × height × 4 bytes × frameCount`

---

## Development Workflow

Since the Xcode project lives externally, changes described in `README.md` apply to the live project at the path above.

### Typical Flow
1. Read the conversation transcript in `README.md` to understand the latest design intent
2. Open the Xcode project at `/Users/wvn/Documents/github/wvn/metal_video_morpher/`
3. Implement or update the relevant `.swift` / `.metal` files
4. Build and run on macOS to test GPU processing
5. Document design decisions or new conversation context back into `README.md`

### Running / Testing
- Build target: macOS app (not iOS)
- Requires a Mac with Metal-capable GPU (all Apple Silicon and most Intel Macs)
- No unit test suite described yet; validate visually via the in-app preview player and performance panel

---

## Git Conventions

- Branch: `claude/add-claude-documentation-raJOk`
- Commit messages should be descriptive and imperative (e.g., "Add real-time performance monitor")
- Push with: `git push -u origin <branch-name>`

---

## Notes for AI Assistants

- The `README.md` is a raw conversation log, **not** structured documentation — treat it as design context, not ground truth
- No actual Swift or Metal code exists in this repository; the described code is referenced/proposed in the README
- When adding new features, follow the established patterns: `ObservableObject` + `@Published`, `async/await`, Metal singleton utilities
- Prefer editing existing files over creating new ones; keep the component count minimal
- Do not add unnecessary abstractions — three similar lines are better than premature helper functions
- The app targets macOS 11.0+; use `#available(macOS 11.0, *)` guards where needed for newer APIs
