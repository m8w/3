// swift-tools-version:5.9
// ──────────────────────────────────────────────────────────────────────────────
//  ButterchurnVisualizer — Swift Package
//
//  HOW TO OPEN IN XCODE (Mac Mini M2):
//    File → Open  →  select this Package.swift
//    Xcode will index the package and show the ButterchurnVisualizer scheme.
//
//  FIRST-TIME SETUP (one-time):
//    1. Select the "ButterchurnVisualizer" scheme in the toolbar.
//    2. Product → Destination → My Mac
//    3. Signing & Capabilities tab → + Capability → "Audio Input"
//       (Adds com.apple.security.device.audio-input — required for any
//        AVAudioEngine input tap, including virtual devices like BlackHole.)
//    4. ⌘R to build and run.
//
//  AUDIO ROUTING (BlackHole 2ch — already set up by default):
//    System Settings → Sound → Input → BlackHole 2ch
//    Audio MIDI Setup → Multi-Output Device → BlackHole 2ch + headphones/speakers
//    See AudioEngine.swift → `inputDeviceName` to change the device.
// ──────────────────────────────────────────────────────────────────────────────

import PackageDescription

let package = Package(
    name: "ButterchurnVisualizer",
    platforms: [
        .macOS(.v13)   // Ventura — native on M2 Mac Mini
    ],
    dependencies: [
        // RTMP broadcast (H.264/AAC → YouTube/Restream). Pinned to the 1.x line;
        // the 2.x rewrite changes the API used in Broadcaster.swift.
        .package(url: "https://github.com/shogo4405/HaishinKit.swift.git",
                 .upToNextMajor(from: "1.6.0")),
    ],
    targets: [
        .executableTarget(
            name: "ButterchurnVisualizer",
            dependencies: [
                .product(name: "HaishinKit", package: "HaishinKit.swift"),
            ],
            path: "Sources/ButterchurnVisualizer",
            exclude: [
                "Milkdrop.metal",
                "Resources/presets_review",
            ],
            resources: [
                // List resources explicitly so SPM never touches the legacy
                // Presets _ Butterchurn / Presets1 folders (filenames with $$
                // in them break SPM's resource processor).
                .process("Resources/butterchurn.min.js"),
                .process("Resources/milkdrop-preset-converter.min.js"),
                .process("Resources/butterchurn_host.html"),
                .process("Resources/mixer_host.html"),
                .process("Resources/curate_host.html"),
                .process("Resources/microtonal_warp.milk"),
                .process("Resources/presets"),
            ]
        )
    ]
)
