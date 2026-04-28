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
    targets: [
        .executableTarget(
            name: "ButterchurnVisualizer",
            path: "Sources/ButterchurnVisualizer",
            exclude: [
                "Milkdrop.metal",   // WebGL (Butterchurn) handles rendering; Metal shader kept for reference
            ],
            resources: [
                // butterchurn_host.html + microtonal_warp.milk + preset JSON folders
                .process("Resources")
            ]
        )
    ]
)
