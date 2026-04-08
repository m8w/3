// swift-tools-version:5.9
// ──────────────────────────────────────────────────────────────────────────────
//  ButterchurnVisualizer — Swift Package
//
//  HOW TO OPEN IN XCODE (Mac Mini M2):
//    File → Open  →  select this Package.swift
//    Xcode will index the package and show the ButterchurnVisualizer scheme.
//
//  FIRST-TIME SETUP (one-time, takes ~60 s):
//    1. Select the "ButterchurnVisualizer" scheme in the toolbar.
//    2. Product → Destination → My Mac
//    3. Signing & Capabilities tab → + Capability → "Microphone"
//       (This adds com.apple.security.device.audio-input to the entitlements.)
//    4. ⌘R to build and run.  Allow microphone access when prompted.
//
//  SYSTEM AUDIO (optional — needs BlackHole or ScreenCaptureKit):
//    See AudioEngine.swift for instructions.
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
            resources: [
                // butterchurn_host.html + microtonal_warp.milk
                .process("Resources")
            ]
        )
    ]
)
