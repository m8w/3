// swift-tools-version:5.9
// ──────────────────────────────────────────────────────────────────────────────
//  ButterchurnVisualizer — Swift Package
//
//  TARGETS
//  ┌─────────────────────┬──────────────────────────────────────────────────┐
//  │ ButterchurnCore     │ Shared library — PresetParser, MilkToJSON,       │
//  │  (library)          │ MicrotonalFFT.  No UI or Metal dependencies.     │
//  ├─────────────────────┼──────────────────────────────────────────────────┤
//  │ ButterchurnVisualizer│ macOS app — WKWebView + AVAudioEngine + Metal.  │
//  │  (executable)       │ Depends on ButterchurnCore.                      │
//  ├─────────────────────┼──────────────────────────────────────────────────┤
//  │ MilkConverter       │ CLI tool — convert .milk ↔ Butterchurn JSON.     │
//  │  (executable)       │ Depends on ButterchurnCore.  No UI needed.       │
//  └─────────────────────┴──────────────────────────────────────────────────┘
//
//  QUICK START — visualizer app (Xcode):
//    File → Open → Package.swift
//    Scheme: ButterchurnVisualizer | Destination: My Mac
//    Signing & Capabilities → + Microphone  →  ⌘R
//
//  QUICK START — converter CLI (Terminal):
//    swift run MilkConverter  MyPreset.milk           # → MyPreset.json
//    swift run MilkConverter  ~/path/to/presets/      # batch folder
//    swift run MilkConverter  MyPreset.json --reverse # → MyPreset.milk
//    swift run MilkConverter  MyPreset.milk --dry-run # print, no write
// ──────────────────────────────────────────────────────────────────────────────

import PackageDescription

let package = Package(
    name: "ButterchurnVisualizer",
    platforms: [
        .macOS(.v13)   // Ventura — native on M2 Mac Mini
    ],
    targets: [

        // ── Shared library ────────────────────────────────────────────────────
        // PresetParser, MilkToJSON, MicrotonalFFT.
        // No AVFoundation / Metal / SwiftUI — safe for CLI use.
        .target(
            name: "ButterchurnCore",
            path: "Sources/ButterchurnCore"
        ),

        // ── macOS visualizer app ──────────────────────────────────────────────
        .executableTarget(
            name: "ButterchurnVisualizer",
            dependencies: ["ButterchurnCore"],
            path: "Sources/ButterchurnVisualizer",
            resources: [
                .process("Resources")   // butterchurn_host.html, microtonal_warp.milk
            ]
        ),

        // ── .milk ↔ Butterchurn JSON CLI ──────────────────────────────────────
        .executableTarget(
            name: "MilkConverter",
            dependencies: ["ButterchurnCore"],
            path: "Sources/MilkConverter"
        ),
    ]
)
