// AudioSettingsView.swift — Audio configuration and live spectrum display

import SwiftUI

struct AudioSettingsView: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                // Live spectrum
                ParameterSection(title: "Live Spectrum") {
                    SpectrumMiniView()
                        .frame(height: 80)
                        .padding(8)
                }

                // Audio source
                ParameterSection(title: "Audio Source") {
                    Picker("Source", selection: $state.audioSource) {
                        ForEach(AudioSource.allCases) { src in
                            Text(src.rawValue).tag(src)
                        }
                    }
                    .pickerStyle(.menu)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                }

                // Sensitivity
                ParameterSection(title: "Sensitivity") {
                    ParamSlider(label: "Overall", value: Binding(
                        get: { Float(state.audioSensitivity) },
                        set: { state.audioSensitivity = Double($0) }
                    ), range: 0.1...5.0)

                    ParamSlider(label: "Bass", value: Binding(
                        get: { Float(state.bassBoost) },
                        set: { state.bassBoost = Double($0) }
                    ), range: 0.1...5.0, tint: .red)

                    ParamSlider(label: "Mid", value: Binding(
                        get: { Float(state.midBoost) },
                        set: { state.midBoost = Double($0) }
                    ), range: 0.1...5.0, tint: .green)

                    ParamSlider(label: "Treble", value: Binding(
                        get: { Float(state.trebleBoost) },
                        set: { state.trebleBoost = Double($0) }
                    ), range: 0.1...5.0, tint: .blue)
                }

                // Beat Detection
                ParameterSection(title: "Beat Detection") {
                    Toggle("Enable Auto-Switch", isOn: $state.beatDetectionEnabled)
                        .padding(.horizontal, 8).padding(.vertical, 4)
                        .font(.system(size: 12))

                    Picker("Mode", selection: Binding(
                        get: { state.beatDetector.hardcutMode },
                        set: { state.beatDetector.hardcutMode = $0 }
                    )) {
                        ForEach(BeatDetector.HardcutMode.allCases, id: \.self) { m in
                            Text(m.rawValue).tag(m)
                        }
                    }
                    .pickerStyle(.menu)
                    .padding(.horizontal, 8).padding(.vertical, 4)
                    .disabled(!state.beatDetectionEnabled)

                    ParamSlider(label: "Bass Thresh", value: Binding(
                        get: { Float(state.beatDetector.hardcutLowThreshold) },
                        set: { state.beatDetector.hardcutLowThreshold = Double($0) }
                    ), range: 0...1)

                    ParamSlider(label: "Treble Thresh", value: Binding(
                        get: { Float(state.beatDetector.hardcutHighThreshold) },
                        set: { state.beatDetector.hardcutHighThreshold = Double($0) }
                    ), range: 0...1)

                    ParamSlider(label: "Min Delay", value: Binding(
                        get: { Float(state.beatDetector.hardcutMinDelay) },
                        set: { state.beatDetector.hardcutMinDelay = Double($0) }
                    ), range: 0.5...10.0)

                    // BPM readout
                    HStack {
                        Text("BPM").font(.system(size: 11)).foregroundColor(.secondary)
                        Spacer()
                        Text(String(format: "%.0f", state.beatDetector.bpm))
                            .font(.system(size: 13, weight: .bold, design: .monospaced))
                            .foregroundColor(.green)
                    }
                    .padding(.horizontal, 8).padding(.vertical, 4)
                }

                // Auto-switch (non-beat mode)
                ParameterSection(title: "Timed Auto-Switch") {
                    Toggle("Enable", isOn: Binding(
                        get: { !state.beatDetectionEnabled },
                        set: { state.beatDetectionEnabled = !$0 }
                    ))
                    .padding(.horizontal, 8).padding(.vertical, 4)
                    .font(.system(size: 12))

                    ParamSlider(label: "Interval", value: Binding(
                        get: { Float(state.autoSwitchInterval) },
                        set: { state.autoSwitchInterval = Double($0) }
                    ), range: 3...120)

                    ParamSlider(label: "Transition", value: Binding(
                        get: { Float(state.transitionDuration) },
                        set: { state.transitionDuration = Double($0) }
                    ), range: 0...5.0)
                }
            }
        }
        .background(Color(hex: "0d0d0d"))
    }
}

// MARK: - Mini spectrum display

struct SpectrumMiniView: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        GeometryReader { geo in
            Canvas { ctx, size in
                // Bass indicator
                let bassH = size.height * CGFloat(state.audioLevel) * 2
                ctx.fill(
                    Path(CGRect(x: 0, y: size.height - bassH, width: 16, height: bassH)),
                    with: .color(.red.opacity(0.7))
                )
                // Beat flash
                if state.beatStrength > 0.8 {
                    ctx.fill(
                        Path(CGRect(x: 0, y: 0, width: size.width, height: size.height)),
                        with: .color(.white.opacity(0.05))
                    )
                }

                // Placeholder spectrum bars
                let barCount = 48
                let barW = (size.width - 20) / CGFloat(barCount)
                for i in 0..<barCount {
                    let level = state.audioLevel * (sin(Double(i) * 0.4) * 0.4 + 0.6)
                    let barH = size.height * CGFloat(level) * 0.9
                    let x = 20 + CGFloat(i) * barW
                    let hue = Double(i) / Double(barCount) * 0.35 + 0.5  // cyan → green
                    ctx.fill(
                        Path(CGRect(x: x + 1, y: size.height - barH, width: barW - 2, height: barH)),
                        with: .color(Color(hue: hue, saturation: 0.9, brightness: 0.9).opacity(0.8))
                    )
                }
            }
            .background(Color(hex: "0a0a0a"))
            .cornerRadius(4)
        }
    }
}
