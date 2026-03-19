// MorphingView.swift — Sidebar panel for morphing techniques and fractal stream
// Controls: preset transition style, fractal overlay, beat-reactive parameters

import SwiftUI

struct MorphingView: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {

                // MARK: - Fractal Stream
                ParameterSection(title: "Fractal Stream") {
                    VStack(alignment: .leading, spacing: 10) {
                        Toggle(isOn: $state.fractalStreamEnabled) {
                            HStack(spacing: 6) {
                                Image(systemName: "sparkles")
                                    .foregroundColor(state.fractalStreamEnabled ? .purple : .secondary)
                                Text("Enable Fractal Overlay")
                                    .font(.system(size: 12))
                            }
                        }
                        .padding(.horizontal, 8)
                        .padding(.top, 4)

                        if state.fractalStreamEnabled {
                            VStack(spacing: 0) {
                                ParamSlider(label: "Blend", value: Binding(
                                    get: { Float(state.fractalBlend) },
                                    set: { state.fractalBlend = Double($0) }
                                ), range: 0...1)

                                HStack(spacing: 8) {
                                    Image(systemName: "info.circle")
                                        .font(.system(size: 10))
                                        .foregroundColor(.secondary)
                                    Text("Audio-reactive Julia set composited over the warp output. Bass drives Julia-c parameter; treble shifts colors.")
                                        .font(.system(size: 10))
                                        .foregroundColor(.secondary)
                                }
                                .padding(.horizontal, 8)
                                .padding(.vertical, 6)
                            }

                            // Live fractal indicator
                            FractalActivityBar()
                                .padding(.horizontal, 8)
                                .padding(.bottom, 6)
                        }
                    }
                }

                // MARK: - Transition Style
                ParameterSection(title: "Preset Transition Style") {
                    VStack(alignment: .leading, spacing: 4) {
                        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 6) {
                            ForEach(MorphingTechnique.allCases) { technique in
                                MorphTechniqueButton(
                                    technique: technique,
                                    isSelected: state.morphingTechnique == technique.rawValue
                                ) {
                                    state.morphingTechnique = technique.rawValue
                                }
                            }
                        }
                        .padding(.horizontal, 8)
                        .padding(.vertical, 6)

                        // Description of selected technique
                        if let selected = MorphingTechnique(rawValue: state.morphingTechnique) {
                            Text(selected.description)
                                .font(.system(size: 10))
                                .foregroundColor(.secondary)
                                .padding(.horizontal, 8)
                                .padding(.bottom, 6)
                        }
                    }
                }

                // MARK: - Transition Settings
                ParameterSection(title: "Transition Settings") {
                    ParamSlider(label: "Duration", value: Binding(
                        get: { Float(state.transitionDuration) },
                        set: { state.transitionDuration = Double($0) }
                    ), range: 0.5...8.0)

                    ParamSlider(label: "Auto Switch", value: Binding(
                        get: { Float(state.autoSwitchInterval) },
                        set: { state.autoSwitchInterval = Double($0) }
                    ), range: 5...120)
                }
            }
        }
        .background(Color(hex: "0d0d0d"))
    }
}

// MARK: - Technique button

struct MorphTechniqueButton: View {
    let technique: MorphingTechnique
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: technique.icon)
                    .font(.system(size: 16))
                    .foregroundColor(isSelected ? .white : .secondary)
                Text(technique.name)
                    .font(.system(size: 9, weight: .medium))
                    .foregroundColor(isSelected ? .white : .secondary)
                    .multilineTextAlignment(.center)
                    .lineLimit(2)
            }
            .frame(maxWidth: .infinity)
            .frame(height: 52)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(isSelected ? Color.accentColor.opacity(0.35) : Color(hex: "1a1a1a"))
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(isSelected ? Color.accentColor : Color(hex: "2a2a2a"), lineWidth: 1)
                    )
            )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Fractal activity bar (real-time audio reactivity indicator)

struct FractalActivityBar: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Live Reactivity")
                .font(.system(size: 9, weight: .semibold))
                .foregroundColor(.secondary)

            HStack(spacing: 4) {
                ReactiveBand(label: "BASS", value: state.beatStrength, color: .purple)
                ReactiveBand(label: "MID",  value: state.audioLevel * 0.8, color: .blue)
                ReactiveBand(label: "VOL",  value: state.audioLevel, color: .cyan)
            }
        }
    }
}

struct ReactiveBand: View {
    let label: String
    let value: Double
    let color: Color

    var body: some View {
        VStack(spacing: 2) {
            ZStack(alignment: .bottom) {
                RoundedRectangle(cornerRadius: 3)
                    .fill(Color(hex: "1a1a1a"))
                    .frame(height: 24)

                RoundedRectangle(cornerRadius: 3)
                    .fill(color.opacity(0.8))
                    .frame(height: CGFloat(value.clamped(to: 0...1)) * 24)
                    .animation(.easeOut(duration: 0.05), value: value)
            }

            Text(label)
                .font(.system(size: 8, weight: .medium, design: .monospaced))
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - MorphingTechnique extensions

extension MorphingTechnique {
    var icon: String {
        switch self {
        case .zoom:            return "arrow.up.left.and.arrow.down.right"
        case .sideWipe:        return "arrow.right"
        case .plasma:          return "wave.3.right"
        case .circle:          return "circle.dotted"
        case .checkerboard:    return "checkerboard.rectangle"
        case .stars:           return "star.fill"
        case .bezierWarp:      return "scribble.variable"
        case .meshMorph:       return "grid"
        case .fractalDissolve: return "aqi.medium"
        case .fractalStream:   return "tornado"
        }
    }

    var description: String {
        switch self {
        case .zoom:
            return "Zoom-based cross-fade — the incoming preset scales in from center."
        case .sideWipe:
            return "Horizontal wipe — new preset slides in from the left edge."
        case .plasma:
            return "Animated noise mask — organic plasma blob reveals the next preset."
        case .circle:
            return "Expanding circle — iris-style reveal from the center outward."
        case .checkerboard:
            return "8×8 checkerboard tiles animate in with staggered timing."
        case .stars:
            return "8-point radial burst — preset switches along star-shaped rays."
        case .bezierWarp:
            return "Cubic Bezier spatial warp — smoothly bends the outgoing frame before dissolving."
        case .meshMorph:
            return "8×8 mesh warp — each grid cell offsets independently for an organic morph."
        case .fractalDissolve:
            return "Mandelbrot escape-time mask — fractal boundary drives the transition threshold."
        case .fractalStream:
            return "Spiral twist — frame rotates outward along IFS-style streamlines."
        }
    }
}
