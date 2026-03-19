// SettingsView.swift — App-wide settings (render quality, display, etc.)

import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                // Display
                ParameterSection(title: "Display") {
                    Picker("Resolution", selection: $state.renderResolution) {
                        ForEach(RenderResolution.allCases, id: \.self) { r in
                            Text(r.rawValue).tag(r)
                        }
                    }
                    .pickerStyle(.menu)
                    .padding(.horizontal, 8).padding(.vertical, 4)

                    Toggle("Show FPS Counter", isOn: $state.showFPSCounter)
                        .padding(.horizontal, 8).padding(.vertical, 4).font(.system(size: 12))

                    Toggle("Show Preset Name", isOn: $state.showPresetName)
                        .padding(.horizontal, 8).padding(.vertical, 4).font(.system(size: 12))

                    ParamSlider(label: "Brightness", value: Binding(
                        get: { Float(state.brightness) },
                        set: { state.brightness = Double($0) }
                    ), range: 0.1...2.0)

                    ParamSlider(label: "Gamma", value: Binding(
                        get: { Float(state.gamma) },
                        set: { state.gamma = Double($0) }
                    ), range: 0.5...3.0)
                }

                // Transition
                ParameterSection(title: "Transitions") {
                    ParamSlider(label: "Duration", value: Binding(
                        get: { Float(state.transitionDuration) },
                        set: { state.transitionDuration = Double($0) }
                    ), range: 0...5.0)

                    Toggle("Hardcut Mode", isOn: $state.hardcutEnabled)
                        .padding(.horizontal, 8).padding(.vertical, 4).font(.system(size: 12))

                    ParamSlider(label: "Hardcut Sensitivity", value: Binding(
                        get: { Float(state.hardcutSensitivity) },
                        set: { state.hardcutSensitivity = Double($0) }
                    ), range: 0...1)
                }

                // About
                ParameterSection(title: "About") {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("MilkDropMac")
                                .font(.system(size: 13, weight: .bold))
                            Spacer()
                            Text("v1.0")
                                .font(.system(size: 11))
                                .foregroundColor(.secondary)
                        }
                        Text("Native macOS MilkDrop 3 visualizer")
                            .font(.system(size: 11))
                            .foregroundColor(.secondary)
                        Text("Metal GPU rendering • Syphon output • BeatDrop-style editor")
                            .font(.system(size: 10))
                            .foregroundColor(Color(hex: "666666"))

                        HStack(spacing: 16) {
                            Link("MilkDrop 3", destination: URL(string: "https://milkdrop3.com")!)
                                .font(.system(size: 11))
                            Link("projectM", destination: URL(string: "https://github.com/projectM-visualizer/projectm")!)
                                .font(.system(size: 11))
                            Link("Syphon", destination: URL(string: "https://syphon.info")!)
                                .font(.system(size: 11))
                        }
                        .padding(.top, 4)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 8)
                }

                // Keyboard shortcuts
                ParameterSection(title: "Keyboard Shortcuts") {
                    VStack(alignment: .leading, spacing: 3) {
                        ShortcutRow("→ / ←",   "Next / Previous preset")
                        ShortcutRow("R",        "Random preset")
                        ShortcutRow("L",        "Lock preset")
                        ShortcutRow("⌘B",       "Toggle beat detection")
                        ShortcutRow("⌘E",       "Open preset editor")
                        ShortcutRow("⌘⇧D",      "Create double preset (.milk2)")
                        ShortcutRow("⌘Enter",   "Recompile preset (in editor)")
                        ShortcutRow("⌘S",       "Save preset")
                        ShortcutRow("⌘I",       "Import presets")
                        ShortcutRow("F",        "Toggle fullscreen")
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 6)
                }
            }
        }
        .background(Color(hex: "0d0d0d"))
    }
}

struct ShortcutRow: View {
    let keys: String
    let action: String

    init(_ keys: String, _ action: String) {
        self.keys = keys
        self.action = action
    }

    var body: some View {
        HStack {
            Text(keys)
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundColor(.accentColor)
                .frame(width: 70, alignment: .leading)
            Text(action)
                .font(.system(size: 11))
                .foregroundColor(.secondary)
        }
    }
}

// MARK: - Quick editor (sidebar version)

struct QuickEditorView: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        VStack(spacing: 8) {
            if let preset = state.presetManager.currentPreset {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Current: \(preset.name)")
                        .font(.system(size: 11, weight: .medium))
                        .lineLimit(1)
                        .padding(.horizontal, 8)
                        .padding(.top, 8)

                    Button(action: {
                        state.editingPreset = preset
                        state.showPresetEditor = true
                    }) {
                        Label("Open Full Editor", systemImage: "chevron.left.forwardslash.chevron.right")
                            .font(.system(size: 12))
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .padding(.horizontal, 8)
                    .keyboardShortcut("e", modifiers: .command)
                }

                Divider().background(Color(hex: "2a2a2a"))

                // Quick sliders for current preset
                ScrollView {
                    VStack(spacing: 0) {
                        ParameterSection(title: "Quick Adjust") {
                            if var params = state.presetManager.currentPreset?.parameters {
                                ParamSlider(label: "Zoom",  value: .constant(params.zoomAmount),  range: 0.1...5)
                                ParamSlider(label: "Warp",  value: .constant(params.warpScale),   range: 0...10)
                                ParamSlider(label: "Decay", value: .constant(params.decay),       range: 0.8...1.0)
                                ParamSlider(label: "Gamma", value: .constant(params.gamma),       range: 0.1...3)
                            }
                        }
                    }
                }
            } else {
                ContentUnavailableView(
                    "No Preset Loaded",
                    systemImage: "waveform.path",
                    description: Text("Select a preset from the Presets tab")
                )
            }
        }
        .background(Color(hex: "0d0d0d"))
    }
}
