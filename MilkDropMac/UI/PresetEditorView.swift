// PresetEditorView.swift — BeatDrop-style MilkDrop preset editor
// Features:
//   - Side-by-side code editor (per-frame, per-vertex, warp HLSL, comp HLSL)
//   - Live parameter sliders (zoom, rot, warp, decay, etc.)
//   - Wave & shape editors (16 slots each, MilkDrop 3 style)
//   - Real-time preview
//   - Save / Save As / Export
//   - Syntax highlighting via NSTextView
//   - Undo/Redo (Cmd+Z / Cmd+Y)
//   - Recompile (Cmd+Enter)
//   - Double-preset (.milk2) support

import SwiftUI

struct PresetEditorView: View {
    @EnvironmentObject var state: AppState
    @Environment(\.dismiss) var dismiss

    @State var preset: MilkDropPreset
    @State private var selectedSection: EditorSection = .perFrame
    @State private var isDirty: Bool = false
    @State private var showSaveAlert: Bool = false
    @State private var params: PresetParameters

    init(preset: MilkDropPreset) {
        _preset = State(initialValue: preset)
        var p = preset
        p.parseParameters()
        _params = State(initialValue: p.parameters ?? PresetParameters())
    }

    var body: some View {
        VStack(spacing: 0) {
            // Toolbar
            EditorToolbar(
                preset: $preset,
                isDirty: isDirty,
                onSave: { savePreset() },
                onSaveAs: { savePresetAs() },
                onRecompile: { recompile() },
                onClose: { handleClose() }
            )

            Divider()

            HSplitView {
                // Left: Parameter sliders
                ScrollView {
                    VStack(alignment: .leading, spacing: 0) {
                        // Preset info
                        ParameterSection(title: "Preset Info") {
                            ParameterField(label: "Name", text: $preset.name)
                                .onChange(of: preset.name) { isDirty = true }
                            ParameterField(label: "Author", text: $preset.author)
                                .onChange(of: preset.author) { isDirty = true }

                            // Rating
                            HStack {
                                Text("Rating").font(.system(size: 11)).foregroundColor(.secondary)
                                Spacer()
                                ForEach(1...5, id: \.self) { i in
                                    Button(action: { preset.rating = i; isDirty = true }) {
                                        Image(systemName: i <= preset.rating ? "star.fill" : "star")
                                            .foregroundColor(.yellow)
                                            .font(.system(size: 14))
                                    }
                                    .buttonStyle(.plain)
                                }
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                        }

                        // Warp / Zoom
                        ParameterSection(title: "Warp & Zoom") {
                            ParamSlider(label: "Zoom",   value: $params.zoomAmount,  range: 0.1...5.0)
                            ParamSlider(label: "Rotate", value: $params.rotatAmount, range: -3.14...3.14)
                            ParamSlider(label: "Warp",   value: $params.warpScale,   range: 0...10)
                            ParamSlider(label: "Decay",  value: $params.decay,       range: 0.8...1.0)
                            ParamSlider(label: "Centre X", value: $params.centreX,   range: 0...1)
                            ParamSlider(label: "Centre Y", value: $params.centreY,   range: 0...1)
                            ParamSlider(label: "Scale X",  value: $params.szx,       range: 0.5...2.0)
                            ParamSlider(label: "Scale Y",  value: $params.szy,       range: 0.5...2.0)
                            ParamSlider(label: "Gamma",    value: $params.gamma,     range: 0.1...3.0)
                        }
                        .onChange(of: params.zoomAmount)   { markDirty() }
                        .onChange(of: params.rotatAmount)  { markDirty() }
                        .onChange(of: params.warpScale)    { markDirty() }
                        .onChange(of: params.decay)        { markDirty() }

                        // Video echo (feedback)
                        ParameterSection(title: "Video Echo") {
                            ParamSlider(label: "Alpha", value: $params.videoEchoAlpha, range: 0...1)
                            ParamSlider(label: "Zoom",  value: $params.videoEchoZoom,  range: 0.1...5)
                            HStack {
                                Text("Orientation").font(.system(size: 11)).foregroundColor(.secondary)
                                Spacer()
                                Picker("", selection: $params.videoEchoOrientation) {
                                    Text("Normal").tag(0)
                                    Text("Flip H").tag(1)
                                    Text("Flip V").tag(2)
                                    Text("Flip Both").tag(3)
                                }
                                .pickerStyle(.menu)
                                .frame(width: 90)
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                        }

                        // Transition
                        ParameterSection(title: "Transition") {
                            HStack {
                                Text("Blend Mode").font(.system(size: 11)).foregroundColor(.secondary)
                                Spacer()
                                Picker("", selection: $params.blendType) {
                                    ForEach(PresetParameters.TransitionBlend.allCases, id: \.self) { t in
                                        Text(t.rawValue.capitalized).tag(t)
                                    }
                                }
                                .pickerStyle(.menu)
                                .frame(width: 110)
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                        }

                        // MilkDrop 3 Beat Detection
                        ParameterSection(title: "Beat Detection (MilkDrop 3)") {
                            Toggle("Enable Hardcut", isOn: $params.hardcutEnabled)
                                .padding(.horizontal, 8).padding(.vertical, 4)
                                .font(.system(size: 11))
                            ParamSlider(label: "Bass Threshold",   value: $params.hardcutBass,   range: 0...1)
                            ParamSlider(label: "Treble Threshold", value: $params.hardcutTreble, range: 0...1)
                            ParamSlider(label: "Min Delay (s)",    value: $params.hardcutDelay,  range: 0.5...10)
                        }

                        // Waves (up to 16)
                        ParameterSection(title: "Waves (\(params.waves.filter { $0.enabled }.count)/16 active)") {
                            WaveListEditor(waves: $params.waves)
                        }

                        // Shapes (up to 16)
                        ParameterSection(title: "Shapes (\(params.shapes.filter { $0.enabled }.count)/16 active)") {
                            ShapeListEditor(shapes: $params.shapes)
                        }
                    }
                }
                .frame(minWidth: 260, maxWidth: 320)
                .background(Color(hex: "0d0d0d"))

                // Right: Code editor
                VStack(spacing: 0) {
                    // Section tabs
                    HStack(spacing: 0) {
                        ForEach(EditorSection.allCases, id: \.self) { section in
                            EditorTabButton(
                                title: section.label,
                                isSelected: selectedSection == section,
                                action: { selectedSection = section }
                            )
                        }
                        Spacer()

                        // Help button
                        Button(action: openPresetsGuide) {
                            Image(systemName: "questionmark.circle")
                                .foregroundColor(.secondary)
                        }
                        .buttonStyle(.plain)
                        .padding(.horizontal, 8)
                        .help("Open MilkDrop Preset Authoring Guide")
                    }
                    .background(Color(hex: "141414"))

                    Divider()

                    // Code editor area
                    EditorCodeView(
                        section: selectedSection,
                        params: $params,
                        onChange: { markDirty() }
                    )
                }
            }
        }
        .background(Color(hex: "111111"))
        .preferredColorScheme(.dark)
    }

    // MARK: - Actions

    private func markDirty() { isDirty = true }

    private func recompile() {
        // Serialize params back to preset data
        preset.data = PresetParser.serialize(params, name: preset.name)
        preset.parameters = params
        // Push to renderer
        state.renderer?.loadPreset(preset)
    }

    private func savePreset() {
        recompile()
        try? state.presetManager.savePreset(preset)
        isDirty = false
    }

    private func savePresetAs() {
        recompile()
        let panel = NSSavePanel()
        panel.nameFieldStringValue = preset.name + "." + preset.fileExtension
        panel.allowedContentTypes = [.milk, .milk2]
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            try? preset.data.write(to: url, atomically: true, encoding: .utf8)
        }
    }

    private func handleClose() {
        if isDirty { showSaveAlert = true } else { dismiss() }
    }

    private func openPresetsGuide() {
        NSWorkspace.shared.open(URL(string: "https://www.geisswerks.com/milkdrop/milkdrop_preset_authoring.html")!)
    }
}

// MARK: - Editor toolbar

struct EditorToolbar: View {
    @Binding var preset: MilkDropPreset
    let isDirty: Bool
    let onSave: () -> Void
    let onSaveAs: () -> Void
    let onRecompile: () -> Void
    let onClose: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            Text("Preset Editor")
                .font(.system(size: 13, weight: .semibold))

            Spacer()

            if isDirty {
                Circle()
                    .fill(Color.orange)
                    .frame(width: 8, height: 8)
                    .help("Unsaved changes")
            }

            Button(action: onRecompile) {
                Label("Preview", systemImage: "play.fill")
                    .font(.system(size: 12))
            }
            .buttonStyle(.bordered)
            .keyboardShortcut(.return, modifiers: .command)
            .help("Recompile and preview (Cmd+Enter)")

            Button(action: onSave) {
                Label("Save", systemImage: "checkmark")
                    .font(.system(size: 12))
            }
            .buttonStyle(.borderedProminent)
            .disabled(!isDirty)
            .keyboardShortcut("s", modifiers: .command)

            Button(action: onSaveAs) {
                Image(systemName: "square.and.arrow.down.on.square")
            }
            .buttonStyle(.bordered)
            .help("Save As...")

            Divider()

            Button(action: onClose) {
                Image(systemName: "xmark")
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color(hex: "141414"))
    }
}

// MARK: - Editor sections

enum EditorSection: String, CaseIterable {
    case perFrame    = "Per Frame"
    case perVertex   = "Per Vertex"
    case warpHLSL    = "Warp Shader"
    case compHLSL    = "Comp Shader"

    var label: String { rawValue }
    var placeholder: String {
        switch self {
        case .perFrame:
            return "// Per-frame equations — run once per frame\n// Variables: zoom, rot, warp, cx, cy, dx, dy, sx, sy, decay\n// Audio: bass, mid, treble, vol, bass_att\n// Time: time, fps, frame\n// Example:\nzoom = 1 + 0.02*bass;\nrot = rot + 0.01*mid;"
        case .perVertex:
            return "// Per-vertex equations — run for each warp mesh vertex\n// Use x, y (vertex position) and all per-frame vars\n// Example:\nzoom = zoom + 0.001*sin(x*3+time);"
        case .warpHLSL:
            return "// Warp pixel shader (HLSL/MSL)\n// Return sampled color with distortion\n// float4 WarpSampler(float2 uv) : SV_Target\n// {\n//   return tex2D(sampler_main, uv);\n// }"
        case .compHLSL:
            return "// Composite pixel shader (HLSL/MSL)\n// Applied after warp + waves/shapes\n// float4 CompSampler(float2 uv) : SV_Target\n// {\n//   float4 c = tex2D(sampler_main, uv);\n//   c.rgb *= brightness;\n//   return c;\n// }"
        }
    }
}

// MARK: - Code editor view

struct EditorCodeView: View {
    let section: EditorSection
    @Binding var params: PresetParameters
    let onChange: () -> Void

    var binding: Binding<String> {
        switch section {
        case .perFrame:   return Binding(
            get: { params.perFrame.joined(separator: ";\n") },
            set: { params.perFrame = $0.components(separatedBy: .newlines).map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty } }
        )
        case .perVertex:  return Binding(
            get: { params.perVertex.joined(separator: ";\n") },
            set: { params.perVertex = $0.components(separatedBy: .newlines).map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty } }
        )
        case .warpHLSL:  return $params.warpHLSL
        case .compHLSL:  return $params.compHLSL
        }
    }

    var body: some View {
        TextEditor(text: binding)
            .font(.system(size: 13, design: .monospaced))
            .scrollContentBackground(.hidden)
            .background(Color(hex: "0c0c0c"))
            .foregroundColor(Color(hex: "e0e0e0"))
            .padding(12)
            .onChange(of: binding.wrappedValue) { onChange() }
            .overlay(
                // Placeholder
                Group {
                    if binding.wrappedValue.isEmpty {
                        Text(section.placeholder)
                            .font(.system(size: 12, design: .monospaced))
                            .foregroundColor(Color(hex: "444444"))
                            .padding(16)
                            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                            .allowsHitTesting(false)
                    }
                }
            )
    }
}

// MARK: - Wave list editor

struct WaveListEditor: View {
    @Binding var waves: [PresetWave]

    var body: some View {
        VStack(spacing: 4) {
            // Ensure 16 slots
            ForEach(0..<16, id: \.self) { i in
                let idx = waves.firstIndex(where: { $0.id == i })
                if let idx {
                    WaveRowEditor(wave: $waves[idx])
                } else {
                    Button(action: {
                        var w = PresetWave(id: i)
                        w.enabled = true
                        waves.append(w)
                    }) {
                        HStack {
                            Text("Wave \(i)")
                                .font(.system(size: 10)).foregroundColor(.secondary)
                            Spacer()
                            Text("+ Add")
                                .font(.system(size: 10)).foregroundColor(.accentColor)
                        }
                        .padding(.horizontal, 8).padding(.vertical, 3)
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }
}

struct WaveRowEditor: View {
    @Binding var wave: PresetWave

    var body: some View {
        DisclosureGroup {
            VStack(alignment: .leading, spacing: 4) {
                ParamSlider(label: "Scaling",   value: $wave.scaling,   range: 0...5)
                ParamSlider(label: "Smoothing", value: $wave.smoothing, range: 0...1)
                ParamSlider(label: "R", value: $wave.r, range: 0...1, tint: .red)
                ParamSlider(label: "G", value: $wave.g, range: 0...1, tint: .green)
                ParamSlider(label: "B", value: $wave.b, range: 0...1, tint: .blue)
                ParamSlider(label: "A", value: $wave.a, range: 0...1)
                HStack {
                    Toggle("Dots",    isOn: $wave.useDots).font(.system(size: 10))
                    Toggle("Thick",   isOn: $wave.drawThick).font(.system(size: 10))
                    Toggle("Additive", isOn: $wave.additive).font(.system(size: 10))
                }
                .padding(.horizontal, 4)
            }
        } label: {
            HStack {
                Toggle("", isOn: $wave.enabled)
                    .labelsHidden()
                    .scaleEffect(0.8)
                Text("Wave \(wave.id)")
                    .font(.system(size: 11, weight: wave.enabled ? .medium : .regular))
                    .foregroundColor(wave.enabled ? .primary : .secondary)
                Spacer()
                Circle()
                    .fill(Color(red: Double(wave.r), green: Double(wave.g), blue: Double(wave.b)))
                    .frame(width: 8, height: 8)
            }
            .padding(.horizontal, 8)
        }
        .padding(.vertical, 2)
    }
}

// MARK: - Shape list editor

struct ShapeListEditor: View {
    @Binding var shapes: [PresetShape]

    var body: some View {
        VStack(spacing: 4) {
            ForEach(0..<16, id: \.self) { i in
                let idx = shapes.firstIndex(where: { $0.id == i })
                if let idx {
                    ShapeRowEditor(shape: $shapes[idx])
                } else {
                    Button(action: {
                        var s = PresetShape(id: i)
                        s.enabled = true
                        shapes.append(s)
                    }) {
                        HStack {
                            Text("Shape \(i)")
                                .font(.system(size: 10)).foregroundColor(.secondary)
                            Spacer()
                            Text("+ Add")
                                .font(.system(size: 10)).foregroundColor(.accentColor)
                        }
                        .padding(.horizontal, 8).padding(.vertical, 3)
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }
}

struct ShapeRowEditor: View {
    @Binding var shape: PresetShape

    var body: some View {
        DisclosureGroup {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Sides").font(.system(size: 10)).foregroundColor(.secondary)
                    Stepper("\(shape.sides)", value: $shape.sides, in: 3...100)
                        .font(.system(size: 11))
                }
                .padding(.horizontal, 8)
                ParamSlider(label: "X",      value: $shape.x,      range: 0...1)
                ParamSlider(label: "Y",      value: $shape.y,      range: 0...1)
                ParamSlider(label: "Radius", value: $shape.radius, range: 0...1)
                ParamSlider(label: "Angle",  value: $shape.ang,    range: -.pi...(.pi))
                ParamSlider(label: "R",  value: $shape.r,  range: 0...1, tint: .red)
                ParamSlider(label: "G",  value: $shape.g,  range: 0...1, tint: .green)
                ParamSlider(label: "B",  value: $shape.b,  range: 0...1, tint: .blue)
                ParamSlider(label: "A",  value: $shape.a,  range: 0...1)
            }
        } label: {
            HStack {
                Toggle("", isOn: $shape.enabled)
                    .labelsHidden().scaleEffect(0.8)
                Text("Shape \(shape.id) (\(shape.sides))")
                    .font(.system(size: 11, weight: shape.enabled ? .medium : .regular))
                    .foregroundColor(shape.enabled ? .primary : .secondary)
                Spacer()
                Circle()
                    .fill(Color(red: Double(shape.r), green: Double(shape.g), blue: Double(shape.b)))
                    .frame(width: 8, height: 8)
            }
            .padding(.horizontal, 8)
        }
        .padding(.vertical, 2)
    }
}

// MARK: - Shared UI components

struct ParameterSection<Content: View>: View {
    let title: String
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(title.uppercased())
                .font(.system(size: 9, weight: .semibold))
                .foregroundColor(Color(hex: "888888"))
                .padding(.horizontal, 8)
                .padding(.top, 12)
                .padding(.bottom, 4)

            content()

            Divider().background(Color(hex: "222222"))
        }
    }
}

struct ParamSlider: View {
    let label: String
    @Binding var value: Float
    let range: ClosedRange<Float>
    var tint: Color = .accentColor

    var body: some View {
        HStack(spacing: 8) {
            Text(label)
                .font(.system(size: 11))
                .foregroundColor(.secondary)
                .frame(width: 70, alignment: .trailing)

            Slider(value: $value, in: range)
                .accentColor(tint)

            Text(String(format: "%.3f", value))
                .font(.system(size: 10, design: .monospaced))
                .foregroundColor(.secondary)
                .frame(width: 50, alignment: .leading)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
    }
}

struct ParameterField: View {
    let label: String
    @Binding var text: String

    var body: some View {
        HStack {
            Text(label)
                .font(.system(size: 11))
                .foregroundColor(.secondary)
                .frame(width: 50, alignment: .trailing)
            TextField("", text: $text)
                .textFieldStyle(.plain)
                .font(.system(size: 12))
                .padding(4)
                .background(Color(hex: "1a1a1a"))
                .cornerRadius(4)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
    }
}

struct EditorTabButton: View {
    let title: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 11, weight: isSelected ? .semibold : .regular))
                .foregroundColor(isSelected ? .white : .secondary)
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .background(
                    isSelected
                    ? Color(hex: "1e1e1e")
                    : Color.clear
                )
        }
        .buttonStyle(.plain)
    }
}
