// ContentView.swift — Main app window layout
// Dark, professional UI inspired by BeatDrop + VDMX

import SwiftUI
import MetalKit

struct ContentView: View {
    @EnvironmentObject var state: AppState
    @State private var showSidebar: Bool = true

    var body: some View {
        HStack(spacing: 0) {
            // Sidebar
            if showSidebar {
                SidebarView()
                    .frame(width: state.sidebarWidth)
                    .background(Color(hex: "0d0d0d"))

                Divider()
                    .background(Color(hex: "2a2a2a"))
            }

            // Main visualizer area
            ZStack(alignment: .topLeading) {
                VisualizerView()
                    .ignoresSafeArea()

                // Overlay: preset name
                if state.showPresetName, let preset = state.presetManager.currentPreset {
                    VStack(alignment: .leading) {
                        PresetNameBanner(name: preset.name)
                    }
                    .padding(12)
                    .allowsHitTesting(false)
                }

                // Overlay: performance HUD
                if state.showFPSCounter {
                    VStack(alignment: .trailing) {
                        PerformanceHUD()
                    }
                    .frame(maxWidth: .infinity, alignment: .trailing)
                    .padding(12)
                    .allowsHitTesting(false)
                }

                // Overlay: audio bars (bottom)
                AudioLevelBar()
                    .frame(maxHeight: .infinity, alignment: .bottom)
                    .allowsHitTesting(false)
            }
        }
        .background(Color.black)
        .preferredColorScheme(.dark)
        .toolbar {
            ToolbarItemGroup(placement: .navigation) {
                Button(action: { withAnimation(.easeInOut(duration: 0.2)) { showSidebar.toggle() } }) {
                    Image(systemName: showSidebar ? "sidebar.left" : "sidebar.squares.left")
                }
                .help("Toggle Sidebar")
            }

            ToolbarItemGroup(placement: .primaryAction) {
                // Previous
                Button(action: { state.presetManager.previousPreset() }) {
                    Image(systemName: "backward.fill")
                }
                .help("Previous Preset")

                // Play/Pause auto-switch
                Button(action: { state.isPresetLocked.toggle() }) {
                    Image(systemName: state.isPresetLocked ? "lock.fill" : "play.fill")
                        .foregroundColor(state.isPresetLocked ? .orange : .primary)
                }
                .help(state.isPresetLocked ? "Unlock Preset" : "Lock Preset")

                // Next
                Button(action: { state.presetManager.nextPreset() }) {
                    Image(systemName: "forward.fill")
                }
                .help("Next Preset")

                // Random
                Button(action: { state.presetManager.randomPreset() }) {
                    Image(systemName: "shuffle")
                }
                .help("Random Preset")

                Divider()

                // Beat detection toggle
                Button(action: { state.beatDetectionEnabled.toggle() }) {
                    Image(systemName: "waveform.path.ecg")
                        .foregroundColor(state.beatDetectionEnabled ? .green : .secondary)
                }
                .help("Toggle Beat Detection")

                // Double preset
                Button(action: { state.createDoublePreset() }) {
                    Image(systemName: "rectangle.split.2x1")
                }
                .help("Create Double Preset (.milk2)")

                Divider()

                // Syphon
                Button(action: { state.syphonEnabled.toggle() }) {
                    HStack(spacing: 4) {
                        Circle()
                            .fill(state.syphonEnabled ? Color.green : Color.red)
                            .frame(width: 8, height: 8)
                        Text("Syphon")
                            .font(.caption)
                    }
                }
                .help(state.syphonEnabled ? "Syphon: Active" : "Syphon: Stopped")

                // Fullscreen
                Button(action: toggleFullscreen) {
                    Image(systemName: "arrow.up.left.and.arrow.down.right")
                }
                .help("Toggle Fullscreen")

                // Editor
                Button(action: { state.showPresetEditor = true }) {
                    Image(systemName: "chevron.left.forwardslash.chevron.right")
                }
                .help("Open Preset Editor")
            }
        }
        .sheet(isPresented: $state.showPresetEditor) {
            if let preset = state.editingPreset ?? state.presetManager.currentPreset {
                PresetEditorView(preset: preset)
                    .environmentObject(state)
                    .frame(minWidth: 900, minHeight: 650)
            }
        }
    }

    private func toggleFullscreen() {
        NSApp.mainWindow?.toggleFullScreen(nil)
    }
}

// MARK: - Visualizer (MTKView wrapper)

struct VisualizerView: NSViewRepresentable {
    @EnvironmentObject var state: AppState

    func makeNSView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = MTLCreateSystemDefaultDevice()
        view.colorPixelFormat = .bgra8Unorm
        view.preferredFramesPerSecond = 60
        view.isPaused = false
        view.enableSetNeedsDisplay = false
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

        if let device = view.device {
            let renderer = MilkDropRenderer(device: device)
            view.delegate = renderer
            context.coordinator.renderer = renderer
            Task { @MainActor in
                state.renderer = renderer
            }
        }
        return view
    }

    func updateNSView(_ view: MTKView, context: Context) {
        guard let renderer = context.coordinator.renderer else { return }
        renderer.updateAudio(state.audioEngine.audioData)
        // Only call loadPreset when the preset actually changes — it resets q-vars
        if let preset = state.presetManager.currentPreset,
           preset.id != context.coordinator.currentPresetID {
            context.coordinator.currentPresetID = preset.id
            renderer.loadPreset(preset)
        }
        renderer.setSyphonEnabled(state.syphonEnabled)
        renderer.fractalEnabled = state.fractalStreamEnabled
        renderer.fractalBlend   = Float(state.fractalBlend)
        renderer.transitionType = state.morphingTechnique
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    class Coordinator {
        var renderer: MilkDropRenderer?
        var currentPresetID: UUID?
    }
}

// MARK: - Sidebar

struct SidebarView: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        VStack(spacing: 0) {
            // Tab bar
            HStack(spacing: 0) {
                ForEach(SidebarTab.allCases, id: \.self) { tab in
                    Button(action: { state.selectedTab = tab }) {
                        VStack(spacing: 3) {
                            Image(systemName: tab.icon)
                                .font(.system(size: 14))
                            Text(tab.rawValue)
                                .font(.system(size: 9))
                        }
                        .foregroundColor(state.selectedTab == tab ? .white : .secondary)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                        .background(state.selectedTab == tab ? Color(hex: "1e1e1e") : Color.clear)
                    }
                    .buttonStyle(.plain)
                }
            }
            .background(Color(hex: "0a0a0a"))

            Divider().background(Color(hex: "2a2a2a"))

            // Tab content
            Group {
                switch state.selectedTab {
                case .presets:  PresetBrowserView()
                case .editor:   QuickEditorView()
                case .audio:    AudioSettingsView()
                case .morph:    MorphingView()
                case .syphon:   SyphonStatusView()
                case .settings: SettingsView()
                }
            }
        }
    }
}

// MARK: - Preset name banner

struct PresetNameBanner: View {
    let name: String
    @State private var opacity: Double = 1

    var body: some View {
        Text(name)
            .font(.system(size: 13, weight: .medium, design: .monospaced))
            .foregroundColor(.white)
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(Color.black.opacity(0.6))
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(Color.white.opacity(0.1), lineWidth: 0.5)
                    )
            )
            .opacity(opacity)
            .onAppear {
                withAnimation(.easeIn(duration: 0.3)) { opacity = 1 }
                DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                    withAnimation(.easeOut(duration: 1)) { opacity = 0 }
                }
            }
            .id(name)  // Recreate on name change
    }
}

// MARK: - Performance HUD

struct PerformanceHUD: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        VStack(alignment: .trailing, spacing: 3) {
            HUDRow(label: "FPS",  value: String(format: "%.0f", state.currentFPS))
            HUDRow(label: "GPU",  value: String(format: "%.0f%%", state.gpuLoad * 100))
            HUDRow(label: "BASS", value: String(format: "%.2f", state.beatStrength))
            HUDRow(label: "VOL",  value: String(format: "%.2f", state.audioLevel))
        }
        .padding(8)
        .background(Color.black.opacity(0.6))
        .cornerRadius(6)
    }
}

struct HUDRow: View {
    let label: String
    let value: String
    var body: some View {
        HStack(spacing: 6) {
            Text(label)
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundColor(.gray)
            Text(value)
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundColor(.green)
                .frame(minWidth: 40, alignment: .trailing)
        }
    }
}

// MARK: - Audio level bar

struct AudioLevelBar: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        HStack(spacing: 2) {
            // Spectrum bars (32 bands)
            ForEach(0..<32, id: \.self) { i in
                let height = min(CGFloat(state.audioLevel) * 60 + CGFloat(i % 3) * 3, 80)
                RoundedRectangle(cornerRadius: 2)
                    .fill(barColor(i: i, level: state.audioLevel))
                    .frame(width: 4, height: height)
                    .animation(.easeOut(duration: 0.06), value: state.audioLevel)
            }
        }
        .frame(maxWidth: .infinity)
        .frame(height: 80, alignment: .bottom)
        .padding(.bottom, 8)
        .opacity(0.7)
    }

    private func barColor(i: Int, level: Double) -> Color {
        let t = Double(i) / 32.0
        if t < 0.5 {
            return Color(red: 0.2 + t * 0.8, green: 0.9, blue: 0.3)
        } else {
            return Color(red: 1.0, green: 0.9 - (t - 0.5) * 1.4, blue: 0.1)
        }
    }
}

// MARK: - Color extension

extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: .alphanumerics.inverted)
        var value: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&value)
        let r = Double((value >> 16) & 0xFF) / 255
        let g = Double((value >> 8)  & 0xFF) / 255
        let b = Double(value & 0xFF) / 255
        self.init(red: r, green: g, blue: b)
    }
}
