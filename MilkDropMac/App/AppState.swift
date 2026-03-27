// AppState.swift — Central app state shared across all views

import SwiftUI
import Combine
import AppKit

@MainActor
class AppState: ObservableObject {
    // MARK: - Core systems
    var presetManager   = PresetManager()
    let audioEngine     = AudioEngine()
    let beatDetector    = BeatDetector()
    var renderer: MilkDropRenderer?

    // MARK: - Syphon
    @Published var syphonEnabled: Bool = true {
        didSet { updateSyphon() }
    }

    // MARK: - Visualizer state
    @Published var isFullscreen: Bool = false
    @Published var isPresetLocked: Bool = false
    @Published var beatDetectionEnabled: Bool = true
    @Published var autoSwitchInterval: Double = 15.0  // seconds
    @Published var transitionDuration: Double = 2.5   // seconds
    @Published var hardcutEnabled: Bool = true
    @Published var hardcutSensitivity: Double = 0.5

    // MARK: - MilkDrop 3 double-preset
    @Published var doublePresetActive: Bool = false
    @Published var doublePresetBlend: Double = 0.5

    // MARK: - Morphing & fractal stream
    @Published var morphingTechnique: Int32 = 6    // Default: bezier warp
    @Published var fractalStreamEnabled: Bool = false
    @Published var fractalBlend: Double = 0.4

    // MARK: - Visual settings
    @Published var renderResolution: RenderResolution = .native
    @Published var fps: Int = 60
    @Published var showFPSCounter: Bool = false
    @Published var showPresetName: Bool = true
    @Published var presetNameFadeTime: Double = 3.0
    @Published var brightness: Double = 1.0
    @Published var gamma: Double = 1.0

    // MARK: - Live param overrides (QuickEditor → renderer, applied after equations)
    @Published var liveZoom:  Float? = nil
    @Published var liveWarp:  Float? = nil
    @Published var liveDecay: Float? = nil
    @Published var liveGamma: Float? = nil

    func clearLiveOverrides() {
        liveZoom = nil; liveWarp = nil; liveDecay = nil; liveGamma = nil
    }

    // MARK: - Audio settings
    @Published var audioSource: AudioSource = .systemDefault
    @Published var audioSensitivity: Double = 1.0
    @Published var bassBoost: Double = 1.0
    @Published var midBoost: Double = 1.0
    @Published var trebleBoost: Double = 1.0

    // MARK: - UI state
    @Published var showPresetEditor: Bool = false
    @Published var showPresetBrowser: Bool = true
    @Published var sidebarWidth: Double = 300
    @Published var selectedTab: SidebarTab = .presets
    @Published var editingPreset: MilkDropPreset?
    @Published var searchText: String = ""

    // MARK: - Performance metrics
    @Published var currentFPS: Double = 0
    @Published var gpuLoad: Double = 0
    @Published var audioLevel: Double = 0
    @Published var beatStrength: Double = 0

    // Processed audio with sensitivity + band boosts applied — passed to renderer
    @Published var processedAudio: AudioData = .silence

    private var cancellables = Set<AnyCancellable>()
    private var autoSwitchTimer: AnyCancellable?

    init() {
        setupBindings()
        setupAutoSwitch()
        audioEngine.start()
    }

    private func setupBindings() {
        // Forward PresetManager changes to AppState so views re-render
        presetManager.objectWillChange
            .sink { [weak self] _ in self?.objectWillChange.send() }
            .store(in: &cancellables)

        // Connect beat detector to preset switching
        beatDetector.$hardcutDetected
            .filter { $0 && self.beatDetectionEnabled && !self.isPresetLocked }
            .sink { [weak self] _ in
                self?.presetManager.nextPreset(transition: .hardcut)
            }
            .store(in: &cancellables)

        // Apply sensitivity + band boosts, then forward to beat detector and renderer
        audioEngine.$audioData
            .sink { [weak self] data in
                guard let self else { return }
                let sens = Float(self.audioSensitivity)
                let bB   = Float(self.bassBoost)
                let mB   = Float(self.midBoost)
                let tB   = Float(self.trebleBoost)
                var d = data
                d.bass     = min(d.bass   * sens * bB, 2.0)
                d.mid      = min(d.mid    * sens * mB, 2.0)
                d.treble   = min(d.treble * sens * tB, 2.0)
                d.rms      = min(d.rms    * sens,      2.0)
                d.bassLevel = min(d.bassLevel * sens * bB, 1.0)
                d.bassAttn  = min(d.bassAttn  * sens * bB, 1.0)
                d.spectrum  = d.spectrum.map { min($0 * sens, 2.0) }
                self.processedAudio = d
                self.beatDetector.process(d)
                self.audioLevel   = Double(d.rms)
                self.beatStrength = Double(d.bassLevel)
            }
            .store(in: &cancellables)

        // Restart audio engine when source changes
        $audioSource
            .dropFirst()
            .sink { [weak self] source in
                guard let self else { return }
                self.audioEngine.stop()
                self.audioEngine.start(source: source)
            }
            .store(in: &cancellables)

        // Update auto-switch interval
        $autoSwitchInterval
            .sink { [weak self] _ in self?.setupAutoSwitch() }
            .store(in: &cancellables)

        $beatDetectionEnabled
            .sink { [weak self] _ in self?.setupAutoSwitch() }
            .store(in: &cancellables)
    }

    private func setupAutoSwitch() {
        autoSwitchTimer?.cancel()
        guard !beatDetectionEnabled else { return } // Beat mode handles its own switching
        autoSwitchTimer = Timer.publish(every: autoSwitchInterval, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                guard let self, !self.isPresetLocked else { return }
                self.presetManager.nextPreset(transition: .smooth)
            }
    }

    private func updateSyphon() {
        renderer?.setSyphonEnabled(syphonEnabled)
    }

    // MARK: - Actions

    func createDoublePreset() {
        guard let current = presetManager.currentPreset else { return }
        let second = presetManager.randomPresetObject()
        let double = MilkDropPreset.makeDoublePreset(a: current, b: second)
        editingPreset = double
        showPresetEditor = true
    }

    func importPresets() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowedContentTypes = [.milk, .milk2]
        panel.begin { [weak self] response in
            guard response == .OK else { return }
            Task { @MainActor in
                self?.presetManager.importPresets(from: panel.urls)
            }
        }
    }

    func savePresetAs() {
        guard let preset = presetManager.currentPreset else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = preset.name
        panel.allowedContentTypes = [.milk]
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            try? preset.data.write(to: url, atomically: true, encoding: .utf8)
        }
    }

    func openPresetFolder() {
        NSWorkspace.shared.open(PresetManager.presetsDirectory)
    }
}

// MARK: - Supporting types

enum MorphingTechnique: Int32, CaseIterable, Identifiable {
    case zoom            = 0
    case sideWipe        = 1
    case plasma          = 2
    case circle          = 3
    case checkerboard    = 4
    case stars           = 5
    case bezierWarp      = 6
    case meshMorph       = 7
    case fractalDissolve = 8
    case fractalStream   = 9

    var id: Int32 { rawValue }

    var name: String {
        switch self {
        case .zoom:            return "Zoom"
        case .sideWipe:        return "Side Wipe"
        case .plasma:          return "Plasma"
        case .circle:          return "Circle"
        case .checkerboard:    return "Checkerboard"
        case .stars:           return "Stars"
        case .bezierWarp:      return "Bezier Warp"
        case .meshMorph:       return "Mesh Morph"
        case .fractalDissolve: return "Fractal Dissolve"
        case .fractalStream:   return "Fractal Stream"
        }
    }
}

enum RenderResolution: String, CaseIterable {
    case quarter = "Quarter"
    case half    = "Half"
    case native  = "Native"
    case double  = "2x (SSAA)"

    var scale: Double {
        switch self {
        case .quarter: return 0.25
        case .half:    return 0.5
        case .native:  return 1.0
        case .double:  return 2.0
        }
    }
}

enum AudioSource: String, CaseIterable, Identifiable {
    case systemDefault  = "System Default"
    case microphone     = "Microphone"
    case lineIn         = "Line In"
    case loopback       = "System Audio (Loopback)"

    var id: String { rawValue }
}

enum SidebarTab: String, CaseIterable {
    case presets  = "Presets"
    case editor   = "Editor"
    case audio    = "Audio"
    case morph    = "Morph"
    case syphon   = "Syphon"
    case settings = "Settings"

    var icon: String {
        switch self {
        case .presets:  return "square.grid.2x2"
        case .editor:   return "chevron.left.forwardslash.chevron.right"
        case .audio:    return "waveform"
        case .morph:    return "sparkles"
        case .syphon:   return "arrow.triangle.branch"
        case .settings: return "gearshape"
        }
    }
}
