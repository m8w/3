// PresetManager.swift — Load, organize, navigate and manage MilkDrop presets

import Foundation
import Combine
import AppKit

@MainActor
class PresetManager: ObservableObject {
    @Published var presets:         [MilkDropPreset] = []
    @Published var currentPreset:   MilkDropPreset?
    @Published var currentIndex:    Int = 0
    @Published var history:         [MilkDropPreset] = []
    @Published var favorites:       Set<UUID> = []
    @Published var isLoading:       Bool = false
    @Published var loadProgress:    Double = 0

    // Filters / search
    @Published var searchText:      String = ""
    @Published var filterFavorites: Bool = false
    @Published var filterRating:    Int = 0
    @Published var sortOrder:       SortOrder = .name

    // Transition state
    private(set) var pendingTransition: TransitionType = .smooth
    @Published var isTransitioning: Bool = false

    // MARK: - Preset directories

    static let presetsDirectory: URL = {
        let app = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let dir = app.appendingPathComponent("MilkDropMac/Presets", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()

    static let bundledPresetsDirectory: URL? = Bundle.main.resourceURL?.appendingPathComponent("Presets")

    // MARK: - Init

    init() {
        Task { await loadAllPresets() }
    }

    // MARK: - Loading

    func loadAllPresets() async {
        isLoading = true
        loadProgress = 0

        var loaded: [MilkDropPreset] = []

        // 1. Built-in presets (embedded as Swift constants — always available regardless
        //    of whether the Xcode bundle resource phase includes the Presets folder)
        loaded.append(contentsOf: BuiltInPresets.all)
        loadProgress = 0.2

        // 2. Bundled presets (folder reference in Xcode bundle, if present)
        if let bundled = PresetManager.bundledPresetsDirectory {
            let bundledPresets = await loadPresets(from: bundled)
            // Skip any that duplicate a built-in by name
            let builtInNames = Set(BuiltInPresets.all.map { $0.name })
            loaded.append(contentsOf: bundledPresets.filter { !builtInNames.contains($0.name) })
            loadProgress = 0.5
        }

        // 3. User presets directory
        let userPresets = await loadPresets(from: PresetManager.presetsDirectory)
        loaded.append(contentsOf: userPresets)

        // 3. Load favorites from UserDefaults
        if let favoriteIDs = UserDefaults.standard.array(forKey: "FavoritePresets") as? [String] {
            favorites = Set(favoriteIDs.compactMap { UUID(uuidString: $0) })
        }

        // Apply favorites
        for i in loaded.indices {
            loaded[i].isFavorite = favorites.contains(loaded[i].id)
        }

        presets = loaded.sorted(by: sortComparator)
        loadProgress = 1.0
        isLoading = false

        // Select a random starting preset (go through setPreset so data is loaded from disk)
        if !presets.isEmpty {
            currentIndex = Int.random(in: 0..<presets.count)
            setPreset(presets[currentIndex], transition: .instant)
        }
    }

    private func loadPresets(from directory: URL) async -> [MilkDropPreset] {
        await Task.detached(priority: .userInitiated) {
            let fm = FileManager.default
            guard let enumerator = fm.enumerator(
                at: directory,
                includingPropertiesForKeys: [.nameKey, .contentModificationDateKey],
                options: [.skipsHiddenFiles]
            ) else { return [] }

            var result: [MilkDropPreset] = []
            for case let url as URL in enumerator.allObjects {
                guard ["milk", "milk2"].contains(url.pathExtension.lowercased()) else { continue }
                let name = url.deletingPathExtension().lastPathComponent
                // data is intentionally empty — loaded lazily when preset is selected
                var preset = MilkDropPreset(name: name, url: url, data: "")
                preset.isDoublePreset = url.pathExtension.lowercased() == "milk2"
                preset.dateModified = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .now
                result.append(preset)
            }
            return result
        }.value
    }

    // MARK: - Navigation

    func nextPreset(transition: TransitionType = .smooth) {
        guard !presets.isEmpty else { return }
        pendingTransition = transition
        let filtered = filteredPresets
        guard !filtered.isEmpty else { return }
        currentIndex = (currentIndex + 1) % filtered.count
        setPreset(filtered[currentIndex], transition: transition)
    }

    func previousPreset(transition: TransitionType = .smooth) {
        guard !presets.isEmpty else { return }
        let filtered = filteredPresets
        guard !filtered.isEmpty else { return }
        currentIndex = (currentIndex - 1 + filtered.count) % filtered.count
        setPreset(filtered[currentIndex], transition: transition)
    }

    func randomPreset(transition: TransitionType = .smooth) {
        let filtered = filteredPresets
        guard !filtered.isEmpty else { return }
        currentIndex = Int.random(in: 0..<filtered.count)
        setPreset(filtered[currentIndex], transition: transition)
    }

    func randomPresetObject() -> MilkDropPreset {
        filteredPresets.randomElement() ?? presets[0]
    }

    func select(_ preset: MilkDropPreset, transition: TransitionType = .smooth) {
        if let idx = filteredPresets.firstIndex(where: { $0.id == preset.id }) {
            currentIndex = idx
        }
        setPreset(preset, transition: transition)
    }

    private func setPreset(_ preset: MilkDropPreset, transition: TransitionType) {
        if let current = currentPreset {
            history.append(current)
            if history.count > 50 { history.removeFirst() }
        }
        pendingTransition = transition
        // Lazy-load preset data from disk if not yet read
        var loaded = preset
        if loaded.data.isEmpty, let url = loaded.url {
            // Try UTF-8 first; fall back to Windows-1252 / Latin-1 for classic
            // MilkDrop preset packs created on Windows (very common in the wild).
            loaded.data = (try? String(contentsOf: url, encoding: .utf8))
                ?? (try? String(contentsOf: url, encoding: .windowsCP1252))
                ?? (try? String(contentsOf: url, encoding: .isoLatin1))
                ?? ""
        }
        currentPreset = loaded
    }

    // MARK: - Filtered / sorted presets

    var filteredPresets: [MilkDropPreset] {
        var result = presets
        if filterFavorites { result = result.filter { $0.isFavorite } }
        if filterRating > 0 { result = result.filter { $0.rating >= filterRating } }
        if !searchText.isEmpty {
            let q = searchText.lowercased()
            result = result.filter {
                $0.name.lowercased().contains(q) ||
                $0.author.lowercased().contains(q) ||
                $0.tags.contains(where: { $0.lowercased().contains(q) })
            }
        }
        return result.sorted(by: sortComparator)
    }

    private var sortComparator: (MilkDropPreset, MilkDropPreset) -> Bool {
        switch sortOrder {
        case .name:         return { $0.name < $1.name }
        case .dateModified: return { $0.dateModified > $1.dateModified }
        case .rating:       return { $0.rating > $1.rating }
        case .author:       return { $0.author < $1.author }
        case .random:       return { _, _ in Bool.random() }
        }
    }

    // MARK: - Favorites

    func toggleFavorite(_ preset: MilkDropPreset) {
        if favorites.contains(preset.id) {
            favorites.remove(preset.id)
        } else {
            favorites.insert(preset.id)
        }
        saveFavorites()
        // Update presets array
        if let i = presets.firstIndex(where: { $0.id == preset.id }) {
            presets[i].isFavorite = favorites.contains(preset.id)
        }
    }

    private func saveFavorites() {
        UserDefaults.standard.set(favorites.map { $0.uuidString }, forKey: "FavoritePresets")
    }

    // MARK: - Import / Save

    func importPresets(from urls: [URL]) {
        var newPresets: [MilkDropPreset] = []
        for url in urls {
            if url.hasDirectoryPath {
                // Recursively import directory
                let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: nil)
                while let fileURL = enumerator?.nextObject() as? URL {
                    if let preset = loadSinglePreset(from: fileURL) {
                        newPresets.append(preset)
                    }
                }
            } else if let preset = loadSinglePreset(from: url) {
                newPresets.append(preset)
            }
        }

        // Copy to user presets directory and add to list
        for preset in newPresets {
            guard let src = preset.url else { continue }
            let dest = PresetManager.presetsDirectory.appendingPathComponent(src.lastPathComponent)
            try? FileManager.default.copyItem(at: src, to: dest)
            presets.append(preset)
        }
        presets = presets.sorted(by: sortComparator)
    }

    func savePreset(_ preset: MilkDropPreset) throws {
        let filename = preset.name.replacingOccurrences(of: "/", with: "-") + ".\(preset.fileExtension)"
        let url = PresetManager.presetsDirectory.appendingPathComponent(filename)
        try preset.data.write(to: url, atomically: true, encoding: .utf8)

        // Update or insert in list
        if let idx = presets.firstIndex(where: { $0.id == preset.id }) {
            presets[idx] = preset.with(\.url, value: url)
        } else {
            presets.append(preset.with(\.url, value: url))
        }
    }

    private func loadSinglePreset(from url: URL) -> MilkDropPreset? {
        guard ["milk", "milk2"].contains(url.pathExtension.lowercased()),
              let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        let name = url.deletingPathExtension().lastPathComponent
        var preset = MilkDropPreset(name: name, url: url, data: text)
        preset.isDoublePreset = url.pathExtension.lowercased() == "milk2"
        return preset
    }

    // MARK: - Types

    enum SortOrder: String, CaseIterable {
        case name, dateModified, rating, author, random
    }
}

enum TransitionType {
    case smooth
    case hardcut
    case instant
}

// MARK: - Built-in presets (embedded as Swift constants)
// These are always available regardless of Xcode bundle resource setup.

enum BuiltInPresets {
    static let all: [MilkDropPreset] = [
        preset("Psychedelic - Rainbow Vortex",    rainbowVortex),
        preset("Psychedelic - Acid Plasma",       acidPlasma),
        preset("Psychedelic - Electric Spiral",   electricSpiral),
        preset("Psychedelic - Deep Space",        deepSpace),
        preset("Psychedelic - Molten",            molten),
        preset("Classic - Hyperspace",            classicHyperspace),
        preset("Classic - Beat Reactor",          beatReactor),
    ]

    private static func preset(_ name: String, _ data: String) -> MilkDropPreset {
        var p = MilkDropPreset(name: name, data: data)
        p.parseParameters()
        return p
    }

    // MARK: - Preset data

    private static let rainbowVortex = """
[preset00]
fRating=5
fGammaAdj=2.8
fDecay=0.97
fZoom=1.06
fRot=0.0
fWarpScale=3.5
fWarpAnimSpeed=1.8
fXCentre=0.5
fYCentre=0.5
szx=1.0
szy=1.0
per_frame_1=zoom = 1.06 + 0.04*bass;
per_frame_2=rot = rot + 0.004 + 0.003*sin(time*0.09);
per_frame_3=warp = 3 + bass*3;
per_frame_4=cx = 0.5 + 0.06*sin(time*0.19);
per_frame_5=cy = 0.5 + 0.06*cos(time*0.13);
per_frame_6=decay = 0.97;
per_frame_7=r = 0.5 + 0.5*sin(time*0.29);
per_frame_8=g = 0.5 + 0.5*sin(time*0.29 + 2.094);
per_frame_9=b = 0.5 + 0.5*sin(time*0.29 + 4.189);
per_frame_10=a = 0.18 + 0.08*bass;
wave_0_enabled=1
wave_0_samples=220
wave_0_scaling=1.0
wave_0_smoothing=0.2
wave_0_r=1.0
wave_0_g=0.0
wave_0_b=1.0
wave_0_a=1.0
wave_0_usedots=0
wave_0_drawthick=1
wave_0_additive=1
wave_0_per_point_1=x = cos(i*6.28318 + time*0.4)*(0.32 + bass*0.12) + 0.5;
wave_0_per_point_2=y = sin(i*6.28318 + time*0.4)*(0.32 + bass*0.12) + 0.5;
wave_0_per_point_3=r = 0.5 + 0.5*sin(i*18.84956 + time*2.1);
wave_0_per_point_4=g = 0.5 + 0.5*sin(i*18.84956 + time*2.1 + 2.094);
wave_0_per_point_5=b = 0.5 + 0.5*sin(i*18.84956 + time*2.1 + 4.189);
wave_0_per_point_6=a = 0.9 + 0.1*bass;
wave_1_enabled=1
wave_1_samples=220
wave_1_scaling=1.0
wave_1_smoothing=0.2
wave_1_r=0.0
wave_1_g=1.0
wave_1_b=0.5
wave_1_a=1.0
wave_1_usedots=0
wave_1_drawthick=1
wave_1_additive=1
wave_1_per_point_1=x = cos(i*12.56637 - time*0.3)*(0.18 + treble*0.08) + 0.5;
wave_1_per_point_2=y = sin(i*12.56637 - time*0.3)*(0.18 + treble*0.08) + 0.5;
wave_1_per_point_3=r = 0.5 + 0.5*sin(i*12.56637 + time*1.7 + 4.189);
wave_1_per_point_4=g = 0.5 + 0.5*sin(i*12.56637 + time*1.7);
wave_1_per_point_5=b = 0.5 + 0.5*sin(i*12.56637 + time*1.7 + 2.094);
wave_1_per_point_6=a = 0.8 + 0.2*treble;
wave_2_enabled=1
wave_2_samples=180
wave_2_scaling=1.0
wave_2_smoothing=0.3
wave_2_r=1.0
wave_2_g=1.0
wave_2_b=0.0
wave_2_a=0.9
wave_2_usedots=0
wave_2_drawthick=1
wave_2_additive=1
wave_2_per_point_1=x = 0.5 + (0.45 + sample*0.2)*cos(i*6.28318*2 + time*0.6);
wave_2_per_point_2=y = 0.5 + (0.45 + sample*0.2)*sin(i*6.28318*2 + time*0.6);
wave_2_per_point_3=r = 0.5 + 0.5*sin(i*6.28318*5 - time*2.5);
wave_2_per_point_4=g = 0.5 + 0.5*sin(i*6.28318*5 - time*2.5 + 2.094);
wave_2_per_point_5=b = 0.5 + 0.5*sin(i*6.28318*5 - time*2.5 + 4.189);
wave_2_per_point_6=a = 0.7 + 0.3*mid;
shape_0_enabled=1
shape_0_sides=6
shape_0_additive=1
shape_0_x=0.5
shape_0_y=0.5
shape_0_radius=0.08
shape_0_r=1.0
shape_0_g=0.5
shape_0_b=0.0
shape_0_a=0.7
shape_0_r2=0.0
shape_0_g2=0.5
shape_0_b2=1.0
shape_0_a2=0.0
shape_0_per_frame_1=radius = 0.04 + 0.07*bass;
shape_0_per_frame_2=ang = ang + 0.03;
shape_0_per_frame_3=r = 0.5 + 0.5*sin(time*1.1);
shape_0_per_frame_4=g = 0.5 + 0.5*sin(time*1.1 + 2.094);
shape_0_per_frame_5=b = 0.5 + 0.5*sin(time*1.1 + 4.189);
"""

    private static let acidPlasma = """
[preset00]
fRating=5
fGammaAdj=2.2
fDecay=0.96
fZoom=1.0
fRot=0.0
fWarpScale=2.0
fWarpAnimSpeed=1.0
fXCentre=0.5
fYCentre=0.5
szx=1.0
szy=1.0
per_frame_1=decay = 0.96 + 0.03*bass;
per_frame_2=warp = 1.5 + bass*2;
per_frame_3=zoom = 1.0 + 0.015*bass;
per_frame_4=rot = rot + 0.001*sin(time*0.07);
per_frame_5=r = 0.5 + 0.5*sin(time*0.23);
per_frame_6=g = 0.5 + 0.5*sin(time*0.23 + 2.094);
per_frame_7=b = 0.5 + 0.5*sin(time*0.23 + 4.189);
per_frame_8=a = 0.12 + 0.06*bass;
per_pixel_1=x = x + 0.016*sin(y*9.1 + time*1.3);
per_pixel_2=y = y + 0.016*cos(x*7.7 + time*1.1);
per_pixel_3=x = x + 0.008*sin(y*3.3 + time*0.7 + 1.57);
per_pixel_4=y = y + 0.008*cos(x*4.1 + time*0.9 + 0.79);
wave_0_enabled=1
wave_0_samples=256
wave_0_scaling=1.2
wave_0_smoothing=0.4
wave_0_r=0.0
wave_0_g=1.0
wave_0_b=1.0
wave_0_a=1.0
wave_0_usedots=0
wave_0_drawthick=1
wave_0_additive=1
wave_0_per_point_1=x = 0.5 + 0.4*sin(i*3.14159 + time*0.8)*cos(i*6.28318*0.5);
wave_0_per_point_2=y = 0.5 + sample*0.3 + 0.35*cos(i*3.14159 + time*0.8)*sin(i*6.28318*0.5);
wave_0_per_point_3=r = 0.5 + 0.5*sin(i*6.28318*4 + time*1.9);
wave_0_per_point_4=g = 0.5 + 0.5*sin(i*6.28318*4 + time*1.9 + 2.094);
wave_0_per_point_5=b = 0.5 + 0.5*sin(i*6.28318*4 + time*1.9 + 4.189);
wave_0_per_point_6=a = 0.85 + 0.15*mid;
wave_1_enabled=1
wave_1_samples=200
wave_1_scaling=1.0
wave_1_smoothing=0.3
wave_1_r=1.0
wave_1_g=0.3
wave_1_b=0.8
wave_1_a=0.9
wave_1_usedots=0
wave_1_drawthick=1
wave_1_additive=1
wave_1_per_point_1=x = 0.5 + i - 0.5;
wave_1_per_point_2=y = 0.5 + sample*0.35 + 0.08*sin(i*6.28318*7 + time*2.3);
wave_1_per_point_3=r = 0.5 + 0.5*sin(i*6.28318*6 + time*1.4 + 1.047);
wave_1_per_point_4=g = 0.5 + 0.5*sin(i*6.28318*6 + time*1.4 + 3.141);
wave_1_per_point_5=b = 0.5 + 0.5*sin(i*6.28318*6 + time*1.4 + 5.236);
wave_1_per_point_6=a = 0.8 + 0.2*bass;
"""

    private static let electricSpiral = """
[preset00]
fRating=5
fGammaAdj=3.0
fDecay=0.96
fZoom=1.08
fRot=0.015
fWarpScale=4.0
fWarpAnimSpeed=2.0
fXCentre=0.5
fYCentre=0.5
szx=1.0
szy=1.0
per_frame_1=zoom = 1.08 + 0.06*bass;
per_frame_2=rot = rot + 0.015 + 0.01*bass;
per_frame_3=warp = 4 + treble*3;
per_frame_4=cx = 0.5 + 0.04*sin(time*0.23 + bass);
per_frame_5=cy = 0.5 + 0.04*cos(time*0.17 + treble);
per_frame_6=decay = 0.96;
per_frame_7=r = 0.5 + 0.5*sin(time*0.41);
per_frame_8=g = 0.5 + 0.5*sin(time*0.41 + 2.094);
per_frame_9=b = 0.5 + 0.5*sin(time*0.41 + 4.189);
per_frame_10=a = 0.22 + 0.1*bass;
per_pixel_1=x = x + 0.01*sin(y*12 + time*1.7);
per_pixel_2=y = y + 0.01*cos(x*12 + time*1.3);
wave_0_enabled=1
wave_0_samples=300
wave_0_scaling=1.0
wave_0_smoothing=0.1
wave_0_r=1.0
wave_0_g=0.2
wave_0_b=0.6
wave_0_a=1.0
wave_0_usedots=0
wave_0_drawthick=1
wave_0_additive=1
wave_0_per_point_1=t = i*6.28318*3 + time*0.7;
wave_0_per_point_2=r2 = 0.05 + i*0.42;
wave_0_per_point_3=x = 0.5 + r2*cos(t);
wave_0_per_point_4=y = 0.5 + r2*sin(t)*(1 + sample*0.4);
wave_0_per_point_5=r = 0.5 + 0.5*sin(i*6.28318*8 + time*2.5);
wave_0_per_point_6=g = 0.5 + 0.5*sin(i*6.28318*8 + time*2.5 + 2.094);
wave_0_per_point_7=b = 0.5 + 0.5*sin(i*6.28318*8 + time*2.5 + 4.189);
wave_0_per_point_8=a = 0.9 + 0.1*bass;
wave_1_enabled=1
wave_1_samples=300
wave_1_scaling=1.0
wave_1_smoothing=0.1
wave_1_r=0.2
wave_1_g=0.8
wave_1_b=1.0
wave_1_a=1.0
wave_1_usedots=0
wave_1_drawthick=1
wave_1_additive=1
wave_1_per_point_1=t = i*6.28318*3 - time*0.5;
wave_1_per_point_2=r2 = 0.04 + i*0.38;
wave_1_per_point_3=x = 0.5 + r2*cos(t + 3.14159);
wave_1_per_point_4=y = 0.5 + r2*sin(t + 3.14159)*(1 + sample*0.3);
wave_1_per_point_5=r = 0.5 + 0.5*sin(i*6.28318*6 - time*2.1 + 4.189);
wave_1_per_point_6=g = 0.5 + 0.5*sin(i*6.28318*6 - time*2.1);
wave_1_per_point_7=b = 0.5 + 0.5*sin(i*6.28318*6 - time*2.1 + 2.094);
wave_1_per_point_8=a = 0.85 + 0.15*treble;
shape_0_enabled=1
shape_0_sides=5
shape_0_additive=1
shape_0_x=0.5
shape_0_y=0.5
shape_0_radius=0.12
shape_0_r=1.0
shape_0_g=0.8
shape_0_b=0.0
shape_0_a=0.5
shape_0_r2=1.0
shape_0_g2=0.0
shape_0_b2=0.8
shape_0_a2=0.0
shape_0_per_frame_1=radius = 0.05 + 0.1*bass;
shape_0_per_frame_2=ang = ang + 0.04;
shape_0_per_frame_3=r = 0.5 + 0.5*sin(time*0.8);
shape_0_per_frame_4=g = 0.5 + 0.5*sin(time*0.8 + 2.094);
shape_0_per_frame_5=b = 0.5 + 0.5*sin(time*0.8 + 4.189);
shape_0_per_frame_6=a = 0.3 + 0.4*bass;
"""

    private static let deepSpace = """
[preset00]
fRating=5
fGammaAdj=2.5
fDecay=0.98
fZoom=1.12
fRot=0.008
fWarpScale=1.5
fWarpAnimSpeed=0.8
fXCentre=0.5
fYCentre=0.5
szx=1.0
szy=1.0
per_frame_1=zoom = 1.12 + 0.08*bass;
per_frame_2=rot = rot + 0.008 + 0.005*sin(time*0.11);
per_frame_3=warp = 1.5 + mid*1.5;
per_frame_4=cx = 0.5 + 0.08*sin(time*0.13);
per_frame_5=cy = 0.5 + 0.08*cos(time*0.17);
per_frame_6=decay = 0.98 - 0.015*treble;
per_frame_7=r = 0.4 + 0.4*sin(time*0.19);
per_frame_8=g = 0.4 + 0.4*sin(time*0.19 + 2.094);
per_frame_9=b = 0.4 + 0.4*sin(time*0.19 + 4.189);
per_frame_10=a = 0.1 + 0.05*bass;
wave_0_enabled=1
wave_0_samples=256
wave_0_scaling=1.0
wave_0_smoothing=0.5
wave_0_r=1.0
wave_0_g=0.5
wave_0_b=0.0
wave_0_a=1.0
wave_0_usedots=0
wave_0_drawthick=1
wave_0_additive=1
wave_0_per_point_1=ang2 = i*6.28318 + time*0.35;
wave_0_per_point_2=r2 = 0.38 + 0.12*sin(i*6.28318*2 + time*0.9) + sample*0.1;
wave_0_per_point_3=x = 0.5 + r2*cos(ang2);
wave_0_per_point_4=y = 0.5 + r2*sin(ang2);
wave_0_per_point_5=r = 0.5 + 0.5*sin(i*6.28318*5 + time*2.3);
wave_0_per_point_6=g = 0.5 + 0.5*sin(i*6.28318*5 + time*2.3 + 2.094);
wave_0_per_point_7=b = 0.5 + 0.5*sin(i*6.28318*5 + time*2.3 + 4.189);
wave_0_per_point_8=a = 0.85;
wave_1_enabled=1
wave_1_samples=256
wave_1_smoothing=0.5
wave_1_r=0.0
wave_1_g=0.8
wave_1_b=1.0
wave_1_a=1.0
wave_1_usedots=0
wave_1_drawthick=1
wave_1_additive=1
wave_1_per_point_1=ang2 = i*6.28318*2 - time*0.27;
wave_1_per_point_2=r2 = 0.22 + 0.07*cos(i*6.28318*3 + time*1.1) + sample*0.08;
wave_1_per_point_3=x = 0.5 + r2*cos(ang2);
wave_1_per_point_4=y = 0.5 + r2*sin(ang2);
wave_1_per_point_5=r = 0.5 + 0.5*sin(i*6.28318*7 - time*1.8 + 4.189);
wave_1_per_point_6=g = 0.5 + 0.5*sin(i*6.28318*7 - time*1.8);
wave_1_per_point_7=b = 0.5 + 0.5*sin(i*6.28318*7 - time*1.8 + 2.094);
wave_1_per_point_8=a = 0.8;
"""

    private static let molten = """
[preset00]
fRating=5
fGammaAdj=2.0
fDecay=0.95
fZoom=1.0
fRot=0.0
fWarpScale=5.0
fWarpAnimSpeed=3.0
fXCentre=0.5
fYCentre=0.5
szx=1.0
szy=1.0
per_frame_1=zoom = 1.0 + 0.025*bass;
per_frame_2=warp = 5 + bass*4;
per_frame_3=rot = rot + 0.002*sin(time*0.07 + bass);
per_frame_4=cx = 0.5 + 0.1*sin(time*0.21);
per_frame_5=cy = 0.5 + 0.1*cos(time*0.17);
per_frame_6=decay = 0.95 + 0.04*bass;
per_frame_7=r = 0.6 + 0.4*sin(time*0.37);
per_frame_8=g = 0.6 + 0.4*sin(time*0.37 + 2.094);
per_frame_9=b = 0.6 + 0.4*sin(time*0.37 + 4.189);
per_frame_10=a = 0.25 + 0.15*bass;
per_pixel_1=x = x + 0.022*sin(y*6.3 + time*1.9 + x*3.1);
per_pixel_2=y = y + 0.022*cos(x*6.3 + time*1.5 + y*2.7);
per_pixel_3=x = x + 0.012*sin(y*11.1 + time*0.8);
per_pixel_4=y = y + 0.012*cos(x*9.7 + time*1.1);
wave_0_enabled=1
wave_0_samples=220
wave_0_scaling=1.3
wave_0_smoothing=0.25
wave_0_r=1.0
wave_0_g=0.4
wave_0_b=0.0
wave_0_a=1.0
wave_0_usedots=0
wave_0_drawthick=1
wave_0_additive=1
wave_0_per_point_1=x = 0.5 + i - 0.5;
wave_0_per_point_2=y = 0.5 + sample*0.4 + 0.07*sin(i*6.28318*9 + time*3.1);
wave_0_per_point_3=r = 0.7 + 0.3*sin(i*6.28318*3 + time*2.7);
wave_0_per_point_4=g = 0.3 + 0.3*sin(i*6.28318*3 + time*2.7 + 2.094);
wave_0_per_point_5=b = 0.0 + 0.5*sin(i*6.28318*3 + time*2.7 + 4.189);
wave_0_per_point_6=a = 0.9 + 0.1*bass;
wave_1_enabled=1
wave_1_samples=220
wave_1_scaling=1.2
wave_1_smoothing=0.25
wave_1_r=0.8
wave_1_g=0.0
wave_1_b=1.0
wave_1_a=1.0
wave_1_usedots=0
wave_1_drawthick=1
wave_1_additive=1
wave_1_per_point_1=x = 0.5 + i - 0.5;
wave_1_per_point_2=y = 0.5 + sample*0.35 + 0.09*sin(i*6.28318*7 - time*2.3);
wave_1_per_point_3=r = 0.4 + 0.3*sin(i*6.28318*4 - time*1.9 + 4.189);
wave_1_per_point_4=g = 0.1 + 0.5*sin(i*6.28318*4 - time*1.9);
wave_1_per_point_5=b = 0.6 + 0.4*sin(i*6.28318*4 - time*1.9 + 2.094);
wave_1_per_point_6=a = 0.85 + 0.15*mid;
wave_2_enabled=1
wave_2_samples=180
wave_2_scaling=1.0
wave_2_smoothing=0.4
wave_2_r=0.0
wave_2_g=1.0
wave_2_b=0.3
wave_2_a=0.9
wave_2_usedots=0
wave_2_drawthick=1
wave_2_additive=1
wave_2_per_point_1=x = 0.5 + cos(i*3.14159 + time*0.9)*(0.43 + sample*0.15);
wave_2_per_point_2=y = 0.5 + sin(i*3.14159 + time*0.9)*(0.43 + sample*0.15);
wave_2_per_point_3=r = 0.0 + 0.6*sin(i*6.28318*6 + time*2.1);
wave_2_per_point_4=g = 0.7 + 0.3*sin(i*6.28318*6 + time*2.1 + 2.094);
wave_2_per_point_5=b = 0.2 + 0.5*sin(i*6.28318*6 + time*2.1 + 4.189);
wave_2_per_point_6=a = 0.8 + 0.2*treble;
"""

    private static let classicHyperspace = """
[preset00]
fRating=4
fGammaAdj=1.8
fDecay=0.98
fZoom=1.01
fRot=0.0
fWarpScale=1.0
fWarpAnimSpeed=1.0
fXCentre=0.5
fYCentre=0.5
szx=1.0
szy=1.0
per_frame_1=zoom = 1.01 + 0.02*bass;
per_frame_2=rot = rot + 0.002*sin(time*0.1);
per_frame_3=warp = 1 + 0.5*bass;
per_frame_4=cx = 0.5 + 0.1*sin(time*0.17);
per_frame_5=cy = 0.5 + 0.1*cos(time*0.13);
per_frame_6=decay = 0.98 - 0.01*treble;
per_frame_7=r = 0.3 + 0.4*bass;
per_frame_8=g = 0.1 + 0.3*mid;
per_frame_9=b = 0.5 + 0.4*treble;
wave_0_enabled=1
wave_0_samples=512
wave_0_scaling=1.0
wave_0_smoothing=0.5
wave_0_r=0.5
wave_0_g=0.8
wave_0_b=1.0
wave_0_a=0.9
wave_0_usedots=0
wave_0_drawthick=1
wave_0_additive=1
wave_0_per_point_1=x = cos(i*3.14159*2)*0.4 + 0.5;
wave_0_per_point_2=y = sin(i*3.14159*2)*0.4*sample*2 + 0.5;
shape_0_enabled=1
shape_0_sides=6
shape_0_additive=1
shape_0_x=0.5
shape_0_y=0.5
shape_0_radius=0.1
shape_0_r=0.8
shape_0_g=0.4
shape_0_b=1.0
shape_0_a=0.3
shape_0_per_frame_1=radius = 0.05 + 0.08*bass;
shape_0_per_frame_2=ang = ang + 0.02;
shape_0_per_frame_3=r = 0.5 + 0.5*sin(time*0.7);
shape_0_per_frame_4=g = 0.2 + 0.3*bass;
"""

    private static let beatReactor = """
[preset00]
fRating=5
fGammaAdj=2.0
fDecay=0.97
fZoom=1.0
fRot=0.0
fWarpScale=2.0
fWarpAnimSpeed=1.5
fXCentre=0.5
fYCentre=0.5
szx=1.0
szy=1.0
per_frame_init_1=r=0; g=0; b=0; q1=0; q2=0;
per_frame_1=q1 = q1 + bass*0.3;
per_frame_2=q2 = q2*0.95 + bass_att*0.05;
per_frame_3=zoom = 1.0 + bass*0.05 + q2*0.02;
per_frame_4=rot = rot + 0.005 + 0.01*mid;
per_frame_5=warp = 2.0 + 3.0*bass;
per_frame_6=decay = 0.96 + 0.02*treble;
per_frame_7=cx = 0.5 + 0.15*sin(q1*0.3);
per_frame_8=cy = 0.5 + 0.15*cos(q1*0.23);
per_frame_9=r = 0.8 + 0.2*sin(time*1.3);
per_frame_10=g = 0.2 + 0.3*cos(time*0.9 + 1.0);
per_frame_11=b = 0.5 + 0.4*sin(time*0.7 + 2.0);
per_pixel_1=zoom = zoom + 0.005*sin(x*6+time) * bass;
per_pixel_2=zoom = zoom + 0.005*cos(y*5-time*0.8) * mid;
wave_0_enabled=1
wave_0_samples=512
wave_0_scaling=1.5
wave_0_smoothing=0.6
wave_0_r=1.0
wave_0_g=0.5
wave_0_b=0.2
wave_0_a=0.8
wave_0_usedots=0
wave_0_drawthick=1
wave_0_additive=1
wave_1_enabled=1
wave_1_samples=256
wave_1_scaling=0.8
wave_1_smoothing=0.3
wave_1_r=0.2
wave_1_g=0.6
wave_1_b=1.0
wave_1_a=0.6
wave_1_usedots=1
wave_1_drawthick=0
wave_1_additive=1
shape_0_enabled=1
shape_0_sides=3
shape_0_additive=1
shape_0_x=0.5
shape_0_y=0.5
shape_0_radius=0.15
shape_0_r=1.0
shape_0_g=0.8
shape_0_b=0.0
shape_0_a=0.4
shape_0_per_frame_1=radius = 0.08 + 0.15*bass_att;
shape_0_per_frame_2=ang = ang + 0.03*mid;
shape_0_per_frame_3=r = 0.5 + 0.5*sin(time*1.1);
shape_0_per_frame_4=g = 0.3 + 0.4*bass;
"""
}
