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

        // 1. Bundled presets
        if let bundled = PresetManager.bundledPresetsDirectory {
            let bundledPresets = await loadPresets(from: bundled)
            loaded.append(contentsOf: bundledPresets)
            loadProgress = 0.5
        }

        // 2. User presets directory
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
            loaded.data = (try? String(contentsOf: url, encoding: .utf8)) ?? ""
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
