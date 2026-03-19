// PresetBrowserView.swift — Grid/list browser for MilkDrop presets
// Features: search, favorites, ratings, tags, drag-and-drop

import SwiftUI

struct PresetBrowserView: View {
    @EnvironmentObject var state: AppState
    @State private var viewStyle: ViewStyle = .list
    @State private var showFilters: Bool = false

    var body: some View {
        VStack(spacing: 0) {
            // Search bar
            HStack(spacing: 8) {
                HStack {
                    Image(systemName: "magnifyingglass")
                        .foregroundColor(.secondary)
                        .font(.system(size: 12))
                    TextField("Search presets...", text: $state.presetManager.searchText)
                        .textFieldStyle(.plain)
                        .font(.system(size: 13))
                    if !state.presetManager.searchText.isEmpty {
                        Button(action: { state.presetManager.searchText = "" }) {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundColor(.secondary)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 5)
                .background(Color(hex: "1a1a1a"))
                .cornerRadius(6)

                Button(action: { showFilters.toggle() }) {
                    Image(systemName: "line.3.horizontal.decrease.circle")
                        .foregroundColor(showFilters ? .accentColor : .secondary)
                }
                .buttonStyle(.plain)
                .help("Filters")

                // View style toggle
                Picker("", selection: $viewStyle) {
                    Image(systemName: "list.bullet").tag(ViewStyle.list)
                    Image(systemName: "square.grid.2x2").tag(ViewStyle.grid)
                }
                .pickerStyle(.segmented)
                .frame(width: 60)
            }
            .padding(8)

            // Filter bar
            if showFilters {
                FilterBarView()
                    .padding(.horizontal, 8)
                    .padding(.bottom, 8)
            }

            // Preset count
            HStack {
                Text("\(state.presetManager.filteredPresets.count) presets")
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)
                Spacer()

                // Sort picker
                Picker("Sort", selection: $state.presetManager.sortOrder) {
                    ForEach(PresetManager.SortOrder.allCases, id: \.self) { order in
                        Text(order.rawValue.capitalized).tag(order)
                    }
                }
                .pickerStyle(.menu)
                .font(.system(size: 11))
                .frame(width: 100)
            }
            .padding(.horizontal, 8)
            .padding(.bottom, 4)

            Divider().background(Color(hex: "2a2a2a"))

            // Preset list
            if state.presetManager.isLoading {
                VStack(spacing: 12) {
                    ProgressView(value: state.presetManager.loadProgress)
                        .progressViewStyle(.linear)
                    Text("Loading presets...")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding()
            } else if state.presetManager.filteredPresets.isEmpty {
                ContentUnavailableView(
                    "No Presets Found",
                    systemImage: "photo.on.rectangle.angled",
                    description: Text("Add .milk files to your presets folder or adjust your filters.")
                )
            } else {
                ScrollViewReader { proxy in
                    ScrollView {
                        if viewStyle == .list {
                            LazyVStack(spacing: 1) {
                                ForEach(state.presetManager.filteredPresets) { preset in
                                    PresetRowView(preset: preset)
                                        .id(preset.id)
                                }
                            }
                        } else {
                            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                                ForEach(state.presetManager.filteredPresets) { preset in
                                    PresetGridCell(preset: preset)
                                        .id(preset.id)
                                }
                            }
                            .padding(8)
                        }
                    }
                    .onChange(of: state.presetManager.currentPreset?.id) { _, id in
                        if let id { withAnimation { proxy.scrollTo(id, anchor: .center) } }
                    }
                }
            }

            Divider().background(Color(hex: "2a2a2a"))

            // Bottom controls
            HStack(spacing: 8) {
                Button(action: { state.importPresets() }) {
                    Label("Import", systemImage: "plus")
                        .font(.system(size: 11))
                }
                .buttonStyle(.bordered)

                Spacer()

                Button(action: { state.openPresetFolder() }) {
                    Image(systemName: "folder")
                }
                .buttonStyle(.plain)
                .help("Open Presets Folder")
            }
            .padding(8)
        }
    }

    enum ViewStyle { case list, grid }
}

// MARK: - Preset row (list view)

struct PresetRowView: View {
    @EnvironmentObject var state: AppState
    let preset: MilkDropPreset
    @State private var hovered: Bool = false

    var isSelected: Bool { state.presetManager.currentPreset?.id == preset.id }

    var body: some View {
        HStack(spacing: 8) {
            // Type indicator
            Image(systemName: preset.isDoublePreset ? "rectangle.split.2x1.fill" : "waveform.path")
                .font(.system(size: 11))
                .foregroundColor(preset.isDoublePreset ? .purple : .accentColor)
                .frame(width: 16)

            // Name
            Text(preset.name)
                .font(.system(size: 12))
                .lineLimit(1)
                .truncationMode(.tail)

            Spacer()

            // Rating stars
            if preset.rating > 0 {
                HStack(spacing: 1) {
                    ForEach(0..<preset.rating, id: \.self) { _ in
                        Image(systemName: "star.fill")
                            .font(.system(size: 8))
                            .foregroundColor(.yellow)
                    }
                }
            }

            // Favorite
            Button(action: { state.presetManager.toggleFavorite(preset) }) {
                Image(systemName: preset.isFavorite ? "heart.fill" : "heart")
                    .font(.system(size: 11))
                    .foregroundColor(preset.isFavorite ? .pink : .secondary)
            }
            .buttonStyle(.plain)
            .opacity(hovered || preset.isFavorite ? 1 : 0)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(
            RoundedRectangle(cornerRadius: 4)
                .fill(isSelected
                      ? Color.accentColor.opacity(0.25)
                      : hovered
                        ? Color(hex: "1e1e1e")
                        : Color.clear)
        )
        .onHover { hovered = $0 }
        .onTapGesture(count: 2) {
            state.presetManager.select(preset, transition: .instant)
        }
        .onTapGesture {
            state.presetManager.select(preset, transition: .smooth)
        }
        .contextMenu { PresetContextMenu(preset: preset) }
        .onDrag { NSItemProvider(object: preset.name as NSString) }
    }
}

// MARK: - Preset grid cell

struct PresetGridCell: View {
    @EnvironmentObject var state: AppState
    let preset: MilkDropPreset
    @State private var hovered: Bool = false

    var isSelected: Bool { state.presetManager.currentPreset?.id == preset.id }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            // Thumbnail placeholder (would show a rendered preview)
            ZStack {
                RoundedRectangle(cornerRadius: 6)
                    .fill(Color(hex: "1a1a1a"))
                    .aspectRatio(16/9, contentMode: .fit)

                Image(systemName: "waveform.path")
                    .font(.system(size: 24))
                    .foregroundColor(.accentColor.opacity(0.4))

                if preset.isDoublePreset {
                    Text("×2")
                        .font(.system(size: 9, weight: .bold))
                        .padding(.horizontal, 4)
                        .padding(.vertical, 2)
                        .background(Color.purple.opacity(0.8))
                        .cornerRadius(3)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
                        .padding(4)
                }
            }
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(isSelected ? Color.accentColor : Color.clear, lineWidth: 2)
            )

            Text(preset.name)
                .font(.system(size: 10))
                .lineLimit(2)
                .foregroundColor(.primary)
        }
        .onHover { hovered = $0 }
        .onTapGesture { state.presetManager.select(preset) }
        .contextMenu { PresetContextMenu(preset: preset) }
    }
}

// MARK: - Context menu

struct PresetContextMenu: View {
    @EnvironmentObject var state: AppState
    let preset: MilkDropPreset

    var body: some View {
        Button("Load Preset") {
            state.presetManager.select(preset, transition: .smooth)
        }
        Button("Load (Instant)") {
            state.presetManager.select(preset, transition: .instant)
        }
        Divider()
        Button(preset.isFavorite ? "Remove from Favorites" : "Add to Favorites") {
            state.presetManager.toggleFavorite(preset)
        }
        Button("Edit Preset") {
            state.editingPreset = preset
            state.showPresetEditor = true
        }
        Button("Create Double Preset with Current") {
            if let current = state.presetManager.currentPreset {
                let double = MilkDropPreset.makeDoublePreset(a: current, b: preset)
                state.editingPreset = double
                state.showPresetEditor = true
            }
        }
        Divider()
        Button("Show in Finder") {
            if let url = preset.url {
                NSWorkspace.shared.activateFileViewerSelecting([url])
            }
        }
        .disabled(preset.url == nil)
    }
}

// MARK: - Filter bar

struct FilterBarView: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        HStack(spacing: 12) {
            Toggle("Favorites", isOn: $state.presetManager.filterFavorites)
                .toggleStyle(.button)
                .controlSize(.small)

            Text("Min ★:")
                .font(.system(size: 11))
                .foregroundColor(.secondary)
            Picker("", selection: $state.presetManager.filterRating) {
                Text("Any").tag(0)
                ForEach(1...5, id: \.self) { r in Text("\(r)★").tag(r) }
            }
            .pickerStyle(.menu)
            .frame(width: 70)

            Spacer()

            Button("Reset") {
                state.presetManager.filterFavorites = false
                state.presetManager.filterRating    = 0
                state.presetManager.searchText      = ""
            }
            .buttonStyle(.plain)
            .font(.system(size: 11))
            .foregroundColor(.secondary)
        }
    }
}
