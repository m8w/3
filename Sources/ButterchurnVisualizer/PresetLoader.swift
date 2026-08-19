import Foundation
import WebKit

// MARK: - PresetLoader
//
// Scans the app-bundle Resources tree for preset files and injects them into
// the running Butterchurn WebView in batches.
//
// Supported formats:
//   .json  — Butterchurn native JSON preset (passed through as-is)
//   .milk  — Milkdrop text preset (parsed by PresetParser, converted by
//             MilkPresetConverter, then injected as Butterchurn JSON)
//
// ── WHERE TO PUT YOUR FILES ───────────────────────────────────────────────────
//  Drop presets into any subfolder of:
//    Sources/ButterchurnVisualizer/Resources/
//  Preferred location (keeps them off the resource root):
//    Sources/ButterchurnVisualizer/Resources/presets/
//
//  After adding files rebuild (`swift run`) — no code changes needed.
// ─────────────────────────────────────────────────────────────────────────────

final class PresetLoader {

    private static let batchSize = 200   // presets per evaluateJavaScript call

    // Inject presets in shuffled batches.
    // Call from WKNavigationDelegate.webView(_:didFinish:).
    //
    // Per-screen mode: if SCREEN1_PRESETS/SCREEN2_PRESETS/SCREEN3_PRESETS point at
    // folders (or Resources/presets/screen1|2|3 exist with files), each screen
    // gets its OWN pool — the three screens can never show the same preset.
    // Otherwise every screen shares the whole bundled set (original behaviour).
    static func injectAll(into webView: WKWebView) {
        DispatchQueue.global(qos: .utility).async {
            let sources = screenSources()
            var perScreen: [[URL]] = [[], [], []]
            var anyPerScreen = false
            for s in 0..<3 {
                guard let dir = sources[s] else { continue }
                let urls = findPresetURLs(in: dir).shuffled()
                perScreen[s] = urls
                if !urls.isEmpty { anyPerScreen = true }
            }

            if anyPerScreen {
                for s in 0..<3 where !perScreen[s].isEmpty {
                    print("[PresetLoader] screen \(s + 1): \(perScreen[s].count) preset(s) (own pool)")
                    injectBatch(urls: perScreen[s], offset: 0, screen: s, into: webView)
                }
                return
            }

            let allURLs = findPresetURLs().shuffled()
            guard !allURLs.isEmpty else {
                print("[PresetLoader] No bundled .json / .milk presets found in Resources/")
                return
            }
            print("[PresetLoader] Found \(allURLs.count) preset(s) — shared pool, batches of \(batchSize)")
            injectBatch(urls: allURLs, offset: 0, screen: -1, into: webView)
        }
    }

    // Resolve the three per-screen preset folders: env override first, then a
    // bundled Resources/presets/screenN folder if it actually holds presets.
    private static func screenSources() -> [URL?] {
        let env = ProcessInfo.processInfo.environment
        func resolve(_ key: String, _ sub: String) -> URL? {
            if let raw = env[key], !raw.isEmpty {
                return URL(fileURLWithPath: (raw as NSString).expandingTildeInPath)
            }
            if let base = Bundle.module.resourceURL {
                let d = base.appendingPathComponent("presets").appendingPathComponent(sub)
                var isDir: ObjCBool = false
                if FileManager.default.fileExists(atPath: d.path, isDirectory: &isDir), isDir.boolValue {
                    return d
                }
            }
            return nil
        }
        return [resolve("SCREEN1_PRESETS", "screen1"),
                resolve("SCREEN2_PRESETS", "screen2"),
                resolve("SCREEN3_PRESETS", "screen3")]
    }

    // MARK: - Hot folder (live injection)
    //
    // If HOT_PRESETS points at a folder, poll it and inject any NEW .json presets
    // into the running mix — so the curator can drop keepers in and they go live
    // without a restart. Point HOT_PRESETS at the curator's CURATE_OUTPUT folder.

    private static let hotQueue = DispatchQueue(label: "hotfolder")
    private static var hotInjected = Set<String>()
    private static var hotBusy = false
    private static var hotTimer: Timer?

    static func watchHotFolder(into webView: WKWebView) {
        guard let raw = ProcessInfo.processInfo.environment["HOT_PRESETS"], !raw.isEmpty else { return }
        let dir = URL(fileURLWithPath: (raw as NSString).expandingTildeInPath)
        print("[HotFolder] watching \(dir.path) for new presets (every 6s)")
        hotTimer?.invalidate()
        hotTimer = Timer.scheduledTimer(withTimeInterval: 6.0, repeats: true) { [weak webView] _ in
            guard let webView else { return }
            hotQueue.async {
                guard !hotBusy else { return }
                hotBusy = true; defer { hotBusy = false }
                // Live-rotation cap (memory safety) — ALL keepers still save to disk
                // and bundle on the next rebuild; this only limits how many are held
                // in the running WebView at once. Raise with HOT_MAX if you have RAM.
                let hotMax = Int(ProcessInfo.processInfo.environment["HOT_MAX"] ?? "6000") ?? 6000
                guard hotInjected.count < hotMax else { return }
                guard let files = try? FileManager.default.contentsOfDirectory(
                    at: dir, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) else { return }
                let fresh = files.filter { $0.pathExtension.lowercased() == "json"
                                           && !hotInjected.contains($0.lastPathComponent) }
                guard !fresh.isEmpty else { return }
                var presets: [String: Any] = [:]
                for url in fresh.prefix(300) {           // cap per tick to avoid a giant eval
                    hotInjected.insert(url.lastPathComponent)
                    if let data = try? Data(contentsOf: url),
                       let obj  = try? JSONSerialization.jsonObject(with: data) {
                        presets[url.deletingPathExtension().lastPathComponent] = obj
                    }
                }
                guard !presets.isEmpty,
                      let data = try? JSONSerialization.data(withJSONObject: presets),
                      let json = String(data: data, encoding: .utf8) else { return }
                DispatchQueue.main.async {
                    webView.evaluateJavaScript("if(typeof window._addPresets==='function')window._addPresets(\(json));",
                                               completionHandler: nil)
                    print("[HotFolder] +\(presets.count) live preset(s) (watched \(hotInjected.count))")
                }
            }
        }
    }

    // MARK: - Reject (live cull)
    //
    // Called when you press X on a preset. Always logs the name to
    // ~/butterchurn_rejected.txt. If REJECT_DIR points at your source presets
    // folder, the matching .json/.milk file(s) are also moved out (to
    // ~/butterchurn_rejected_presets/) so a rebuild won't bring them back.

    static func rejectPreset(named name: String) {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let logURL = home.appendingPathComponent("butterchurn_rejected.txt")
        let line = name + "\n"
        if let fh = try? FileHandle(forWritingTo: logURL) {
            fh.seekToEndOfFile(); if let d = line.data(using: .utf8) { fh.write(d) }; try? fh.close()
        } else {
            try? line.write(to: logURL, atomically: true, encoding: .utf8)
        }

        guard let raw = ProcessInfo.processInfo.environment["REJECT_DIR"], !raw.isEmpty else {
            print("[Reject] logged '\(name)' (set REJECT_DIR to also delete its file)")
            return
        }
        let src = URL(fileURLWithPath: (raw as NSString).expandingTildeInPath)
        let trash = home.appendingPathComponent("butterchurn_rejected_presets")
        DispatchQueue.global(qos: .utility).async {
            try? FileManager.default.createDirectory(at: trash, withIntermediateDirectories: true)
            guard let en = FileManager.default.enumerator(at: src,
                includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) else { return }
            var moved = 0
            for case let url as URL in en {
                let ext = url.pathExtension.lowercased()
                guard ext == "json" || ext == "milk",
                      url.deletingPathExtension().lastPathComponent == name else { continue }
                let dest = trash.appendingPathComponent("\(moved)_\(url.lastPathComponent)")
                try? FileManager.default.removeItem(at: dest)
                try? FileManager.default.moveItem(at: url, to: dest)
                moved += 1
            }
            print("[Reject] '\(name)' → removed \(moved) file(s) from presets")
        }
    }

    // MARK: - Private

    // screen: -1 → shared pool (_addPresets); 0..2 → that screen's pool (_addPresetsFor).
    private static func injectBatch(urls: [URL], offset: Int, screen: Int, into webView: WKWebView) {
        let slice = Array(urls[offset ..< min(offset + batchSize, urls.count)])

        var presets: [String: Any] = [:]
        for url in slice {
            let name = url.deletingPathExtension().lastPathComponent
            if let obj = loadPreset(at: url) {
                presets[name] = obj
            }
        }

        guard !presets.isEmpty,
              let data    = try? JSONSerialization.data(withJSONObject: presets),
              let jsonStr = String(data: data, encoding: .utf8)
        else {
            scheduleNext(urls: urls, offset: offset, screen: screen, into: webView)
            return
        }

        let js = screen >= 0
            ? "if(typeof window._addPresetsFor==='function')window._addPresetsFor(\(screen),\(jsonStr));"
            : "if(typeof window._addPresets==='function')window._addPresets(\(jsonStr));"
        DispatchQueue.main.async {
            webView.evaluateJavaScript(js) { _, err in
                if let err { print("[PresetLoader] batch error: \(err)") }
                scheduleNext(urls: urls, offset: offset, screen: screen, into: webView)
            }
        }
    }

    // Decode a single preset file — JSON pass-through or .milk conversion.
    private static func loadPreset(at url: URL) -> Any? {
        switch url.pathExtension.lowercased() {
        case "json":
            guard let data = try? Data(contentsOf: url),
                  let obj  = try? JSONSerialization.jsonObject(with: data)
            else { return nil }
            return obj

        case "milk":
            guard let preset = try? PresetParser.load(from: url) else { return nil }
            return MilkPresetConverter.toButterchurnDict(preset)

        default:
            return nil
        }
    }

    private static func scheduleNext(urls: [URL], offset: Int, screen: Int, into webView: WKWebView) {
        let next = offset + batchSize
        guard next < urls.count else {
            let tag = screen >= 0 ? "screen \(screen + 1)" : "shared"
            print("[PresetLoader] All \(urls.count) \(tag) preset(s) injected")
            return
        }
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 0.15) {
            injectBatch(urls: urls, offset: next, screen: screen, into: webView)
        }
    }

    /// Walk the entire Resources bundle tree (or PRESETS_ONLY) for the shared pool.
    private static func findPresetURLs() -> [URL] {
        // PRESETS_ONLY=/path → load ONLY that folder (isolated batch review),
        // ignoring the bundled presets entirely. Otherwise load the whole bundle.
        let root: URL
        if let only = ProcessInfo.processInfo.environment["PRESETS_ONLY"], !only.isEmpty {
            root = URL(fileURLWithPath: (only as NSString).expandingTildeInPath)
            print("[PresetLoader] PRESETS_ONLY — loading only \(root.path)")
        } else if let resourceURL = Bundle.module.resourceURL {
            root = resourceURL
        } else {
            return []
        }
        return findPresetURLs(in: root)
    }

    /// Recursively collect every .json / .milk file under `root`.
    private static func findPresetURLs(in root: URL) -> [URL] {
        guard let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { return [] }

        let supported: Set<String> = ["json", "milk"]
        return (enumerator.allObjects as? [URL] ?? [])
            .filter { supported.contains($0.pathExtension.lowercased()) }
    }
}
