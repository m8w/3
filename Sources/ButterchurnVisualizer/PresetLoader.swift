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

    // Inject all bundled presets in shuffled batches.
    // Call from WKNavigationDelegate.webView(_:didFinish:).
    static func injectAll(into webView: WKWebView) {
        DispatchQueue.global(qos: .utility).async {
            let allURLs = findPresetURLs().shuffled()
            guard !allURLs.isEmpty else {
                print("[PresetLoader] No bundled .json / .milk presets found in Resources/")
                return
            }
            print("[PresetLoader] Found \(allURLs.count) preset(s) — injecting in batches of \(batchSize)")
            injectBatch(urls: allURLs, offset: 0, into: webView)
        }
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

    private static func injectBatch(urls: [URL], offset: Int, into webView: WKWebView) {
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
            scheduleNext(urls: urls, offset: offset, into: webView)
            return
        }

        let js = "if(typeof window._addPresets==='function')window._addPresets(\(jsonStr));"
        DispatchQueue.main.async {
            webView.evaluateJavaScript(js) { _, err in
                if let err { print("[PresetLoader] batch error: \(err)") }
                scheduleNext(urls: urls, offset: offset, into: webView)
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

    private static func scheduleNext(urls: [URL], offset: Int, into webView: WKWebView) {
        let next = offset + batchSize
        guard next < urls.count else {
            print("[PresetLoader] All \(urls.count) preset(s) injected")
            return
        }
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 0.15) {
            injectBatch(urls: urls, offset: next, into: webView)
        }
    }

    /// Walk the entire Resources bundle tree and return URLs of every .json and .milk file.
    private static func findPresetURLs() -> [URL] {
        guard let resourceURL = Bundle.module.resourceURL else { return [] }

        guard let enumerator = FileManager.default.enumerator(
            at: resourceURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { return [] }

        let supported: Set<String> = ["json", "milk"]
        return (enumerator.allObjects as? [URL] ?? [])
            .filter { supported.contains($0.pathExtension.lowercased()) }
    }
}
