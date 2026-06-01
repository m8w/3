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
