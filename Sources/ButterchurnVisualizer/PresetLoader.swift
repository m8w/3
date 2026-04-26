import Foundation
import WebKit

// MARK: - PresetLoader
//
// Scans the entire app-bundle Resources tree for .json preset files,
// then injects ALL of them into the running Butterchurn WebView in
// batches so the JS side is never handed a single huge payload.
//
// ── WHERE TO PUT YOUR FILES ───────────────────────────────────────────────────
//  Drop .json presets into any subfolder of:
//    Sources/ButterchurnVisualizer/Resources/
//  Examples:
//    Resources/Presets _ Butterchurn/flexi - bubbles.json
//    Resources/Presets1/martin - gargoyle.json
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
                print("[PresetLoader] No bundled .json presets found in Resources/")
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
            guard let data   = try? Data(contentsOf: url),
                  let parsed = try? JSONSerialization.jsonObject(with: data)
            else { continue }
            presets[url.deletingPathExtension().lastPathComponent] = parsed
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

    private static func scheduleNext(urls: [URL], offset: Int, into webView: WKWebView) {
        let next = offset + batchSize
        guard next < urls.count else {
            print("[PresetLoader] All \(urls.count) preset(s) injected")
            return
        }
        // Small delay between batches to keep the main thread responsive.
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 0.15) {
            injectBatch(urls: urls, offset: next, into: webView)
        }
    }

    /// Walk the entire Resources bundle tree and return URLs of every .json file.
    private static func findPresetURLs() -> [URL] {
        guard let resourceURL = Bundle.module.resourceURL else { return [] }

        let dedicatedURL = resourceURL.appendingPathComponent("presets")
        let searchURL    = FileManager.default.fileExists(atPath: dedicatedURL.path)
                         ? dedicatedURL : resourceURL

        guard let enumerator = FileManager.default.enumerator(
            at: searchURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { return [] }

        return (enumerator.allObjects as? [URL] ?? [])
            .filter { $0.pathExtension.lowercased() == "json" }
    }
}
