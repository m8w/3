import Foundation
import WebKit

// MARK: - PresetLoader
//
// Scans the entire app-bundle Resources tree for .json preset files,
// then injects a random sample into the running Butterchurn WebView.
//
// ── WHERE TO PUT YOUR FILES ───────────────────────────────────────────────────
//
//  1. CONVERTED PRESETS (.json)
//     Drop them into any subfolder of:
//       Sources/ButterchurnVisualizer/Resources/
//     Subfolders and arbitrary names are fine — the loader walks the whole tree.
//     Examples:
//       Resources/Presets _ Butterchurn/flexi - bubbles.json
//       Resources/Presets1/martin - gargoyle.json
//
//  2. TEXTURES (.jpg .png .bmp .tga)
//     Drop them into:
//       Sources/ButterchurnVisualizer/Resources/textures/
//     Butterchurn's CDN build has ~30 built-in textures.
//     Custom textures require a patched Butterchurn build — see README.
//
//  3. REBUILD
//     After adding files, press ⌘B in Xcode (or run `swift run`) so they
//     get bundled. No code changes needed — PresetLoader finds them automatically.
//
// ── PERFORMANCE NOTE ─────────────────────────────────────────────────────────
//  With thousands of presets only MAX_INJECT are loaded per launch (random
//  sample). Restart the app to get a fresh batch.
// ─────────────────────────────────────────────────────────────────────────────

final class PresetLoader {

    private static let maxInject = 500

    // Inject a random sample of bundled presets into `webView`.
    // Call this from WKNavigationDelegate.webView(_:didFinish:).
    static func injectAll(into webView: WKWebView) {
        let allURLs = findPresetURLs()
        guard !allURLs.isEmpty else {
            print("[PresetLoader] No bundled .json presets found in Resources/")
            return
        }

        let sampleURLs = allURLs.count > maxInject
            ? Array(allURLs.shuffled().prefix(maxInject))
            : allURLs

        print("[PresetLoader] Found \(allURLs.count) preset(s) — injecting \(sampleURLs.count)")

        var presets: [String: Any] = [:]
        for url in sampleURLs {
            do {
                let data   = try Data(contentsOf: url)
                let parsed = try JSONSerialization.jsonObject(with: data)
                let name   = url.deletingPathExtension().lastPathComponent
                presets[name] = parsed
            } catch {
                print("[PresetLoader] skipping \(url.lastPathComponent): \(error.localizedDescription)")
            }
        }

        guard !presets.isEmpty,
              let data    = try? JSONSerialization.data(withJSONObject: presets),
              let jsonStr = String(data: data, encoding: .utf8)
        else { return }

        let js = "if(typeof window._addPresets==='function')window._addPresets(\(jsonStr));"
        webView.evaluateJavaScript(js) { _, err in
            if let err { print("[PresetLoader] inject error: \(err)") }
        }
    }

    // MARK: - Private

    /// Walk the entire Resources bundle tree and return URLs of every .json file.
    private static func findPresetURLs() -> [URL] {
        guard let resourceURL = Bundle.module.resourceURL else { return [] }

        // Prefer a dedicated presets/ subfolder if it exists; otherwise search everything.
        let dedicatedURL = resourceURL.appendingPathComponent("presets")
        let searchURL    = FileManager.default.fileExists(atPath: dedicatedURL.path)
                         ? dedicatedURL
                         : resourceURL

        guard let enumerator = FileManager.default.enumerator(
            at: searchURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { return [] }

        return (enumerator.allObjects as? [URL] ?? [])
            .filter { $0.pathExtension.lowercased() == "json" }
    }
}
