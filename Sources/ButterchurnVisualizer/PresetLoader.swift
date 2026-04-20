import Foundation
import WebKit

// MARK: - PresetLoader
//
// Scans the app bundle's Resources/presets/ folder for .json preset files,
// then injects them into the running Butterchurn WebView via JavaScript.
//
// ── WHERE TO PUT YOUR FILES ───────────────────────────────────────────────────
//
//  1. CONVERTED PRESETS (.json)
//     Drop them into:
//       Sources/ButterchurnVisualizer/Resources/presets/
//     Subfolders are fine — the loader walks the entire tree.
//     Example layout:
//       Resources/presets/Flexi/flexi - bubbles.json
//       Resources/presets/Martin/martin - gargoyle.json
//       Resources/presets/aderrasi - atrium.json
//
//  2. TEXTURES (.jpg .png .bmp .tga)
//     Drop them into:
//       Sources/ButterchurnVisualizer/Resources/textures/
//     Butterchurn's standard build has ~30 built-in textures.
//     Custom textures require a patched Butterchurn build — see README.
//
//  3. REBUILD
//     After adding files, press ⌘B in Xcode so they get bundled.
//     No code changes needed — PresetLoader finds them automatically.
// ─────────────────────────────────────────────────────────────────────────────

final class PresetLoader {

    // Inject all bundled presets into `webView` as soon as the page is ready.
    // Call this from WKNavigationDelegate.webView(_:didFinish:).
    static func injectAll(into webView: WKWebView) {
        let presets = loadBundledPresets()
        guard !presets.isEmpty else {
            print("[PresetLoader] No bundled .json presets found in Resources/presets/")
            return
        }
        print("[PresetLoader] Injecting \(presets.count) bundled preset(s)")

        // Encode the full dictionary as JSON and call window._addPresets({...})
        // in the WebView.  The HTML side merges these into the preset rotation.
        guard let data = try? JSONSerialization.data(withJSONObject: presets),
              let jsonStr = String(data: data, encoding: .utf8)
        else { return }

        let js = "if(typeof window._addPresets==='function')window._addPresets(\(jsonStr));"
        webView.evaluateJavaScript(js, completionHandler: { _, err in
            if let err { print("[PresetLoader] inject error: \(err)") }
        })
    }

    // MARK: - Private

    /// Recursively scan Resources/presets/ for .json files.
    /// Returns a flat dictionary: { "PresetName": <preset object> }
    private static func loadBundledPresets() -> [String: Any] {
        var result: [String: Any] = [:]

        guard let presetsURL = Bundle.module.resourceURL?
                .appendingPathComponent("presets")
        else { return result }

        guard let enumerator = FileManager.default.enumerator(
            at: presetsURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { return result }

        for case let fileURL as URL in enumerator {
            guard fileURL.pathExtension.lowercased() == "json" else { continue }

            do {
                let data   = try Data(contentsOf: fileURL)
                let parsed = try JSONSerialization.jsonObject(with: data)
                let name   = fileURL.deletingPathExtension().lastPathComponent
                result[name] = parsed
            } catch {
                print("[PresetLoader] skipping \(fileURL.lastPathComponent): \(error.localizedDescription)")
            }
        }

        return result
    }
}
