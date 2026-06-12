import SwiftUI
import WebKit
import CryptoKit

// MARK: - Curator
//
// Offline preset-culling pipeline. Reads a folder of MilkDrop .milk presets
// (e.g. a 130k MegaPack), converts each to Butterchurn JSON in a WKWebView,
// renders it for ~1s with synthetic beat-driven audio, scores the result for
// colourfulness + spatial contrast + motion + temporal colour change, and
// writes the keepers (high-dynamic, colourful) as .json into the output dir.
//
// ── HOW TO RUN ────────────────────────────────────────────────────────────────
//   From the repo root:
//     CURATE_INPUT="$HOME/Downloads/MilkDrop 130k+ Presets MegaPack 2026 2" swift run
//
//   Defaults if CURATE_INPUT is unset but CURATE=1 is set:
//     input  = ~/Downloads/MilkDrop 130k+ Presets MegaPack 2026 2
//     output = <cwd>/Sources/ButterchurnVisualizer/Resources/presets/curated
//
//   Keepers land in Resources/presets/curated/ — a normal `swift run` afterward
//   bundles and plays them automatically. The run is resumable: re-launching
//   skips presets already recorded in the output's _manifest.tsv.
//
//   Tunables (env): CURATE_OUTPUT, CURATE_FRAMES, CURATE_FPS, CURATE_WARM,
//     CURATE_LIMIT, CURATE_MIN_SCORE, CURATE_MIN_MOTION, CURATE_MIN_COLOR,
//     CURATE_MIN_LUMASTD, CURATE_MIN_TCC, CURATE_TIMEOUT
// ─────────────────────────────────────────────────────────────────────────────

// Lazily constructs the Curator from the environment so the App can decide at
// launch whether to show the visualizer or the curation UI.
@MainActor
final class CuratorBox: ObservableObject {
    let instance: Curator? = Curator.fromEnvironment()
}

@MainActor
final class Curator: NSObject, ObservableObject, WKNavigationDelegate, WKScriptMessageHandler {

    // Published progress (drives CurateView).
    @Published var total     = 0
    @Published var processed = 0
    @Published var kept      = 0
    @Published var failed    = 0
    @Published var current   = ""
    @Published var status    = "starting…"
    @Published var finished  = false

    let inputDir:  URL
    let outputDir: URL
    private let manifestURL: URL
    private let cfg: [String: Any]
    private let limit: Int
    private let timeoutSecs: Double

    private(set) var webView: WKWebView!
    private var seen = Set<String>()      // content hashes already processed
    private var usedNames = Set<String>()
    private var manifest: FileHandle?

    // MARK: Environment bootstrap

    static func fromEnvironment() -> Curator? {
        let env = ProcessInfo.processInfo.environment
        let enabled = env["CURATE_INPUT"] != nil || env["CURATE"] == "1"
        guard enabled else { return nil }

        let home = FileManager.default.homeDirectoryForCurrentUser
        let defaultIn = home
            .appendingPathComponent("Downloads")
            .appendingPathComponent("MilkDrop 130k+ Presets MegaPack 2026 2")
        let inputDir = env["CURATE_INPUT"].map { URL(fileURLWithPath: $0) } ?? defaultIn

        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let defaultOut = cwd
            .appendingPathComponent("Sources/ButterchurnVisualizer/Resources/presets/curated")
        let outputDir = env["CURATE_OUTPUT"].map { URL(fileURLWithPath: $0) } ?? defaultOut

        return Curator(inputDir: inputDir, outputDir: outputDir, env: env)
    }

    private init(inputDir: URL, outputDir: URL, env: [String: String]) {
        self.inputDir  = inputDir
        self.outputDir = outputDir
        self.manifestURL = outputDir.appendingPathComponent("_manifest.tsv")
        self.limit = env["CURATE_LIMIT"].flatMap(Int.init) ?? Int.max
        self.timeoutSecs = env["CURATE_TIMEOUT"].flatMap(Double.init) ?? 20.0

        func d(_ k: String, _ v: Double) -> Double { env[k].flatMap(Double.init) ?? v }
        func i(_ k: String, _ v: Int) -> Int { env[k].flatMap(Int.init) ?? v }
        self.cfg = [
            "frames": i("CURATE_FRAMES", 30),
            "fps":    i("CURATE_FPS", 30),
            "warm":   i("CURATE_WARM", 6),
            "thresholds": [
                "minScore":        d("CURATE_MIN_SCORE", 22),
                "minMotion":       d("CURATE_MIN_MOTION", 1.2),
                "minColorfulness": d("CURATE_MIN_COLOR", 8),
                "minLumaStd":      d("CURATE_MIN_LUMASTD", 5),
                "minTcc":          d("CURATE_MIN_TCC", 6),
            ],
        ]
        super.init()
        buildWebView()
    }

    // MARK: WebView setup

    private func buildWebView() {
        let config = WKWebViewConfiguration()
        let prefs = WKWebpagePreferences()
        prefs.allowsContentJavaScript = true
        config.defaultWebpagePreferences = prefs
        config.userContentController.add(self, name: "log")

        // Inject butterchurn + the milk→butterchurn converter at documentStart,
        // both read from the app bundle (no network needed).
        for res in ["butterchurn.min", "milkdrop-preset-converter.min"] {
            guard let url = Bundle.module.url(forResource: res, withExtension: "js"),
                  let src = try? String(contentsOf: url, encoding: .utf8) else {
                print("[Curator] FATAL: \(res).js missing from bundle")
                continue
            }
            config.userContentController.addUserScript(
                WKUserScript(source: src, injectionTime: .atDocumentStart, forMainFrameOnly: true))
        }

        webView = WKWebView(frame: NSRect(x: 0, y: 0, width: 320, height: 240), configuration: config)
        webView.navigationDelegate = self

        guard let host = Bundle.module.url(forResource: "curate_host", withExtension: "html"),
              let html = try? String(contentsOf: host, encoding: .utf8) else {
            print("[Curator] FATAL: curate_host.html missing"); return
        }
        webView.loadHTMLString(html, baseURL: URL(string: "https://milkdrop.local"))
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        Task { await self.run() }
    }

    func userContentController(_ u: WKUserContentController, didReceive m: WKScriptMessage) {
        if let s = m.body as? String { print("[JS] \(s)") }
    }

    // MARK: Main loop

    private func run() async {
        guard await waitReady() else {
            status = "butterchurn/converter failed to load — check console"; return
        }
        try? FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
        loadManifest()

        status = "scanning \(inputDir.lastPathComponent)…"
        let files = enumerateMilk()
        total = files.count
        print("[Curator] \(files.count) .milk file(s) under \(inputDir.path)")
        if files.isEmpty {
            status = "no .milk files found in \(inputDir.path)"; finished = true; return
        }

        var done = 0
        for url in files {
            if done >= limit { break }
            done += 1

            guard let text = try? String(contentsOf: url, encoding: .utf8) else { continue }
            let hash = sha(text)
            if !seen.insert(hash).inserted { continue }   // duplicate content — skip

            current = url.lastPathComponent
            let result = await curateWithTimeout(text)

            processed += 1
            var keepFlag = false, score = 0.0
            if let r = result, (r["ok"] as? Bool) == true {
                score = (r["score"] as? Double) ?? 0
                if (r["keep"] as? Bool) == true, let json = r["preset"] as? String {
                    let outName = write(json: json, baseName: url.deletingPathExtension().lastPathComponent, hash: hash)
                    kept += 1; keepFlag = true
                    appendManifest(hash: hash, keep: true, score: score, out: outName, name: current)
                } else {
                    appendManifest(hash: hash, keep: false, score: score, out: "", name: current)
                }
            } else {
                failed += 1
                appendManifest(hash: hash, keep: false, score: 0, out: "", name: current)
                // A timeout can wedge the JS thread — reload to be safe.
                if result == nil { await reloadAndWait() }
            }

            if processed % 50 == 0 || keepFlag {
                status = "\(processed)/\(total) · kept \(kept) · failed \(failed)"
            }
            if processed % 200 == 0 {
                print("[Curator] \(processed)/\(total) processed · kept \(kept) · failed \(failed)")
            }
        }

        try? manifest?.close()
        status = "done — \(kept) kept of \(processed) processed (\(failed) failed)"
        print("[Curator] \(status)")
        print("[Curator] keepers written to \(outputDir.path)")
        finished = true
    }

    // MARK: JS bridge

    // Returns the JS result dict, or nil on hard JS failure / timeout (both of
    // which warrant a page reload to clear any wedged promise).
    private func curateWithTimeout(_ milk: String) async -> [String: Any]? {
        await withCheckedContinuation { (cont: CheckedContinuation<[String: Any]?, Never>) in
            var done = false
            func finish(_ v: [String: Any]?) {
                guard !done else { return }
                done = true
                cont.resume(returning: v)
            }
            let timeout = DispatchWorkItem { finish(nil) }
            DispatchQueue.main.asyncAfter(deadline: .now() + timeoutSecs, execute: timeout)

            webView.callAsyncJavaScript(
                "return await window.__curate(milk, cfg);",
                arguments: ["milk": milk, "cfg": cfg],
                in: nil, in: .page
            ) { result in
                timeout.cancel()
                switch result {
                case .success(let v): finish(v as? [String: Any])
                case .failure:        finish(nil)
                }
            }
        }
    }

    private func waitReady() async -> Bool {
        for _ in 0..<160 {   // up to ~40s (converter bundle parse can be slow)
            if let r = try? await webView.evaluateJavaScript("window.__curateReady === true") as? Bool, r {
                return true
            }
            try? await Task.sleep(nanoseconds: 250_000_000)
        }
        return false
    }

    private func reloadAndWait() async {
        guard let host = Bundle.module.url(forResource: "curate_host", withExtension: "html"),
              let html = try? String(contentsOf: host, encoding: .utf8) else { return }
        webView.loadHTMLString(html, baseURL: URL(string: "https://milkdrop.local"))
        _ = await waitReady()
    }

    // MARK: IO helpers

    private func enumerateMilk() -> [URL] {
        guard let en = FileManager.default.enumerator(
            at: inputDir, includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]) else { return [] }
        return (en.allObjects as? [URL] ?? []).filter { $0.pathExtension.lowercased() == "milk" }
    }

    private func write(json: String, baseName: String, hash: String) -> String {
        var safe = baseName.replacingOccurrences(of: "/", with: "_")
        if safe.count > 120 { safe = String(safe.prefix(120)) }
        var name = safe + ".json"
        if !usedNames.insert(name).inserted {                 // collision → disambiguate
            name = safe + "__" + String(hash.prefix(6)) + ".json"
            usedNames.insert(name)
        }
        try? json.write(to: outputDir.appendingPathComponent(name), atomically: true, encoding: .utf8)
        return name
    }

    private func loadManifest() {
        if let data = try? String(contentsOf: manifestURL, encoding: .utf8) {
            for line in data.split(separator: "\n") {
                let cols = line.split(separator: "\t", omittingEmptySubsequences: false)
                if let h = cols.first { seen.insert(String(h)) }
                if cols.count >= 4, cols[1] == "1" { kept += 1; usedNames.insert(String(cols[3])) }
            }
            processed = seen.count
            print("[Curator] resuming — \(seen.count) already processed, \(kept) previously kept")
        }
        if !FileManager.default.fileExists(atPath: manifestURL.path) {
            FileManager.default.createFile(atPath: manifestURL.path, contents: nil)
        }
        manifest = try? FileHandle(forWritingTo: manifestURL)
        _ = try? manifest?.seekToEnd()
    }

    private func appendManifest(hash: String, keep: Bool, score: Double, out: String, name: String) {
        let line = "\(hash)\t\(keep ? 1 : 0)\t\(String(format: "%.1f", score))\t\(out)\t\(name)\n"
        if let d = line.data(using: .utf8) { manifest?.write(d) }
    }

    private func sha(_ s: String) -> String {
        SHA256.hash(data: Data(s.utf8)).map { String(format: "%02x", $0) }.joined()
    }
}
