import Foundation

// MARK: - Parsed preset model

/// In-memory representation of a .milk file.
/// All numbered equation lines (per_frame_1=, per_pixel_1=, …) are stored in
/// order.  `baseVals` holds every other key=value pair (zoom, decay, etc.).
public struct MilkPreset {

    /// Non-equation scalars: zoom, rot, decay, fGammaAdj, …
    public var baseVals: [String: String] = [:]

    /// per_frame_init_N= lines, in order.
    public var initEqs:     [String] = []
    /// per_frame_N= lines, in order.
    public var perFrameEqs: [String] = []
    /// per_pixel_N= lines, in order.
    public var perPixelEqs: [String] = []
    /// wave_K_per_frame_N= grouped by wave index K.
    public var waveFrameEqs:  [Int: [String]] = [:]
    /// wave_K_per_point_N= grouped by wave index K.
    public var wavePointEqs:  [Int: [String]] = [:]

    public init() {}
}

// MARK: - Parser

public enum PresetParseError: Error {
    case missingPresetSection
    case malformedLine(String)
}

public struct PresetParser {

    // Regex patterns for the numbered equation keys.
    private static let initRE      = try! NSRegularExpression(pattern: #"^per_frame_init_(\d+)$"#)
    private static let perFrameRE  = try! NSRegularExpression(pattern: #"^per_frame_(\d+)$"#)
    private static let perPixelRE  = try! NSRegularExpression(pattern: #"^per_pixel_(\d+)$"#)
    private static let waveFrameRE = try! NSRegularExpression(pattern: #"^wave_(\d+)_per_frame_(\d+)$"#)
    private static let wavePointRE = try! NSRegularExpression(pattern: #"^wave_(\d+)_per_point_(\d+)$"#)

    /// Load and parse a .milk file at `url`.
    public static func load(from url: URL) throws -> MilkPreset {
        let raw = try String(contentsOf: url, encoding: .utf8)
        return try parse(string: raw)
    }

    /// Parse the text of a .milk file.
    public static func parse(string: String) throws -> MilkPreset {
        var preset = MilkPreset()
        var inSection = false

        // Temporary sorted storage: (index, equation) so we can sort after parsing.
        var initPairs:     [(Int, String)] = []
        var framePairs:    [(Int, String)] = []
        var pixelPairs:    [(Int, String)] = []
        var waveFramePairs: [Int: [(Int, String)]] = [:]
        var wavePointPairs: [Int: [(Int, String)]] = [:]

        for rawLine in string.components(separatedBy: .newlines) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)

            // Skip comments and blank lines.
            if line.isEmpty || line.hasPrefix(";") || line.hasPrefix("#") { continue }

            // Section header.
            if line.hasPrefix("[") {
                inSection = line.lowercased().hasPrefix("[preset")
                continue
            }
            guard inSection else { continue }

            // Split on the first "=" only.
            guard let eqRange = line.range(of: "=") else {
                throw PresetParseError.malformedLine(line)
            }
            let key   = String(line[line.startIndex ..< eqRange.lowerBound])
                .trimmingCharacters(in: .whitespaces)
            let value = String(line[eqRange.upperBound...])
                .trimmingCharacters(in: .whitespaces)

            let keyLower = key.lowercased()
            let keyRange = NSRange(keyLower.startIndex..., in: keyLower)

            // ── per_frame_init_N ──────────────────────────────────────────
            if let m = initRE.firstMatch(in: keyLower, range: keyRange),
               let r = Range(m.range(at: 1), in: keyLower),
               let n = Int(keyLower[r]) {
                initPairs.append((n, value))
                continue
            }
            // ── per_frame_N ───────────────────────────────────────────────
            if let m = perFrameRE.firstMatch(in: keyLower, range: keyRange),
               let r = Range(m.range(at: 1), in: keyLower),
               let n = Int(keyLower[r]) {
                framePairs.append((n, value))
                continue
            }
            // ── per_pixel_N ───────────────────────────────────────────────
            if let m = perPixelRE.firstMatch(in: keyLower, range: keyRange),
               let r = Range(m.range(at: 1), in: keyLower),
               let n = Int(keyLower[r]) {
                pixelPairs.append((n, value))
                continue
            }
            // ── wave_K_per_frame_N ────────────────────────────────────────
            if let m = waveFrameRE.firstMatch(in: keyLower, range: keyRange),
               let rK = Range(m.range(at: 1), in: keyLower),
               let rN = Range(m.range(at: 2), in: keyLower),
               let k  = Int(keyLower[rK]),
               let n  = Int(keyLower[rN]) {
                waveFramePairs[k, default: []].append((n, value))
                continue
            }
            // ── wave_K_per_point_N ────────────────────────────────────────
            if let m = wavePointRE.firstMatch(in: keyLower, range: keyRange),
               let rK = Range(m.range(at: 1), in: keyLower),
               let rN = Range(m.range(at: 2), in: keyLower),
               let k  = Int(keyLower[rK]),
               let n  = Int(keyLower[rN]) {
                wavePointPairs[k, default: []].append((n, value))
                continue
            }

            // Everything else goes into baseVals.
            preset.baseVals[key] = value
        }

        // Sort all equation arrays by their numeric index, then strip index.
        preset.initEqs     = initPairs.sorted  { $0.0 < $1.0 }.map { $0.1 }
        preset.perFrameEqs = framePairs.sorted { $0.0 < $1.0 }.map { $0.1 }
        preset.perPixelEqs = pixelPairs.sorted { $0.0 < $1.0 }.map { $0.1 }
        for (k, pairs) in waveFramePairs {
            preset.waveFrameEqs[k] = pairs.sorted { $0.0 < $1.0 }.map { $0.1 }
        }
        for (k, pairs) in wavePointPairs {
            preset.wavePointEqs[k] = pairs.sorted { $0.0 < $1.0 }.map { $0.1 }
        }

        return preset
    }
}

// MARK: - Serialiser

public struct PresetSerialiser {

    /// Serialise `preset` back to .milk text.
    public static func serialise(_ preset: MilkPreset) -> String {
        var lines: [String] = ["[preset00]"]

        // Base scalar values first (sorted for deterministic output).
        for key in preset.baseVals.keys.sorted() {
            lines.append("\(key)=\(preset.baseVals[key]!)")
        }

        // Equation blocks.
        for (i, eq) in preset.initEqs.enumerated() {
            lines.append("per_frame_init_\(i + 1)=\(eq)")
        }
        for (i, eq) in preset.perFrameEqs.enumerated() {
            lines.append("per_frame_\(i + 1)=\(eq)")
        }
        for (i, eq) in preset.perPixelEqs.enumerated() {
            lines.append("per_pixel_\(i + 1)=\(eq)")
        }
        for k in preset.waveFrameEqs.keys.sorted() {
            for (i, eq) in (preset.waveFrameEqs[k] ?? []).enumerated() {
                lines.append("wave_\(k)_per_frame_\(i + 1)=\(eq)")
            }
        }
        for k in preset.wavePointEqs.keys.sorted() {
            for (i, eq) in (preset.wavePointEqs[k] ?? []).enumerated() {
                lines.append("wave_\(k)_per_point_\(i + 1)=\(eq)")
            }
        }

        return lines.joined(separator: "\n")
    }
}

// MARK: - Q-injection helpers

/// Tools for modifying a parsed preset so that q1–q32 are initialised
/// to the live FFT values before Butterchurn runs its per_frame equations.
///
/// Two strategies are provided:
///
///  1. **Static bake** (`bakeQInits`): rewrites the `per_frame_init` block with
///     literal float values captured at a single moment.  Good for saving a
///     "snapshot" preset.
///
///  2. **JS live-patch** (`jsLivePatch`): returns a one-liner of JavaScript that
///     you call via `webView.evaluateJavaScript(_:)` every audio frame to push
///     the latest values into the running Butterchurn preset without reloading it.
public struct QInjector {

    // MARK: Strategy 1 — bake literal values into per_frame_init

    /// Replace the q-init lines in `preset.initEqs` with the given 32 values.
    /// Call `PresetSerialiser.serialise(_:)` afterwards to get a .milk string.
    public static func bakeQInits(into preset: inout MilkPreset,
                                   values: [Float]) {
        precondition(values.count == 32)
        // Remove any existing q assignments from init block.
        preset.initEqs.removeAll { eq in
            (1...32).contains(where: { eq.contains("q\($0)=") })
        }
        // Append one assignment per bin.
        for i in 0..<32 {
            preset.initEqs.append("q\(i + 1)=\(values[i]);")
        }
    }

    // MARK: Strategy 2 — live JavaScript patch (preferred for real-time use)

    /// Generates a JavaScript snippet that sets `q1`–`q32` on the active
    /// Butterchurn preset's `globalVars` object.  Evaluate this every audio
    /// callback (≈ 60 Hz) via `WKWebView.evaluateJavaScript(_:)`.
    ///
    /// Your `butterchurn_host.html` must expose the visualizer globally, e.g.:
    /// ```js
    /// window._viz = butterchurn.createVisualizer(…);
    /// ```
    public static func jsLivePatch(values: [Float],
                                   vizGlobal: String = "window._viz") -> String {
        precondition(values.count == 32)
        // Build the q array literal.
        let arr = values.map { String(format: "%.6f", $0) }.joined(separator: ",")
        return """
        (function(){
          var v=\(vizGlobal);
          if(!v||!v.preset)return;
          var g=v.preset.globalVars;
          if(!g)return;
          var q=[\(arr)];
          for(var i=0;i<32;i++) g['q'+(i+1)]=q[i];
        })();
        """
    }

    // MARK: Convenience: append q1=…;q32=… block to per_frame equations

    /// Prepend a q-assignment block to `preset.perFrameEqs` so the values are
    /// refreshed every frame from `per_frame_init` variables `_q1`–`_q32`.
    /// Use this when you want the preset file itself to be self-contained
    /// (values come from the init block, not from live JS injection).
    public static func appendQRefreshBlock(to preset: inout MilkPreset) {
        // Remove any existing refresh lines.
        preset.perFrameEqs.removeAll { $0.contains("=_q") }
        var block: [String] = []
        for i in 1...32 {
            block.append("q\(i)=_q\(i);")
        }
        preset.perFrameEqs.insert(contentsOf: block, at: 0)
    }
}

// MARK: - Quick usage example (not compiled into production — remove if needed)

#if DEBUG
private func _exampleUsage() throws {
    // 1. Parse the preset.
    let url = Bundle.main.url(forResource: "microtonal_warp", withExtension: "milk")!
    var preset = try PresetParser.load(from: url)

    // 2. Snapshot current FFT values into the init block (offline bake).
    let snapshot: [Float] = Array(repeating: 0, count: 32)   // replace with real FFT output
    QInjector.bakeQInits(into: &preset, values: snapshot)
    let milkText = PresetSerialiser.serialise(preset)
    print(milkText)

    // 3. OR: in your render loop, push values live to the WebView.
    //    let fftValues = fftProcessor.magnitudes(samples: ptr, count: n)
    //    let js = QInjector.jsLivePatch(values: fftValues)
    //    webView.evaluateJavaScript(js, completionHandler: nil)
}
#endif
