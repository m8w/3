import Foundation

// MARK: - Butterchurn JSON model
//
// This is the exact shape Butterchurn expects when you call
//   visualizer.loadPreset(preset, blendTime)
// in JavaScript.  All field names are snake_case to match the JS side.

public struct BCPreset: Codable {

    // ── Top-level ────────────────────────────────────────────────────────────
    public var name:         String
    public var baseVals:     BCBaseVals

    // Equation strings — each is the full block for that section, semicolons
    // separating individual assignments (same syntax as per_frame_N= lines).
    public var initEQs:      String
    public var perFrameEQs:  String
    public var perPixelEQs:  String

    // Optional waveform / shape objects (up to 8 waves, 4 shapes in Milkdrop).
    public var waves:        [BCWave]
    public var shapes:       [BCShape]

    public init(name: String = "Untitled") {
        self.name        = name
        self.baseVals    = BCBaseVals()
        self.initEQs     = ""
        self.perFrameEQs = ""
        self.perPixelEQs = ""
        self.waves       = []
        self.shapes      = []
    }
}

// MARK: - Base values (visual scalars)

public struct BCBaseVals: Codable {
    // Core warp / transform
    public var zoom:   Double = 1.0
    public var rot:    Double = 0.0
    public var cx:     Double = 0.5
    public var cy:     Double = 0.5
    public var dx:     Double = 0.0
    public var dy:     Double = 0.0
    public var warp:   Double = 1.0
    public var sx:     Double = 1.0
    public var sy:     Double = 1.0
    public var decay:  Double = 0.98

    // Video echo (Milkdrop's composite layer)
    public var echo_zoom:   Double = 1.0
    public var echo_alpha:  Double = 0.0
    public var echo_orient: Int    = 0

    // Wave
    public var wave_mode:      Int    = 0
    public var additiveWaves:  Int    = 0
    public var waveDots:       Int    = 0
    public var waveThick:      Int    = 0
    public var wave_r:         Double = 0.5
    public var wave_g:         Double = 0.5
    public var wave_b:         Double = 0.5
    public var wave_x:         Double = 0.5
    public var wave_y:         Double = 0.5
    public var wave_a:         Double = 0.8
    public var wave_scale:     Double = 1.0
    public var wave_smoothing: Double = 0.75
    public var wave_mystery:   Double = 0.0

    // Outer / inner border
    public var ob_size: Double = 0.01
    public var ob_r:    Double = 0.0
    public var ob_g:    Double = 0.0
    public var ob_b:    Double = 0.0
    public var ob_a:    Double = 0.0
    public var ib_size: Double = 0.01
    public var ib_r:    Double = 0.0
    public var ib_g:    Double = 0.0
    public var ib_b:    Double = 0.0
    public var ib_a:    Double = 0.0

    // Motion vectors
    public var mv_x:  Double = 12.0
    public var mv_y:  Double = 9.0
    public var mv_l:  Double = 0.9
    public var mv_r:  Double = 0.0
    public var mv_g:  Double = 0.0
    public var mv_b:  Double = 0.0
    public var mv_a:  Double = 0.0
    public var mv_dx: Double = 0.0
    public var mv_dy: Double = 0.0

    // Image / post-processing
    public var gammaAdj:           Double = 1.0
    public var fShader:            Double = 0.0
    public var bTexWrap:           Int    = 1
    public var bDarkenCenter:      Int    = 0
    public var bRedBlueStereo:     Int    = 0
    public var bBrighten:          Int    = 0
    public var bDarken:            Int    = 0
    public var bSolarize:          Int    = 0
    public var bInvert:            Int    = 0
    public var modWaveAlphaByVol:  Int    = 0
    public var modWaveAlphaStart:  Double = 0.75
    public var modWaveAlphaEnd:    Double = 0.95
    public var maximizeWaveColor:  Int    = 0
    public var warpAnimSpeed:      Double = 1.0
    public var warpScale:          Double = 1.33
    public var zoomExponent:       Double = 1.0
    public var fRating:            Double = 3.0

    public init() {}
}

// MARK: - Wave object

public struct BCWave: Codable {
    public var enabled:    Int    = 1
    public var bSpectrum:  Int    = 0
    public var bUseDots:   Int    = 0
    public var bDrawThick: Int    = 0
    public var bAdditive:  Int    = 0
    public var samples:    Int    = 512
    public var sep:        Int    = 0
    public var smoothing:  Double = 0.5
    public var scaling:    Double = 1.0
    public var r:          Double = 1.0
    public var g:          Double = 1.0
    public var b:          Double = 1.0
    public var a:          Double = 1.0

    // Equation blocks for this wave.
    public var per_frame_init_eqs_compat: String = ""
    public var per_frame_eqs_compat:      String = ""
    public var per_point_eqs_compat:      String = ""

    public init() {}
}

// MARK: - Shape object

public struct BCShape: Codable {
    public var enabled:      Int    = 1
    public var sides:        Int    = 4
    public var additive:     Int    = 0
    public var thickOutline: Int    = 0
    public var textured:     Int    = 0
    public var num_inst:     Int    = 1
    public var x:            Double = 0.5
    public var y:            Double = 0.5
    public var rad:          Double = 0.1
    public var ang:          Double = 0.0
    public var r:            Double = 1.0
    public var g:            Double = 0.0
    public var b:            Double = 0.0
    public var a:            Double = 1.0
    public var r2:           Double = 1.0
    public var g2:           Double = 1.0
    public var b2:           Double = 0.0
    public var a2:           Double = 0.0
    public var border_r:     Double = 1.0
    public var border_g:     Double = 1.0
    public var border_b:     Double = 1.0
    public var border_a:     Double = 0.1

    public var per_frame_init_eqs_compat: String = ""
    public var per_frame_eqs_compat:      String = ""
    public var per_point_eqs_compat:      String = ""

    public init() {}
}

// MARK: - Converter

/// Converts a parsed `MilkPreset` to a `BCPreset` (Butterchurn JSON format).
public struct MilkToJSON {

    // ── Scalar mapping: .milk key → BCBaseVals keyPath ───────────────────────
    // .milk files use a mix of camelCase (fDecay, fGammaAdj) and lowercase
    // (zoom, rot, wave_r) keys.  We normalise all keys to lowercase on ingest.
    private static let scalarMap: [String: WritableKeyPath<BCBaseVals, Double>] = [
        "zoom":                  \.zoom,
        "rot":                   \.rot,
        "cx":                    \.cx,
        "cy":                    \.cy,
        "dx":                    \.dx,
        "dy":                    \.dy,
        "warp":                  \.warp,
        "sx":                    \.sx,
        "sy":                    \.sy,
        "fdecay":                \.decay,
        "fvideoeachozoom":       \.echo_zoom,   // fVideoEchoZoom
        "fvideoechoalpha":       \.echo_alpha,
        "fgammaadj":             \.gammaAdj,
        "fshader":               \.fShader,
        "fwavealpha":            \.wave_a,
        "fwavescale":            \.wave_scale,
        "fwavesmoothing":        \.wave_smoothing,
        "fwaveparam":            \.wave_mystery,
        "fwarpanimspeed":        \.warpAnimSpeed,
        "fwarpscale":            \.warpScale,
        "fzoomexponent":         \.zoomExponent,
        "frating":               \.fRating,
        "fmodwavealphastart":    \.modWaveAlphaStart,
        "fmodwavealphaend":      \.modWaveAlphaEnd,
        "wave_r":                \.wave_r,
        "wave_g":                \.wave_g,
        "wave_b":                \.wave_b,
        "wave_x":                \.wave_x,
        "wave_y":                \.wave_y,
        "ob_size":               \.ob_size,
        "ob_r":                  \.ob_r,
        "ob_g":                  \.ob_g,
        "ob_b":                  \.ob_b,
        "ob_a":                  \.ob_a,
        "ib_size":               \.ib_size,
        "ib_r":                  \.ib_r,
        "ib_g":                  \.ib_g,
        "ib_b":                  \.ib_b,
        "ib_a":                  \.ib_a,
        "nmotionvectorsx":       \.mv_x,
        "nmotionvectorsy":       \.mv_y,
        "mv_dx":                 \.mv_dx,
        "mv_dy":                 \.mv_dy,
        "mv_l":                  \.mv_l,
        "mv_r":                  \.mv_r,
        "mv_g":                  \.mv_g,
        "mv_b":                  \.mv_b,
        "mv_a":                  \.mv_a,
    ]

    private static let intMap: [String: WritableKeyPath<BCBaseVals, Int>] = [
        "nvideoechoorientation":    \.echo_orient,
        "nwavemode":                \.wave_mode,
        "badditivewaves":           \.additiveWaves,
        "bwavedots":                \.waveDots,
        "bwavethick":               \.waveThick,
        "btexwrap":                 \.bTexWrap,
        "bdarkencenter":            \.bDarkenCenter,
        "bredbluestero":            \.bRedBlueStereo,
        "bbrighten":                \.bBrighten,
        "bdarken":                  \.bDarken,
        "bsolarize":                \.bSolarize,
        "binvert":                  \.bInvert,
        "bmodwavealphabyvolume":    \.modWaveAlphaByVol,
        "bmaximizewavecolor":       \.maximizeWaveColor,
    ]

    // ── Public convert API ───────────────────────────────────────────────────

    /// Convert a `MilkPreset` (from `PresetParser`) to a Butterchurn `BCPreset`.
    /// - Parameter preset:  Parsed .milk preset.
    /// - Parameter name:    Optional display name; defaults to the preset's
    ///                      filename (caller should pass it).
    public static func convert(_ milk: MilkPreset, name: String = "Converted") -> BCPreset {
        var bc   = BCPreset(name: name)

        // ── 1. Scalar base values ────────────────────────────────────────────
        for (rawKey, rawVal) in milk.baseVals {
            let key = rawKey.lowercased()
                           .replacingOccurrences(of: "_", with: "")  // strip underscores for looser matching

            // Try Double map.
            if let kp = scalarMap[key] ?? scalarMap[rawKey.lowercased()],
               let v  = Double(rawVal) {
                bc.baseVals[keyPath: kp] = v
                continue
            }
            // Try Int map.
            if let kp = intMap[key] ?? intMap[rawKey.lowercased()],
               let v  = Int(rawVal) {
                bc.baseVals[keyPath: kp] = v
                continue
            }
        }

        // ── 2. Equation blocks ───────────────────────────────────────────────
        bc.initEQs     = joinEQs(milk.initEqs)
        bc.perFrameEQs = joinEQs(milk.perFrameEqs)
        bc.perPixelEQs = joinEQs(milk.perPixelEqs)

        // ── 3. Waves ─────────────────────────────────────────────────────────
        for idx in milk.waveFrameEqs.keys.sorted() {
            var w = buildWave(index: idx, milk: milk)
            bc.waves.append(w)
        }
        // Also create wave entries for point-only waves.
        for idx in milk.wavePointEqs.keys.sorted() {
            if bc.waves.indices.contains(idx - 1) { continue }
            bc.waves.append(buildWave(index: idx, milk: milk))
        }

        // ── 4. Shapes ────────────────────────────────────────────────────────
        bc.shapes = buildShapes(from: milk.baseVals)

        return bc
    }

    /// Serialise a `BCPreset` to pretty-printed JSON data.
    public static func toJSON(_ preset: BCPreset, pretty: Bool = true) throws -> Data {
        let enc = JSONEncoder()
        enc.outputFormatting = pretty ? [.prettyPrinted, .sortedKeys] : []
        return try enc.encode(preset)
    }

    /// Full pipeline: .milk file URL → JSON string.
    public static func convertFile(at url: URL) throws -> String {
        let milk = try PresetParser.load(from: url)
        let name = url.deletingPathExtension().lastPathComponent
        let bc   = convert(milk, name: name)
        let data = try toJSON(bc)
        return String(decoding: data, as: UTF8.self)
    }

    /// Full pipeline: .milk string → JSON string.
    public static func convertString(_ milkText: String,
                                     name: String = "Converted") throws -> String {
        let milk = try PresetParser.parse(string: milkText)
        let bc   = convert(milk, name: name)
        let data = try toJSON(bc)
        return String(decoding: data, as: UTF8.self)
    }

    // MARK: - Private helpers

    /// Join an array of equation strings into one semicolon-separated block.
    private static func joinEQs(_ eqs: [String]) -> String {
        eqs
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
    }

    /// Build a `BCWave` from .milk wave_K_* fields.
    private static func buildWave(index: Int, milk: MilkPreset) -> BCWave {
        var w = BCWave()
        let prefix = "wave_\(index)_"

        for (key, val) in milk.baseVals where key.lowercased().hasPrefix(prefix) {
            let sub = key.dropFirst(prefix.count).lowercased()
            switch sub {
            case "enabled":    w.enabled    = Int(val) ?? 1
            case "bspectrum":  w.bSpectrum  = Int(val) ?? 0
            case "busedots":   w.bUseDots   = Int(val) ?? 0
            case "bdrawthick": w.bDrawThick = Int(val) ?? 0
            case "badditive":  w.bAdditive  = Int(val) ?? 0
            case "samples":    w.samples    = Int(val) ?? 512
            case "sep":        w.sep        = Int(val) ?? 0
            case "smoothing":  w.smoothing  = Double(val) ?? 0.5
            case "scaling":    w.scaling    = Double(val) ?? 1.0
            case "r":          w.r          = Double(val) ?? 1.0
            case "g":          w.g          = Double(val) ?? 1.0
            case "b":          w.b          = Double(val) ?? 1.0
            case "a":          w.a          = Double(val) ?? 1.0
            default: break
            }
        }

        w.per_frame_init_eqs_compat = ""   // .milk doesn't have per-wave init
        w.per_frame_eqs_compat      = joinEQs(milk.waveFrameEqs[index] ?? [])
        w.per_point_eqs_compat      = joinEQs(milk.wavePointEqs[index] ?? [])

        return w
    }

    /// Extract shape_K_* fields from baseVals into BCShape objects.
    private static func buildShapes(from vals: [String: String]) -> [BCShape] {
        // Collect unique shape indices first.
        var indices = Set<Int>()
        let re = try! NSRegularExpression(pattern: #"^shape_(\d+)_"#)
        for key in vals.keys {
            let range = NSRange(key.startIndex..., in: key)
            if let m = re.firstMatch(in: key, range: range),
               let r = Range(m.range(at: 1), in: key),
               let n = Int(key[r]) {
                indices.insert(n)
            }
        }

        return indices.sorted().map { idx in
            var s = BCShape()
            let prefix = "shape_\(idx)_"
            for (key, val) in vals where key.lowercased().hasPrefix(prefix) {
                let sub = key.dropFirst(prefix.count).lowercased()
                switch sub {
                case "enabled":      s.enabled      = Int(val) ?? 1
                case "sides":        s.sides        = Int(val) ?? 4
                case "additive":     s.additive     = Int(val) ?? 0
                case "thickoutline": s.thickOutline = Int(val) ?? 0
                case "textured":     s.textured     = Int(val) ?? 0
                case "num_inst":     s.num_inst     = Int(val) ?? 1
                case "x":            s.x            = Double(val) ?? 0.5
                case "y":            s.y            = Double(val) ?? 0.5
                case "rad":          s.rad          = Double(val) ?? 0.1
                case "ang":          s.ang          = Double(val) ?? 0.0
                case "r":            s.r            = Double(val) ?? 1.0
                case "g":            s.g            = Double(val) ?? 0.0
                case "b":            s.b            = Double(val) ?? 0.0
                case "a":            s.a            = Double(val) ?? 1.0
                case "r2":           s.r2           = Double(val) ?? 1.0
                case "g2":           s.g2           = Double(val) ?? 1.0
                case "b2":           s.b2           = Double(val) ?? 0.0
                case "a2":           s.a2           = Double(val) ?? 0.0
                case "border_r":     s.border_r     = Double(val) ?? 1.0
                case "border_g":     s.border_g     = Double(val) ?? 1.0
                case "border_b":     s.border_b     = Double(val) ?? 1.0
                case "border_a":     s.border_a     = Double(val) ?? 0.1
                default: break
                }
            }
            return s
        }
    }
}

// MARK: - Reverse: Butterchurn JSON → .milk

/// Converts a Butterchurn JSON preset back to .milk format.
/// Useful for editing a JSON preset in Milkdrop and round-tripping.
public struct JSONToMilk {

    /// Decode JSON data and produce a .milk string.
    public static func convert(_ jsonData: Data, name: String? = nil) throws -> String {
        let bc   = try JSONDecoder().decode(BCPreset.self, from: jsonData)
        return toMilk(bc, name: name ?? bc.name)
    }

    public static func toMilk(_ bc: BCPreset, name: String) -> String {
        var lines = ["[preset00]"]

        // ── Base vals ────────────────────────────────────────────────────────
        let bv = bc.baseVals
        func d(_ v: Double) -> String { String(format: "%f", v) }

        lines += [
            "fRating=\(d(bv.fRating))",
            "fGammaAdj=\(d(bv.gammaAdj))",
            "fDecay=\(d(bv.decay))",
            "fVideoEchoZoom=\(d(bv.echo_zoom))",
            "fVideoEchoAlpha=\(d(bv.echo_alpha))",
            "nVideoEchoOrientation=\(bv.echo_orient)",
            "nWaveMode=\(bv.wave_mode)",
            "bAdditiveWaves=\(bv.additiveWaves)",
            "bWaveDots=\(bv.waveDots)",
            "bWaveThick=\(bv.waveThick)",
            "bModWaveAlphaByVolume=\(bv.modWaveAlphaByVol)",
            "bMaximizeWaveColor=\(bv.maximizeWaveColor)",
            "bTexWrap=\(bv.bTexWrap)",
            "bDarkenCenter=\(bv.bDarkenCenter)",
            "bRedBlueStereo=\(bv.bRedBlueStereo)",
            "bBrighten=\(bv.bBrighten)",
            "bDarken=\(bv.bDarken)",
            "bSolarize=\(bv.bSolarize)",
            "bInvert=\(bv.bInvert)",
            "fWaveAlpha=\(d(bv.wave_a))",
            "fWaveScale=\(d(bv.wave_scale))",
            "fWaveSmoothing=\(d(bv.wave_smoothing))",
            "fWaveParam=\(d(bv.wave_mystery))",
            "fWarpAnimSpeed=\(d(bv.warpAnimSpeed))",
            "fWarpScale=\(d(bv.warpScale))",
            "fZoomExponent=\(d(bv.zoomExponent))",
            "fShader=\(d(bv.fShader))",
            "zoom=\(d(bv.zoom))",
            "rot=\(d(bv.rot))",
            "cx=\(d(bv.cx))",
            "cy=\(d(bv.cy))",
            "dx=\(d(bv.dx))",
            "dy=\(d(bv.dy))",
            "warp=\(d(bv.warp))",
            "sx=\(d(bv.sx))",
            "sy=\(d(bv.sy))",
            "wave_r=\(d(bv.wave_r))",
            "wave_g=\(d(bv.wave_g))",
            "wave_b=\(d(bv.wave_b))",
            "wave_x=\(d(bv.wave_x))",
            "wave_y=\(d(bv.wave_y))",
            "ob_size=\(d(bv.ob_size))",
            "ob_r=\(d(bv.ob_r))",
            "ob_g=\(d(bv.ob_g))",
            "ob_b=\(d(bv.ob_b))",
            "ob_a=\(d(bv.ob_a))",
            "ib_size=\(d(bv.ib_size))",
            "ib_r=\(d(bv.ib_r))",
            "ib_g=\(d(bv.ib_g))",
            "ib_b=\(d(bv.ib_b))",
            "ib_a=\(d(bv.ib_a))",
            "nMotionVectorsX=\(d(bv.mv_x))",
            "nMotionVectorsY=\(d(bv.mv_y))",
            "mv_dx=\(d(bv.mv_dx))",
            "mv_dy=\(d(bv.mv_dy))",
            "mv_l=\(d(bv.mv_l))",
            "mv_r=\(d(bv.mv_r))",
            "mv_g=\(d(bv.mv_g))",
            "mv_b=\(d(bv.mv_b))",
            "mv_a=\(d(bv.mv_a))",
        ]

        // ── Equation blocks → numbered lines ─────────────────────────────────
        func emitEQs(prefix: String, block: String) {
            let stmts = block
                .components(separatedBy: CharacterSet(charactersIn: ";\n"))
                .map    { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty && !$0.hasPrefix(";") && !$0.hasPrefix("//") }
            for (i, stmt) in stmts.enumerated() {
                lines.append("\(prefix)\(i + 1)=\(stmt);")
            }
        }

        emitEQs(prefix: "per_frame_init_", block: bc.initEQs)
        emitEQs(prefix: "per_frame_",      block: bc.perFrameEQs)
        emitEQs(prefix: "per_pixel_",      block: bc.perPixelEQs)

        // ── Waves ─────────────────────────────────────────────────────────────
        for (k, w) in bc.waves.enumerated() {
            let p = "wave_\(k + 1)_"
            lines += [
                "\(p)enabled=\(w.enabled)",
                "\(p)bSpectrum=\(w.bSpectrum)",
                "\(p)bUseDots=\(w.bUseDots)",
                "\(p)bDrawThick=\(w.bDrawThick)",
                "\(p)bAdditive=\(w.bAdditive)",
                "\(p)samples=\(w.samples)",
                "\(p)sep=\(w.sep)",
                "\(p)smoothing=\(d(w.smoothing))",
                "\(p)scaling=\(d(w.scaling))",
                "\(p)r=\(d(w.r))",
                "\(p)g=\(d(w.g))",
                "\(p)b=\(d(w.b))",
                "\(p)a=\(d(w.a))",
            ]
            emitEQs(prefix: "\(p)per_frame_", block: w.per_frame_eqs_compat)
            emitEQs(prefix: "\(p)per_point_", block: w.per_point_eqs_compat)
        }

        // ── Shapes ────────────────────────────────────────────────────────────
        for (k, s) in bc.shapes.enumerated() {
            let p = "shape_\(k + 1)_"
            lines += [
                "\(p)enabled=\(s.enabled)",
                "\(p)sides=\(s.sides)",
                "\(p)additive=\(s.additive)",
                "\(p)thickOutline=\(s.thickOutline)",
                "\(p)textured=\(s.textured)",
                "\(p)num_inst=\(s.num_inst)",
                "\(p)x=\(d(s.x))", "\(p)y=\(d(s.y))",
                "\(p)rad=\(d(s.rad))", "\(p)ang=\(d(s.ang))",
                "\(p)r=\(d(s.r))", "\(p)g=\(d(s.g))",
                "\(p)b=\(d(s.b))", "\(p)a=\(d(s.a))",
                "\(p)r2=\(d(s.r2))", "\(p)g2=\(d(s.g2))",
                "\(p)b2=\(d(s.b2))", "\(p)a2=\(d(s.a2))",
                "\(p)border_r=\(d(s.border_r))",
                "\(p)border_g=\(d(s.border_g))",
                "\(p)border_b=\(d(s.border_b))",
                "\(p)border_a=\(d(s.border_a))",
            ]
            emitEQs(prefix: "\(p)per_frame_", block: s.per_frame_eqs_compat)
            emitEQs(prefix: "\(p)per_point_", block: s.per_point_eqs_compat)
        }

        return lines.joined(separator: "\n")
    }
}
