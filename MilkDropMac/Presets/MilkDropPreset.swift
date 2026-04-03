// MilkDropPreset.swift — Model and parser for .milk and .milk2 preset files

import Foundation
import UniformTypeIdentifiers

// MARK: - UTType extensions for .milk files

extension UTType {
    static let milk  = UTType(filenameExtension: "milk",  conformingTo: .text)!
    static let milk2 = UTType(filenameExtension: "milk2", conformingTo: .text)!
}

// MARK: - Preset model

struct MilkDropPreset: Identifiable, Hashable, Codable {
    let id: UUID
    var name: String
    var url: URL?
    var data: String          // Raw .milk text
    var isFavorite: Bool = false
    var isDoublePreset: Bool = false  // .milk2
    var tags: [String] = []
    var rating: Int = 0       // 0–5 stars
    var author: String = ""
    var dateModified: Date = .now

    // Parsed parameters (lazy, populated on demand)
    var parameters: PresetParameters?

    init(id: UUID = .init(), name: String, url: URL? = nil, data: String) {
        self.id   = id
        self.name = name
        self.url  = url
        self.data = data
    }

    // MARK: - Factory

    static func makeDoublePreset(a: MilkDropPreset, b: MilkDropPreset) -> MilkDropPreset {
        // MilkDrop 3 .milk2: concatenates two presets with a separator
        let combined = """
[preset_a]
\(a.data)

[preset_b]
\(b.data)
"""
        return MilkDropPreset(
            name: "\(a.name) ⟷ \(b.name)",
            data: combined
        ).with(\.isDoublePreset, value: true)
    }

    func with<V>(_ keyPath: WritableKeyPath<MilkDropPreset, V>, value: V) -> MilkDropPreset {
        var copy = self
        copy[keyPath: keyPath] = value
        return copy
    }

    // MARK: - Parsing

    mutating func parseParameters() {
        parameters = PresetParser.parse(data)
    }

    var fileExtension: String { isDoublePreset ? "milk2" : "milk" }
}

// MARK: - Preset parameters (the actual MilkDrop variables)

struct PresetParameters: Codable, Equatable, Hashable {
    // General
    var rating: Float           = 3.0
    var gamma:  Float           = 1.0
    var videoEchoAlpha: Float   = 0.0
    var videoEchoZoom:  Float   = 1.0
    var videoEchoOrientation: Int = 0
    var fVideoEchoAlpha: Float  = 0.0

    // Warp / feedback
    var warpScale:       Float  = 1.0
    var warpSpeed:       Float  = 1.0
    var zoomAmount:      Float  = 1.0    // 1 = no zoom
    var rotatAmount:     Float  = 0.0    // radians per frame
    var warpX:          Float   = 0.0
    var warpY:          Float   = 0.0
    var centreX:        Float   = 0.5
    var centreY:        Float   = 0.5
    var szx:            Float   = 1.0    // X scale
    var szy:            Float   = 1.0    // Y scale
    var decay:          Float   = 0.98   // feedback decay

    // Color
    var r:   Float = 0, g: Float = 0, b: Float = 0  // Ambient color
    var a:   Float = 0                               // Ambient alpha

    // Equations (raw text blocks)
    var perFrameInit:    String = ""
    var perFrame:        [String] = []    // up to 64 per_frame_N= lines
    var perVertex:       [String] = []    // up to 64 per_pixel_N= lines
    var warpHLSL:        String = ""      // MilkDrop 2+: warp pixel shader
    var compHLSL:        String = ""      // MilkDrop 2+: composite pixel shader

    // Waves (up to 16 in MilkDrop 3)
    var waves: [PresetWave] = []

    // Shapes (up to 16 in MilkDrop 3)
    var shapes: [PresetShape] = []

    // MilkDrop 3 extras
    var useDoublePreset: Bool = false
    var hardcutEnabled:  Bool = false
    var hardcutBass:     Float = 0.8
    var hardcutTreble:   Float = 0.5
    var hardcutDelay:    Float = 3.0

    // Transition
    var blendType: TransitionBlend = .zoom

    enum TransitionBlend: String, Codable, CaseIterable, Equatable, Hashable {
        case zoom, side, plasma, cercle, plasma2, plasma3, snail,
             triangle, donuts, corner, patches, checkerboard, bubbles, stars, cisor
    }
}

// MARK: - Wave definition

struct PresetWave: Codable, Identifiable, Equatable, Hashable {
    var id: Int
    var enabled: Bool    = false
    var samples: Int     = 512
    var sep:     Int     = 0
    var scaling: Float   = 1.0
    var smoothing: Float = 0.5
    var r: Float = 1, g: Float = 1, b: Float = 1, a: Float = 1
    var useDots:       Bool = false
    var drawThick:     Bool = false
    var additive:      Bool = false
    var perPoint:      [String] = []
}

// MARK: - Shape definition

struct PresetShape: Codable, Identifiable, Equatable, Hashable {
    var id: Int
    var enabled:    Bool  = false
    var sides:      Int   = 4
    var additive:   Bool  = false
    var thickOutline: Bool = false
    var textured:   Bool  = false
    var x:          Float = 0.5
    var y:          Float = 0.5
    var radius:     Float = 0.1
    var ang:        Float = 0.0
    var tex_ang:    Float = 0.0
    var tex_zoom:   Float = 1.0
    var r: Float = 1, g: Float = 1, b: Float = 1, a: Float = 1
    var r2: Float = 1, g2: Float = 1, b2: Float = 1, a2: Float = 1
    var border_r: Float = 1, border_g: Float = 1, border_b: Float = 1, border_a: Float = 0.5
    var perFrame:   [String] = []
}

// MARK: - Parser

enum PresetParser {
    static func parse(_ text: String) -> PresetParameters {
        var params = PresetParameters()
        let lines = text.components(separatedBy: .newlines)
        var waveDict:  [Int: [String: String]] = [:]
        var shapeDict: [Int: [String: String]] = [:]

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty, !trimmed.hasPrefix("["),
                  let eqIdx = trimmed.firstIndex(of: "=") else { continue }

            let key   = String(trimmed[..<eqIdx]).lowercased()
            let value = String(trimmed[trimmed.index(after: eqIdx)...])

            // Per-frame equations
            if key.hasPrefix("per_frame_") && !key.hasPrefix("per_frame_init") {
                params.perFrame.append(value)
                continue
            }
            if key == "per_frame_init_1" || key == "per_frame_init" {
                params.perFrameInit = value
                continue
            }
            // Per-vertex (pixel) equations
            if key.hasPrefix("per_pixel_") {
                params.perVertex.append(value)
                continue
            }
            // HLSL shaders
            if key.hasPrefix("warp_") && key.hasSuffix("_hlsl") {
                params.warpHLSL += value + "\n"
                continue
            }
            if key.hasPrefix("comp_") && key.hasSuffix("_hlsl") {
                params.compHLSL += value + "\n"
                continue
            }
            // Waves: wave_N_XXX=
            if key.hasPrefix("wave_"), let (waveIdx, waveKey) = parseIndexedKey(key, prefix: "wave_") {
                waveDict[waveIdx, default: [:]][waveKey] = value
                continue
            }
            // Shapes: shape_N_XXX=
            if key.hasPrefix("shape_"), let (shapeIdx, shapeKey) = parseIndexedKey(key, prefix: "shape_") {
                shapeDict[shapeIdx, default: [:]][shapeKey] = value
                continue
            }

            // Scalar parameters
            switch key {
            case "frating":            params.rating           = Float(value) ?? 3
            case "fgammadj":           params.gamma            = Float(value) ?? 1
            case "fdecay":             params.decay            = Float(value) ?? 0.98
            case "fvideoechodecay":    params.videoEchoAlpha   = Float(value) ?? 0    // was wrongly mapped to decay
            case "fzoom":              params.zoomAmount       = Float(value) ?? 1
            case "frot":               params.rotatAmount      = Float(value) ?? 0
            case "fwarpscale":         params.warpScale        = Float(value) ?? 1
            case "fwarpanimspeed":     params.warpSpeed        = Float(value) ?? 1
            case "fxcenter", "fxcent": params.centreX          = Float(value) ?? 0.5
            case "fycenter", "fycent": params.centreY          = Float(value) ?? 0.5
            case "frotcx":             params.centreX          = Float(value) ?? 0.5
            case "frotcy":             params.centreY          = Float(value) ?? 0.5
            case "szx":                params.szx              = Float(value) ?? 1
            case "szy":                params.szy              = Float(value) ?? 1
            case "fvideoechodecayalpha", "fvideoechodecay2": params.videoEchoAlpha = Float(value) ?? 0
            case "fvideoechodecayzoom":  params.videoEchoZoom  = Float(value) ?? 1
            case "ivideoechodecayorientation": params.videoEchoOrientation = Int(value) ?? 0
            case "bgammadj":
                break  // legacy
            default: break
            }
        }

        // Build wave objects
        for (idx, dict) in waveDict {
            // Helper: try multiple key variants (real presets use bEnabled, nSamples, fR, etc.)
            func ival(_ keys: String...) -> Int? {
                for k in keys { if let v = dict[k].flatMap({ Int($0) }) { return v } }
                return nil
            }
            func fval(_ keys: String...) -> Float? {
                for k in keys { if let v = dict[k].flatMap({ Float($0) }) { return v } }
                return nil
            }
            var w = PresetWave(id: idx)
            w.enabled   = ival("enabled",  "benabled",   "nenabled")  == 1
            w.samples   = ival("samples",  "nsamples")                ?? 512
            w.sep       = ival("sep",      "nsep")                    ?? 0
            w.scaling   = fval("scaling",  "fscaling")                ?? 1
            w.smoothing = fval("smoothing","fsmoothing")              ?? 0.5
            w.r         = fval("r",        "fr")                      ?? 1
            w.g         = fval("g",        "fg")                      ?? 1
            w.b         = fval("b",        "fb")                      ?? 1
            w.a         = fval("a",        "fa")                      ?? 1
            w.useDots   = ival("usedots",  "busedots",   "nusedots")  == 1
            w.drawThick = ival("drawthick","bdrawthick", "ndrawthick") == 1
            w.additive  = ival("additive", "badditive",  "nadditive")  == 1
            // Per-point equations
            var i = 1
            while let eq = dict["per_point_\(i)"] {
                w.perPoint.append(eq)
                i += 1
            }
            params.waves.append(w)
        }
        params.waves.sort { $0.id < $1.id }

        // Build shape objects
        for (idx, dict) in shapeDict {
            func ival(_ keys: String...) -> Int? {
                for k in keys { if let v = dict[k].flatMap({ Int($0) }) { return v } }
                return nil
            }
            func fval(_ keys: String...) -> Float? {
                for k in keys { if let v = dict[k].flatMap({ Float($0) }) { return v } }
                return nil
            }
            var s = PresetShape(id: idx)
            s.enabled      = ival("enabled",     "benabled",      "nenabled")   == 1
            s.sides        = ival("sides",        "nsides")                     ?? 4
            s.additive     = ival("additive",     "badditive")                  == 1
            s.thickOutline = ival("thickoutline", "bthickoutline")              == 1
            s.textured     = ival("textured",     "btextured")                  == 1
            s.x            = fval("x",            "fx")                         ?? 0.5
            s.y            = fval("y",            "fy")                         ?? 0.5
            s.radius       = fval("radius",       "fradius")                    ?? 0.1
            s.ang          = fval("ang",          "fang")                       ?? 0
            s.tex_ang      = fval("tex_ang",      "ftex_ang")                   ?? 0
            s.tex_zoom     = fval("tex_zoom",     "ftex_zoom")                  ?? 1
            s.r  = fval("r",  "fr")  ?? 1;  s.g  = fval("g",  "fg")  ?? 1
            s.b  = fval("b",  "fb")  ?? 1;  s.a  = fval("a",  "fa")  ?? 1
            s.r2 = fval("r2", "fr2") ?? 1;  s.g2 = fval("g2", "fg2") ?? 1
            s.b2 = fval("b2", "fb2") ?? 1;  s.a2 = fval("a2", "fa2") ?? 1
            s.border_r = fval("border_r", "fborder_r") ?? 1
            s.border_g = fval("border_g", "fborder_g") ?? 1
            s.border_b = fval("border_b", "fborder_b") ?? 1
            s.border_a = fval("border_a", "fborder_a") ?? 0.5
            var i = 1
            while let eq = dict["per_frame_\(i)"] {
                s.perFrame.append(eq)
                i += 1
            }
            params.shapes.append(s)
        }
        params.shapes.sort { $0.id < $1.id }

        return params
    }

    private static func parseIndexedKey(_ key: String, prefix: String) -> (Int, String)? {
        let remainder = String(key.dropFirst(prefix.count))
        guard let underscoreIdx = remainder.firstIndex(of: "_") else { return nil }
        let indexStr = String(remainder[..<underscoreIdx])
        let subKey   = String(remainder[remainder.index(after: underscoreIdx)...])
        guard let index = Int(indexStr) else { return nil }
        return (index, subKey)
    }

    // MARK: - Serializer (write .milk file)

    static func serialize(_ params: PresetParameters, name: String) -> String {
        var lines: [String] = ["[preset00]", "fRating=\(params.rating)"]
        lines.append("fGammaAdj=\(params.gamma)")
        lines.append("fDecay=\(params.decay)")
        lines.append("fZoom=\(params.zoomAmount)")
        lines.append("fRot=\(params.rotatAmount)")
        lines.append("fWarpScale=\(params.warpScale)")
        lines.append("fWarpAnimSpeed=\(params.warpSpeed)")
        lines.append("fXCentre=\(params.centreX)")
        lines.append("fYCentre=\(params.centreY)")
        lines.append("szx=\(params.szx)")
        lines.append("szy=\(params.szy)")
        lines.append("fVideoEchoDecayAlpha=\(params.videoEchoAlpha)")
        lines.append("fVideoEchoDecayZoom=\(params.videoEchoZoom)")
        lines.append("iVideoEchoDecayOrientation=\(params.videoEchoOrientation)")

        // Per-frame equations
        if !params.perFrameInit.isEmpty {
            lines.append("per_frame_init_1=\(params.perFrameInit)")
        }
        for (i, eq) in params.perFrame.enumerated() {
            lines.append("per_frame_\(i + 1)=\(eq)")
        }
        for (i, eq) in params.perVertex.enumerated() {
            lines.append("per_pixel_\(i + 1)=\(eq)")
        }

        // HLSL shaders (split into lines)
        if !params.warpHLSL.isEmpty {
            let shaderLines = params.warpHLSL.components(separatedBy: .newlines)
            for (i, l) in shaderLines.enumerated() {
                lines.append("warp_\(i + 1)_hlsl=\(l)")
            }
        }
        if !params.compHLSL.isEmpty {
            let shaderLines = params.compHLSL.components(separatedBy: .newlines)
            for (i, l) in shaderLines.enumerated() {
                lines.append("comp_\(i + 1)_hlsl=\(l)")
            }
        }

        // Waves
        for wave in params.waves where wave.enabled {
            let p = "wave_\(wave.id)_"
            lines.append("\(p)enabled=1")
            lines.append("\(p)samples=\(wave.samples)")
            lines.append("\(p)sep=\(wave.sep)")
            lines.append("\(p)scaling=\(wave.scaling)")
            lines.append("\(p)smoothing=\(wave.smoothing)")
            lines.append("\(p)r=\(wave.r)"); lines.append("\(p)g=\(wave.g)")
            lines.append("\(p)b=\(wave.b)"); lines.append("\(p)a=\(wave.a)")
            lines.append("\(p)usedots=\(wave.useDots ? 1 : 0)")
            lines.append("\(p)drawthick=\(wave.drawThick ? 1 : 0)")
            lines.append("\(p)additive=\(wave.additive ? 1 : 0)")
            for (i, eq) in wave.perPoint.enumerated() {
                lines.append("\(p)per_point_\(i + 1)=\(eq)")
            }
        }

        // Shapes
        for shape in params.shapes where shape.enabled {
            let p = "shape_\(shape.id)_"
            lines.append("\(p)enabled=1")
            lines.append("\(p)sides=\(shape.sides)")
            lines.append("\(p)additive=\(shape.additive ? 1 : 0)")
            lines.append("\(p)thickoutline=\(shape.thickOutline ? 1 : 0)")
            lines.append("\(p)textured=\(shape.textured ? 1 : 0)")
            lines.append("\(p)x=\(shape.x)"); lines.append("\(p)y=\(shape.y)")
            lines.append("\(p)radius=\(shape.radius)")
            lines.append("\(p)ang=\(shape.ang)")
            for (i, eq) in shape.perFrame.enumerated() {
                lines.append("\(p)per_frame_\(i + 1)=\(eq)")
            }
        }

        return lines.joined(separator: "\n")
    }
}
