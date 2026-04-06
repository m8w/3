// EquationEvaluator.swift — MilkDrop EEL2-compatible expression evaluator
// Handles per_frame, per_frame_init, shape per-frame, and wave per-point equations.

import Foundation
import simd

// MARK: - Token

private enum Token: Equatable {
    case number(Double)
    case ident(String)
    case op(Character)       // + - * / % ^ = < > ! & |
    case lparen, rparen
    case comma
    case semicolon
    case eof
}

// MARK: - Lexer

private struct Lexer {
    private let src: [Character]
    private var pos: Int = 0

    init(_ s: String) { src = Array(s) }

    private var current: Character { pos < src.count ? src[pos] : "\0" }
    private var next: Character { pos + 1 < src.count ? src[pos + 1] : "\0" }

    mutating func tokenize() -> [Token] {
        var tokens: [Token] = []
        while pos < src.count {
            skipWhitespace()
            guard pos < src.count else { break }
            let c = current
            if c == "/" && next == "/" {
                // Line comment
                while pos < src.count && src[pos] != "\n" { pos += 1 }
                continue
            }
            if c.isNumber || (c == "." && next.isNumber) {
                tokens.append(readNumber())
            } else if c.isLetter || c == "_" {
                tokens.append(readIdent())
            } else if c == ";" {
                tokens.append(.semicolon); pos += 1
            } else if c == "(" {
                tokens.append(.lparen); pos += 1
            } else if c == ")" {
                tokens.append(.rparen); pos += 1
            } else if c == "," {
                tokens.append(.comma); pos += 1
            } else if c == "=" && next != "=" {
                tokens.append(.op("=")); pos += 1
            } else if c == "=" && next == "=" {
                tokens.append(.op("~")); pos += 2   // == → ~
            } else if c == "!" && next == "=" {
                tokens.append(.op("≠")); pos += 2   // != → ≠
            } else if c == "<" && next == "=" {
                tokens.append(.op("≤")); pos += 2
            } else if c == ">" && next == "=" {
                tokens.append(.op("≥")); pos += 2
            } else if c == "&" && next == "&" {
                tokens.append(.op("∧")); pos += 2   // && → ∧
            } else if c == "|" && next == "|" {
                tokens.append(.op("∨")); pos += 2   // || → ∨
            } else if "<>+-*/%^!&|".contains(c) {
                tokens.append(.op(c)); pos += 1
            } else {
                // Skip unknown characters
                pos += 1
            }
        }
        tokens.append(.eof)
        return tokens
    }

    private mutating func skipWhitespace() {
        while pos < src.count && src[pos].isWhitespace { pos += 1 }
    }

    private mutating func readNumber() -> Token {
        var s = ""
        while pos < src.count && (src[pos].isNumber || src[pos] == ".") {
            s.append(src[pos]); pos += 1
        }
        // Scientific notation: e.g. 1e-3
        if pos < src.count && (src[pos] == "e" || src[pos] == "E") {
            s.append(src[pos]); pos += 1
            if pos < src.count && (src[pos] == "+" || src[pos] == "-") {
                s.append(src[pos]); pos += 1
            }
            while pos < src.count && src[pos].isNumber {
                s.append(src[pos]); pos += 1
            }
        }
        return .number(Double(s) ?? 0)
    }

    private mutating func readIdent() -> Token {
        var s = ""
        while pos < src.count && (src[pos].isLetter || src[pos].isNumber || src[pos] == "_") {
            s.append(src[pos]); pos += 1
        }
        return .ident(s)
    }
}

// MARK: - Parser / Evaluator

private struct Parser {
    let tokens: [Token]
    var pos: Int = 0
    var vars: [String: Double]

    var current: Token { pos < tokens.count ? tokens[pos] : .eof }

    mutating func consume() { pos += 1 }

    mutating func expect(_ t: Token) {
        if current == t { consume() }
    }

    // Evaluate all semicolon-separated statements, return last value
    mutating func evalStatements() -> Double {
        var result = 0.0
        while current != .eof {
            result = parseAssign()
            // consume optional semicolons
            while current == .semicolon { consume() }
        }
        return result
    }

    // assignment: ident = expr
    mutating func parseAssign() -> Double {
        // Look-ahead: if next token is `=` (not ==), treat as assignment
        if case .ident(let name) = current {
            if pos + 1 < tokens.count, case .op("=") = tokens[pos + 1] {
                consume() // consume ident
                consume() // consume =
                let val = parseAssign() // right-associative
                vars[name] = val
                return val
            }
        }
        return parseOr()
    }

    mutating func parseOr() -> Double {
        var left = parseAnd()
        while case .op("∨") = current {
            consume()
            let right = parseAnd()
            left = (left != 0 || right != 0) ? 1 : 0
        }
        return left
    }

    mutating func parseAnd() -> Double {
        var left = parseNot()
        while case .op("∧") = current {
            consume()
            let right = parseNot()
            left = (left != 0 && right != 0) ? 1 : 0
        }
        return left
    }

    mutating func parseNot() -> Double {
        if case .op("!") = current {
            consume()
            let val = parseNot()
            return val == 0 ? 1 : 0
        }
        return parseCompare()
    }

    mutating func parseCompare() -> Double {
        var left = parseAdd()
        loop: while true {
            switch current {
            case .op("<"):  consume(); let r = parseAdd(); left = left < r  ? 1 : 0
            case .op(">"):  consume(); let r = parseAdd(); left = left > r  ? 1 : 0
            case .op("≤"):  consume(); let r = parseAdd(); left = left <= r ? 1 : 0
            case .op("≥"):  consume(); let r = parseAdd(); left = left >= r ? 1 : 0
            case .op("~"):  consume(); let r = parseAdd(); left = left == r ? 1 : 0
            case .op("≠"):  consume(); let r = parseAdd(); left = left != r ? 1 : 0
            default: break loop
            }
        }
        return left
    }

    mutating func parseAdd() -> Double {
        var left = parseMul()
        loop: while true {
            switch current {
            case .op("+"):  consume(); left += parseMul()
            case .op("-"):  consume(); left -= parseMul()
            default: break loop
            }
        }
        return left
    }

    mutating func parseMul() -> Double {
        var left = parsePower()
        loop: while true {
            switch current {
            case .op("*"):  consume(); left *= parsePower()
            case .op("/"):
                consume()
                let r = parsePower()
                left = r == 0 ? 0 : left / r
            case .op("%"):
                consume()
                let r = parsePower()
                left = r == 0 ? 0 : left.truncatingRemainder(dividingBy: r)
            default: break loop
            }
        }
        return left
    }

    mutating func parsePower() -> Double {
        let base = parseUnary()
        if case .op("^") = current {
            consume()
            let exp = parsePower() // right-associative
            return pow(base, exp)
        }
        return base
    }

    mutating func parseUnary() -> Double {
        if case .op("-") = current { consume(); return -parsePrimary() }
        if case .op("+") = current { consume(); return  parsePrimary() }
        return parsePrimary()
    }

    mutating func parsePrimary() -> Double {
        switch current {
        case .number(let v):
            consume()
            return v
        case .ident(let name):
            consume()
            // Function call?
            if case .lparen = current {
                consume() // (
                var args: [Double] = []
                if current != .rparen {
                    args.append(parseAssign())
                    while case .comma = current {
                        consume()
                        args.append(parseAssign())
                    }
                }
                expect(.rparen)
                return callFunction(name: name, args: args)
            }
            // Variable lookup
            return vars[name] ?? 0
        case .lparen:
            consume()
            let v = parseAssign()
            expect(.rparen)
            return v
        default:
            // Unrecognized token (e.g. bare '&' or '|') — must consume to prevent
            // evalStatements from looping forever waiting for .eof
            consume()
            return 0
        }
    }

    func callFunction(name: String, args: [Double]) -> Double {
        let a0 = args.count > 0 ? args[0] : 0
        let a1 = args.count > 1 ? args[1] : 0
        let a2 = args.count > 2 ? args[2] : 0
        switch name.lowercased() {
        case "sin":      return sin(a0)
        case "cos":      return cos(a0)
        case "tan":      return tan(a0)
        case "asin":     return asin(max(-1, min(1, a0)))
        case "acos":     return acos(max(-1, min(1, a0)))
        case "atan":     return atan(a0)
        case "atan2":    return atan2(a0, a1)
        case "sinh":     return sinh(a0)
        case "cosh":     return cosh(a0)
        case "tanh":     return tanh(a0)
        case "sqrt":     return a0 < 0 ? 0 : sqrt(a0)
        case "sqr":      return a0 * a0
        case "pow":      return a1 == 0 ? 1 : pow(a0, a1)
        case "exp":      return Foundation.exp(a0)
        case "log":      return a0 <= 0 ? -1e10 : Foundation.log(a0)
        case "log10":    return a0 <= 0 ? -1e10 : Foundation.log10(a0)
        case "floor":    return Foundation.floor(a0)
        case "ceil":     return Foundation.ceil(a0)
        case "round":    return Foundation.round(a0)
        case "int":      return a0 >= 0 ? Foundation.floor(a0) : Foundation.ceil(a0)
        case "abs":      return Swift.abs(a0)
        case "sign":     return a0 > 0 ? 1 : (a0 < 0 ? -1 : 0)
        case "min":      return Swift.min(a0, a1)
        case "max":      return Swift.max(a0, a1)
        case "clamp":    return Swift.max(a1, Swift.min(a2, a0))
        case "mod":      return a1 == 0 ? 0 : a0.truncatingRemainder(dividingBy: a1)
        case "if":       return a0 != 0 ? a1 : a2
        case "above":    return a0 > a1 ? 1 : 0
        case "below":    return a0 < a1 ? 1 : 0
        case "equal":    return a0 == a1 ? 1 : 0
        case "lerp":     return a0 + (a1 - a0) * a2
        case "sigmoid":  return 1.0 / (1.0 + Foundation.exp(-a0 * a1))
        case "rand":     return Double.random(in: 0..<1) * a0
        case "band":     return (a0 != 0 && a1 != 0) ? 1 : 0
        case "bor":      return (a0 != 0 || a1 != 0) ? 1 : 0
        case "bnot":     return a0 == 0 ? 1 : 0
        case "getosc":   return a0   // placeholder — returns frequency arg
        default:         return 0
        }
    }
}

// MARK: - Public EquationEvaluator

class EquationEvaluator {
    // Persistent variables: q1–q32 survive between frames; all others reset each frame
    var variables: [String: Double] = [:]
    private var qVars: [String: Double] = [:]

    init() {
        for i in 1...32 { qVars["q\(i)"] = 0 }
    }

    // Called once when a preset loads: reset q-vars, run per_frame_init equations
    func initPreset(initEquations: String, uniforms: inout MilkDropUniforms, audio: AudioData) {
        // Reset persistent q-vars
        for i in 1...32 { qVars["q\(i)"] = 0 }

        // Build variable environment
        var env = makeBaseEnv(uniforms: uniforms, audio: audio)
        env.merge(qVars) { _, new in new }

        // Run init equations
        env = runCode(initEquations, vars: env)

        // Harvest q-vars back
        harvestQVars(from: env)
        // Write uniforms back from init results
        writeUniforms(from: env, into: &uniforms)
    }

    // Called every frame: evaluate per_frame equations array
    func evaluate(equations: [String], uniforms: inout MilkDropUniforms, audio: AudioData) {
        var env = makeBaseEnv(uniforms: uniforms, audio: audio)
        env.merge(qVars) { _, new in new }

        for eq in equations {
            env = runCode(eq, vars: env)
        }

        harvestQVars(from: env)
        writeUniforms(from: env, into: &uniforms)
    }

    // Evaluate shape per-frame equations (scoped to shape properties)
    func evaluateShape(equations: [String], shape: inout PresetShape, uniforms: MilkDropUniforms, audio: AudioData) {
        var env = makeBaseEnv(uniforms: uniforms, audio: audio)
        env.merge(qVars) { _, new in new }

        // Inject shape variables
        env["sides"]   = Double(shape.sides)
        env["radius"]  = Double(shape.radius)
        env["ang"]     = Double(shape.ang)
        env["x"]       = Double(shape.x)
        env["y"]       = Double(shape.y)
        env["r"]       = Double(shape.r)
        env["g"]       = Double(shape.g)
        env["b"]       = Double(shape.b)
        env["a"]       = Double(shape.a)
        env["r2"]      = Double(shape.r2)
        env["g2"]      = Double(shape.g2)
        env["b2"]      = Double(shape.b2)
        env["a2"]      = Double(shape.a2)
        env["thickoutline"] = shape.thickOutline ? 1.0 : 0.0
        env["additive"]     = shape.additive     ? 1.0 : 0.0
        env["textured"]     = shape.textured     ? 1.0 : 0.0

        for eq in equations {
            env = runCode(eq, vars: env)
        }

        // Write back shape vars
        shape.sides   = Int(env["sides"]  ?? Double(shape.sides))
        shape.radius  = Float(env["radius"] ?? Double(shape.radius))
        shape.ang     = Float(env["ang"]    ?? Double(shape.ang))
        shape.x       = Float(env["x"]      ?? Double(shape.x))
        shape.y       = Float(env["y"]      ?? Double(shape.y))
        shape.r       = Float(env["r"]      ?? Double(shape.r))
        shape.g       = Float(env["g"]      ?? Double(shape.g))
        shape.b       = Float(env["b"]      ?? Double(shape.b))
        shape.a       = Float(env["a"]      ?? Double(shape.a))
        shape.r2      = Float(env["r2"]     ?? Double(shape.r2))
        shape.g2      = Float(env["g2"]     ?? Double(shape.g2))
        shape.b2      = Float(env["b2"]     ?? Double(shape.b2))
        shape.a2      = Float(env["a2"]     ?? Double(shape.a2))
    }

    // Evaluate wave per-point equations, returns updated (x, y, r, g, b, a) per sample
    func evaluateWavePoints(equations: [String], wave: PresetWave,
                            uniforms: MilkDropUniforms, audio: AudioData,
                            sampleCount: Int) -> [(x: Float, y: Float, r: Float, g: Float, b: Float, a: Float)] {
        var baseEnv = makeBaseEnv(uniforms: uniforms, audio: audio)
        baseEnv.merge(qVars) { _, new in new }

        var result: [(x: Float, y: Float, r: Float, g: Float, b: Float, a: Float)] = []
        result.reserveCapacity(sampleCount)

        for idx in 0..<sampleCount {
            let t = sampleCount > 1 ? Double(idx) / Double(sampleCount - 1) : 0
            var env = baseEnv
            let sIdx = min(idx, audio.waveform.count - 1)
            let sampleVal = idx < audio.waveform.count ? Double(audio.waveform[sIdx]) : 0
            env["i"]      = t
            env["x"]      = t   // default: left→right sweep 0..1
            env["y"]      = 0.5 + sampleVal * Double(wave.scaling) * 0.3
            env["r"]      = Double(wave.r)
            env["g"]      = Double(wave.g)
            env["b"]      = Double(wave.b)
            env["a"]      = Double(wave.a)
            env["sample"] = sampleVal
            env["value1"] = sampleVal
            env["value2"] = idx < audio.spectrum.count ? Double(audio.spectrum[sIdx]) : 0

            for eq in equations {
                env = runCode(eq, vars: env)
            }

            result.append((
                x: Float(env["x"] ?? 0.5),
                y: Float(env["y"] ?? 0.5),
                r: Float(env["r"] ?? Double(wave.r)),
                g: Float(env["g"] ?? Double(wave.g)),
                b: Float(env["b"] ?? Double(wave.b)),
                a: Float(env["a"] ?? Double(wave.a))
            ))
        }
        return result
    }

    // Evaluate per_vertex equations for one mesh vertex.
    // Returns the UV to sample from the previous frame for that vertex position.
    //
    // MilkDrop 2 per_pixel convention:
    //   1. Seed env["x"]/env["y"] with the SCREEN GRID position (0..1).
    //   2. Seed env["rad"]/env["ang"] relative to warp center.
    //   3. Run equations — they may modify zoom/rot/warp/dx/dy/sx/sy per-vertex,
    //      and/or directly set x/y to the desired sample UV.
    //   4. After equations, re-compute sample UV using the (possibly modified) params.
    //      If equations directly modified x or y, those override the computed UV.
    //
    // This matches the real MilkDrop per-pixel pipeline where equations like
    //   zoom = zoom + 0.005*sin(x*6+time)*bass
    // spatially vary the zoom based on screen position, producing a non-uniform warp.
    func evaluateVertex(equations: [String], x: Float, y: Float,
                        uniforms: MilkDropUniforms, audio: AudioData) -> SIMD2<Float> {
        var env = makeBaseEnv(uniforms: uniforms, audio: audio)
        env.merge(qVars) { _, new in new }

        let asp = uniforms.aspect > 0 ? Double(uniforms.aspect) : 1.0
        let ucx = env["cx"] ?? 0.5, ucy = env["cy"] ?? 0.5
        let xd = Double(x), yd = Double(y)

        // Seed screen position and polar coords — available to equations
        env["x"]   = xd
        env["y"]   = yd
        env["rad"] = sqrt((xd - ucx) * (xd - ucx) + (yd - ucy) * (yd - ucy)) * 2
        env["ang"] = atan2(yd - ucy, xd - ucx)

        // Run per-vertex equations — may modify zoom/rot/warp/dx/dy/sx/sy and/or x,y
        for eq in equations {
            env = runCode(eq, vars: env)
        }

        // Recompute sample UV using the (possibly per-vertex modified) warp params.
        // This is identical to warp_fragment but uses env values over uniform values
        // so that per-vertex parameter modifications take effect.
        let finalCX = env["cx"] ?? ucx
        let finalCY = env["cy"] ?? ucy

        var uvC = SIMD2<Double>(xd - finalCX, yd - finalCY) * SIMD2<Double>(asp, 1.0)
        let zoom = max(env["zoom"] ?? Double(uniforms.zoom), 0.001)
        uvC /= zoom
        let rot = env["rot"] ?? Double(uniforms.rot)
        let cosR = cos(rot), sinR = sin(rot)
        uvC = SIMD2<Double>(uvC.x * cosR - uvC.y * sinR,
                            uvC.x * sinR + uvC.y * cosR)
        uvC *= SIMD2<Double>(env["sx"] ?? Double(uniforms.sx),
                             env["sy"] ?? Double(uniforms.sy))
        uvC += SIMD2<Double>(env["dx"] ?? Double(uniforms.dx),
                             env["dy"] ?? Double(uniforms.dy)) * 2.0

        var sampleUV = uvC / SIMD2<Double>(asp, 1.0) + SIMD2<Double>(finalCX, finalCY)
        let warpAmt = env["warp"] ?? Double(uniforms.warp)
        let t1 = Double(uniforms.time * uniforms.warpSpeed)
        let wf0 = sin(t1 * 1.413 + 3.681), wf1 = cos(t1 * 1.731 - 1.869)
        let wf2 = sin(t1 * 2.197 + 0.292), wf3 = cos(t1 * 0.792 - 3.141)
        sampleUV.x += warpAmt * 0.0035 * (wf0 * sin(t1 * 0.53 + 3.0 * sampleUV.y)
                                         + wf1 * cos(t1 * 0.87 + 2.5 * sampleUV.x))
        sampleUV.y += warpAmt * 0.0035 * (wf2 * cos(t1 * 0.67 - 4.0 * sampleUV.x)
                                         + wf3 * sin(t1 * 1.09 + 3.0 * sampleUV.y + 0.5))

        return SIMD2<Float>(Float(sampleUV.x), Float(sampleUV.y))
    }

    // MARK: - Private helpers

    // Cache tokenized form of each unique equation string.
    // Re-tokenizing 512 samples × 64 per_point equations per frame (~32K calls)
    // was the source of the UI freeze — the token array is now built once per
    // unique string and reused for every evaluation.
    private var tokenCache: [String: [Token]] = [:]

    private func runCode(_ code: String, vars: [String: Double]) -> [String: Double] {
        let trimmed = code.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return vars }
        let tokens: [Token]
        if let cached = tokenCache[trimmed] {
            tokens = cached
        } else {
            var lexer = Lexer(trimmed)
            tokens = lexer.tokenize()
            tokenCache[trimmed] = tokens
        }
        var parser = Parser(tokens: tokens, vars: vars)
        _ = parser.evalStatements()
        return parser.vars
    }

    private func makeBaseEnv(uniforms: MilkDropUniforms, audio: AudioData) -> [String: Double] {
        var env: [String: Double] = [:]

        // Warp parameters (readable + writable by equations)
        env["zoom"]   = Double(uniforms.zoom)
        env["rot"]    = Double(uniforms.rot)
        env["warp"]   = Double(uniforms.warp)
        env["cx"]     = Double(uniforms.cx)
        env["cy"]     = Double(uniforms.cy)
        env["dx"]     = Double(uniforms.dx)
        env["dy"]     = Double(uniforms.dy)
        env["sx"]     = Double(uniforms.sx)
        env["sy"]     = Double(uniforms.sy)
        env["decay"]  = Double(uniforms.decay)
        env["gamma"]  = Double(uniforms.gamma)

        // Per-frame color overlay (set by equations like r=0.3+0.4*bass)
        env["r"]      = Double(uniforms.r)
        env["g"]      = Double(uniforms.g)
        env["b"]      = Double(uniforms.b)
        env["a"]      = Double(uniforms.amb)

        // Video echo / composite pass
        env["echo_zoom"]    = Double(uniforms.videoEchoZoom)
        env["echo_alpha"]   = Double(uniforms.videoEchoAlpha)
        env["echo_orient"]  = Double(uniforms.videoEchoOrientation)

        // Audio
        env["bass"]         = Double(audio.bass)
        env["mid"]          = Double(audio.mid)
        env["treb"]         = Double(audio.treble)
        env["vol"]          = Double(audio.rms)
        env["bass_att"]     = Double(audio.bassAttn)
        env["mid_att"]      = Double(audio.mid)
        env["treb_att"]     = Double(audio.treble)

        // Time / frame
        env["time"]         = Double(uniforms.time)
        env["frame"]        = Double(uniforms.frame)
        env["fps"]          = Double(uniforms.fps)
        env["progress"]     = Double(uniforms.progress)

        // Common math constants
        env["pi"]  = Double.pi
        env["e"]   = M_E
        env["phi"] = 1.6180339887

        return env
    }

    private func writeUniforms(from env: [String: Double], into uniforms: inout MilkDropUniforms) {
        if let v = env["zoom"]  { uniforms.zoom  = Float(v) }
        if let v = env["rot"]   { uniforms.rot   = Float(v) }
        if let v = env["warp"]    { uniforms.warp    = Float(v) }
        if let v = env["cx"]      { uniforms.cx      = Float(v) }
        if let v = env["cy"]      { uniforms.cy      = Float(v) }
        if let v = env["dx"]      { uniforms.dx      = Float(v) }
        if let v = env["dy"]      { uniforms.dy      = Float(v) }
        if let v = env["sx"]      { uniforms.sx      = Float(v) }
        if let v = env["sy"]      { uniforms.sy      = Float(v) }
        if let v = env["decay"]   { uniforms.decay   = Float(v) }
        if let v = env["gamma"]   { uniforms.gamma   = Float(v) }
        if let v = env["r"]       { uniforms.r       = Float(v) }
        if let v = env["g"]       { uniforms.g       = Float(v) }
        if let v = env["b"]       { uniforms.b       = Float(v) }
        if let v = env["a"]       { uniforms.amb     = Float(v) }

        if let v = env["echo_zoom"]   { uniforms.videoEchoZoom        = Float(v) }
        if let v = env["echo_alpha"]  { uniforms.videoEchoAlpha       = Float(v) }
        if let v = env["echo_orient"] { uniforms.videoEchoOrientation = Int32(v) }

        // Write q1–q32 to uniforms q array
        for i in 0..<32 {
            if let v = env["q\(i+1)"] {
                uniforms.q.0 = i == 0 ? Float(v) : uniforms.q.0  // handled below
            }
        }
        writeQVarsToUniforms(env: env, into: &uniforms)
    }

    private func writeQVarsToUniforms(env: [String: Double], into uniforms: inout MilkDropUniforms) {
        // MilkDropUniforms.q is a tuple of 32 Floats — write via withUnsafeMutablePointer
        withUnsafeMutableBytes(of: &uniforms.q) { ptr in
            let floatPtr = ptr.bindMemory(to: Float.self)
            for i in 0..<min(32, floatPtr.count) {
                if let v = env["q\(i+1)"] {
                    floatPtr[i] = Float(v)
                }
            }
        }
    }

    private func harvestQVars(from env: [String: Double]) {
        for i in 1...32 {
            let key = "q\(i)"
            qVars[key] = env[key] ?? 0
        }
    }
}
