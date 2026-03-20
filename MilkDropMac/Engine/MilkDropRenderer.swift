// MilkDropRenderer.swift — Metal-based MilkDrop rendering engine
// Implements the full MilkDrop rendering pipeline:
//   1. Evaluate per-frame equations (zoom, rot, warp, cx, cy, dx, dy, etc.)
//   2. Warp pass: distort feedback texture using mesh
//   3. Wave rendering: draw audio waveform
//   4. Shape rendering: draw custom shapes
//   5. Composite pass: final output with gamma/brightness
//   6. Syphon output: share frame to other apps

import Metal
import MetalKit
import simd
import Foundation

// MARK: - Uniforms that mirror MilkDrop.metal

struct MilkDropUniforms {
    var time:                Float = 0
    var fps:                 Float = 60
    var frame:               Float = 0
    var progress:            Float = 0

    var bass:                Float = 0
    var mid:                 Float = 0
    var treble:              Float = 0
    var vol:                 Float = 0
    var bass_att:            Float = 0
    var mid_att:             Float = 0
    var treble_att:          Float = 0
    var vol_att:             Float = 0

    var zoom:                Float = 1
    var rot:                 Float = 0
    var warp:                Float = 1
    var cx:                  Float = 0.5
    var cy:                  Float = 0.5
    var dx:                  Float = 0
    var dy:                  Float = 0
    var sx:                  Float = 1
    var sy:                  Float = 1
    var decay:               Float = 0.98
    var gamma:               Float = 1
    var videoEchoAlpha:      Float = 0
    var videoEchoZoom:       Float = 1
    var videoEchoOrientation: Int32 = 0

    var q: (Float, Float, Float, Float, Float, Float, Float, Float,
            Float, Float, Float, Float, Float, Float, Float, Float,
            Float, Float, Float, Float, Float, Float, Float, Float,
            Float, Float, Float, Float, Float, Float, Float, Float) = (
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    )

    var resolution:          SIMD2<Float> = .zero
    var aspect:              Float = 1
}

// MARK: - Renderer

class MilkDropRenderer: NSObject, MTKViewDelegate {
    // Metal
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var library: MTLLibrary?

    // Pipeline states
    var warpPipeline:      MTLRenderPipelineState?
    var wavePipeline:      MTLRenderPipelineState?
    var shapePipeline:     MTLRenderPipelineState?
    var compositePipeline: MTLRenderPipelineState?
    var blendPipeline:     MTLRenderPipelineState?
    var fractalPipeline:   MTLRenderPipelineState?
    var copyPipeline:      MTLRenderPipelineState?

    // Framebuffers
    var warpTextureA: MTLTexture?   // Ping-pong feedback buffers
    var warpTextureB: MTLTexture?
    var waveTexture:  MTLTexture?
    var shapeTexture: MTLTexture?
    var outputTexture: MTLTexture?
    var transitionTextureA: MTLTexture?  // For preset blending
    var transitionTextureB: MTLTexture?
    var fractalTexture: MTLTexture?      // Fractal stream overlay
    var pingPong: Bool = false

    // Equation evaluator
    var evaluator = EquationEvaluator()

    // State
    var uniforms = MilkDropUniforms()
    var currentPreset: MilkDropPreset?
    var nextPreset: MilkDropPreset?
    var transitionProgress: Float = 0
    var transitionDuration: Float = 2.5
    var transitionType: Int32 = 0
    var isTransitioning: Bool = false

    // Fractal stream
    var fractalEnabled: Bool = false
    var fractalBlend: Float = 0.4

    // Syphon
    var syphonServer: SyphonBridgeWrapper?
    var syphonEnabled: Bool = true

    // Timing
    var startTime: CFTimeInterval = CACurrentMediaTime()
    var frameCount: Int = 0

    // Audio data
    var audioData: AudioData = .silence

    // MARK: - Init

    init?(device: MTLDevice) {
        self.device = device
        guard let queue = device.makeCommandQueue() else { return nil }
        self.commandQueue = queue
        super.init()

        setupLibrary()
        setupPipelines()
        setupSyphon()
    }

    private func setupLibrary() {
        library = device.makeDefaultLibrary()
    }

    private func setupPipelines() {
        guard let lib = library else { return }

        let quad = lib.makeFunction(name: "quad_vertex")

        // Warp pipeline
        warpPipeline = makePipeline(
            vertex: quad,
            fragment: lib.makeFunction(name: "warp_fragment"),
            pixelFormat: .bgra8Unorm
        )

        // Wave pipeline (line rendering)
        wavePipeline = makePipeline(
            vertex: lib.makeFunction(name: "wave_vertex"),
            fragment: lib.makeFunction(name: "wave_fragment"),
            pixelFormat: .bgra8Unorm,
            blending: true
        )

        // Shape pipeline
        shapePipeline = makePipeline(
            vertex: lib.makeFunction(name: "shape_vertex"),
            fragment: lib.makeFunction(name: "shape_fragment"),
            pixelFormat: .bgra8Unorm,
            blending: true
        )

        // Composite pipeline
        compositePipeline = makePipeline(
            vertex: quad,
            fragment: lib.makeFunction(name: "composite_fragment"),
            pixelFormat: .bgra8Unorm
        )

        // Blend pipeline
        blendPipeline = makePipeline(
            vertex: quad,
            fragment: lib.makeFunction(name: "blend_fragment"),
            pixelFormat: .bgra8Unorm
        )

        // Fractal stream pipeline (additive blending for glow)
        fractalPipeline = makePipeline(
            vertex: quad,
            fragment: lib.makeFunction(name: "fractal_stream_fragment"),
            pixelFormat: .bgra8Unorm,
            blending: true
        )

        // Present pipeline: copy finalTexture to drawable via render pass
        copyPipeline = makePipeline(
            vertex: quad,
            fragment: lib.makeFunction(name: "copy_fragment"),
            pixelFormat: .bgra8Unorm
        )
    }

    private func makePipeline(
        vertex: MTLFunction?,
        fragment: MTLFunction?,
        pixelFormat: MTLPixelFormat,
        blending: Bool = false
    ) -> MTLRenderPipelineState? {
        guard let v = vertex, let f = fragment else { return nil }
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction   = v
        desc.fragmentFunction = f
        desc.colorAttachments[0].pixelFormat = pixelFormat
        if blending {
            let att = desc.colorAttachments[0]!
            att.isBlendingEnabled = true
            att.sourceRGBBlendFactor      = .sourceAlpha
            att.destinationRGBBlendFactor = .oneMinusSourceAlpha
            att.sourceAlphaBlendFactor    = .one
            att.destinationAlphaBlendFactor = .zero
        }
        return try? device.makeRenderPipelineState(descriptor: desc)
    }

    private func setupSyphon() {
        syphonServer = SyphonBridgeWrapper(device: device, name: "MilkDropMac")
    }

    // MARK: - Framebuffer resize

    func setupTextures(size: CGSize) {
        let w = Int(size.width)
        let h = Int(size.height)
        warpTextureA     = makeTexture(w: w, h: h)
        warpTextureB     = makeTexture(w: w, h: h)
        waveTexture      = makeTexture(w: w, h: h)
        shapeTexture     = makeTexture(w: w, h: h)
        outputTexture    = makeTexture(w: w, h: h)
        transitionTextureA = makeTexture(w: w, h: h)
        transitionTextureB = makeTexture(w: w, h: h)
        fractalTexture   = makeTexture(w: w, h: h)
    }

    private func makeTexture(w: Int, h: Int) -> MTLTexture {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: w, height: h,
            mipmapped: false
        )
        desc.usage = [.renderTarget, .shaderRead]
        desc.storageMode = .private
        return device.makeTexture(descriptor: desc)!
    }

    // MARK: - MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        setupTextures(size: size)
        uniforms.resolution = SIMD2<Float>(Float(size.width), Float(size.height))
        uniforms.aspect     = Float(size.width / size.height)
    }

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let cmdBuf = commandQueue.makeCommandBuffer() else { return }

        let now = CACurrentMediaTime()
        uniforms.time  = Float(now - startTime)
        uniforms.frame = Float(frameCount)
        frameCount += 1

        // Update audio data into uniforms
        uniforms.bass    = audioData.bass
        uniforms.mid     = audioData.mid
        uniforms.treble  = audioData.treble
        uniforms.vol     = audioData.rms
        uniforms.bass_att = audioData.bassAttn

        // Evaluate per-frame equations from current preset
        if let preset = currentPreset {
            evaluatePreset(preset)
        }

        // Ping-pong feedback textures
        let (readTex, writeTex) = pingPong
            ? (warpTextureB!, warpTextureA!)
            : (warpTextureA!, warpTextureB!)
        pingPong.toggle()

        // 1. Warp pass: distort previous composite (readTex) → writeTex
        renderWarpPass(cmd: cmdBuf, input: readTex, output: writeTex)

        // 2. Wave pass (into waveTexture)
        if let waveTex = waveTexture {
            renderWavePass(cmd: cmdBuf, output: waveTex)
        }

        // 3. Shape pass
        if let shapeTex = shapeTexture {
            renderShapePass(cmd: cmdBuf, output: shapeTex)
        }

        // 4a. Fractal pass (if enabled) → fractalTexture
        if fractalEnabled, let fracTex = fractalTexture {
            renderFractalPass(cmd: cmdBuf, output: fracTex)
        }

        // 4b. Composite pass: warp + waves + shapes → readTex
        // readTex is safe to write now (warp is done reading it).
        // Writing back to readTex completes the feedback loop:
        // next frame the warp will distort this full composite.
        let waveTex   = waveTexture
        let shapeTex  = shapeTexture
        if let w = waveTex, let s = shapeTex {
            renderCompositePass(cmd: cmdBuf, warp: writeTex, wave: w, shape: s, output: readTex)
        }

        // 5. Transition blend (if active)
        var finalTexture: MTLTexture = readTex
        if isTransitioning,
           let txA = transitionTextureA, let txB = transitionTextureB,
           let blendOut = outputTexture {
            renderBlendPass(cmd: cmdBuf, a: txA, b: txB, output: blendOut)
            finalTexture = blendOut
        }

        // 6. Blit to drawable — guard against size mismatch (window resize race)
        let drawTex = drawable.texture
        if finalTexture.width != drawTex.width || finalTexture.height != drawTex.height {
            setupTextures(size: CGSize(width: drawTex.width, height: drawTex.height))
            uniforms.resolution = SIMD2<Float>(Float(drawTex.width), Float(drawTex.height))
            uniforms.aspect = Float(drawTex.width) / Float(max(drawTex.height, 1))
            cmdBuf.commit()
            return
        }
        if let pipeline = copyPipeline {
            let desc = makeRenderPassDesc(drawTex)
            if let enc = cmdBuf.makeRenderCommandEncoder(descriptor: desc) {
                enc.setRenderPipelineState(pipeline)
                enc.setFragmentTexture(finalTexture, index: 0)
                drawQuad(enc: enc)
                enc.endEncoding()
            }
        }

        // 7. Syphon publish
        if syphonEnabled {
            syphonServer?.publishTexture(finalTexture, commandBuffer: cmdBuf)
        }

        cmdBuf.present(drawable)
        cmdBuf.commit()
    }

    // MARK: - Render passes

    private func renderWarpPass(cmd: MTLCommandBuffer, input: MTLTexture, output: MTLTexture) {
        guard let pipeline = warpPipeline else { return }
        let desc = makeRenderPassDesc(output)
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: desc) else { return }
        enc.setRenderPipelineState(pipeline)

        var u = uniforms
        enc.setFragmentBytes(&u, length: MemoryLayout<MilkDropUniforms>.size, index: 0)
        enc.setFragmentTexture(input, index: 0)

        // Full-screen quad
        drawQuad(enc: enc)
        enc.endEncoding()
    }

    private func renderWavePass(cmd: MTLCommandBuffer, output: MTLTexture) {
        guard let pipeline = wavePipeline,
              let preset = currentPreset,
              let params = preset.parameters else { return }

        let desc = makeRenderPassDesc(output, clear: true, clearColor: MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0))
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: desc) else { return }
        enc.setRenderPipelineState(pipeline)

        for wave in params.waves where wave.enabled {
            renderWave(wave: wave, audioData: audioData, enc: enc)
        }
        enc.endEncoding()
    }

    private func renderWave(wave: PresetWave, audioData: AudioData, enc: MTLRenderCommandEncoder) {
        let sampleCount = min(wave.samples, audioData.waveform.count)
        guard sampleCount > 1 else { return }

        // Build vertex array
        var positions = [SIMD2<Float>]()
        for i in 0..<sampleCount {
            let t = Float(i) / Float(sampleCount - 1)
            let amp = audioData.waveform[i] * wave.scaling
            let x = t
            let y = 0.5 + amp * 0.3
            positions.append(SIMD2<Float>(x, y))
        }

        struct WaveUniforms {
            var color:       SIMD4<Float>
            var thickness:   Float
            var drawThick:   Int32
            var additive:    Int32
            var useDots:     Int32
            var smoothing:   Float
            var sampleCount: Int32
        }
        var wu = WaveUniforms(
            color:       SIMD4<Float>(wave.r, wave.g, wave.b, wave.a),
            thickness:   wave.drawThick ? 2.0 : 1.0,
            drawThick:   wave.drawThick ? 1 : 0,
            additive:    wave.additive ? 1 : 0,
            useDots:     wave.useDots ? 1 : 0,
            smoothing:   wave.smoothing,
            sampleCount: Int32(sampleCount)
        )

        enc.setVertexBytes(&positions, length: positions.count * MemoryLayout<SIMD2<Float>>.stride, index: 0)
        enc.setVertexBytes(&wu, length: MemoryLayout<WaveUniforms>.size, index: 1)

        if wave.useDots {
            enc.drawPrimitives(type: .point, vertexStart: 0, vertexCount: positions.count)
        } else {
            enc.drawPrimitives(type: .lineStrip, vertexStart: 0, vertexCount: positions.count)
        }
    }

    private func renderShapePass(cmd: MTLCommandBuffer, output: MTLTexture) {
        guard let pipeline = shapePipeline,
              let params = currentPreset?.parameters else { return }

        let desc = makeRenderPassDesc(output, clear: true, clearColor: MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0))
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: desc) else { return }
        enc.setRenderPipelineState(pipeline)

        for shape in params.shapes where shape.enabled {
            renderShape(shape: shape, enc: enc)
        }
        enc.endEncoding()
    }

    private func renderShape(shape: PresetShape, enc: MTLRenderCommandEncoder) {
        let sides = max(shape.sides, 3)
        // Build triangle list (Metal has no triangleFan)
        var positions = [SIMD2<Float>]()
        let center = SIMD2<Float>(shape.x, shape.y)
        let aspectInv: Float = uniforms.aspect > 0 ? 1 / uniforms.aspect : 1
        for i in 0..<sides {
            let a1 = (Float(i)     / Float(sides)) * 2 * .pi + shape.ang
            let a2 = (Float(i + 1) / Float(sides)) * 2 * .pi + shape.ang
            positions.append(center)
            positions.append(SIMD2<Float>(shape.x + cos(a1) * shape.radius * aspectInv, shape.y + sin(a1) * shape.radius))
            positions.append(SIMD2<Float>(shape.x + cos(a2) * shape.radius * aspectInv, shape.y + sin(a2) * shape.radius))
        }

        struct ShapeUniforms {
            var color, color2, borderColor: SIMD4<Float>
            var center: SIMD2<Float>
            var radius: Float
            var angle:  Float
            var sides:  Int32
            var additive: Int32
            var thickOutline: Int32
        }
        var su = ShapeUniforms(
            color:        SIMD4<Float>(shape.r,  shape.g,  shape.b,  shape.a),
            color2:       SIMD4<Float>(shape.r2, shape.g2, shape.b2, shape.a2),
            borderColor:  SIMD4<Float>(shape.border_r, shape.border_g, shape.border_b, shape.border_a),
            center:       SIMD2<Float>(shape.x, shape.y),
            radius:       shape.radius,
            angle:        shape.ang,
            sides:        Int32(sides),
            additive:     shape.additive ? 1 : 0,
            thickOutline: shape.thickOutline ? 1 : 0
        )

        var u = uniforms
        enc.setVertexBytes(&positions, length: positions.count * MemoryLayout<SIMD2<Float>>.stride, index: 0)
        enc.setVertexBytes(&su, length: MemoryLayout<ShapeUniforms>.size, index: 1)
        enc.setVertexBytes(&u, length: MemoryLayout<MilkDropUniforms>.size, index: 2)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: positions.count)
    }

    private func renderCompositePass(
        cmd: MTLCommandBuffer,
        warp: MTLTexture, wave: MTLTexture, shape: MTLTexture,
        output: MTLTexture
    ) {
        guard let pipeline = compositePipeline else { return }
        let desc = makeRenderPassDesc(output)
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: desc) else { return }
        enc.setRenderPipelineState(pipeline)

        struct CompositeUniforms {
            var brightness: Float; var gamma: Float
            var videoEchoAlpha: Float; var videoEchoZoom: Float; var videoEchoOrientation: Int32
            var resolution: SIMD2<Float>; var time: Float; var bass: Float; var treble: Float
            var q: (Float,Float,Float,Float,Float,Float,Float,Float,
                    Float,Float,Float,Float,Float,Float,Float,Float,
                    Float,Float,Float,Float,Float,Float,Float,Float,
                    Float,Float,Float,Float,Float,Float,Float,Float)
            var fractalBlend: Float
            var fractalEnabled: Int32
        }
        var cu = CompositeUniforms(
            brightness: 1.0,
            gamma:  uniforms.gamma,
            videoEchoAlpha: uniforms.videoEchoAlpha,
            videoEchoZoom: uniforms.videoEchoZoom,
            videoEchoOrientation: uniforms.videoEchoOrientation,
            resolution: uniforms.resolution,
            time: uniforms.time,
            bass: uniforms.bass,
            treble: uniforms.treble,
            q: uniforms.q,
            fractalBlend: fractalBlend,
            fractalEnabled: fractalEnabled ? 1 : 0
        )

        enc.setFragmentBytes(&cu, length: MemoryLayout<CompositeUniforms>.size, index: 0)
        enc.setFragmentTexture(warp,  index: 0)
        enc.setFragmentTexture(wave,  index: 1)
        enc.setFragmentTexture(shape, index: 2)
        if fractalEnabled, let fracTex = fractalTexture {
            enc.setFragmentTexture(fracTex, index: 3)
        }
        drawQuad(enc: enc)
        enc.endEncoding()
    }

    private func renderFractalPass(cmd: MTLCommandBuffer, output: MTLTexture) {
        guard let pipeline = fractalPipeline else { return }
        let desc = makeRenderPassDesc(output, clear: true, clearColor: MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0))
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: desc) else { return }
        enc.setRenderPipelineState(pipeline)

        var u = uniforms
        enc.setVertexBytes(&u, length: MemoryLayout<MilkDropUniforms>.size, index: 0)
        enc.setFragmentBytes(&u, length: MemoryLayout<MilkDropUniforms>.size, index: 0)
        drawQuad(enc: enc)
        enc.endEncoding()
    }

    private func renderBlendPass(cmd: MTLCommandBuffer, a: MTLTexture, b: MTLTexture, output: MTLTexture) {
        guard let pipeline = blendPipeline else { return }
        let desc = makeRenderPassDesc(output)
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: desc) else { return }
        enc.setRenderPipelineState(pipeline)

        struct BlendUniforms {
            var blend: Float; var blendType: Int32; var time: Float
            var resolution: SIMD2<Float>
        }
        var bu = BlendUniforms(
            blend: transitionProgress,
            blendType: transitionType,
            time: uniforms.time,
            resolution: uniforms.resolution
        )
        enc.setFragmentBytes(&bu, length: MemoryLayout<BlendUniforms>.size, index: 0)
        enc.setFragmentTexture(a, index: 0)
        enc.setFragmentTexture(b, index: 1)
        drawQuad(enc: enc)
        enc.endEncoding()
    }

    // MARK: - Equation evaluation

    private func evaluatePreset(_ preset: MilkDropPreset) {
        guard let params = preset.parameters else { return }
        // Copy static params into uniforms
        uniforms.zoom  = params.zoomAmount
        uniforms.rot   = params.rotatAmount
        uniforms.warp  = params.warpScale
        uniforms.cx    = params.centreX
        uniforms.cy    = params.centreY
        uniforms.sx    = params.szx
        uniforms.sy    = params.szy
        uniforms.decay = params.decay
        uniforms.gamma = params.gamma

        // Evaluate per-frame equations (modifies uniforms via evaluator)
        evaluator.evaluate(equations: params.perFrame, uniforms: &uniforms, audio: audioData)
    }

    // MARK: - Helpers

    private func makeRenderPassDesc(
        _ texture: MTLTexture,
        clear: Bool = false,
        clearColor: MTLClearColor = .init(red: 0, green: 0, blue: 0, alpha: 1)
    ) -> MTLRenderPassDescriptor {
        let desc = MTLRenderPassDescriptor()
        desc.colorAttachments[0].texture    = texture
        desc.colorAttachments[0].loadAction = clear ? .clear : .load
        desc.colorAttachments[0].storeAction = .store
        if clear { desc.colorAttachments[0].clearColor = clearColor }
        return desc
    }

    private func drawQuad(enc: MTLRenderCommandEncoder) {
        // Full-screen quad: two triangles covering NDC [-1,1]x[-1,1]
        struct QuadVertex { var pos: SIMD2<Float>; var uv: SIMD2<Float> }
        var quad: [QuadVertex] = [
            .init(pos: .init(-1,-1), uv: .init(0,1)),
            .init(pos: .init( 1,-1), uv: .init(1,1)),
            .init(pos: .init(-1, 1), uv: .init(0,0)),
            .init(pos: .init(-1, 1), uv: .init(0,0)),
            .init(pos: .init( 1,-1), uv: .init(1,1)),
            .init(pos: .init( 1, 1), uv: .init(1,0)),
        ]
        enc.setVertexBytes(&quad, length: quad.count * MemoryLayout<QuadVertex>.stride, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
    }

    // MARK: - Public API

    func loadPreset(_ preset: MilkDropPreset) {
        var mutablePreset = preset
        mutablePreset.parseParameters()
        currentPreset = mutablePreset
    }

    func beginTransition(to preset: MilkDropPreset, type: Int32 = 0, duration: Float = 2.5) {
        var mutablePreset = preset
        mutablePreset.parseParameters()
        nextPreset = mutablePreset
        transitionType     = type
        transitionDuration = duration
        transitionProgress = 0
        isTransitioning    = true
    }

    func updateAudio(_ data: AudioData) {
        audioData = data
    }

    func setSyphonEnabled(_ enabled: Bool) {
        syphonEnabled = enabled
    }
}

// MARK: - Equation Evaluator (simplified MilkDrop expression parser)

/// Evaluates per-frame MilkDrop equations that modify visual parameters.
/// Full implementation would use projectM-eval or a complete expression parser.
class EquationEvaluator {
    // Variable table
    var variables: [String: Double] = [:]

    func evaluate(equations: [String], uniforms: inout MilkDropUniforms, audio: AudioData) {
        // Seed variables from current uniforms
        variables["zoom"]   = Double(uniforms.zoom)
        variables["rot"]    = Double(uniforms.rot)
        variables["warp"]   = Double(uniforms.warp)
        variables["cx"]     = Double(uniforms.cx)
        variables["cy"]     = Double(uniforms.cy)
        variables["dx"]     = Double(uniforms.dx)
        variables["dy"]     = Double(uniforms.dy)
        variables["sx"]     = Double(uniforms.sx)
        variables["sy"]     = Double(uniforms.sy)
        variables["decay"]  = Double(uniforms.decay)
        variables["bass"]   = Double(audio.bass)
        variables["mid"]    = Double(audio.mid)
        variables["treble"] = Double(audio.treble)
        variables["time"]   = Double(uniforms.time)
        variables["fps"]    = Double(uniforms.fps)
        variables["frame"]  = Double(uniforms.frame)
        variables["pi"]     = Double.pi
        variables["e"]      = M_E

        // Evaluate each equation "var = expr"
        for eq in equations {
            guard let eqIdx = eq.firstIndex(of: "=") else { continue }
            let lhs = String(eq[..<eqIdx]).trimmingCharacters(in: .whitespaces)
            let rhs = String(eq[eq.index(after: eqIdx)...]).trimmingCharacters(in: .whitespaces)
            if let value = evaluateExpression(rhs) {
                variables[lhs] = value
            }
        }

        // Write back modified variables to uniforms
        uniforms.zoom  = Float(variables["zoom"]   ?? Double(uniforms.zoom))
        uniforms.rot   = Float(variables["rot"]    ?? Double(uniforms.rot))
        uniforms.warp  = Float(variables["warp"]   ?? Double(uniforms.warp))
        uniforms.cx    = Float(variables["cx"]     ?? Double(uniforms.cx))
        uniforms.cy    = Float(variables["cy"]     ?? Double(uniforms.cy))
        uniforms.dx    = Float(variables["dx"]     ?? Double(uniforms.dx))
        uniforms.dy    = Float(variables["dy"]     ?? Double(uniforms.dy))
        uniforms.sx    = Float(variables["sx"]     ?? 1.0)
        uniforms.sy    = Float(variables["sy"]     ?? 1.0)
        uniforms.decay = Float(variables["decay"]  ?? Double(uniforms.decay))
    }

    // Simplified expression evaluator (handles basic MilkDrop math)
    private func evaluateExpression(_ expr: String) -> Double? {
        // For production use, integrate projectM-eval library (libns-eel2 / projectM-eval)
        // This simplified evaluator handles common patterns:
        //   - Constants: 1.0, 0.5, pi
        //   - Variables: bass, mid, treble, zoom, etc.
        //   - Arithmetic: +, -, *, /
        //   - Functions: sin(), cos(), abs(), sqrt(), pow(), min(), max(), if()
        let trimmed = expr.trimmingCharacters(in: .whitespaces)
        if let val = Double(trimmed) { return val }
        if let val = variables[trimmed] { return val }
        // For full equation support, integrate: https://github.com/projectM-visualizer/projectm-eval
        return nil
    }
}
