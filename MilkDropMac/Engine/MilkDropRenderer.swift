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
    var warpSpeed:           Float = 1
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

    // Per-frame color overlay (r,g,b set by per-frame equations; amb = alpha/strength)
    var r:   Float = 0
    var g:   Float = 0
    var b:   Float = 0
    var amb: Float = 0
}

// MARK: - Renderer

class MilkDropRenderer: NSObject, MTKViewDelegate {
    // Metal
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var library: MTLLibrary?

    // Pipeline states
    var warpPipeline:          MTLRenderPipelineState?
    var meshWarpPipeline:      MTLRenderPipelineState?   // per_vertex mesh warp
    var wavePipeline:          MTLRenderPipelineState?   // alpha blend
    var waveAdditivePipeline:  MTLRenderPipelineState?   // additive blend
    var shapePipeline:         MTLRenderPipelineState?
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
    var lastFrameTime: CFTimeInterval = CACurrentMediaTime()
    var frameCount: Int = 0

    // Audio data
    var audioData: AudioData = .silence

    // Live param overrides — set by QuickEditor, applied after per-frame equations
    var liveZoom:  Float? = nil
    var liveWarp:  Float? = nil
    var liveDecay: Float? = nil
    var liveGamma: Float? = nil

    // FPS reporting
    var onFPSUpdate: ((Double) -> Void)?
    private var smoothedFPS: Double = 60

    // Seed feedback textures on next draw (ensures warp has something to distort)
    private var needsFeedbackSeed: Bool = true

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

        // Mesh warp pipeline (for presets with per_vertex equations)
        meshWarpPipeline = makePipeline(
            vertex: lib.makeFunction(name: "mesh_vertex"),
            fragment: lib.makeFunction(name: "mesh_warp_fragment"),
            pixelFormat: .bgra8Unorm
        )

        // Wave pipeline (alpha blend)
        wavePipeline = makePipeline(
            vertex: lib.makeFunction(name: "wave_vertex"),
            fragment: lib.makeFunction(name: "wave_fragment"),
            pixelFormat: .bgra8Unorm,
            blending: true
        )

        // Wave additive pipeline (src + dst — for wave.additive = true)
        waveAdditivePipeline = makePipeline(
            vertex: lib.makeFunction(name: "wave_vertex"),
            fragment: lib.makeFunction(name: "wave_fragment"),
            pixelFormat: .bgra8Unorm,
            blending: true,
            additive: true
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
        blending: Bool = false,
        additive: Bool = false
    ) -> MTLRenderPipelineState? {
        guard let v = vertex, let f = fragment else { return nil }
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction   = v
        desc.fragmentFunction = f
        desc.colorAttachments[0].pixelFormat = pixelFormat
        if blending {
            let att = desc.colorAttachments[0]!
            att.isBlendingEnabled = true
            if additive {
                att.sourceRGBBlendFactor      = .sourceAlpha
                att.destinationRGBBlendFactor = .one          // src*alpha + dst
                att.sourceAlphaBlendFactor    = .one
                att.destinationAlphaBlendFactor = .one
            } else {
                att.sourceRGBBlendFactor      = .sourceAlpha
                att.destinationRGBBlendFactor = .oneMinusSourceAlpha
                att.sourceAlphaBlendFactor    = .one
                att.destinationAlphaBlendFactor = .zero
            }
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
        needsFeedbackSeed = true   // New textures are blank — must re-seed the feedback loop
        uniforms.resolution = SIMD2<Float>(Float(size.width), Float(size.height))
        uniforms.aspect     = Float(size.width / size.height)
    }

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let cmdBuf = commandQueue.makeCommandBuffer() else { return }

        let now = CACurrentMediaTime()
        let dt  = Float(now - lastFrameTime)
        lastFrameTime = now
        uniforms.time  = Float(now - startTime)
        uniforms.frame = Float(frameCount)
        frameCount += 1

        // Smooth FPS and report to caller
        let instantFPS = dt > 0 ? 1.0 / Double(dt) : 60.0
        smoothedFPS = smoothedFPS * 0.9 + instantFPS * 0.1
        onFPSUpdate?(smoothedFPS)

        // Update audio data into uniforms
        uniforms.bass     = audioData.bass
        uniforms.mid      = audioData.mid
        uniforms.treble   = audioData.treble
        uniforms.vol      = audioData.rms
        uniforms.bass_att = audioData.bassAttn

        // Advance transition progress
        if isTransitioning {
            transitionProgress += dt / max(transitionDuration, 0.001)
            if transitionProgress >= 1.0 {
                // Transition complete — commit incoming preset as current
                transitionProgress = 0
                isTransitioning    = false
                if let next = nextPreset {
                    currentPreset = next
                    evaluator.initPreset(initEquations: next.parameters?.perFrameInit ?? "",
                                         uniforms: &uniforms, audio: audioData)
                }
                nextPreset = nil
            }
        }

        // Evaluate per-frame equations from current preset
        if let preset = currentPreset {
            evaluatePreset(preset)
        }

        // Seed feedback textures when a new preset is loaded
        seedFeedbackIfNeeded(cmd: cmdBuf)

        // Fixed feedback textures: A holds the composite output (warp reads this),
        // B is the warp scratch (composite reads this). No ping-pong — toggling caused
        // the composite output to become the warp's *write* target next frame instead of
        // its *read* target, so wave/shape content never fed back into the warp loop.
        let readTex  = warpTextureA!   // composite output from previous frame → warp input
        let writeTex = warpTextureB!   // warp writes here → composite reads this

        // 1. Warp pass: distort previous composite (readTex) → writeTex
        // Use mesh warp when per_vertex equations are present, otherwise full-screen quad
        let perVertex = currentPreset?.parameters?.perVertex ?? []
        if perVertex.isEmpty {
            renderWarpPass(cmd: cmdBuf, input: readTex, output: writeTex)
        } else {
            renderMeshWarpPass(cmd: cmdBuf, input: readTex, output: writeTex, equations: perVertex)
        }

        // 2. Wave pass (into waveTexture)
        if let waveTex = waveTexture {
            renderWavePass(cmd: cmdBuf, output: waveTex)
        }

        // 3. Shape pass (writeTex passed so textured shapes can sample the warped frame)
        if let shapeTex = shapeTexture {
            renderShapePass(cmd: cmdBuf, output: shapeTex, warpTex: writeTex)
        }

        // 4a. Fractal pass (if enabled) → fractalTexture
        if fractalEnabled, let fracTex = fractalTexture {
            renderFractalPass(cmd: cmdBuf, output: fracTex)
        }

        // 4b. Composite pass: warp + waves + shapes → readTex
        let waveTex  = waveTexture
        let shapeTex = shapeTexture
        if let w = waveTex, let s = shapeTex {
            renderCompositePass(cmd: cmdBuf, warp: writeTex, wave: w, shape: s, output: readTex)
        }

        // 5. Transition blend (if active): capture outgoing into txA, render incoming into txB
        var finalTexture: MTLTexture = readTex
        if isTransitioning,
           let txA = transitionTextureA, let txB = transitionTextureB,
           let blendOut = outputTexture {
            // txA = current outgoing frame (copy readTex)
            renderCopy(cmd: cmdBuf, src: readTex, dst: txA)
            // txB = incoming preset rendered independently
            if let next = nextPreset {
                renderNextPreset(next, cmd: cmdBuf, output: txB)
            }
            uniforms.progress = transitionProgress
            renderBlendPass(cmd: cmdBuf, a: txA, b: txB, output: blendOut)
            finalTexture = blendOut
        }

        // 6. Blit to drawable — guard against size mismatch (window resize race)
        let drawTex = drawable.texture
        if finalTexture.width != drawTex.width || finalTexture.height != drawTex.height {
            setupTextures(size: CGSize(width: drawTex.width, height: drawTex.height))
            needsFeedbackSeed = true   // New textures are blank — re-seed next frame
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
        enc.setFragmentBytes(&u, length: MemoryLayout<MilkDropUniforms>.stride, index: 0)
        enc.setFragmentTexture(input, index: 0)

        // Full-screen quad
        drawQuad(enc: enc)
        enc.endEncoding()
    }

    // Mesh warp pass: evaluates per_vertex equations to build a warped mesh
    private func renderMeshWarpPass(cmd: MTLCommandBuffer, input: MTLTexture, output: MTLTexture,
                                    equations: [String]) {
        guard let pipeline = meshWarpPipeline else {
            renderWarpPass(cmd: cmd, input: input, output: output)
            return
        }

        let meshW = 32, meshH = 24
        struct MeshVtx { var screenPos: SIMD2<Float>; var sampleUV: SIMD2<Float> }

        // Build vertices: for each grid point compute sample UV via per_vertex equations
        var vertices = [MeshVtx]()
        vertices.reserveCapacity(meshW * meshH)
        for row in 0..<meshH {
            for col in 0..<meshW {
                let x = Float(col) / Float(meshW - 1)
                let y = Float(row) / Float(meshH - 1)
                let sampleUV = evaluator.evaluateVertex(
                    equations: equations, x: x, y: y,
                    uniforms: uniforms, audio: audioData
                )
                vertices.append(MeshVtx(screenPos: SIMD2<Float>(x, y), sampleUV: sampleUV))
            }
        }

        // Build index buffer (two triangles per quad)
        var indices = [UInt32]()
        indices.reserveCapacity((meshW - 1) * (meshH - 1) * 6)
        for row in 0..<(meshH - 1) {
            for col in 0..<(meshW - 1) {
                let tl = UInt32(row * meshW + col)
                let tr = tl + 1
                let bl = tl + UInt32(meshW)
                let br = bl + 1
                indices += [tl, tr, bl, bl, tr, br]
            }
        }

        let desc = makeRenderPassDesc(output)
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: desc) else { return }
        enc.setRenderPipelineState(pipeline)

        var u = uniforms
        guard let vertexBuf = device.makeBuffer(bytes: &vertices,
                                                length: vertices.count * MemoryLayout<MeshVtx>.stride,
                                                options: .storageModeShared) else {
            enc.endEncoding(); return
        }
        enc.setVertexBuffer(vertexBuf, offset: 0, index: 0)
        enc.setVertexBytes(&u, length: MemoryLayout<MilkDropUniforms>.stride, index: 1)
        enc.setFragmentTexture(input, index: 0)
        enc.setFragmentBytes(&u, length: MemoryLayout<MilkDropUniforms>.stride, index: 0)

        guard let indexBuf = device.makeBuffer(bytes: indices,
                                               length: indices.count * MemoryLayout<UInt32>.stride,
                                               options: .storageModeShared) else {
            enc.endEncoding(); return
        }
        enc.drawIndexedPrimitives(type: .triangle,
                                  indexCount: indices.count,
                                  indexType: .uint32,
                                  indexBuffer: indexBuf,
                                  indexBufferOffset: 0)
        enc.endEncoding()
    }

    private func renderWavePass(cmd: MTLCommandBuffer, output: MTLTexture) {
        guard let wavePipe = wavePipeline else { return }

        let desc = makeRenderPassDesc(output, clear: true, clearColor: MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0))
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: desc) else { return }

        let enabledWaves = currentPreset?.parameters?.waves.filter { $0.enabled } ?? []

        if enabledWaves.isEmpty {
            // No preset waves — draw a bright fallback waveform so the feedback loop
            // always has visual content to work with regardless of preset.
            enc.setRenderPipelineState(wavePipe)
            renderFallbackWave(enc: enc)
        } else {
            for wave in enabledWaves {
                let pipe = wave.additive ? (waveAdditivePipeline ?? wavePipe) : wavePipe
                enc.setRenderPipelineState(pipe)
                renderWave(wave: wave, audioData: audioData, enc: enc)
            }
        }
        enc.endEncoding()
    }

    // Drawn when the preset has no waves defined. Uses a triangle-strip band so it's
    // several percent of screen height and visible regardless of audio level. When audio
    // is silent the wave animates synthetically so the feedback loop always has content.
    private func renderFallbackWave(enc: MTLRenderCommandEncoder) {
        let count = 256
        let halfThick: Float = 0.022   // Band half-height in UV space (~4% total height)

        // Scale factor: always normalise to a minimum amplitude so the band is wide
        // even with very quiet audio or complete silence.
        let rmsGain = max(audioData.rms * 5.0, 0.18)
        let isSilent = rmsGain < 0.19

        var positions = [SIMD2<Float>]()
        positions.reserveCapacity(count * 2)

        for i in 0..<count {
            let t = Float(i) / Float(count - 1)
            let amp: Float
            if isSilent {
                // Animated multi-harmonic sine — always active even with no audio input
                amp = sin(t * .pi * 8  + uniforms.time * 1.8) * 0.20
                   + sin(t * .pi * 5  - uniforms.time * 1.1) * 0.12
                   + sin(t * .pi * 13 + uniforms.time * 0.7) * 0.08
            } else {
                let raw = audioData.waveform.count > i ? audioData.waveform[i] : 0
                amp = (raw / rmsGain).clamped(to: -1...1)
            }
            let y = 0.5 + amp * 0.35
            positions.append(SIMD2<Float>(t, y + halfThick))   // top edge
            positions.append(SIMD2<Float>(t, y - halfThick))   // bottom edge
        }

        struct WaveUniforms {
            var color: SIMD4<Float>; var thickness: Float
            var drawThick: Int32; var additive: Int32; var useDots: Int32
            var smoothing: Float; var sampleCount: Int32; var perPointColors: Int32
        }
        var wu = WaveUniforms(
            color: SIMD4<Float>(1.0, 0.9, 0.85, 0.07), thickness: halfThick * 2,
            drawThick: 0, additive: 0, useDots: 0,
            smoothing: 0, sampleCount: Int32(count * 2), perPointColors: 0
        )
        var dummy = SIMD4<Float>(0.3, 0.75, 1.0, 0.95)
        enc.setVertexBytes(&positions, length: positions.count * MemoryLayout<SIMD2<Float>>.stride, index: 0)
        enc.setVertexBytes(&wu, length: MemoryLayout<WaveUniforms>.stride, index: 1)
        enc.setVertexBytes(&dummy, length: MemoryLayout<SIMD4<Float>>.stride, index: 2)
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: positions.count)
    }

    private func renderWave(wave: PresetWave, audioData: AudioData, enc: MTLRenderCommandEncoder) {
        // Cap per_point evaluation to 128 samples: draw(in:) runs on the main thread
        // (MTKView display link), so 512 samples × many equations causes UI freezes.
        // 128 samples is visually indistinguishable from 512 for waveform rendering.
        let rawSamples = min(wave.samples, audioData.waveform.count)
        let sampleCount = wave.perPoint.isEmpty ? rawSamples : min(rawSamples, 128)
        guard sampleCount > 1 else { return }

        // Build vertex array — use per-point equations if the wave defines them
        var positions  = [SIMD2<Float>]()
        var colors     = [SIMD4<Float>]()
        var hasPerPointColors = false

        if !wave.perPoint.isEmpty {
            let pts = evaluator.evaluateWavePoints(
                equations: wave.perPoint,
                wave: wave,
                uniforms: uniforms,
                audio: audioData,
                sampleCount: sampleCount
            )
            for pt in pts {
                positions.append(SIMD2<Float>(pt.x, pt.y))
                colors.append(SIMD4<Float>(pt.r, pt.g, pt.b, pt.a))
            }
            hasPerPointColors = true
        } else {
            // Default: horizontal waveform sweep.
            // wave.sep splits into two halves offset vertically by sep/512 screen units.
            let half = sampleCount / 2
            let sepOffset = Float(wave.sep) / 512.0
            for i in 0..<sampleCount {
                let t   = Float(i % half) / Float(max(half - 1, 1))
                let amp = audioData.waveform[i] * wave.scaling
                let y   = (i < half)
                    ? 0.5 - sepOffset + amp * 0.3
                    : 0.5 + sepOffset + amp * 0.3
                positions.append(SIMD2<Float>(t, y))
            }
        }

        struct WaveUniforms {
            var color:           SIMD4<Float>
            var thickness:       Float
            var drawThick:       Int32
            var additive:        Int32
            var useDots:         Int32
            var smoothing:       Float
            var sampleCount:     Int32
            var perPointColors:  Int32
        }
        var wu = WaveUniforms(
            color:          SIMD4<Float>(wave.r, wave.g, wave.b, wave.a),
            thickness:      wave.drawThick ? 2.0 : 1.0,
            drawThick:      wave.drawThick ? 1 : 0,
            additive:       wave.additive ? 1 : 0,
            useDots:        wave.useDots ? 1 : 0,
            smoothing:      wave.smoothing,
            sampleCount:    Int32(sampleCount),
            perPointColors: hasPerPointColors ? 1 : 0
        )

        enc.setVertexBytes(&positions, length: positions.count * MemoryLayout<SIMD2<Float>>.stride, index: 0)
        enc.setVertexBytes(&wu, length: MemoryLayout<WaveUniforms>.stride, index: 1)
        // Always bind a color buffer at index 2; use dummy entry when not using per-point colors
        if hasPerPointColors {
            enc.setVertexBytes(&colors, length: colors.count * MemoryLayout<SIMD4<Float>>.stride, index: 2)
        } else {
            var dummy = SIMD4<Float>(wave.r, wave.g, wave.b, wave.a)
            enc.setVertexBytes(&dummy, length: MemoryLayout<SIMD4<Float>>.stride, index: 2)
        }

        if wave.useDots {
            enc.drawPrimitives(type: .point, vertexStart: 0, vertexCount: positions.count)
        } else {
            enc.drawPrimitives(type: .lineStrip, vertexStart: 0, vertexCount: positions.count)
        }
    }

    private func renderShapePass(cmd: MTLCommandBuffer, output: MTLTexture, warpTex: MTLTexture) {
        guard let pipeline = shapePipeline,
              let params = currentPreset?.parameters else { return }

        let desc = makeRenderPassDesc(output, clear: true, clearColor: MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0))
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: desc) else { return }
        enc.setRenderPipelineState(pipeline)
        // Bind warp texture once for all textured shapes in this pass
        enc.setFragmentTexture(warpTex, index: 0)

        for shape in params.shapes where shape.enabled {
            var s = shape
            if !s.perFrame.isEmpty {
                evaluator.evaluateShape(equations: s.perFrame, shape: &s,
                                        uniforms: uniforms, audio: audioData)
            }
            renderShape(shape: s, enc: enc)
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
            var center:       SIMD2<Float>
            var radius:       Float
            var angle:        Float
            var sides:        Int32
            var additive:     Int32
            var thickOutline: Int32
            var textured:     Int32
            var tex_ang:      Float
            var tex_zoom:     Float
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
            thickOutline: shape.thickOutline ? 1 : 0,
            textured:     shape.textured ? 1 : 0,
            tex_ang:      shape.tex_ang,
            tex_zoom:     shape.tex_zoom
        )

        var u = uniforms
        enc.setVertexBytes(&positions, length: positions.count * MemoryLayout<SIMD2<Float>>.stride, index: 0)
        enc.setVertexBytes(&su, length: MemoryLayout<ShapeUniforms>.stride, index: 1)
        enc.setVertexBytes(&u, length: MemoryLayout<MilkDropUniforms>.stride, index: 2)
        enc.setFragmentBytes(&su, length: MemoryLayout<ShapeUniforms>.stride, index: 0)
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
            // Per-frame color overlay (r,g,b from equations, amb = strength)
            var r: Float; var g: Float; var b: Float; var amb: Float
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
            fractalEnabled: fractalEnabled ? 1 : 0,
            r: uniforms.r, g: uniforms.g, b: uniforms.b, amb: uniforms.amb
        )

        enc.setFragmentBytes(&cu, length: MemoryLayout<CompositeUniforms>.stride, index: 0)
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
        enc.setFragmentBytes(&u, length: MemoryLayout<MilkDropUniforms>.stride, index: 0)
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
        enc.setFragmentBytes(&bu, length: MemoryLayout<BlendUniforms>.stride, index: 0)
        enc.setFragmentTexture(a, index: 0)
        enc.setFragmentTexture(b, index: 1)
        drawQuad(enc: enc)
        enc.endEncoding()
    }

    // MARK: - Helpers for transitions

    // Copy src texture into dst (used to snapshot the outgoing frame)
    private func renderCopy(cmd: MTLCommandBuffer, src: MTLTexture, dst: MTLTexture) {
        guard let pipeline = copyPipeline else { return }
        let desc = makeRenderPassDesc(dst)
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: desc) else { return }
        enc.setRenderPipelineState(pipeline)
        enc.setFragmentTexture(src, index: 0)
        drawQuad(enc: enc)
        enc.endEncoding()
    }

    // Render the incoming preset into a standalone texture (no feedback loop — single clear frame)
    private func renderNextPreset(_ preset: MilkDropPreset, cmd: MTLCommandBuffer, output: MTLTexture) {
        guard let params = preset.parameters,
              let waveTex = waveTexture, let shapeTex = shapeTexture else { return }

        // Snapshot uniforms so we don't disturb the current preset's state
        var nextUniforms = uniforms
        nextUniforms.zoom  = params.zoomAmount
        nextUniforms.rot   = params.rotatAmount
        nextUniforms.warp  = params.warpScale
        nextUniforms.cx    = params.centreX
        nextUniforms.cy    = params.centreY
        nextUniforms.sx    = params.szx
        nextUniforms.sy    = params.szy
        nextUniforms.decay = params.decay
        nextUniforms.gamma = params.gamma

        // Render: clear output with black then composite waves+shapes over it
        let clearDesc = makeRenderPassDesc(output, clear: true)
        if let enc = cmd.makeRenderCommandEncoder(descriptor: clearDesc) {
            enc.endEncoding()
        }
        renderCompositePass(cmd: cmd, warp: output, wave: waveTex, shape: shapeTex, output: output)
    }

    // MARK: - Equation evaluation

    private func evaluatePreset(_ preset: MilkDropPreset) {
        guard let params = preset.parameters else { return }
        // Seed static params into uniforms (equations may override these)
        uniforms.zoom               = params.zoomAmount
        uniforms.rot                = params.rotatAmount
        uniforms.warp               = params.warpScale
        uniforms.cx                 = params.centreX
        uniforms.cy                 = params.centreY
        uniforms.dx                 = params.warpX
        uniforms.dy                 = params.warpY
        uniforms.sx                 = params.szx
        uniforms.sy                 = params.szy
        uniforms.decay              = params.decay
        uniforms.gamma              = params.gamma
        uniforms.warpSpeed          = params.warpSpeed
        uniforms.videoEchoAlpha     = params.videoEchoAlpha
        uniforms.videoEchoZoom      = params.videoEchoZoom
        uniforms.videoEchoOrientation = Int32(params.videoEchoOrientation)
        uniforms.r                  = params.r
        uniforms.g                  = params.g
        uniforms.b                  = params.b
        uniforms.amb                = params.a

        // Evaluate per-frame equations (modifies uniforms via evaluator)
        evaluator.evaluate(equations: params.perFrame, uniforms: &uniforms, audio: audioData)

        // Apply live overrides from QuickEditor — these win over equations
        if let v = liveZoom  { uniforms.zoom  = v }
        if let v = liveWarp  { uniforms.warp  = v }
        if let v = liveDecay { uniforms.decay = v }
        if let v = liveGamma { uniforms.gamma = v }
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
        // quad_vertex generates positions from [[vertex_id]] — no vertex buffer needed
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
    }

    // MARK: - Public API

    func loadPreset(_ preset: MilkDropPreset) {
        var mutablePreset = preset
        mutablePreset.parseParameters()
        currentPreset = mutablePreset
        // Run per_frame_init equations and reset q-vars for the new preset
        evaluator.initPreset(initEquations: mutablePreset.parameters?.perFrameInit ?? "", uniforms: &uniforms, audio: audioData)
        // Seed feedback so the warp has non-black content to distort
        needsFeedbackSeed = true
    }

    // Seed both warp ping-pong textures with a subtle dark gradient so
    // the warp feedback loop has something visible to distort from frame 1.
    private func seedFeedbackIfNeeded(cmd: MTLCommandBuffer) {
        guard needsFeedbackSeed,
              let texA = warpTextureA, let texB = warpTextureB else { return }
        needsFeedbackSeed = false

        // Seed both ping-pong textures to a bright purple so the warp feedback loop
        // starts with enough luminance to remain visible for several seconds while
        // the wave pass builds up steady-state content.
        for tex in [texA, texB] {
            let desc = MTLRenderPassDescriptor()
            desc.colorAttachments[0].texture     = tex
            desc.colorAttachments[0].loadAction  = .clear
            desc.colorAttachments[0].storeAction = .store
            desc.colorAttachments[0].clearColor  = MTLClearColor(red: 0.75, green: 0.15, blue: 0.45, alpha: 1)
            guard let enc = cmd.makeRenderCommandEncoder(descriptor: desc) else { continue }
            enc.endEncoding()
        }
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

