//
//  ConsistentMorphProcessor.swift
//  metal_video_morpher
//
//  Drives the spacetime-consistent morph pipeline implemented in Shaders.metal.
//
//  Per output frame at index t:
//    1. (Optionally) compute optical flow from frame t -> t+1 with Vision.
//    2. Run consistentMorphKernel composing tricubic Bezier FFD,
//       spacetime tetrahedral barycentric warp, and flow advection.
//

import Foundation
import Metal
import MetalKit
import CoreVideo
import Vision
import simd

// MARK: - Parameter struct (mirrors ConsistentMorphParams in Shaders.metal) ---

struct ConsistentMorphParams {
    var width: UInt32
    var height: UInt32
    var frameIndex: UInt32
    var frameCount: UInt32

    var morphStrength: Float
    var flowWeight: Float
    var ffdWeight: Float
    var tetWeight: Float

    var tetCount: UInt32
    var useFlow: UInt32
}

// MARK: - Tetra entry --------------------------------------------------------

/// One spacetime tet: 4 source vertices in (x_norm, y_norm, t_norm), 4 target
/// vertices in the same space. Pack into 8 SIMD3<Float> for the GPU.
struct SpacetimeTet {
    var sourceVerts: (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)
    var targetVerts: (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)

    var packed: [SIMD3<Float>] {
        return [sourceVerts.0, sourceVerts.1, sourceVerts.2, sourceVerts.3,
                targetVerts.0, targetVerts.1, targetVerts.2, targetVerts.3]
    }
}

// MARK: - Processor ----------------------------------------------------------

final class ConsistentMorphProcessor {

    enum MorphError: Error {
        case noMetalDevice
        case shaderNotFound
        case textureCreationFailed
        case bufferCreationFailed
    }

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let flowPipeline: MTLComputePipelineState
    private let textureCache: CVMetalTextureCache

    // Tunables (the UI binds to these).
    var morphStrength: Float = 0.7
    var flowWeight: Float    = 1.0
    var ffdWeight: Float     = 1.0
    var tetWeight: Float     = 1.0
    var useFlow: Bool        = true

    // Geometry inputs.
    var bezierLattice: BezierLattice = BezierLattice.identity()
    var tets: [SpacetimeTet] = []

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MorphError.noMetalDevice
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MorphError.noMetalDevice
        }
        self.queue = queue

        let library = try device.makeDefaultLibrary(bundle: .main)
        guard let consistentFn = library.makeFunction(name: "consistentMorphKernel"),
              let flowFn       = library.makeFunction(name: "flowAdvectKernel") else {
            throw MorphError.shaderNotFound
        }
        self.pipeline     = try device.makeComputePipelineState(function: consistentFn)
        self.flowPipeline = try device.makeComputePipelineState(function: flowFn)

        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache)
        guard let cache else { throw MorphError.textureCreationFailed }
        self.textureCache = cache
    }

    // MARK: Public entry point

    /// Run the composed morph kernel.
    /// - Parameters:
    ///   - source: input frame as a CVPixelBuffer (BGRA8).
    ///   - flow:   optional precomputed flow texture (RG float, in pixels).
    ///   - frameIndex / frameCount: spacetime coordinate of this frame.
    /// - Returns: a freshly allocated MTLTexture containing the morphed frame.
    func morph(source: CVPixelBuffer,
               flow: MTLTexture?,
               frameIndex: Int,
               frameCount: Int) throws -> MTLTexture
    {
        let width  = CVPixelBufferGetWidth(source)
        let height = CVPixelBufferGetHeight(source)

        let srcTex = try makeTexture(from: source, format: .bgra8Unorm)
        let outTex = try makeWritableTexture(width: width, height: height,
                                             format: .bgra8Unorm)

        // A zero-flow stand-in if caller didn't provide one.
        let flowTex = flow ?? (try makeWritableTexture(width: 1, height: 1,
                                                      format: .rg16Float))

        // Pack control buffers.
        var lattice = bezierLattice.controlPointsAsSIMD()
        guard lattice.count == 64 else { throw MorphError.bufferCreationFailed }

        var packedTets: [SIMD3<Float>] = []
        packedTets.reserveCapacity(tets.count * 8)
        for t in tets { packedTets.append(contentsOf: t.packed) }

        guard let latticeBuf = device.makeBuffer(bytes: &lattice,
                                                 length: MemoryLayout<SIMD3<Float>>.stride * 64,
                                                 options: .storageModeShared) else {
            throw MorphError.bufferCreationFailed
        }

        let tetBuf: MTLBuffer = packedTets.isEmpty
            ? device.makeBuffer(length: MemoryLayout<SIMD3<Float>>.stride * 8,
                                options: .storageModeShared)!
            : device.makeBuffer(bytes: &packedTets,
                                length: MemoryLayout<SIMD3<Float>>.stride * packedTets.count,
                                options: .storageModeShared)!

        var params = ConsistentMorphParams(
            width: UInt32(width),
            height: UInt32(height),
            frameIndex: UInt32(frameIndex),
            frameCount: UInt32(max(frameCount, 1)),
            morphStrength: morphStrength,
            flowWeight: useFlow ? flowWeight : 0,
            ffdWeight: ffdWeight,
            tetWeight: tetWeight,
            tetCount: UInt32(tets.count),
            useFlow: useFlow ? 1 : 0
        )

        guard let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            throw MorphError.bufferCreationFailed
        }
        enc.setComputePipelineState(pipeline)
        enc.setTexture(srcTex,  index: 0)
        enc.setTexture(flowTex, index: 1)
        enc.setTexture(outTex,  index: 2)
        enc.setBuffer(latticeBuf, offset: 0, index: 0)
        enc.setBuffer(tetBuf,     offset: 0, index: 1)
        enc.setBytes(&params, length: MemoryLayout<ConsistentMorphParams>.stride, index: 2)

        let tg = MTLSize(width: 16, height: 16, depth: 1)
        let grid = MTLSize(
            width:  (width  + tg.width  - 1) / tg.width,
            height: (height + tg.height - 1) / tg.height,
            depth: 1
        )
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        return outTex
    }

    // MARK: Optical-flow estimation (Vision.VNGenerateOpticalFlowRequest)

    /// Computes a dense flow field between two CVPixelBuffer frames and
    /// returns it as an RG-float MTLTexture in pixel units.
    func computeFlow(from a: CVPixelBuffer, to b: CVPixelBuffer) throws -> MTLTexture {
        let request = VNGenerateOpticalFlowRequest(targetedCVPixelBuffer: b, options: [:])
        request.computationAccuracy = .medium
        let handler = VNImageRequestHandler(cvPixelBuffer: a, options: [:])
        try handler.perform([request])

        guard let observation = request.results?.first as? VNPixelBufferObservation else {
            throw MorphError.bufferCreationFailed
        }
        let flowPB = observation.pixelBuffer
        return try makeTexture(from: flowPB, format: .rg32Float)
    }

    // MARK: - Texture helpers

    private func makeTexture(from pb: CVPixelBuffer,
                             format: MTLPixelFormat) throws -> MTLTexture
    {
        let w = CVPixelBufferGetWidth(pb)
        let h = CVPixelBufferGetHeight(pb)
        var cvtex: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, textureCache, pb, nil,
            format, w, h, 0, &cvtex
        )
        guard status == kCVReturnSuccess,
              let cvtex,
              let tex = CVMetalTextureGetTexture(cvtex) else {
            throw MorphError.textureCreationFailed
        }
        return tex
    }

    private func makeWritableTexture(width: Int, height: Int,
                                     format: MTLPixelFormat) throws -> MTLTexture
    {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: width, height: height, mipmapped: false
        )
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        guard let tex = device.makeTexture(descriptor: desc) else {
            throw MorphError.textureCreationFailed
        }
        return tex
    }
}
