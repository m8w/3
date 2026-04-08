import Metal
import simd

// MARK: - Uniform struct (matches GPU layout exactly)

/// Mirror of the `MilkdropUniforms` struct in `Milkdrop.metal`.
/// Swift's layout for 32 contiguous Floats is identical to Metal's
/// `float q[32]`, so we can copy with a single `setVertexBytes` /
/// `setFragmentBytes` call — no intermediate buffer needed.
public struct MilkdropUniforms {

    // q1…q32 — 32 arbitrary-Hz magnitudes from MicrotonalFFT.
    // Stored as a fixed-size tuple so the compiler can guarantee
    // contiguous, 4-byte-aligned layout without padding.
    public var q: (
        Float, Float, Float, Float,   //  q1– q4
        Float, Float, Float, Float,   //  q5– q8
        Float, Float, Float, Float,   //  q9–q12
        Float, Float, Float, Float,   // q13–q16
        Float, Float, Float, Float,   // q17–q20
        Float, Float, Float, Float,   // q21–q24
        Float, Float, Float, Float,   // q25–q28
        Float, Float, Float, Float    // q29–q32
    )

    // Convenience globals passed alongside the q array.
    public var time:       Float  // seconds since launch
    public var sampleRate: Float  // e.g. 44100
    public var _pad0:      Float  // keeps struct 16-byte aligned
    public var _pad1:      Float

    public init() {
        q = (0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,
             0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0)
        time       = 0
        sampleRate = 44100
        _pad0      = 0
        _pad1      = 0
    }

    // MARK: Loading q values from an array

    /// Copy 32 normalised magnitudes (from MicrotonalFFT) into the tuple fields.
    /// Index 0 → q1, index 31 → q32.
    public mutating func loadQ(_ values: [Float]) {
        precondition(values.count == 32, "Expected exactly 32 q values")
        withUnsafeMutableBytes(of: &q) { raw in
            let floats = raw.bindMemory(to: Float.self)
            for i in 0..<32 { floats[i] = values[i] }
        }
    }

    /// Read back a single q value by 1-based index (q1 = index 1).
    public func q(at index: Int) -> Float {
        precondition((1...32).contains(index))
        return withUnsafeBytes(of: q) { raw in
            raw.bindMemory(to: Float.self)[index - 1]
        }
    }
}

// MARK: - Encoder

/// Wraps `MilkdropUniforms` binding for both render and compute pipelines.
public struct MilkdropUniformEncoder {

    // The index to use for the uniform buffer in the shader.
    // Match this to `[[buffer(N)]]` in your Metal shader.
    public var bufferIndex: Int

    public init(bufferIndex: Int = 0) {
        self.bufferIndex = bufferIndex
    }

    // MARK: Render pipeline

    /// Bind `uniforms` to the **vertex** stage of `encoder`.
    public func bindVertex(_ uniforms: inout MilkdropUniforms,
                           to encoder: MTLRenderCommandEncoder) {
        withUnsafeBytes(of: uniforms) { ptr in
            encoder.setVertexBytes(
                ptr.baseAddress!,
                length: MemoryLayout<MilkdropUniforms>.stride,
                index:  bufferIndex
            )
        }
    }

    /// Bind `uniforms` to the **fragment** stage of `encoder`.
    public func bindFragment(_ uniforms: inout MilkdropUniforms,
                             to encoder: MTLRenderCommandEncoder) {
        withUnsafeBytes(of: uniforms) { ptr in
            encoder.setFragmentBytes(
                ptr.baseAddress!,
                length: MemoryLayout<MilkdropUniforms>.stride,
                index:  bufferIndex
            )
        }
    }

    // MARK: Compute pipeline

    /// Bind `uniforms` to a **compute** kernel.
    public func bindCompute(_ uniforms: inout MilkdropUniforms,
                            to encoder: MTLComputeCommandEncoder) {
        withUnsafeBytes(of: uniforms) { ptr in
            encoder.setBytes(
                ptr.baseAddress!,
                length: MemoryLayout<MilkdropUniforms>.stride,
                index:  bufferIndex
            )
        }
    }
}

// MARK: - Convenience: build uniforms from FFT output in one call

extension MilkdropUniforms {

    /// Create a ready-to-bind `MilkdropUniforms` from the raw 32-value output
    /// of `MicrotonalFFT.magnitudes(…)`.
    ///
    /// - Parameters:
    ///   - q:          32 magnitudes, already normalised to [0, 1].
    ///   - time:       Seconds since launch (from CACurrentMediaTime()).
    ///   - sampleRate: Audio sample rate (for reference in shaders).
    public static func make(q values: [Float],
                            time: Float,
                            sampleRate: Float = 44100) -> MilkdropUniforms {
        var u = MilkdropUniforms()
        u.loadQ(values)
        u.time       = time
        u.sampleRate = sampleRate
        return u
    }
}
