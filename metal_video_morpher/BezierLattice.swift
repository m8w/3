//
//  BezierLattice.swift
//  metal_video_morpher
//
//  4×4×4 control lattice for the tricubic Bezier free-form deformation in
//  Shaders.metal :: ffd_tricubic. Each control point stores a *displacement*
//  (Δx, Δy, Δt) in normalized coords, so the identity warp is the all-zero
//  lattice (and no animation is required to leave the source untouched).
//

import Foundation
import simd

struct BezierLattice {
    // Stored in i + 4j + 16k order, matching the GPU-side indexing.
    var controlPoints: [SIMD3<Float>]

    init(controlPoints: [SIMD3<Float>]) {
        precondition(controlPoints.count == 64,
                     "BezierLattice requires exactly 64 control points (4x4x4).")
        self.controlPoints = controlPoints
    }

    static func identity() -> BezierLattice {
        BezierLattice(controlPoints: Array(repeating: SIMD3<Float>(repeating: 0),
                                           count: 64))
    }

    // MARK: - Indexing helpers

    static func index(i: Int, j: Int, k: Int) -> Int {
        precondition((0..<4).contains(i) && (0..<4).contains(j) && (0..<4).contains(k))
        return i + 4*j + 16*k
    }

    subscript(i: Int, j: Int, k: Int) -> SIMD3<Float> {
        get { controlPoints[Self.index(i: i, j: j, k: k)] }
        set { controlPoints[Self.index(i: i, j: j, k: k)] = newValue }
    }

    /// SIMD3 array view, ready to be uploaded to a Metal buffer.
    func controlPointsAsSIMD() -> [SIMD3<Float>] { controlPoints }

    // MARK: - Convenience generators

    /// Sinusoidal "breathing" warp — handy default to verify the pipeline.
    /// Amplitude is in normalized units (e.g. 0.05 = 5% of the image).
    static func breathing(amplitude: Float = 0.05,
                          temporalCycles: Float = 1.0) -> BezierLattice
    {
        var lat = identity()
        for k in 0..<4 {
            // w in [0,1] across the time axis of the control lattice.
            let w = Float(k) / 3.0
            let phase = sin(2.0 * .pi * temporalCycles * w)
            for j in 0..<4 {
                for i in 0..<4 {
                    // Push interior control points outward along (i,j) toward the
                    // image center — produces a soft pulsing zoom over time.
                    let cx = (Float(i) - 1.5) / 1.5
                    let cy = (Float(j) - 1.5) / 1.5
                    lat[i, j, k] = SIMD3<Float>(cx, cy, 0) * (amplitude * phase)
                }
            }
        }
        return lat
    }

    /// Linear interpolation between two lattices. Useful for keyframed FFD.
    static func lerp(_ a: BezierLattice,
                     _ b: BezierLattice,
                     _ alpha: Float) -> BezierLattice {
        var out = identity()
        for n in 0..<64 {
            out.controlPoints[n] = mix(a.controlPoints[n],
                                       b.controlPoints[n],
                                       t: clamp(alpha, min: 0, max: 1))
        }
        return out
    }
}

// MARK: - simd helpers (Swift's stdlib doesn't ship a generic `mix`).

private func mix(_ a: SIMD3<Float>, _ b: SIMD3<Float>, t: Float) -> SIMD3<Float> {
    return a + (b - a) * t
}

private func clamp(_ x: Float, min lo: Float, max hi: Float) -> Float {
    return Swift.min(Swift.max(x, lo), hi)
}
