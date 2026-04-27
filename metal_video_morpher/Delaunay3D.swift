//
//  Delaunay3D.swift
//  metal_video_morpher
//
//  Bowyer–Watson 3D Delaunay tetrahedralization.
//
//  Used to generate the spacetime tet mesh consumed by consistentMorphKernel.
//  Inputs are points in (x_norm, y_norm, t_norm) ∈ [0,1]³.  We add a large
//  super-tetrahedron, insert points one by one removing tets whose
//  circumsphere contains the new point, then strip any tet that touches a
//  super vertex.
//
//  Complexity: O(n^2) worst case, fine for the 100s-of-keypoints regime.
//

import Foundation
import simd

struct Tet: Equatable {
    var a, b, c, d: Int      // indices into points

    func vertices() -> [Int] { [a, b, c, d] }

    var faces: [[Int]] {
        // Each face is a sorted triple of vertex indices.
        return [
            [a, b, c].sorted(),
            [a, b, d].sorted(),
            [a, c, d].sorted(),
            [b, c, d].sorted()
        ]
    }
}

struct Delaunay3D {

    /// Tetrahedralize the given normalized points.  Caller is expected to
    /// pass values inside (a generously padded) [0,1]³.
    static func tetrahedralize(points: [SIMD3<Float>]) -> [Tet] {
        guard points.count >= 4 else { return [] }

        // Super-tetrahedron — large enough to contain the unit cube.
        let big: Float = 10.0
        var pts = points
        let s0 = pts.count
        pts.append(SIMD3<Float>(-big, -big, -big))
        pts.append(SIMD3<Float>( big, -big, -big))
        pts.append(SIMD3<Float>(  0,  big, -big))
        pts.append(SIMD3<Float>(  0,   0,  big))

        var tets: [Tet] = [Tet(a: s0, b: s0+1, c: s0+2, d: s0+3)]

        for i in 0..<s0 {
            let p = pts[i]

            // Find tets whose circumsphere contains p.
            var bad: [Int] = []
            for (idx, t) in tets.enumerated() where inCircumsphere(p, t, pts) {
                bad.append(idx)
            }

            // Build the polyhedral cavity boundary: faces shared by exactly one bad tet.
            var faceCount: [[Int]: Int] = [:]
            var faceOwner: [[Int]: [Int]] = [:]
            for bi in bad {
                for f in tets[bi].faces {
                    faceCount[f, default: 0] += 1
                    var owners = faceOwner[f] ?? []
                    owners.append(bi)
                    faceOwner[f] = owners
                }
            }

            // Remove bad tets (highest index first to keep indices stable).
            for bi in bad.sorted(by: >) { tets.remove(at: bi) }

            // For each boundary face, form a new tet with p.
            for (f, n) in faceCount where n == 1 {
                tets.append(Tet(a: f[0], b: f[1], c: f[2], d: i))
            }
        }

        // Strip any tet touching a super-tet vertex.
        let superRange = s0..<(s0 + 4)
        return tets.filter { t in
            !superRange.contains(t.a) &&
            !superRange.contains(t.b) &&
            !superRange.contains(t.c) &&
            !superRange.contains(t.d)
        }
    }

    // MARK: - Predicates

    /// True iff p lies strictly inside the circumsphere of tet t.
    private static func inCircumsphere(_ p: SIMD3<Float>,
                                       _ t: Tet,
                                       _ pts: [SIMD3<Float>]) -> Bool {
        let a = pts[t.a], b = pts[t.b], c = pts[t.c], d = pts[t.d]
        guard let (center, r2) = circumsphere(a, b, c, d) else { return false }
        let dx = p - center
        return simd_length_squared(dx) < r2 - 1e-9
    }

    /// Circumsphere center and squared radius of a tet, or nil if degenerate.
    private static func circumsphere(_ a: SIMD3<Float>,
                                     _ b: SIMD3<Float>,
                                     _ c: SIMD3<Float>,
                                     _ d: SIMD3<Float>) -> (SIMD3<Float>, Float)?
    {
        // Solve the 3x3 linear system from |x-a|^2 = |x-b|^2 = |x-c|^2 = |x-d|^2.
        let ab = b - a, ac = c - a, ad = d - a
        let M = simd_float3x3(rows: [ab, ac, ad])
        let det = M.determinant
        if abs(det) < 1e-12 { return nil }

        let rhs = SIMD3<Float>(
            0.5 * simd_dot(ab, ab),
            0.5 * simd_dot(ac, ac),
            0.5 * simd_dot(ad, ad)
        )
        let center = a + M.inverse * rhs
        let r2 = simd_length_squared(center - a)
        return (center, r2)
    }
}

// Convenience matrix initializer for (rows: [v1, v2, v3]).
private extension simd_float3x3 {
    init(rows: [SIMD3<Float>]) {
        precondition(rows.count == 3)
        self.init(columns: (
            SIMD3<Float>(rows[0].x, rows[1].x, rows[2].x),
            SIMD3<Float>(rows[0].y, rows[1].y, rows[2].y),
            SIMD3<Float>(rows[0].z, rows[1].z, rows[2].z)
        ))
    }
}
