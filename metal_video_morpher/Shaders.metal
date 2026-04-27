//
//  Shaders.metal
//  metal_video_morpher
//
//  Spacetime-consistent morph kernels.
//
//  Pipeline (per output pixel at (x, y, t)):
//
//      sample_xy =  x
//                 - D_xy(u(x), v(y), w(t))     // tricubic Bezier FFD
//                 - delta_tet(x, y, t)         // spacetime tetra barycentric
//                 - flow_offset(x, y, t)       // optical-flow advection
//
//      out(x, t) = bilinear_sample( I_0, sample_xy )
//
//  See EQUATIONS.md for the full derivation.
//

#include <metal_stdlib>
using namespace metal;

// MARK: - Bernstein basis ----------------------------------------------------

// Cubic Bernstein polynomials B_i^3(u), i = 0..3.
inline float4 bernstein3(float u) {
    const float v = 1.0f - u;
    return float4(v*v*v,
                  3.0f * u * v * v,
                  3.0f * u * u * v,
                  u*u*u);
}

// Quadratic Bernstein polynomials B_i^2(u), i = 0..2.  (Optional triquadratic mode.)
inline float3 bernstein2(float u) {
    const float v = 1.0f - u;
    return float3(v*v, 2.0f*u*v, u*u);
}

// MARK: - Tricubic Bezier FFD -----------------------------------------------
//
// Control lattice is laid out as P_{ijk} with i = u-axis (4), j = v-axis (4),
// k = w-axis (4). Index = i + 4*j + 16*k. Each entry is a float3 displacement
// in normalized image coords (xy) and time (z, usually 0 for purely spatial).

inline float3 ffd_tricubic(constant float3 *P,        // 64 control points
                           float u, float v, float w)
{
    const float4 Bu = bernstein3(clamp(u, 0.0f, 1.0f));
    const float4 Bv = bernstein3(clamp(v, 0.0f, 1.0f));
    const float4 Bw = bernstein3(clamp(w, 0.0f, 1.0f));

    float3 acc = float3(0.0f);
    for (uint k = 0; k < 4; ++k) {
        float3 plane = float3(0.0f);
        for (uint j = 0; j < 4; ++j) {
            float3 row = float3(0.0f);
            for (uint i = 0; i < 4; ++i) {
                row += Bu[i] * P[i + 4u*j + 16u*k];
            }
            plane += Bv[j] * row;
        }
        acc += Bw[k] * plane;
    }
    return acc;
}

// MARK: - Spacetime tetrahedral barycentric warp -----------------------------
//
// Each tet is described by:
//   - 4 source vertices  v_i = (x, y, t)
//   - 4 target vertices  v_i*
// We pack into a flat buffer: stride = 8 float3 per tet (4 source then 4 target).
//
// The CPU side culls tets to candidates near (x, y, t). For simplicity here we
// linear-scan up to `tetCount` tets; for production swap in a uniform-grid
// acceleration structure.

inline float4 barycentric_3d(float3 q, float3 a, float3 b, float3 c, float3 d) {
    // Solve:  q = la*a + lb*b + lc*c + ld*d, sum = 1.
    // Equivalent to expressing (q - d) in basis (a-d, b-d, c-d).
    const float3 va = a - d;
    const float3 vb = b - d;
    const float3 vc = c - d;
    const float3 vq = q - d;

    const float3 cbc = cross(vb, vc);
    const float det = dot(va, cbc);
    if (fabs(det) < 1e-9f) return float4(-1.0f);   // degenerate tet

    const float invDet = 1.0f / det;
    const float la = dot(vq, cbc) * invDet;
    const float lb = dot(vq, cross(vc, va)) * invDet;
    const float lc = dot(vq, cross(va, vb)) * invDet;
    const float ld = 1.0f - la - lb - lc;
    return float4(la, lb, lc, ld);
}

inline float3 tetra_warp_offset(constant float3 *tets,    // 8 * tetCount entries
                                uint tetCount,
                                float3 q)
{
    for (uint t = 0; t < tetCount; ++t) {
        const uint base = 8u * t;
        const float3 a = tets[base + 0];
        const float3 b = tets[base + 1];
        const float3 c = tets[base + 2];
        const float3 d = tets[base + 3];

        const float4 lam = barycentric_3d(q, a, b, c, d);
        if (all(lam >= -1e-4f)) {
            const float3 A = tets[base + 4];
            const float3 B = tets[base + 5];
            const float3 C = tets[base + 6];
            const float3 D = tets[base + 7];
            const float3 qStar = lam.x*A + lam.y*B + lam.z*C + lam.w*D;
            return qStar - q;     // displacement
        }
    }
    return float3(0.0f);
}

// MARK: - Composed consistent-morph kernel -----------------------------------

struct ConsistentMorphParams {
    uint   width;
    uint   height;
    uint   frameIndex;
    uint   frameCount;

    float  morphStrength;     // global scale on FFD + tetra
    float  flowWeight;        // scale on flow advection
    float  ffdWeight;
    float  tetWeight;

    uint   tetCount;
    uint   useFlow;           // 0/1
};

kernel void consistentMorphKernel(
    texture2d<float, access::sample>      sourceFrame      [[ texture(0) ]],
    texture2d<float, access::sample>      flowField        [[ texture(1) ]],   // RG = (vx, vy) in pixels
    texture2d<float, access::write>       outFrame         [[ texture(2) ]],
    constant float3                      *bezierLattice    [[ buffer(0) ]],    // 64 entries
    constant float3                      *tetBuffer        [[ buffer(1) ]],    // 8 * tetCount
    constant ConsistentMorphParams       &P                [[ buffer(2) ]],
    uint2                                 gid              [[ thread_position_in_grid ]]
) {
    if (gid.x >= P.width || gid.y >= P.height) return;

    constexpr sampler S(coord::normalized,
                        address::clamp_to_edge,
                        filter::linear);

    const float u = (float(gid.x) + 0.5f) / float(P.width);
    const float v = (float(gid.y) + 0.5f) / float(P.height);
    const float w = (P.frameCount > 1)
                  ? (float(P.frameIndex) / float(P.frameCount - 1u))
                  : 0.0f;

    // 1. Tricubic Bezier FFD displacement (xy in normalized coords).
    float3 dFFD = ffd_tricubic(bezierLattice, u, v, w);

    // 2. Spacetime tetrahedral correction.
    const float3 q = float3(u, v, w);
    float3 dTet = (P.tetCount > 0u)
                ? tetra_warp_offset(tetBuffer, P.tetCount, q)
                : float3(0.0f);

    // 3. Optical-flow advection (sampled in normalized coords, returned px).
    float2 dFlow = float2(0.0f);
    if (P.useFlow != 0u) {
        const float2 flowPx = flowField.sample(S, float2(u, v)).rg;
        dFlow = flowPx / float2(P.width, P.height);
    }

    // Weighted sum, then scale by global morph strength.
    float2 sampleUV = float2(u, v)
                    - P.morphStrength * (P.ffdWeight * dFFD.xy + P.tetWeight * dTet.xy)
                    - P.flowWeight    * dFlow;

    const float4 c = sourceFrame.sample(S, sampleUV);
    outFrame.write(c, gid);
}

// MARK: - Standalone flow-only advection (handy for debugging) ---------------

kernel void flowAdvectKernel(
    texture2d<float, access::sample> sourceFrame [[ texture(0) ]],
    texture2d<float, access::sample> flowField   [[ texture(1) ]],
    texture2d<float, access::write>  outFrame    [[ texture(2) ]],
    constant float                  &dt          [[ buffer(0) ]],
    uint2                            gid         [[ thread_position_in_grid ]]
) {
    const uint W = outFrame.get_width();
    const uint H = outFrame.get_height();
    if (gid.x >= W || gid.y >= H) return;

    constexpr sampler S(coord::normalized,
                        address::clamp_to_edge,
                        filter::linear);

    const float2 uv = (float2(gid) + 0.5f) / float2(W, H);
    const float2 vpx = flowField.sample(S, uv).rg;
    const float2 back = uv - dt * vpx / float2(W, H);
    outFrame.write(sourceFrame.sample(S, back), gid);
}
