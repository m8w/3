//  Milkdrop.metal
//  Microtonal Milkdrop-style visualizer
//
//  Pipeline overview:
//    1. WarpPass   — compute kernel that writes a UV-distortion (warp) texture
//                    driven by q1…q32.  Distortion is band-specific: each of
//                    the 32 frequency bins bends a different region of the grid.
//    2. FragmentPass — fullscreen quad that samples the warp texture and the
//                    previous frame's feedback texture to produce the final image.

#include <metal_stdlib>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// Shared uniform layout — must match MilkdropUniforms in MilkdropUniforms.swift
// ─────────────────────────────────────────────────────────────────────────────
struct MilkdropUniforms {
    float q[32];      // q1…q32: normalised magnitudes at 32 microtonal frequencies
    float time;       // seconds since launch
    float sampleRate;
    float _pad0;
    float _pad1;
};

// ─────────────────────────────────────────────────────────────────────────────
// Math helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Smooth 2-D Perlin-style hash — cheap, no texture lookup.
static float2 hash2(float2 p) {
    float3 p3 = fract(float3(p.xyx) * float3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

/// Value noise on a 2-D grid (returns [-1, 1]).
static float vnoise(float2 uv) {
    float2 i = floor(uv);
    float2 f = fract(uv);
    float2 u = f * f * (3.0 - 2.0 * f);   // Hermite smooth-step
    float a = hash2(i).x;
    float b = hash2(i + float2(1,0)).x;
    float c = hash2(i + float2(0,1)).x;
    float d = hash2(i + float2(1,1)).x;
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 2.0 - 1.0;
}

/// Sum 3 octaves of value noise — used to build the warp field.
static float fbm(float2 uv) {
    float v = 0.0;
    float amp = 0.5;
    for (int i = 0; i < 3; i++) {
        v   += amp * vnoise(uv);
        uv  *= 2.1;
        amp *= 0.5;
    }
    return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// PASS 1 — Warp map compute kernel
//
// Writes an RG16Float texture where each texel stores (du, dv): the
// displacement that the fragment pass will add to its UV before sampling
// the feedback buffer.  Each of the 32 q-bins modulates a specific spatial
// frequency and direction, so microtonal energy bends the grid in its own way.
// ─────────────────────────────────────────────────────────────────────────────
kernel void buildWarpMap(
    texture2d<float, access::write> warpOut  [[texture(0)]],
    constant MilkdropUniforms&      u        [[buffer(0)]],
    uint2                           gid      [[thread_position_in_grid]]
)
{
    const uint2 size = uint2(warpOut.get_width(), warpOut.get_height());
    if (any(gid >= size)) return;

    // Normalised UV in [-1, 1]
    float2 uv = (float2(gid) / float2(size)) * 2.0 - 1.0;
    uv.x *= float(size.x) / float(size.y);   // correct for aspect ratio

    float du = 0.0, dv = 0.0;

    // ── 32-bin contribution ───────────────────────────────────────────────────
    // Each bin i contributes a sinusoidal warp whose:
    //   • spatial frequency scales with i (low bins → broad waves, high → fine)
    //   • direction rotates around the circle as i increases
    //   • amplitude is q[i]
    //
    // This gives the microtonal scale a direct spatial signature:
    // a 31-EDO cluster will light up neighbouring bins simultaneously,
    // producing tightly-grouped ripple patterns distinct from 12-EDO clusters.

    for (int i = 0; i < 32; i++) {
        float mag   = u.q[i];
        if (mag < 1e-4) continue;   // skip silent bins (saves ALU)

        // Spatial frequency: bin 0 → large waves, bin 31 → fine detail.
        float freq  = 1.5 + float(i) * 0.6;

        // Direction: evenly space 32 directions around the circle.
        float angle = float(i) * (M_PI_F * 2.0 / 32.0);
        float2 dir  = float2(cos(angle), sin(angle));

        // Phase: slow temporal drift so the field evolves when audio is static.
        float phase = u.time * (0.08 + float(i) * 0.005);

        // Warp contribution for this bin.
        float wave  = sin(dot(uv, dir) * freq + phase);
        du += mag * dir.x * wave * 0.12;
        dv += mag * dir.y * wave * 0.12;
    }

    // ── FBM background warp ───────────────────────────────────────────────────
    // A gentle base turbulence that remains active even during silence, giving
    // the visualizer the classic Milkdrop "breathing" feel.
    float t    = u.time * 0.07;
    float bx   = fbm(uv * 1.8 + float2(t, 0.0)) * 0.04;
    float by   = fbm(uv * 1.8 + float2(0.0, t)) * 0.04;

    float2 warp = float2(du + bx, dv + by);
    warpOut.write(float4(warp, 0.0, 1.0), gid);
}

// ─────────────────────────────────────────────────────────────────────────────
// PASS 2 — Vertex shader (fullscreen triangle)
// ─────────────────────────────────────────────────────────────────────────────
struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

// One large triangle covers the entire clip space without a vertex buffer.
vertex VertexOut fullscreenVertex(uint vid [[vertex_id]])
{
    const float2 positions[3] = {
        float2(-1, -1),
        float2( 3, -1),
        float2(-1,  3)
    };
    VertexOut out;
    out.position = float4(positions[vid], 0, 1);
    out.uv       = positions[vid] * 0.5 + 0.5;
    out.uv.y     = 1.0 - out.uv.y;   // flip Y for Metal texture coordinates
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// PASS 2 — Fragment shader
//
// Samples the warp map, distorts the UV, then samples the previous frame's
// feedback texture.  Adds an emissive glow layer driven by the loudest q bins.
// ─────────────────────────────────────────────────────────────────────────────
fragment float4 milkdropFragment(
    VertexOut               in       [[stage_in]],
    constant MilkdropUniforms& u     [[buffer(0)]],
    texture2d<float>        feedback [[texture(0)]],  // previous frame
    texture2d<float>        warpMap  [[texture(1)]]   // output of buildWarpMap
)
{
    constexpr sampler smp(filter::linear, address::repeat);

    float2 uv = in.uv;

    // ── 1. Fetch warp displacement ────────────────────────────────────────────
    float2 warp = warpMap.sample(smp, uv).rg;

    // ── 2. Zoom + rotate around centre ───────────────────────────────────────
    // Compute aggregate bass (q1-q5) and treble (q27-q32) energies.
    float bass  = (u.q[0]+u.q[1]+u.q[2]+u.q[3]+u.q[4]) * 0.2;
    float treble= (u.q[27]+u.q[28]+u.q[29]+u.q[30]+u.q[31]) * 0.2;

    float2 centre = float2(0.5, 0.5);
    float2 rel    = uv - centre;

    // Rotation: treble transients spin the field; bass gives a slow drift.
    float rotAngle = treble * 0.18 * sin(u.time * 1.3)
                   + bass   * 0.04 * u.time;
    float cosA = cos(rotAngle), sinA = sin(rotAngle);
    rel = float2(cosA * rel.x - sinA * rel.y,
                 sinA * rel.x + cosA * rel.y);

    // Zoom: slight inward pull creates the Milkdrop "tunnel" decay.
    float zoom = 0.985 - bass * 0.015 + treble * 0.010;
    uv = centre + rel * zoom + warp;

    // ── 3. Sample previous frame (feedback) ───────────────────────────────────
    float4 prev = feedback.sample(smp, uv);

    // ── 4. Decay: prevent DC accumulation (equivalent to Milkdrop decay knob).
    prev.rgb *= 0.96;

    // ── 5. Emissive glow from the 32 q-bins ──────────────────────────────────
    // Each bin paints a coloured streak at a unique angular position, so
    // microtonal clusters appear as distinct colour arcs in the field.
    float2 fragUV = in.uv * 2.0 - 1.0;
    float  r = length(fragUV);
    float  a = atan2(fragUV.y, fragUV.x);   // angle in [-π, π]

    float3 glow = float3(0.0);
    for (int i = 0; i < 32; i++) {
        float mag = u.q[i];
        if (mag < 0.01) continue;

        // Angular "slot" for this bin.
        float binAngle = float(i) * (M_PI_F * 2.0 / 32.0) - M_PI_F;
        float angleDiff = abs(fmod(a - binAngle + 3.0 * M_PI_F, 2.0 * M_PI_F) - M_PI_F);
        float stripe    = exp(-angleDiff * angleDiff * 80.0);  // narrow Gaussian arc

        // Radial envelope: glow at r ≈ 0.3–0.8, fades at centre and edge.
        float radEnv = smoothstep(0.25, 0.45, r) * smoothstep(0.9, 0.6, r);

        // Colour: HSV-like mapping from bin index.
        float  hue = float(i) / 32.0;
        float3 col = 0.5 + 0.5 * cos(2.0 * M_PI_F * (hue + float3(0, 0.33, 0.67)));

        glow += col * stripe * radEnv * mag * 0.6;
    }

    float4 result = float4(prev.rgb + glow, 1.0);
    return result;
}
