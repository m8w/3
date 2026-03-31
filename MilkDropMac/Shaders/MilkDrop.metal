// MilkDrop.metal — Metal shaders for MilkDrop-style visualization
// Implements: warp mesh, feedback, wave rendering, composite pass
// Designed to match the MilkDrop 2/3 rendering pipeline on Metal

#include <metal_stdlib>
using namespace metal;

// MARK: - Shared types

struct VertexOut {
    float4 position [[position]];
    float2 texcoord;
    float4 color;
};

// Full-screen quad vertex shader. Generates positions from vertex_id — no vertex
// buffer or vertex descriptor needed, so no Metal validation issues at draw time.
vertex VertexOut quad_vertex(uint vid [[vertex_id]]) {
    const float2 positions[6] = {
        float2(-1,-1), float2( 1,-1), float2(-1, 1),
        float2(-1, 1), float2( 1,-1), float2( 1, 1)
    };
    const float2 uvs[6] = {
        float2(0,1), float2(1,1), float2(0,0),
        float2(0,0), float2(1,1), float2(1,0)
    };
    VertexOut out;
    out.position = float4(positions[vid], 0.0, 1.0);
    out.texcoord = uvs[vid];
    out.color    = float4(1.0);
    return out;
}

struct MilkDropUniforms {
    // Time
    float time;
    float fps;
    float frame;
    float progress;         // 0–1 preset transition blend

    // Audio
    float bass;
    float mid;
    float treble;
    float vol;
    float bass_att;         // Attenuated bass
    float mid_att;
    float treble_att;
    float vol_att;

    // Preset parameters
    float zoom;
    float rot;
    float warp;
    float cx;
    float cy;
    float dx;
    float dy;
    float sx;
    float sy;
    float decay;
    float gamma;
    float warpSpeed;
    float videoEchoAlpha;
    float videoEchoZoom;
    int   videoEchoOrientation;

    // MilkDrop user variables (q1–q32 from per-frame equations)
    float q[32];

    // Resolution
    float2 resolution;
    float  aspect;           // width / height
};

// MARK: - Warp pass: distorts the feedback texture

// Warp (per-vertex) — applies zoom, rotation, warp distortion, feedback
fragment float4 warp_fragment(
    VertexOut in            [[stage_in]],
    texture2d<float> prev   [[texture(0)]],   // Previous frame (feedback)
    constant MilkDropUniforms &u [[buffer(0)]]
) {
    constexpr sampler tex_sampler(address::repeat, filter::linear);

    // UV in -1..1 space centered at (cx, cy)
    float2 uv = in.texcoord;
    float2 uvCentered = (uv - float2(u.cx, u.cy)) * float2(u.aspect, 1.0);

    // Zoom
    float zoom = u.zoom;
    zoom = max(zoom, 0.001);
    uvCentered /= zoom;

    // Rotation
    float c = cos(u.rot);
    float s = sin(u.rot);
    uvCentered = float2(
        uvCentered.x * c - uvCentered.y * s,
        uvCentered.x * s + uvCentered.y * c
    );

    // Scale
    uvCentered *= float2(u.sx, u.sy);

    // Translation
    uvCentered += float2(u.dx, u.dy) * 2.0;

    // Warp (psychedelic swirling distortion)
    float warpAmt = u.warp;
    float t = u.time * u.warpSpeed * 0.5;
    float warpX = sin(t * 1.11 + uvCentered.y * 3.0) * warpAmt * 0.03;
    float warpY = cos(t * 0.93 + uvCentered.x * 2.5) * warpAmt * 0.03;
    uvCentered += float2(warpX, warpY);

    // Un-center
    float2 sampleUV = uvCentered / float2(u.aspect, 1.0) + float2(u.cx, u.cy);

    // Sample feedback texture
    float4 color = prev.sample(tex_sampler, sampleUV);

    // Apply decay only — gamma is applied once in the composite pass,
    // not here, to avoid double-gamma crushing everything to black.
    color.rgb *= u.decay;

    return color;
}

// MARK: - Wave rendering

struct WaveUniforms {
    float4 color;         // RGBA — used when perPointColors == 0
    float  thickness;
    int    drawThick;
    int    additive;
    int    useDots;
    float  smoothing;
    int    sampleCount;
    int    perPointColors;  // 1 = use per-vertex color buffer, 0 = use uniform color
};

vertex VertexOut wave_vertex(
    uint vid                        [[vertex_id]],
    constant float2 *positions      [[buffer(0)]],
    constant WaveUniforms &wave     [[buffer(1)]],
    constant float4 *colors         [[buffer(2)]]   // per-vertex colors (may be unused)
) {
    VertexOut out;
    float2 pos = positions[vid];
    out.position = float4(pos * 2.0 - 1.0, 0, 1);
    out.texcoord = pos;
    out.color    = wave.perPointColors != 0 ? colors[vid] : wave.color;
    return out;
}

fragment float4 wave_fragment(VertexOut in [[stage_in]]) {
    return in.color;
}

// MARK: - Shape rendering

struct ShapeUniforms {
    float4 color;
    float4 color2;          // Inner color for gradient
    float4 borderColor;
    float2 center;          // 0..1 UV space
    float  radius;
    float  angle;
    int    sides;
    int    additive;
    int    thickOutline;
    int    textured;        // 1 = sample warp texture
    float  tex_ang;         // Texture rotation angle
    float  tex_zoom;        // Texture zoom (1 = normal)
};

vertex VertexOut shape_vertex(
    uint vid                        [[vertex_id]],
    constant float2 *positions      [[buffer(0)]],
    constant ShapeUniforms &shape   [[buffer(1)]],
    constant MilkDropUniforms &u    [[buffer(2)]]
) {
    VertexOut out;
    float2 pos = positions[vid];
    out.position = float4(pos * 2.0 - 1.0, 0, 1);

    if (shape.textured != 0) {
        // Compute texture UV: rotate offset from center by tex_ang, scale by tex_zoom
        float2 offset = pos - shape.center;
        float c = cos(-shape.tex_ang);
        float s = sin(-shape.tex_ang);
        float2 rotated = float2(offset.x * c - offset.y * s,
                                offset.x * s + offset.y * c);
        out.texcoord = rotated / max(shape.tex_zoom, 0.001) + 0.5;
    } else {
        out.texcoord = pos;
    }

    // Radial gradient: center vs. edge
    float dist = length(pos - shape.center);
    float t = saturate(dist / max(shape.radius, 0.0001));
    out.color = mix(shape.color, shape.color2, t);
    return out;
}

fragment float4 shape_fragment(
    VertexOut in                    [[stage_in]],
    texture2d<float> warpTex        [[texture(0)]],
    constant ShapeUniforms &shape   [[buffer(0)]]
) {
    if (shape.textured != 0) {
        constexpr sampler s(address::repeat, filter::linear);
        float4 tex = warpTex.sample(s, in.texcoord);
        return float4(tex.rgb * in.color.rgb, in.color.a);
    }
    return in.color;
}

// MARK: - Composite pass (final output)

struct CompositeUniforms {
    float brightness;
    float gamma;
    float videoEchoAlpha;
    float videoEchoZoom;
    int   videoEchoOrientation;
    float2 resolution;
    float  time;
    float  bass;
    float  treble;
    // q variables for composite shader
    float q[32];
    // Fractal stream overlay
    float fractalBlend;
    int   fractalEnabled;
    // Per-frame color overlay (r,g,b set by EEL per-frame equations; amb = strength)
    float r; float g; float b; float amb;
};

fragment float4 composite_fragment(
    VertexOut in                    [[stage_in]],
    texture2d<float> warpTex        [[texture(0)]],   // Warped frame
    texture2d<float> waveTex        [[texture(1)]],   // Wave layer
    texture2d<float> shapeTex       [[texture(2)]],   // Shapes layer
    texture2d<float> fractalTex     [[texture(3)]],   // Fractal overlay
    constant CompositeUniforms &u   [[buffer(0)]]
) {
    constexpr sampler s(address::clamp_to_edge, filter::linear);

    float2 uv = in.texcoord;
    float4 warp  = warpTex.sample(s, uv);
    float4 waves = waveTex.sample(s, uv);
    float4 shapes = shapeTex.sample(s, uv);

    // Composite: warp base + additive waves + shapes
    float4 color = warp;
    color.rgb = mix(color.rgb, waves.rgb, waves.a);
    color.rgb = mix(color.rgb, shapes.rgb, shapes.a);

    // Fractal stream overlay (additive blend for glow effect)
    if (u.fractalEnabled != 0) {
        float4 fractal = fractalTex.sample(s, uv);
        color.rgb += fractal.rgb * fractal.a * u.fractalBlend;
    }

    // Per-frame color overlay: r,g,b set by EEL per-frame equations, amb is strength.
    // Additive so bright presets glow with the animated palette. Skipped when amb==0.
    if (u.amb > 0.0) {
        color.rgb += float3(u.r, u.g, u.b) * u.amb;
    }

    // Gamma (MilkDrop fGammaAdj): simple brightness multiplier, NOT a power curve.
    // Higher gamma = brighter image. Applied once here, never in the warp pass.
    color.rgb *= u.gamma;
    color.rgb *= u.brightness;

    // Vignette (subtle)
    float2 c = uv - 0.5;
    float vignette = 1.0 - dot(c, c) * 0.5;
    color.rgb *= vignette;

    return saturate(color);
}

// MARK: - Preset blend / transition

struct BlendUniforms {
    float  blend;           // 0 = preset A, 1 = preset B
    int    blendType;       // 0=zoom, 1=side, 2=plasma, etc.
    float  time;
    float2 resolution;
};

fragment float4 blend_fragment(
    VertexOut in                [[stage_in]],
    texture2d<float> texA       [[texture(0)]],   // Outgoing preset
    texture2d<float> texB       [[texture(1)]],   // Incoming preset
    constant BlendUniforms &u   [[buffer(0)]]
) {
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = in.texcoord;
    float t = u.blend;

    float4 colA = texA.sample(s, uv);
    float4 colB = texB.sample(s, uv);
    float4 result;

    switch (u.blendType) {
        case 0: { // Zoom blend
            float zoom = mix(1.0, 0.5, t);
            float2 zUV = (uv - 0.5) * zoom + 0.5;
            float4 zB = texB.sample(s, zUV);
            result = mix(colA, zB, smoothstep(0.3, 0.7, t));
            break;
        }
        case 1: { // Side wipe
            float edge = t;
            result = uv.x < edge ? colB : colA;
            break;
        }
        case 2: { // Plasma (animated noise-based mask)
            float n = sin(uv.x * 10 + u.time) * cos(uv.y * 8 - u.time * 0.7);
            float mask = step(n * 0.5 + 0.5, t);
            result = mix(colA, colB, mask);
            break;
        }
        case 3: { // Cercle (expanding circle)
            float dist = length(uv - 0.5) * 2.0;
            float mask = step(dist, t * 1.414);
            result = mix(colA, colB, mask);
            break;
        }
        case 4: { // Checkerboard
            float cx = floor(uv.x * 8);
            float cy = floor(uv.y * 8);
            float checker = fmod(cx + cy, 2.0);
            float edge = checker < 0.5 ? t * 2.0 : t * 2.0 - 1.0;
            result = mix(colA, colB, saturate(edge));
            break;
        }
        case 5: { // Stars (radial burst)
            float2 d = uv - 0.5;
            float angle = atan2(d.y, d.x);
            float star = sin(angle * 8) * 0.1 + 0.9;
            float dist2 = length(d) * 2.0 / star;
            result = mix(colA, colB, step(dist2, t));
            break;
        }
        case 6: { // Bezier warp morph
            float mt = 1.0 - t;
            float mt2 = mt * mt;
            float t2 = t * t;
            // Control points orbit with time for organic feel
            float2 ctrl1 = float2(0.5 + sin(u.time * 0.7) * 0.3, 0.2 + cos(u.time * 0.5) * 0.2);
            float2 ctrl2 = float2(0.5 + cos(u.time * 0.6) * 0.3, 0.8 + sin(u.time * 0.4) * 0.2);
            // Warp sample UV via Bezier offset
            float2 warpOffset = (ctrl1 * 3.0 * mt2 * t + ctrl2 * 3.0 * mt * t2) - uv;
            float2 warpedUV = uv + warpOffset * t * (1.0 - t) * 2.0;
            warpedUV = clamp(warpedUV, 0.0, 1.0);
            float4 warpedA = texA.sample(s, warpedUV);
            result = mix(warpedA, colB, smoothstep(0.2, 0.8, t));
            break;
        }
        case 7: { // Mesh morph (grid-based warp)
            float2 cell = floor(uv * 8.0) / 8.0;
            float2 morphOff = float2(
                sin(cell.x * 6.28 + cell.y * 4.19 + u.time) * 0.08,
                cos(cell.x * 5.13 + cell.y * 7.31 + u.time * 0.8) * 0.08
            );
            float2 uvA = clamp(uv + morphOff * (1.0 - t), 0.0, 1.0);
            float2 uvB = clamp(uv - morphOff * t, 0.0, 1.0);
            result = mix(texA.sample(s, uvA), texB.sample(s, uvB), smoothstep(0.1, 0.9, t));
            break;
        }
        case 8: { // Fractal dissolve
            float2 fc = (uv - 0.5) * 3.5;
            float2 z2 = float2(0.0);
            int fiter = 0;
            for (int i = 0; i < 24; i++) {
                z2 = float2(z2.x * z2.x - z2.y * z2.y + fc.x,
                            2.0 * z2.x * z2.y + fc.y);
                if (dot(z2, z2) > 4.0) break;
                fiter++;
            }
            float mask = float(fiter) / 24.0;
            result = mix(colA, colB, step(mask, t));
            break;
        }
        case 9: { // Fractal stream (spiral twist)
            float2 p = uv * 2.0 - 1.0;
            float radius = length(p);
            float angle = atan2(p.y, p.x);
            float twist = sin(radius * 4.0 * 3.14159 - u.time * 2.0) * t * 1.5;
            float2 twistedUV = float2(
                0.5 + 0.5 * cos(angle + twist) * radius,
                0.5 + 0.5 * sin(angle + twist) * radius
            );
            twistedUV = clamp(twistedUV, 0.0, 1.0);
            result = mix(texA.sample(s, twistedUV), colB, smoothstep(0.15, 0.85, t));
            break;
        }
        default: { // Simple cross-fade
            result = mix(colA, colB, t);
            break;
        }
    }

    return result;
}

// MARK: - Fractal stream visualizer (audio-reactive Julia set)

fragment float4 fractal_stream_fragment(
    VertexOut in [[stage_in]],
    constant MilkDropUniforms &u [[buffer(0)]]
) {
    float2 uv = in.texcoord * 2.0 - 1.0;
    uv.x *= u.aspect;

    // Julia set — c parameter driven by audio + time
    float2 c = float2(
        -0.70 + sin(u.time * 0.17) * (0.25 + u.bass * 0.35),
         0.27 + cos(u.time * 0.11) * (0.18 + u.treble * 0.28)
    );

    // Zoom driven by bass
    float zoom = 1.8 / max(u.zoom * 0.6, 0.4);
    float2 z = uv * zoom;

    int iterations = 0;
    const int maxIter = 96;
    while (iterations < maxIter && dot(z, z) < 4.0) {
        z = float2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        iterations++;
    }

    if (iterations == maxIter) {
        // Interior: deep glow pulsed by vol
        float glow = 0.04 + u.vol * 0.12;
        return float4(glow * 0.3, glow * 0.1, glow * 0.8, 0.9);
    }

    // Smooth escape time for banding-free coloring
    float escape = float(iterations) - log2(log2(dot(z, z)));
    float t = fract(escape * 0.04 + u.time * 0.015);

    // Audio-reactive palette
    float3 col;
    col.r = 0.5 + 0.5 * sin(t * 6.2832 * 1.0 + u.bass   * 5.0 + u.time * 0.4);
    col.g = 0.5 + 0.5 * sin(t * 6.2832 * 1.7 + u.mid    * 3.5 + u.time * 0.3 + 2.1);
    col.b = 0.5 + 0.5 * sin(t * 6.2832 * 2.3 + u.treble * 2.5 + u.time * 0.6 + 4.2);

    // Brightness responds to overall volume
    float brightness = 0.25 + u.vol * 0.75;
    float alpha = 0.6 + u.bass * 0.3;

    return float4(col * brightness, alpha);
}

// MARK: - Mesh warp pass (per_vertex equations pre-compute per-vertex sample UVs)

struct MeshVertex {
    float2 screenPos;   // 0..1 grid position
    float2 sampleUV;    // Pre-computed UV to sample from previous frame
};

vertex VertexOut mesh_vertex(
    uint vid                        [[vertex_id]],
    constant MeshVertex *vertices   [[buffer(0)]],
    constant MilkDropUniforms &u    [[buffer(1)]]
) {
    MeshVertex v = vertices[vid];
    VertexOut out;
    out.position = float4(v.screenPos * 2.0 - 1.0, 0, 1);
    out.texcoord = v.sampleUV;
    out.color    = float4(1);
    return out;
}

fragment float4 mesh_warp_fragment(
    VertexOut in                    [[stage_in]],
    texture2d<float> prev           [[texture(0)]],
    constant MilkDropUniforms &u    [[buffer(0)]]
) {
    constexpr sampler s(address::repeat, filter::linear);
    float4 color = prev.sample(s, in.texcoord);
    color.rgb *= u.decay;   // gamma applied once in composite, not here
    return color;
}

// MARK: - Present pass: copy finalTexture to drawable (avoids blit format constraints)

fragment float4 copy_fragment(
    VertexOut in           [[stage_in]],
    texture2d<float> src   [[texture(0)]]
) {
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    return src.sample(s, in.texcoord);
}

// MARK: - Spectrum / FFT visualization overlay

vertex VertexOut spectrum_vertex(
    uint vid                        [[vertex_id]],
    constant float2 *positions      [[buffer(0)]],
    constant float4 *colors         [[buffer(1)]]
) {
    VertexOut out;
    out.position = float4(positions[vid] * 2.0 - 1.0, 0, 1);
    out.texcoord = positions[vid];
    out.color    = colors[vid];
    return out;
}

fragment float4 spectrum_fragment(VertexOut in [[stage_in]]) {
    return in.color;
}
