// MilkDrop.metal — Metal shaders for MilkDrop-style visualization
// Implements: warp mesh, feedback, wave rendering, composite pass
// Designed to match the MilkDrop 2/3 rendering pipeline on Metal

#include <metal_stdlib>
using namespace metal;

// MARK: - Shared types

struct VertexIn {
    float2 position [[attribute(0)]];
    float2 texcoord [[attribute(1)]];
};

struct VertexOut {
    float4 position [[position]];
    float2 texcoord;
    float4 color;
};

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

// Full-screen quad vertex shader
vertex VertexOut warp_vertex(
    VertexIn in [[stage_in]],
    constant MilkDropUniforms &u [[buffer(0)]]
) {
    VertexOut out;
    out.position = float4(in.position, 0, 1);
    out.texcoord = in.texcoord;
    out.color = float4(1);
    return out;
}

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
    float t = u.time * u.warp * 0.5;
    float warpX = sin(t * 1.11 + uvCentered.y * 3.0) * warpAmt * 0.03;
    float warpY = cos(t * 0.93 + uvCentered.x * 2.5) * warpAmt * 0.03;
    uvCentered += float2(warpX, warpY);

    // Un-center
    float2 sampleUV = uvCentered / float2(u.aspect, 1.0) + float2(u.cx, u.cy);

    // Sample feedback texture
    float4 color = prev.sample(tex_sampler, sampleUV);

    // Apply gamma and decay
    color.rgb = pow(max(color.rgb, 0.0), float3(u.gamma));
    color.rgb *= u.decay;

    return color;
}

// MARK: - Wave rendering

struct WaveUniforms {
    float4 color;         // RGBA
    float  thickness;
    int    drawThick;
    int    additive;
    int    useDots;
    float  smoothing;
    int    sampleCount;   // How many waveform points
};

vertex VertexOut wave_vertex(
    uint vid                        [[vertex_id]],
    constant float2 *positions      [[buffer(0)]],
    constant WaveUniforms &wave     [[buffer(1)]]
) {
    VertexOut out;
    float2 pos = positions[vid];
    out.position = float4(pos * 2.0 - 1.0, 0, 1);
    out.texcoord = pos;
    out.color    = wave.color;
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
    out.texcoord = pos;

    // Radial gradient: center vs. edge
    float dist = length(pos - shape.center);
    float t = saturate(dist / shape.radius);
    out.color = mix(shape.color, shape.color2, t);
    return out;
}

fragment float4 shape_fragment(VertexOut in [[stage_in]]) {
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
};

vertex VertexOut composite_vertex(
    VertexIn in [[stage_in]],
    constant CompositeUniforms &u [[buffer(0)]]
) {
    VertexOut out;
    out.position = float4(in.position, 0, 1);
    out.texcoord = in.texcoord;
    out.color    = float4(1);
    return out;
}

fragment float4 composite_fragment(
    VertexOut in                    [[stage_in]],
    texture2d<float> warpTex        [[texture(0)]],   // Warped frame
    texture2d<float> waveTex        [[texture(1)]],   // Wave layer
    texture2d<float> shapeTex       [[texture(2)]],   // Shapes layer
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

    // Gamma / brightness
    color.rgb = pow(max(color.rgb, 0.0), float3(u.gamma));
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

vertex VertexOut blend_vertex(
    VertexIn in [[stage_in]],
    constant BlendUniforms &u [[buffer(0)]]
) {
    VertexOut out;
    out.position = float4(in.position, 0, 1);
    out.texcoord = in.texcoord;
    out.color    = float4(1);
    return out;
}

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
        default: { // Simple cross-fade
            result = mix(colA, colB, t);
            break;
        }
    }

    return result;
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
