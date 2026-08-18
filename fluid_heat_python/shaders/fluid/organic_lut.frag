#version 330 core
// Heat -> colour transfer. Cross-fades an incandescent palette (fresh heat)
// against an organic one (lingering, cooling density), adds reaction-diffusion
// veins, and composites the flow-warped asemic / archive-skin layer.

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_tex0;   // fluid state  R=u G=v B=T A=D
uniform sampler2D u_tex1;   // asemic / skin layer (RGBA)
uniform sampler2D u_tex2;   // reaction-diffusion (R=U G=V)
uniform vec2  u_texel;

uniform float u_heat_gain;
uniform float u_density_gain;
uniform float u_asemic_mix;
uniform float u_asemic_flow;
uniform float u_glow;
uniform float u_exposure;
uniform float u_organic_bias;
uniform float u_vein_gain;
uniform float u_decay_mix;

vec3 incandescent(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.0);
    vec3 c1 = vec3(0x1a, 0x00, 0x33) / 255.0;
    vec3 c2 = vec3(0xe6, 0x3e, 0x00) / 255.0;
    vec3 c3 = vec3(0xff, 0xcc, 0x00) / 255.0;
    vec3 c4 = vec3(1.0);
    if (t < 0.25)      return mix(c0, c1, t / 0.25);
    else if (t < 0.50) return mix(c1, c2, (t - 0.25) / 0.25);
    else if (t < 0.75) return mix(c2, c3, (t - 0.50) / 0.25);
    else               return mix(c3, c4, (t - 0.75) / 0.25);
}

vec3 organic(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0x05, 0x0a, 0x08) / 255.0;   // near-black forest
    vec3 c1 = vec3(0x2e, 0x1a, 0x45) / 255.0;   // vein purple
    vec3 c2 = vec3(0x24, 0x55, 0x33) / 255.0;   // moss
    vec3 c3 = vec3(0x8a, 0xb0, 0x4a) / 255.0;   // lichen
    vec3 c4 = vec3(0xe8, 0xe0, 0xb4) / 255.0;   // bone
    if (t < 0.25)      return mix(c0, c1, t / 0.25);
    else if (t < 0.50) return mix(c1, c2, (t - 0.25) / 0.25);
    else if (t < 0.75) return mix(c2, c3, (t - 0.50) / 0.25);
    else               return mix(c3, c4, (t - 0.75) / 0.25);
}

void main() {
    vec4 s = texture(u_tex0, v_uv);
    vec2 vel = s.rg;
    float T = clamp(s.b * u_heat_gain, 0.0, 1.0);
    float D = clamp(s.a * u_density_gain, 0.0, 1.0);

    // decayed cells (density lingering with low heat) read as organic
    float decay = clamp(D - T, 0.0, 1.0);
    float org_w = clamp(decay * u_decay_mix + u_organic_bias * (1.0 - T) * D, 0.0, 1.0);

    // Veins are gated by the fluid's own presence - reaction-diffusion should
    // read as growth *through* the medium, not as free-floating pattern.
    float V = texture(u_tex2, v_uv).g;
    float presence = smoothstep(0.02, 0.35, max(D, T));
    float veins = smoothstep(0.2, 0.55, V) * u_vein_gain * presence;

    // asemic / skin uv warped by the local velocity - the fluid carries it
    vec2 uv1 = clamp(v_uv + vel * u_asemic_flow, 0.0, 1.0);
    vec4 asemic = texture(u_tex1, uv1);

    vec3 hot  = incandescent(T);
    vec3 cool = organic(clamp(D + veins * 0.4, 0.0, 1.0));
    vec3 pal  = mix(hot, cool, org_w);
    float lum = D + T * 0.5 + veins * 0.35;
    vec3 col = pal * lum;

    col = mix(col, col + cool * veins * 0.6, 0.7);

    vec3 scrib = asemic.rgb * (0.4 + 0.9 * mix(hot, cool, org_w));
    col = mix(col, col + scrib, u_asemic_mix * asemic.a);

    // neighbourhood glow
    vec4 n = 0.25 * (
        texture(u_tex0, v_uv + vec2( 2.0 * u_texel.x, 0.0)) +
        texture(u_tex0, v_uv + vec2(-2.0 * u_texel.x, 0.0)) +
        texture(u_tex0, v_uv + vec2(0.0,  2.0 * u_texel.y)) +
        texture(u_tex0, v_uv + vec2(0.0, -2.0 * u_texel.y))
    );
    float gT = clamp(n.b * u_heat_gain, 0.0, 1.0);
    float gD = clamp(n.a * u_density_gain, 0.0, 1.0);
    float gw = clamp(max(gD - gT, 0.0) * u_decay_mix
                     + u_organic_bias * (1.0 - gT) * gD, 0.0, 1.0);
    col += mix(incandescent(gT), organic(gD), gw) * n.a * u_glow;

    col *= u_exposure;
    col = col / (1.0 + col);              // Reinhard
    out_color = vec4(col, clamp(D + T, 0.0, 1.0));
}
