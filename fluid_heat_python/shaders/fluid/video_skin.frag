#version 330 core
// Channel A ("skin", the 53k archive). The clip is UV-warped by the local
// fluid velocity so it breathes with the flow, tinted by the current heat
// colour, and mixed into the rendered fluid by density x heat.

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_tex0;   // fluid state
uniform sampler2D u_tex1;   // colour pass (from organic_lut)
uniform sampler2D u_tex2;   // channel A clip
uniform float u_skin_mix;
uniform float u_warp;
uniform float u_tint;
uniform float u_contrast;
uniform float u_heat_mask;

vec3 apply_contrast(vec3 c, float k) {
    return clamp((c - 0.5) * k + 0.5, 0.0, 1.0);
}

void main() {
    vec4 fluid = texture(u_tex0, v_uv);
    vec4 base  = texture(u_tex1, v_uv);

    vec2 uv2 = clamp(v_uv + fluid.rg * u_warp, 0.0, 1.0);
    vec3 skin = apply_contrast(texture(u_tex2, uv2).rgb, u_contrast);

    vec3 tinted = mix(skin, skin * base.rgb * 2.0, u_tint);

    float T = clamp(fluid.b * u_heat_mask, 0.0, 1.0);
    float D = clamp(fluid.a, 0.0, 1.0);
    float m = u_skin_mix * (0.4 + 0.6 * (T + D * 0.5));

    out_color = vec4(mix(base.rgb, base.rgb + tinted * D, m), base.a);
}
