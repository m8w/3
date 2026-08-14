#version 330 core
// Vorticity confinement - re-inject the small-scale rotation that numerical
// diffusion eats, so the flow keeps curling expressively.

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_tex0;
uniform vec2  u_texel;
uniform float u_epsilon;
uniform float u_dt;

float curl(vec2 uv) {
    float L = texture(u_tex0, uv - vec2(u_texel.x, 0.0)).g;
    float R = texture(u_tex0, uv + vec2(u_texel.x, 0.0)).g;
    float B = texture(u_tex0, uv - vec2(0.0, u_texel.y)).r;
    float T = texture(u_tex0, uv + vec2(0.0, u_texel.y)).r;
    return 0.5 * ((R - L) - (T - B));
}

void main() {
    vec4 s = texture(u_tex0, v_uv);

    float wL = abs(curl(v_uv - vec2(u_texel.x, 0.0)));
    float wR = abs(curl(v_uv + vec2(u_texel.x, 0.0)));
    float wB = abs(curl(v_uv - vec2(0.0, u_texel.y)));
    float wT = abs(curl(v_uv + vec2(0.0, u_texel.y)));

    vec2 eta = 0.5 * vec2(wR - wL, wT - wB);
    eta /= (length(eta) + 1e-5);

    float w = curl(v_uv);
    vec2 force = u_epsilon * vec2(eta.y * w, -eta.x * w);
    out_color = vec4(s.rg + force * u_dt, s.b, s.a);
}
