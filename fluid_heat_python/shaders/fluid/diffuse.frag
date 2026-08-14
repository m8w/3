#version 330 core
// Per-channel diffusion: viscosity on velocity, thermal diffusivity on T,
// molecular diffusion on density. 5-tap Laplacian relaxation.

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_tex0;
uniform vec2  u_texel;
uniform float u_k_vel;
uniform float u_k_T;
uniform float u_k_D;

void main() {
    vec4 C = texture(u_tex0, v_uv);
    vec4 L = texture(u_tex0, v_uv - vec2(u_texel.x, 0.0));
    vec4 R = texture(u_tex0, v_uv + vec2(u_texel.x, 0.0));
    vec4 B = texture(u_tex0, v_uv - vec2(0.0, u_texel.y));
    vec4 T = texture(u_tex0, v_uv + vec2(0.0, u_texel.y));
    vec4 avg = 0.25 * (L + R + B + T);
    vec4 k = vec4(u_k_vel, u_k_vel, u_k_T, u_k_D);
    out_color = C + (avg - C) * k;
}
