#version 330 core
// One Jacobi relaxation step of the pressure Poisson equation:
//   p' = (p_L + p_R + p_B + p_T + alpha * div) * rbeta

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_tex0;   // pressure (R)
uniform sampler2D u_tex1;   // divergence (R)
uniform vec2  u_texel;
uniform float u_alpha;      // -1.0
uniform float u_rbeta;      //  0.25

void main() {
    float pL = texture(u_tex0, v_uv - vec2(u_texel.x, 0.0)).r;
    float pR = texture(u_tex0, v_uv + vec2(u_texel.x, 0.0)).r;
    float pB = texture(u_tex0, v_uv - vec2(0.0, u_texel.y)).r;
    float pT = texture(u_tex0, v_uv + vec2(0.0, u_texel.y)).r;
    float b  = texture(u_tex1, v_uv).r;
    out_color = vec4((pL + pR + pB + pT + u_alpha * b) * u_rbeta, 0.0, 0.0, 1.0);
}
