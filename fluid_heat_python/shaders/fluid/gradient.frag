#version 330 core
// Subtract the pressure gradient to project the velocity field back onto
// the divergence-free subspace. This is what makes it read as "fluid".

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_tex0;   // velocity state (RG vel, B T, A D)
uniform sampler2D u_tex1;   // pressure (R)
uniform vec2  u_texel;
uniform float u_scale;

void main() {
    vec4 s = texture(u_tex0, v_uv);
    // FORWARD difference - the discrete adjoint of the backward-difference
    // divergence. See the note in divergence.frag.
    float pC = texture(u_tex1, v_uv).r;
    float pR = texture(u_tex1, v_uv + vec2(u_texel.x, 0.0)).r;
    float pT = texture(u_tex1, v_uv + vec2(0.0, u_texel.y)).r;
    vec2 grad = vec2(pR - pC, pT - pC);
    out_color = vec4(s.rg - grad * u_scale, s.b, s.a);
}
