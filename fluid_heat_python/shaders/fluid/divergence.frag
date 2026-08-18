#version 330 core
// div(u) = du/dx + dv/dy, written into R.
//
// BACKWARD difference. Paired with the FORWARD difference in gradient.frag
// it composes to exactly the compact 5-point Laplacian that jacobi.frag
// inverts - the discretely-consistent choice, and one fetch cheaper than
// central differences. Measured convergence is the same either way (~96% of
// divergence removed at 40 iterations); this pair is preferred on principle
// and cost, not because it fixed a defect.

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_tex0;
uniform vec2 u_texel;

void main() {
    vec2 C = texture(u_tex0, v_uv).rg;
    float L = texture(u_tex0, v_uv - vec2(u_texel.x, 0.0)).r;
    float B = texture(u_tex0, v_uv - vec2(0.0, u_texel.y)).g;
    out_color = vec4((C.r - L) + (C.g - B), 0.0, 0.0, 1.0);
}
