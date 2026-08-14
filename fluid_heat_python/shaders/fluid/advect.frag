#version 330 core
// Semi-Lagrangian advection (Stam, Stable Fluids). Trace backwards along the
// velocity field and resample. Per-channel dissipation.

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_tex0;   // quantity to advect (previous state)
uniform sampler2D u_tex1;   // velocity source
uniform float u_dt;
uniform float u_diss_v;
uniform float u_diss_T;
uniform float u_diss_D;

void main() {
    vec2 vel = texture(u_tex1, v_uv).rg;
    // velocity is in normalized-units/second, so the back-trace is direct
    vec2 back = clamp(v_uv - vel * u_dt, 0.0, 1.0);
    vec4 s = texture(u_tex0, back);       // hardware bilinear
    out_color = vec4(s.rg * u_diss_v, s.b * u_diss_T, s.a * u_diss_D);
}
