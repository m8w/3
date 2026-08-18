#version 330 core
// Boussinesq buoyancy: heat lifts, density sinks.
//   f = (alpha * (T - T_amb) - beta * D) * up

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_tex0;
uniform float u_alpha;
uniform float u_beta;
uniform float u_dt;
uniform float u_T_amb;
uniform vec2  u_up;

void main() {
    vec4 s = texture(u_tex0, v_uv);
    float force = (u_alpha * (s.b - u_T_amb) - u_beta * s.a) * u_dt;
    out_color = vec4(s.rg + u_up * force, s.b, s.a);
}
