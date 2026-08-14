#version 330 core
// Heat-modulated viscosity. Hot cells thin out and accelerate; cold cells
// thicken into a syrupy drag (cooling lava / drying ink).

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_tex0;
uniform float u_visc_cold;
uniform float u_visc_hot;
uniform float u_T_knee;

void main() {
    vec4 s = texture(u_tex0, v_uv);
    float T = clamp(s.b, 0.0, 1.0);
    float m = smoothstep(u_T_knee * 0.4, u_T_knee + 0.5, T);
    float k = mix(u_visc_cold, u_visc_hot, m);
    out_color = vec4(s.rg * k, s.b, s.a);
}
