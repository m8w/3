#version 330 core
// Fragment - blend incandescent + organic 5-stop palettes weighted by
// (1 - pulse) * organic_bias, lit by per-vertex normal + rim, Reinhard tonemap.

in vec3 v_world_pos;
in vec3 v_normal_eye;
in vec3 v_view_dir;

uniform float u_heat_gain;
uniform float u_organic_bias;
uniform float u_rim;
uniform float u_ambient;
uniform float u_pulse;

out vec4 out_color;

vec3 incandescent(float t) {
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
    vec3 c0 = vec3(0x05, 0x0a, 0x08) / 255.0;
    vec3 c1 = vec3(0x2e, 0x1a, 0x45) / 255.0;
    vec3 c2 = vec3(0x24, 0x55, 0x33) / 255.0;
    vec3 c3 = vec3(0x8a, 0xb0, 0x4a) / 255.0;
    vec3 c4 = vec3(0xe8, 0xe0, 0xb4) / 255.0;
    if (t < 0.25)      return mix(c0, c1, t / 0.25);
    else if (t < 0.50) return mix(c1, c2, (t - 0.25) / 0.25);
    else if (t < 0.75) return mix(c2, c3, (t - 0.50) / 0.25);
    else               return mix(c3, c4, (t - 0.75) / 0.25);
}

void main() {
    float radial = length(v_world_pos.xz);
    float h = clamp(0.55 + v_world_pos.y * 0.5
                   - radial * 0.35 + u_pulse * 0.25, 0.0, 1.0);
    float w = clamp(u_organic_bias + (1.0 - u_pulse) * 0.4, 0.0, 1.0);
    vec3 pal = mix(incandescent(h * u_heat_gain),
                   organic(clamp(h + 0.15, 0.0, 1.0)),
                   w);

    vec3 L = normalize(vec3(0.4, 0.9, 0.35));
    float lam = max(0.0, dot(v_normal_eye, L));
    float rimf = pow(1.0 - max(0.0, dot(v_normal_eye, v_view_dir)), 3.0);

    vec3 col = pal * (u_ambient + lam) + pal * rimf * u_rim;
    col = col / (1.0 + col);
    out_color = vec4(col, 1.0);
}
