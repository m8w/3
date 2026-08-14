#version 330 core
// Vertex program - 3D value-noise displacement scaled by audio pulse.
// Passes world position + eye-space normal + view dir to the fragment stage.

in vec3 in_position;
in vec3 in_normal;

uniform mat4 u_mvp;
uniform mat4 u_mv;
uniform mat3 u_normal_mat;

uniform float u_pulse;
uniform float u_displace;
uniform float u_noise_scale;
uniform float u_time;

out vec3 v_world_pos;
out vec3 v_normal_eye;
out vec3 v_view_dir;

float hash13(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.x + p.y) * p.z);
}

float vnoise3(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    float n000 = hash13(i);
    float n100 = hash13(i + vec3(1, 0, 0));
    float n010 = hash13(i + vec3(0, 1, 0));
    float n110 = hash13(i + vec3(1, 1, 0));
    float n001 = hash13(i + vec3(0, 0, 1));
    float n101 = hash13(i + vec3(1, 0, 1));
    float n011 = hash13(i + vec3(0, 1, 1));
    float n111 = hash13(i + vec3(1, 1, 1));
    return mix(
        mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
        mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
        u.z);
}

void main() {
    vec3 p = in_position;
    vec3 n = normalize(in_normal);

    float d = vnoise3(p * u_noise_scale + vec3(u_time * 0.15))
            + 0.5 * vnoise3(p * u_noise_scale * 2.13 - vec3(u_time * 0.08));
    d = (d - 0.75) * 2.0;
    float amp = u_displace * (0.4 + 0.6 * u_pulse);
    vec3 pd = p + n * d * amp;

    vec4 mv = u_mv * vec4(pd, 1.0);
    v_world_pos = pd;
    v_normal_eye = normalize(u_normal_mat * n);
    v_view_dir = -normalize(mv.xyz);

    gl_Position = u_mvp * vec4(pd, 1.0);
}
