#version 330 core
// Pseudo-volumetric lift. Raymarches the 2D colour field at rotating slice
// angles to fake depth, so the smoke reads as 3D without a real 3D texture.

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_tex0;    // colour pass (RGB + A = density)
uniform float u_time;
uniform int   u_steps;
uniform float u_depth;
uniform float u_shear;
uniform float u_swirl;

void main() {
    vec2 p0 = v_uv - 0.5;
    vec3 acc = vec3(0.0);
    float alpha = 0.0;
    int N = clamp(u_steps, 1, 96);

    for (int i = 0; i < 96; ++i) {
        if (i >= N) break;
        float t = float(i) / float(N);
        float z = (t - 0.5) * u_depth;
        float s = sin(z * u_swirl + u_time * 0.2);
        float c = cos(z * u_swirl + u_time * 0.2);
        vec2 p = vec2(p0.x * c - p0.y * s, p0.x * s + p0.y * c);
        p += vec2(u_shear * z, 0.0);
        vec2 samp = clamp(p + 0.5, 0.0, 1.0);

        vec4 c0 = texture(u_tex0, samp);
        float a = (1.0 - alpha) * c0.a * (0.6 / float(N)) * 8.0;
        acc += c0.rgb * a;
        alpha += a;
        if (alpha > 0.98) break;
    }
    out_color = vec4(acc, 1.0);
}
