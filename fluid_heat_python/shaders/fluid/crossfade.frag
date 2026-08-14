#version 330 core
// Flow-warped A/B dissolve between two archive clips ("asemic ghosting").
// The destination's luminance gradient warps the source while it fades out,
// so one clip appears to melt into the next rather than cutting.

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_tex0;   // outgoing clip
uniform sampler2D u_tex1;   // incoming clip
uniform vec2  u_texel;
uniform float u_t;          // 0 = all A, 1 = all B
uniform float u_warp;

float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

void main() {
    vec3 b = texture(u_tex1, v_uv).rgb;

    float Lx1 = luma(texture(u_tex1, v_uv + vec2(u_texel.x, 0.0)).rgb);
    float Lx0 = luma(texture(u_tex1, v_uv - vec2(u_texel.x, 0.0)).rgb);
    float Ly1 = luma(texture(u_tex1, v_uv + vec2(0.0, u_texel.y)).rgb);
    float Ly0 = luma(texture(u_tex1, v_uv - vec2(0.0, u_texel.y)).rgb);
    vec2 g = vec2(Lx1 - Lx0, Ly1 - Ly0);

    vec2 warped = clamp(v_uv + g * u_warp * (1.0 - u_t), 0.0, 1.0);
    vec3 a = texture(u_tex0, warped).rgb;

    float k = smoothstep(0.0, 1.0, clamp(u_t, 0.0, 1.0));
    out_color = vec4(mix(a, b, k), 1.0);
}
