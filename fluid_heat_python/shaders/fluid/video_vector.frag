#version 330 core
// Channel B ("nerves", the 10k archive). The clip's luminance gradient
// becomes a pure velocity field that steers the fluid - no heat, no density,
// only direction. Dark areas damp the flow; bright areas accelerate it.

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_tex0;   // fluid state
uniform sampler2D u_tex1;   // channel B clip
uniform vec2  u_texel;
uniform float u_force;
uniform float u_damp;
uniform float u_curl;

float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

void main() {
    vec4 s = texture(u_tex0, v_uv);

    float Lx1 = luma(texture(u_tex1, v_uv + vec2(u_texel.x, 0.0)).rgb);
    float Lx0 = luma(texture(u_tex1, v_uv - vec2(u_texel.x, 0.0)).rgb);
    float Ly1 = luma(texture(u_tex1, v_uv + vec2(0.0, u_texel.y)).rgb);
    float Ly0 = luma(texture(u_tex1, v_uv - vec2(0.0, u_texel.y)).rgb);
    vec2 g = 0.5 * vec2(Lx1 - Lx0, Ly1 - Ly0);

    // rotate the gradient 90 degrees for tangential (curl-like) flow
    vec2 tangent = vec2(-g.y, g.x);
    vec2 dir = mix(g, tangent, u_curl);

    float L = luma(texture(u_tex1, v_uv).rgb);
    float attenuate = mix(1.0 - u_damp, 1.0, L);

    out_color = vec4(s.rg * attenuate + dir * u_force, s.b, s.a);
}
