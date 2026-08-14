#version 330 core
// Gray-Scott reaction-diffusion riding on the fluid. Heat raises the feed
// rate (hot cells eat substrate faster); density seeds the product. Grows
// Turing veins wherever the fluid has recently been active.
//   state: R = U (substrate), G = V (product)

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_tex0;   // reaction state (RG)
uniform sampler2D u_tex1;   // fluid state (R=u G=v B=T A=D)
uniform vec2  u_texel;
uniform float u_Du;
uniform float u_Dv;
uniform float u_F;
uniform float u_k;
uniform float u_heat_feed;
uniform float u_dt;

vec2 laplacian(vec2 uv) {
    vec2 C  = texture(u_tex0, uv).rg;
    vec2 L  = texture(u_tex0, uv + vec2(-u_texel.x, 0.0)).rg;
    vec2 R  = texture(u_tex0, uv + vec2( u_texel.x, 0.0)).rg;
    vec2 B  = texture(u_tex0, uv + vec2(0.0, -u_texel.y)).rg;
    vec2 T  = texture(u_tex0, uv + vec2(0.0,  u_texel.y)).rg;
    vec2 BL = texture(u_tex0, uv + vec2(-u_texel.x, -u_texel.y)).rg;
    vec2 BR = texture(u_tex0, uv + vec2( u_texel.x, -u_texel.y)).rg;
    vec2 TL = texture(u_tex0, uv + vec2(-u_texel.x,  u_texel.y)).rg;
    vec2 TR = texture(u_tex0, uv + vec2( u_texel.x,  u_texel.y)).rg;
    return -C + 0.2 * (L + R + B + T) + 0.05 * (BL + BR + TL + TR);
}

void main() {
    vec2 uv_ = texture(u_tex0, v_uv).rg;
    vec4 fluid = texture(u_tex1, v_uv);

    float U = uv_.r;
    float V = uv_.g;

    float Flocal = u_F + fluid.b * u_heat_feed;
    float Vnudge = 0.002 * fluid.a * (1.0 - V);

    vec2 lp = laplacian(v_uv);
    float uvv = U * V * V;
    float dU = u_Du * lp.r - uvv + Flocal * (1.0 - U);
    float dV = u_Dv * lp.g + uvv - (Flocal + u_k) * V + Vnudge;

    out_color = vec4(clamp(U + dU * u_dt, 0.0, 1.0),
                     clamp(V + dV * u_dt, 0.0, 1.0),
                     0.0, 1.0);
}
