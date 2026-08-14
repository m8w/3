#version 330 core
// Trivial pass-through used to present a framebuffer to the screen and to
// copy between targets when a pass needs an unmodified source.

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_tex0;
uniform float u_gain;

void main() {
    out_color = vec4(texture(u_tex0, v_uv).rgb * u_gain, 1.0);
}
