#version 330 core
// Audio-driven impulse injection: 8 frequency bins -> 8 spatial jets, each
// wandering through a pseudo-Perlin field so the "nerve endings" feel alive.
// State packing: R=u  G=v  B=T(emperature)  A=D(ensity)

in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_tex0;      // current state
uniform float u_bins[8];
uniform float u_gain;
uniform float u_heat_amt;
uniform float u_force_amt;
uniform float u_density_amt;
uniform float u_time;
uniform float u_jitter;
uniform float u_swarm_rate;
uniform float u_max_vel;      // CFL clamp: advection steps |vel| * dt per frame

// (cx, cy, radius, angle_degrees) per jet
const vec4 JETS[8] = vec4[8](
    vec4(0.20, 0.15, 0.08,  90.0),   // 0 sub-bass    bottom-left  -> up
    vec4(0.50, 0.10, 0.10,  90.0),   // 1 bass        bottom-mid   -> up
    vec4(0.80, 0.15, 0.08,  90.0),   // 2 low-mid     bottom-right -> up
    vec4(0.20, 0.50, 0.06,   0.0),   // 3 mid         left wall    -> right
    vec4(0.80, 0.50, 0.06, 180.0),   // 4 upper-mid   right wall   -> left
    vec4(0.30, 0.80, 0.05, -45.0),   // 5 presence    top-left     -> down-right
    vec4(0.70, 0.80, 0.05, 225.0),   // 6 brilliance  top-right    -> down-left
    vec4(0.50, 0.55, 0.12,  90.0)    // 7 air         centre       -> up
);

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

void main() {
    vec4 src = texture(u_tex0, v_uv);
    vec2 vel = src.rg;
    float T = src.b;
    float D = src.a;

    for (int i = 0; i < 8; ++i) {
        float amp = clamp(u_bins[i] * u_gain, 0.0, 4.0);
        if (amp <= 0.001) continue;

        vec4 j = JETS[i];
        float fi = float(i);

        // Brownian drift of the injection site
        vec2 n = vec2(
            vnoise(vec2(u_time * u_swarm_rate + fi * 3.7, fi * 1.3)) - 0.5,
            vnoise(vec2(u_time * u_swarm_rate * 1.13 + fi * 5.1, fi * 2.1)) - 0.5
        );
        vec2 centre = j.xy + n * u_jitter;

        float d = distance(v_uv, centre);
        float falloff = exp(-(d * d) / (j.z * j.z));

        // direction wobbles too
        float rad = radians(j.w) + (vnoise(vec2(u_time * 0.8 + fi, fi)) - 0.5) * 0.6;
        vec2 dir = vec2(cos(rad), sin(rad));

        // Velocity is additive (it needs to be able to cancel itself out),
        // but T and D inject *toward* saturation so a sustained loud signal
        // settles at 1.0 instead of integrating without bound.
        vel += dir * amp * u_force_amt * falloff;
        T   += amp * u_heat_amt    * falloff * max(0.0, 1.0 - T);
        D   += amp * u_density_amt * falloff * max(0.0, 1.0 - D);
    }

    // CFL guard - keeps the semi-Lagrangian back-trace inside a sane radius
    float speed = length(vel);
    if (speed > u_max_vel) vel *= u_max_vel / speed;

    out_color = vec4(vel, clamp(T, 0.0, 1.0), clamp(D, 0.0, 1.0));
}
