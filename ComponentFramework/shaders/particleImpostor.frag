#version 450
in flat int vIsGhost;
in float vHeight;
in float vSpeed;
in float vPressure;
in float vDensity;

uniform int  colorMode;
uniform vec2 vizRange;
uniform vec2 heightMinMax;

out vec4 outColor;

float remap01(float v, float lo, float hi){
    return clamp((v - lo) / max(1e-6, hi - lo), 0.0, 1.0);
}

vec3 heightPalette(float t) {
    vec3 c1 = vec3(0.05, 0.15, 0.85);
    vec3 c2 = vec3(0.25, 0.60, 0.90);
    vec3 c3 = vec3(0.80, 0.30, 0.40);
    vec3 c4 = vec3(0.95, 0.10, 0.10);
    if (t < 0.33) {
        float u = t / 0.33;
        return mix(c1, c2, u);
    } else if (t < 0.66) {
        float u = (t - 0.33) / 0.33;
        return mix(c2, c3, u);
    } else {
        float u = (t - 0.66) / 0.34;
        return mix(c3, c4, u);
    }
}

vec3 turbo(float t) {
    t = clamp(t, 0.0, 1.0);
    return vec3(0.1357 + 4.0*t - 4.5*t*t,
                0.0000 + 2.0*t - 1.0*t*t,
                0.6667 - 1.5*t + 1.0*t*t);
}

void main() {
    if (vIsGhost == 1) discard;
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    if (dot(uv, uv) > 1.0) discard;

    vec3 col;
    if (colorMode == 0) {
        float t = remap01(vHeight, heightMinMax.x, heightMinMax.y);
        col = heightPalette(t);
    } else if (colorMode == 1) {
        col = turbo(remap01(vSpeed, vizRange.x, vizRange.y));
    } else if (colorMode == 2) {
        col = turbo(remap01(vPressure, vizRange.x, vizRange.y));
    } else if (colorMode == 3) {
        col = turbo(remap01(vDensity, vizRange.x, vizRange.y));
    } else {
        col = vec3(1.0); // instanceColor not used in impostor path; fallback white
    }

    outColor = vec4(col, 1.0);
}