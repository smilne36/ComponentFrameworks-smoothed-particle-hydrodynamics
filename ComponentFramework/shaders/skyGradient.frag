#version 450
uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform vec3 skyHorizonColor;
uniform vec3 skyZenithColor;
uniform vec3 sunDirWorld;
uniform vec3 sunColor;
uniform int  uSkyMode;   // 0 gradient 1 nebula 2 starfield 3 aurora 4 sunset
uniform float uTime;

in vec2 vTexCoord;
layout(location=0) out vec4 outColor;

float hash21(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vnoise(vec2 p){
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i), b = hash21(i + vec2(1,0)), c = hash21(i + vec2(0,1)), d = hash21(i + vec2(1,1));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
float fbm(vec2 p){ float s = 0.0, a = 0.5; for (int i = 0; i < 5; ++i){ s += a * vnoise(p); p *= 2.02; a *= 0.5; } return s; }

// Fullscreen procedural sky, drawn into the background before the scene
// (depth writes off). Mode 0 is the original gradient + sun glow.
void main() {
    vec2 ndc = vTexCoord * 2.0 - 1.0;
    vec3 viewRay = vec3(ndc.x / projectionMatrix[0][0],
                        ndc.y / projectionMatrix[1][1],
                        -1.0);
    vec3 dir = normalize(transpose(mat3(viewMatrix)) * viewRay);

    float t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 col = mix(skyHorizonColor, skyZenithColor, pow(t, 0.7));
    float s = max(dot(dir, normalize(sunDirWorld)), 0.0);
    col += sunColor * pow(s, 128.0) * 0.8 + sunColor * pow(s, 8.0) * 0.08;

    // Planar sky coords (stable away from the poles) for the procedural modes
    vec2 sky = dir.xz / (abs(dir.y) + 0.35);

    if (uSkyMode == 1) {                       // Nebula
        float n  = fbm(sky * 2.5 + uTime * 0.02);
        float n2 = fbm(sky * 5.0 - uTime * 0.015);
        vec3  neb = mix(skyHorizonColor, skyZenithColor, n);
        neb += sunColor * smoothstep(0.55, 0.95, n2) * 0.6;
        col = mix(col, neb, 0.85);
        vec2 id = floor(sky * 180.0);
        col += vec3(step(0.986, hash21(id))) * (0.5 + 0.5 * sin(uTime * 3.0 + hash21(id) * 30.0));
    } else if (uSkyMode == 2) {                // Starfield
        col *= 0.28;
        vec2 id = floor(sky * 120.0);
        float h = hash21(id);
        float tw = 0.5 + 0.5 * sin(uTime * 4.0 + h * 40.0);
        col += vec3(step(0.99, h) * tw);
        col += vec3(0.6, 0.7, 1.0) * step(0.997, hash21(id + 7.0)) * tw;   // blue giants
    } else if (uSkyMode == 3) {                // Aurora
        float band = fbm(vec2(sky.x * 3.0 + uTime * 0.10, sky.y * 1.5));
        float curtain = smoothstep(0.2, 0.9, band) * smoothstep(0.0, 0.6, dir.y);
        vec3  aur = mix(vec3(0.10, 0.90, 0.50), vec3(0.40, 0.20, 0.90), band);
        col += aur * curtain * 0.9;
    } else if (uSkyMode == 4) {                // Sunset
        float g = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
        col = mix(vec3(1.0, 0.50, 0.22), vec3(0.28, 0.10, 0.42), pow(g, 0.8));
        col += vec3(1.0, 0.60, 0.30) * pow(s, 40.0) * 1.5;
        col += vec3(1.0, 0.40, 0.20) * pow(s, 4.0)  * 0.15;
    }

    outColor = vec4(col, 1.0);
}
