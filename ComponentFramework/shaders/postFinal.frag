#version 450
in vec2 vTexCoord;
out vec4 outColor;

uniform sampler2D baseTex;      // scene color, or the trail buffer when trails are on
uniform sampler2D bloomTex;     // blurred half-res brights
uniform vec2  uResolution;      // output size in pixels
uniform float uKaleidoSegments; // < 2 = off
uniform float uKaleidoAngle;    // radians
uniform float uChromatic;       // radial RGB split strength
uniform float uVignette;        // 0..1 edge darkening
uniform float uGrain;           // 0..0.2 film grain
uniform float uBloomStrength;   // 0 = bloom off
uniform float uTime;            // deterministic post clock (grain seed)

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Mirror-fold the view into N wedge pairs around the screen center
// (aspect-true). seg = pi/N so N periods tile the full circle exactly --
// seamless for every N, odd or even.
vec2 kaleido(vec2 uv) {
    if (uKaleidoSegments < 2.0) return uv;
    float aspect = uResolution.x / max(uResolution.y, 1.0);
    vec2  c   = (uv - 0.5) * vec2(aspect, 1.0);
    float r   = length(c);
    float a   = atan(c.y, c.x) + uKaleidoAngle;
    float seg = 3.14159265 / uKaleidoSegments;
    a = mod(a, 2.0 * seg);
    if (a > seg) a = 2.0 * seg - a;
    a -= uKaleidoAngle;
    return vec2(cos(a), sin(a)) * r / vec2(aspect, 1.0) + 0.5;
}

vec3 sampleChromatic(vec2 uv) {
    if (uChromatic <= 0.0) return texture(baseTex, uv).rgb;
    vec2 d = (uv - 0.5) * uChromatic * 0.01;   // split grows toward the edges
    return vec3(texture(baseTex, uv + d).r,
                texture(baseTex, uv).g,
                texture(baseTex, uv - d).b);
}

void main() {
    vec2 uv  = clamp(kaleido(vTexCoord), vec2(0.0), vec2(1.0));
    vec3 col = sampleChromatic(uv);
    col += texture(bloomTex, uv).rgb * uBloomStrength;   // bloom folds with the kaleidoscope

    if (uVignette > 0.0) {
        vec2 v = vTexCoord - 0.5;
        col *= 1.0 - uVignette * smoothstep(0.25, 0.75, dot(v, v) * 2.0);
    }
    if (uGrain > 0.0)
        col += (hash12(vTexCoord * uResolution + fract(uTime * 0.7919) * 1024.0) - 0.5) * uGrain;

    outColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
