#version 450
in vec2 vTexCoord;
out vec4 outColor;

// Depth of field: gather blur whose radius grows with distance from the
// focus plane (circle of confusion). Depth comes from the scene render's
// depth attachment (impostor/mesh paths write real depth).
uniform sampler2D sceneTex;
uniform sampler2D depthTex;
uniform vec2  uResolution;
uniform float uNear;
uniform float uFar;
uniform float uFocusDist;   // view-space distance in focus
uniform float uAperture;    // CoC scale; 0 = off

float viewZ(vec2 uv) {
    float d = texture(depthTex, uv).r;
    float zn = d * 2.0 - 1.0;
    return 2.0 * uNear * uFar / (uFar + uNear - zn * (uFar - uNear));
}

void main() {
    vec3 base = texture(sceneTex, vTexCoord).rgb;
    if (uAperture <= 0.0) { outColor = vec4(base, 1.0); return; }

    float z   = viewZ(vTexCoord);
    float coc = uAperture * abs(z - uFocusDist) / max(z, 0.1)
              * (uResolution.y / 1080.0) * 10.0;      // pixels of blur
    coc = clamp(coc, 0.0, 14.0);
    if (coc < 0.5) { outColor = vec4(base, 1.0); return; }

    // 12-tap poisson disc gather
    const vec2 taps[12] = vec2[](
        vec2(-0.326, -0.406), vec2(-0.840, -0.074), vec2(-0.696,  0.457),
        vec2(-0.203,  0.621), vec2( 0.962, -0.195), vec2( 0.473, -0.480),
        vec2( 0.519,  0.767), vec2( 0.185, -0.893), vec2( 0.507,  0.064),
        vec2( 0.896,  0.412), vec2(-0.322, -0.933), vec2(-0.792, -0.598));
    vec2 px = coc / uResolution;
    vec3 acc = base;
    for (int i = 0; i < 12; ++i)
        acc += texture(sceneTex, vTexCoord + taps[i] * px).rgb;
    outColor = vec4(acc / 13.0, 1.0);
}
