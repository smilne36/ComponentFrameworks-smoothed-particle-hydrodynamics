#version 450
uniform sampler2D smoothDepthTex;
uniform sampler2D thicknessTex;
uniform sampler2D backgroundTex;
uniform mat4      projectionMatrix;
uniform mat4      viewMatrix;
uniform vec2      screenSize;
uniform vec3      sunDirWorld;        // world-space sun direction (normalized)
uniform vec3      sunColor;
uniform float     specularPower;
uniform float     specularStrength;
uniform vec3      extinction;         // Beer-Lambert coefficients per RGB channel
uniform float     thicknessScale;
uniform float     refractionStrength;
uniform float     fresnelBias;
uniform vec3      deepWaterColor;
uniform vec3      envReflectColor;   // reflection tint applied to the sky gradient
uniform sampler2D foamTex;           // additive foam accumulation (unit 3)
uniform float     foamAmount;
uniform float     exposure;
uniform vec3      skyHorizonColor;
uniform vec3      skyZenithColor;

in vec2 vTexCoord;
layout(location=0) out vec4 outColor;

// ==== BEGIN SHARED COLOR ADJUST (subset of the shared palette block) ====
uniform float hueShift;     // degrees
uniform float satMul;
uniform float brightMul;
uniform float contrastMul;
uniform int   invertColor;

// Branchless RGB<->HSV (Sam Hocevar, public domain)
vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    return vec3(abs(q.z + (q.w - q.y) / (6.0*d + 1e-10)), d / (q.x + 1e-10), q.x);
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec3 applyColorAdjust(vec3 c) {
    vec3 hsv = rgb2hsv(clamp(c, 0.0, 1.0));
    hsv.x = fract(hsv.x + hueShift / 360.0);
    hsv.y = clamp(hsv.y * satMul, 0.0, 1.0);
    c = hsv2rgb(hsv) * brightMul;
    c = (c - 0.5) * contrastMul + 0.5;
    if (invertColor == 1) c = vec3(1.0) - c;
    return clamp(c, 0.0, 1.0);
}
// ==== END SHARED COLOR ADJUST ====

// Same gradient as skyGradient.frag (keep in sync) — used for reflections.
vec3 skyGradient(vec3 dir) {
    float t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 col = mix(skyHorizonColor, skyZenithColor, pow(t, 0.7));
    float s = max(dot(dir, normalize(sunDirWorld)), 0.0);
    col += sunColor * pow(s, 128.0) * 0.8;
    return col;
}

// Narkowicz ACES filmic fit
vec3 acesTonemap(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// HDR -> display: exposure, tonemap, gamma, then the user grade.
vec3 finishColor(vec3 c) {
    c = acesTonemap(c * exposure);
    c = pow(c, vec3(1.0 / 2.2));
    return applyColorAdjust(c);
}

// Reconstruct view-space position from stored view-space Z and screen UV
vec3 viewPosFromZ(vec2 uv, float vz) {
    vec2 ndc = uv * 2.0 - 1.0;
    return vec3(
        ndc.x / projectionMatrix[0][0] * (-vz),
        ndc.y / projectionMatrix[1][1] * (-vz),
        vz
    );
}

void main() {
    float vz = texture(smoothDepthTex, vTexCoord).r;

    // No fluid at this pixel — show background (same display chain for a consistent frame)
    if (vz == 0.0) {
        vec4 bg = texture(backgroundTex, vTexCoord);
        outColor = vec4(finishColor(bg.rgb), bg.a);
        return;
    }

    vec2 px = 1.0 / screenSize;

    // Reconstruct surface position
    vec3 pos = viewPosFromZ(vTexCoord, vz);

    // Depth-aware normal reconstruction: per axis take the forward or backward
    // difference with the smaller depth change (avoids straddling silhouettes),
    // falling back to whichever side has valid fluid.
    float vzL = texture(smoothDepthTex, vTexCoord - vec2(px.x, 0.0)).r;
    float vzR = texture(smoothDepthTex, vTexCoord + vec2(px.x, 0.0)).r;
    float vzD = texture(smoothDepthTex, vTexCoord - vec2(0.0, px.y)).r;
    float vzU = texture(smoothDepthTex, vTexCoord + vec2(0.0, px.y)).r;

    vec3 dX = vec3(0.0);
    if (vzR != 0.0) dX = viewPosFromZ(vTexCoord + vec2(px.x, 0.0), vzR) - pos;
    if (vzL != 0.0) {
        vec3 dxB = pos - viewPosFromZ(vTexCoord - vec2(px.x, 0.0), vzL);
        if (vzR == 0.0 || abs(dxB.z) < abs(dX.z)) dX = dxB;
    }
    vec3 dY = vec3(0.0);
    if (vzU != 0.0) dY = viewPosFromZ(vTexCoord + vec2(0.0, px.y), vzU) - pos;
    if (vzD != 0.0) {
        vec3 dyB = pos - viewPosFromZ(vTexCoord - vec2(0.0, px.y), vzD);
        if (vzU == 0.0 || abs(dyB.z) < abs(dY.z)) dY = dyB;
    }

    vec3 N  = (length(dX) > 1e-5 && length(dY) > 1e-5)
              ? normalize(cross(dX, dY))
              : vec3(0.0, 0.0, 1.0);
    if (N.z < 0.0) N = -N; // ensure normal faces camera

    // View direction (from surface toward camera origin)
    vec3 V = normalize(-pos);

    // Fresnel (Schlick)
    float cosN = max(0.0, dot(N, V));
    float F    = fresnelBias + (1.0 - fresnelBias) * pow(1.0 - cosN, 5.0);

    // Transform sun to view space for shading
    vec3 sunView = normalize(mat3(viewMatrix) * normalize(sunDirWorld));

    // Blinn-Phong specular
    vec3  H    = normalize(sunView + V);
    float spec = pow(max(0.0, dot(N, H)), specularPower);

    // Refraction: sample background at normal-distorted UV
    vec2 refractUV = clamp(vTexCoord + N.xy * refractionStrength, vec2(0.001), vec2(0.999));
    vec3 bgSample  = texture(backgroundTex, refractUV).rgb;

    // Beer-Lambert absorption from fluid thickness
    float thick    = max(0.0, texture(thicknessTex, vTexCoord).r * thicknessScale);
    vec3  transmit = exp(-extinction * thick);

    // Transmitted color = background tinted by water absorption
    float avgTrans = dot(transmit, vec3(1.0 / 3.0));
    vec3 transmitted = mix(deepWaterColor, bgSample * transmit, clamp(avgTrans, 0.0, 1.0));

    // Environment reflection: sample the sky gradient along the reflected ray
    vec3 Rw = transpose(mat3(viewMatrix)) * reflect(-V, N);
    vec3 envColor = skyGradient(Rw) * envReflectColor;

    // Fresnel blend: thin/normal-incidence → refracted; thick/grazing → reflected
    vec3 surfaceColor = mix(transmitted, envColor, F);

    // Sun specular highlight (HDR: tonemapped below instead of clamped)
    surfaceColor += sunColor * spec * specularStrength;

    // Foam: lift toward white where aerated particles accumulated
    float foamF = 1.0 - exp(-texture(foamTex, vTexCoord).r * foamAmount);
    surfaceColor = mix(surfaceColor, vec3(0.95), clamp(foamF, 0.0, 1.0));

    outColor = vec4(finishColor(surfaceColor), 1.0);
}
