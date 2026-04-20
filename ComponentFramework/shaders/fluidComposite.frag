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

in vec2 vTexCoord;
layout(location=0) out vec4 outColor;

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

    // No fluid at this pixel — show background as-is
    if (vz == 0.0) {
        outColor = texture(backgroundTex, vTexCoord);
        return;
    }

    vec2 px = 1.0 / screenSize;

    // Reconstruct surface position
    vec3 pos = viewPosFromZ(vTexCoord, vz);

    // Forward-difference normal reconstruction
    float vzR = texture(smoothDepthTex, vTexCoord + vec2(px.x, 0.0)).r;
    float vzU = texture(smoothDepthTex, vTexCoord + vec2(0.0, px.y)).r;
    vec3 posR = (vzR != 0.0) ? viewPosFromZ(vTexCoord + vec2(px.x, 0.0), vzR) : pos;
    vec3 posU = (vzU != 0.0) ? viewPosFromZ(vTexCoord + vec2(0.0, px.y), vzU) : pos;

    vec3 dX = posR - pos;
    vec3 dY = posU - pos;
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

    // Simplified environment reflection (overcast sky blue)
    vec3 envReflect = vec3(0.05, 0.12, 0.28);

    // Fresnel blend: thin/normal-incidence → refracted; thick/grazing → reflected
    vec3 surfaceColor = mix(transmitted, envReflect, F);

    // Sun specular highlight
    surfaceColor += sunColor * spec * specularStrength;

    outColor = vec4(clamp(surfaceColor, 0.0, 1.0), 1.0);
}
