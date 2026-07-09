#version 450
uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform vec3 skyHorizonColor;
uniform vec3 skyZenithColor;
uniform vec3 sunDirWorld;
uniform vec3 sunColor;

in vec2 vTexCoord;
layout(location=0) out vec4 outColor;

// Fullscreen procedural sky: vertical gradient + soft sun glow.
// Drawn into the background FBO before terrain/lines (depth writes off).
void main() {
    // World-space view ray through this pixel
    vec2 ndc = vTexCoord * 2.0 - 1.0;
    vec3 viewRay = vec3(ndc.x / projectionMatrix[0][0],
                        ndc.y / projectionMatrix[1][1],
                        -1.0);
    vec3 dir = normalize(transpose(mat3(viewMatrix)) * viewRay);

    float t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 col = mix(skyHorizonColor, skyZenithColor, pow(t, 0.7));

    float s = max(dot(dir, normalize(sunDirWorld)), 0.0);
    col += sunColor * pow(s, 128.0) * 0.8 + sunColor * pow(s, 8.0) * 0.08;

    outColor = vec4(col, 1.0);
}
