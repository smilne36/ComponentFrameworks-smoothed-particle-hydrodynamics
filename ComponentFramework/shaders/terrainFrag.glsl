#version 450
in vec3 vWorldPos;
in vec3 vNorm;

out vec4 outColor;

uniform vec3 sunDirWorld;
uniform vec3 sunColor;

void main() {
    float h = vWorldPos.y;

    // Height-blended terrain colours
    vec3 wetRock  = vec3(0.22, 0.19, 0.15);   // wet channel floor
    vec3 dryRock  = vec3(0.42, 0.37, 0.30);   // rocky bank
    vec3 soil     = vec3(0.34, 0.28, 0.20);   // earthy upper bank
    vec3 grass    = vec3(0.20, 0.36, 0.13);   // high ground vegetation

    // Normalise height to roughly [0,1] across a 6-unit vertical range
    float t = clamp(h * 0.18 + 0.35, 0.0, 1.0);

    vec3 color;
    if      (t < 0.25) color = mix(wetRock, dryRock, t * 4.0);
    else if (t < 0.55) color = mix(dryRock, soil,    (t - 0.25) * (1.0 / 0.30));
    else               color = mix(soil,    grass,    (t - 0.55) * (1.0 / 0.45));

    // Simple Lambertian + ambient
    float NdotL = max(0.15, dot(normalize(vNorm), normalize(sunDirWorld)));
    outColor = vec4(color * sunColor * NdotL, 1.0);
}
