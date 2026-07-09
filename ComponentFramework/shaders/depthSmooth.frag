#version 450
uniform sampler2D depthTex;
uniform vec2      screenSize;
uniform vec2      filterDir;        // (1,0) horizontal pass, (0,1) vertical pass
uniform float     particleRadius;   // world units
uniform float     worldFilterScale; // kernel half-width, in particle radii
uniform float     surfaceMerge;     // narrow-range band, in particle radii
uniform float     projScaleY;       // P[1][1] * targetHeight * 0.5

in vec2 vTexCoord;
layout(location=0) out float outSmooth;

// Narrow-range depth filter (Truong et al. style), separable:
// the kernel width is a world-space size projected to pixels (so near fluid
// smooths as much as far fluid), and instead of a bilateral range weight,
// samples are constrained to a band around the center depth — samples from
// nearer, unrelated surfaces are skipped (keeps silhouettes crisp) and
// farther samples are clamped into the band (melts particle blobs together).
void main() {
    float center = texture(depthTex, vTexCoord).r;
    if (center == 0.0) { outSmooth = 0.0; return; }

    // View-space Z is negative in front of the camera: larger value = nearer.
    float pxRadius = worldFilterScale * particleRadius * projScaleY / max(0.001, -center);
    float halfKf   = clamp(pxRadius, 1.0, 32.0);
    int   halfK    = int(halfKf);
    float sigmaS   = max(1.0, halfKf * 0.4);

    float band = surfaceMerge * particleRadius;
    float nearBound = center + band;   // anything nearer belongs to another surface
    float farBound  = center - band;   // anything farther gets pulled into the band

    float sum = 0.0, wsum = 0.0;
    for (int i = -halfK; i <= halfK; ++i) {
        vec2  off = filterDir * float(i) / screenSize;
        float d   = texture(depthTex, vTexCoord + off).r;
        if (d == 0.0) continue;        // background
        if (d > nearBound) continue;   // closer surface: don't bleed its silhouette
        d = max(d, farBound);          // merge: clamp far samples toward the surface

        float ws = exp(-float(i * i) / (2.0 * sigmaS * sigmaS));
        sum  += d * ws;
        wsum += ws;
    }
    outSmooth = (wsum > 1e-6) ? sum / wsum : center;
}
