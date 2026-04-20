#version 450
uniform sampler2D depthTex;
uniform vec2      screenSize;
uniform vec2      filterDir;    // (1,0) horizontal pass, (0,1) vertical pass
uniform float     filterRadius; // half-kernel size in pixels
uniform float     depthFalloff; // bilateral depth sigma (view-space units)

in vec2 vTexCoord;
layout(location=0) out float outSmooth;

void main() {
    float center = texture(depthTex, vTexCoord).r;
    if (center == 0.0) { outSmooth = 0.0; return; }

    float sigmaS = max(1.0, filterRadius * 0.4);
    int   halfK  = int(filterRadius);

    float sum = 0.0, wsum = 0.0;
    for (int i = -halfK; i <= halfK; ++i) {
        vec2  off = filterDir * float(i) / screenSize;
        float d   = texture(depthTex, vTexCoord + off).r;
        if (d == 0.0) continue;

        float ws = exp(-float(i * i) / (2.0 * sigmaS * sigmaS));
        float dd = d - center;
        float wd = exp(-(dd * dd) / (2.0 * depthFalloff * depthFalloff));
        float w  = ws * wd;
        sum  += d * w;
        wsum += w;
    }
    outSmooth = (wsum > 1e-6) ? sum / wsum : center;
}
