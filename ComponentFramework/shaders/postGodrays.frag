#version 450
in vec2 vTexCoord;
out vec4 outColor;

// Volumetric light shafts (radial "god ray" blur). Marches a fixed number of
// samples from each pixel toward a screen-space light position, accumulating
// the bright-pass texture with per-step decay so the brights streak into rays.
uniform sampler2D srcTex;     // bloom bright pass (half res)
uniform vec2  uLightPos;      // light position in screen UV [0,1]
uniform float uDecay;         // per-sample falloff (0.9 - 0.98)
uniform float uDensity;       // how far the shafts reach (step scale)

const int NUM_SAMPLES = 24;

void main() {
    vec2  delta = (vTexCoord - uLightPos) * (uDensity / float(NUM_SAMPLES));
    vec2  coord = vTexCoord;
    float illum = 1.0;
    vec3  acc   = vec3(0.0);
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        coord -= delta;                       // step toward the light
        acc   += texture(srcTex, coord).rgb * illum;
        illum *= uDecay;
    }
    outColor = vec4(acc / float(NUM_SAMPLES), 1.0);
}
