#version 450
in vec2 vTexCoord;
out vec4 outColor;

// Long-exposure trails: keep the brighter of "now" and the decayed history.
// decay = exp(-ln2 * dt / halfLife), computed on the CPU per frame so the
// fade rate is framerate-independent and reel-deterministic.
uniform sampler2D sceneTex;    // this frame
uniform sampler2D historyTex;  // previous trail buffer
uniform float decay;

void main() {
    vec3 cur  = texture(sceneTex,   vTexCoord).rgb;
    vec3 hist = texture(historyTex, vTexCoord).rgb * decay;
    outColor  = vec4(max(cur, hist), 1.0);
}
