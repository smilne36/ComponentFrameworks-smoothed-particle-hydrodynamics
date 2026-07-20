#version 450
in vec2 vTexCoord;
out vec4 outColor;

// Bloom bright-pass: keep only pixels above the luma threshold, with a soft
// knee so glow fades in instead of popping.
uniform sampler2D srcTex;
uniform float threshold;
uniform float knee;

void main() {
    vec3  c   = texture(srcTex, vTexCoord).rgb;
    float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
    outColor  = vec4(c * smoothstep(threshold, threshold + max(knee, 1e-4), lum), 1.0);
}
