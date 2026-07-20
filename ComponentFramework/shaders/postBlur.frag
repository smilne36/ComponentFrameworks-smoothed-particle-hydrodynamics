#version 450
in vec2 vTexCoord;
out vec4 outColor;

// Separable 9-tap gaussian; dispatched twice (horizontal then vertical).
// uDir carries the texel step pre-scaled by the resolution-relative radius,
// so the glow width looks identical at window, reel, and supersampled sizes.
uniform sampler2D srcTex;
uniform vec2 uDir;

void main() {
    const float w[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    vec3 c = texture(srcTex, vTexCoord).rgb * w[0];
    for (int i = 1; i < 5; ++i) {
        c += texture(srcTex, vTexCoord + uDir * float(i)).rgb * w[i];
        c += texture(srcTex, vTexCoord - uDir * float(i)).rgb * w[i];
    }
    outColor = vec4(c, 1.0);
}
