#version 450
in flat int vIsGhost;
out vec4 outColor;

void main() {
    if (vIsGhost == 1) discard;
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) discard; // round sprite
    // Simple shading
    vec3 col = mix(vec3(0.15,0.4,1.0), vec3(0.9,0.5,1.0), clamp(1.0 - r2, 0.0, 1.0));
    outColor = vec4(col, 1.0);
}