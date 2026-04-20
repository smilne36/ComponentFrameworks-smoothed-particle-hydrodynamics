#version 450
in vec3 vViewPos;
in flat int vIsGhost;

layout(location=0) out float outThick;

void main() {
    if (vIsGhost == 1) discard;

    vec2 disc = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(disc, disc);
    if (r2 > 1.0) discard;

    // Gaussian soft blob contribution — accumulated additively
    outThick = exp(-4.0 * r2) * 0.05;
}
