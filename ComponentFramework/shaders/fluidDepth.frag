#version 450
uniform mat4 projectionMatrix;
uniform float particleRadius;

in vec3 vViewPos;
in flat int vIsGhost;

layout(location=0) out float outViewZ;

void main() {
    if (vIsGhost == 1) discard;

    // gl_PointCoord Y increases downward; flip to get view-space Y (up = positive)
    vec2 disc = vec2(gl_PointCoord.x, 1.0 - gl_PointCoord.y) * 2.0 - 1.0;
    float r2 = dot(disc, disc);
    if (r2 > 1.0) discard;

    // Sphere surface hit point in view space (billboard facing camera)
    float nz = sqrt(1.0 - r2);
    vec3 hit = vViewPos + vec3(disc, nz) * particleRadius;

    // Write correct sphere depth to depth buffer
    vec4 clip = projectionMatrix * vec4(hit, 1.0);
    gl_FragDepth = (clip.z / clip.w + 1.0) * 0.5;

    // View-space Z (negative for geometry in front of camera)
    outViewZ = hit.z;
}
