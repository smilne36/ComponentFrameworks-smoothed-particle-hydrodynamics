#version 450
layout(location=0) in vec2 dummy;

uniform mat4 projectionMatrix, viewMatrix;
uniform float particleRadius;
uniform float viewportH;

struct Particle {
    vec4 pos; vec4 vel; vec4 acc;
    float density; float pressure; float padA; float padB;
    ivec4 flags;
};
layout(std430, binding=0) buffer ParticleBuf { Particle particles[]; };

out vec3 vViewPos;
out flat int vIsGhost;

void main() {
    Particle p = particles[gl_VertexID];
    vIsGhost = p.flags.x;

    vec4 vp = viewMatrix * vec4(p.pos.xyz, 1.0);
    vViewPos = vp.xyz;
    gl_Position = projectionMatrix * vp;

    // Screen-space diameter: 2r * (proj_y_scale / -viewZ) * (H/2)
    float sz = 2.0 * particleRadius * projectionMatrix[1][1]
               / max(0.001, -vp.z) * (viewportH * 0.5);
    gl_PointSize = max(2.0, sz);
}
