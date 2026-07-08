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

out flat int vIsGhost;
out vec3 vWorldPos;
out vec3 vViewPos;
out vec3 vVel;
out float vPressure;
out float vDensity;

void main() {
    int idx = gl_VertexID;
    Particle p = particles[idx];
    vIsGhost  = p.flags.x;
    vWorldPos = p.pos.xyz;
    vVel      = p.vel.xyz;
    vPressure = p.pressure;
    vDensity  = p.density;

    vec4 viewPos = viewMatrix * vec4(p.pos.xyz, 1.0);
    vViewPos = viewPos.xyz;
    gl_Position = projectionMatrix * viewPos;

    // Screen-space diameter: 2r * (proj_y_scale / -viewZ) * (H/2)
    float sz = 2.0 * particleRadius * projectionMatrix[1][1]
               / max(0.001, -viewPos.z) * (viewportH * 0.5);
    gl_PointSize = max(1.0, sz);
}
