#version 450
layout(location=0) in vec2 dummy;

uniform mat4 projectionMatrix, viewMatrix;
uniform float particleRadius;
uniform int   colorMode;        // same semantics
uniform vec2  vizRange;
uniform vec2  heightMinMax;

struct Particle {
    vec4 pos; vec4 vel; vec4 acc;
    float density; float pressure; float padA; float padB;
    ivec4 flags;
};
layout(std430, binding=0) buffer ParticleBuf { Particle particles[]; };

out flat int vIsGhost;
out float vHeight;
out float vSpeed;
out float vPressure;
out float vDensity;

void main() {
    int idx = gl_VertexID;
    Particle p = particles[idx];
    vIsGhost = p.flags.x;
    vHeight  = p.pos.y;
    vSpeed   = length(p.vel.xyz);
    vPressure = p.pressure;
    vDensity  = p.density;

    vec4 viewPos = viewMatrix * vec4(p.pos.xyz, 1.0);
    gl_Position = projectionMatrix * viewPos;
    gl_PointSize = particleRadius; // actual size refined in frag if needed
}