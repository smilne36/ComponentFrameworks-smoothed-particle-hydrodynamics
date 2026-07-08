#version 450

layout(location=0) in vec3 vertexPosition;
layout(location=1) in vec3 vertexNormal;
layout(location=2) in vec2 vertexUV;
layout(location=5) in vec4 instancePos;
layout(location=4) in vec3 instanceColor;

uniform mat4 projectionMatrix, viewMatrix, modelMatrix;
uniform int  useSSBO = 1;

struct Particle {
    vec4 pos; vec4 vel; vec4 acc;
    float density; float pressure; float padA; float padB;
    ivec4 flags;
};
layout(std430, binding=0) buffer ParticleBuf { Particle particles[]; };

flat out int fragGhost;
out vec3 vWorldPos;
out vec3 vViewPos;
out vec3 vNormal;
out vec3 vVel;
out float vPressure;
out float vDensity;
out vec3 vInstanceColor;

void main() {
    int idx = gl_InstanceID;
    Particle p = particles[idx];
    fragGhost = p.flags.x;

    vec3 basePos = (useSSBO == 1) ? p.pos.xyz : instancePos.xyz;

    vec4 world = modelMatrix * vec4(vertexPosition, 1.0);
    world.xyz += basePos;
    vec4 viewPos = viewMatrix * world;
    gl_Position = projectionMatrix * viewPos;

    vWorldPos = basePos;   // particle centre: gives flat per-particle color drives
    vViewPos  = viewPos.xyz;
    vNormal   = mat3(viewMatrix) * vertexNormal;   // model is uniform scale, so no inverse-transpose needed
    vVel      = p.vel.xyz;
    vPressure = p.pressure;
    vDensity  = p.density;
    vInstanceColor = instanceColor;
}
