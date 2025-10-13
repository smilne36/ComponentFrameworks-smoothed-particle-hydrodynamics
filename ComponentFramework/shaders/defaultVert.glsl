#version 450

layout(location=0) in vec3 vertexPosition;
layout(location=1) in vec3 vertexNormal;
layout(location=2) in vec2 vertexUV;

layout(location=4) in vec3 instanceColor;
layout(location=5) in vec4 instancePos;
layout(location=6) in vec3 instanceVel;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix;

// 0=Depth, 1=Speed, 2=Pressure, 3=Density, 4=InstanceColor
uniform int colorMode = 1;
uniform vec2 vizRange = vec2(0.0, 10.0);
uniform int useSSBO = 1;

out vec3 fragColor;
flat out int fragGhost;

struct Particle {
    vec4 pos; vec4 vel; vec4 acc;
    float density; float pressure; float padA; float padB;
    ivec4 flags; // x=isGhost, y=isActive, z=padC, w=pad0
};

layout(std430, binding=0) buffer ParticleBuf { Particle particles[]; };

float remap01(float v, float lo, float hi) {
    return clamp((v - lo) / max(1e-8, hi - lo), 0.0, 1.0);
}
vec3 turbo(float t) {
    t = clamp(t, 0.0, 1.0);
    float r = clamp(abs(2.0*t - 0.5), 0.0, 1.0);
    float g = clamp(1.0 - abs(2.0*t - 1.0), 0.0, 1.0);
    float b = clamp(abs(2.0*t - 1.5), 0.0, 1.0);
    return vec3(r, g, b);
}

void main() {
    vec3 ipos;
    vec3 ivel;
    float density = 0.0;
    float pressure = 0.0;
    int isGhost = 0;

    if (useSSBO == 1) {
        Particle p = particles[gl_InstanceID];
        ipos = p.pos.xyz;
        ivel = p.vel.xyz;
        density = p.density;
        pressure = p.pressure;
        isGhost = p.flags.x;
    } else {
        ipos = instancePos.xyz;
        ivel = instanceVel.xyz;
    }

    mat4 M = modelMatrix;
    M[3].xyz += ipos;

    gl_Position = projectionMatrix * viewMatrix * M * vec4(vertexPosition, 1.0);

    if (colorMode == 4) {
        fragColor = instanceColor;
    } else if (colorMode == 0) {
        float t = remap01(ipos.y, -7.0, 7.0);
        fragColor = mix(vec3(0.1,0.4,1.0), vec3(0.5,1.0,1.0), t);
    } else if (colorMode == 1) {
        float speed = length(ivel);
        fragColor = turbo(remap01(speed, vizRange.x, vizRange.y));
    } else if (colorMode == 2) {
        fragColor = turbo(remap01(pressure, vizRange.x, vizRange.y));
    } else {
        fragColor = turbo(remap01(density, vizRange.x, vizRange.y));
    }

    fragGhost = isGhost;
}