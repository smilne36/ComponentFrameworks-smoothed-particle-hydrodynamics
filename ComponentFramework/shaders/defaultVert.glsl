#version 450

layout(location=0) in vec3 vertexPosition;
layout(location=1) in vec3 vertexNormal;
layout(location=2) in vec2 vertexUV;

// Instance data (optional when useSSBO==1)
layout(location=5) in vec4 instancePos;
layout(location=4) in vec3 instanceColor;

uniform mat4 projectionMatrix, viewMatrix, modelMatrix;

// 0 = Height, 1 = Speed, 2 = Pressure, 3 = Density, 4 = InstanceColor
uniform int  colorMode = 1;
uniform vec2 vizRange  = vec2(0.0, 1.0);

// Controls
uniform vec2 heightMinMax = vec2(-7.0, 7.0);
uniform int  useSSBO = 1; // 1 = use SSBO positions, 0 = use instancePos

// SSBO with particle data
struct Particle {
    vec4 pos; vec4 vel; vec4 acc;
    float density; float pressure; float padA; float padB;
    ivec4 flags; // x=isGhost
};
layout(std430, binding=0) buffer ParticleBuf { Particle particles[]; };

out vec3 fragColor;
flat out int fragGhost;

float remap01(float v, float lo, float hi) {
    return clamp((v - lo) / max(1e-6, hi - lo), 0.0, 1.0);
}

vec3 turbo(float t) {
    t = clamp(t, 0.0, 1.0);
    return vec3(0.1357 + 4.0*t - 4.5*t*t,
                0.0000 + 2.0*t - 1.0*t*t,
                0.6667 - 1.5*t + 1.0*t*t);
}

void main() {
    int idx = gl_InstanceID;
    Particle p = particles[idx];
    fragGhost  = p.flags.x;

    // Select world-space instance position
    vec3 basePos = (useSSBO == 1) ? p.pos.xyz : instancePos.xyz;

    // Transform: translate mesh instance by basePos
    vec4 world = modelMatrix * vec4(vertexPosition, 1.0);
    world.xyz += basePos;
    gl_Position = projectionMatrix * viewMatrix * world;

    // Color selection
    if (colorMode == 4) {
        fragColor = instanceColor;
    } else if (colorMode == 0) {
        float t = remap01(basePos.y, heightMinMax.x, heightMinMax.y);
        fragColor = mix(vec3(0.1,0.4,1.0), vec3(0.5,1.0,1.0), t);
    } else if (colorMode == 1) {
        float speed = length(p.vel.xyz);
        fragColor = turbo(remap01(speed, vizRange.x, vizRange.y));
    } else if (colorMode == 2) {
        fragColor = turbo(remap01(p.pressure, vizRange.x, vizRange.y));
    } else {
        fragColor = turbo(remap01(p.density,  vizRange.x, vizRange.y));
    }
}