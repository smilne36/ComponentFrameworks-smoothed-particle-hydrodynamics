#version 450
layout(location=0) in vec2 dummy; // not used

uniform mat4 projectionMatrix, viewMatrix;
uniform float particleRadius; // world-space
uniform int   colorMode;      // reuse your modes if you want

struct Particle {
    vec4 pos; vec4 vel; vec4 acc;
    float density; float pressure; float padA; float padB;
    ivec4 flags;
};
layout(std430, binding=0) buffer ParticleBuf { Particle particles[]; };

out flat int vIsGhost;
out vec3 vAttrib; // pack what you want for color (e.g., vel, pressure)

void main() {
    int idx = gl_VertexID; // when drawing with glDrawArraysInstanced base+instance, or use gl_InstanceID
    Particle p = particles[idx];
    vIsGhost = p.flags.x;
    vAttrib = p.vel.xyz; // example

    vec4 viewPos = viewMatrix * vec4(p.pos.xyz, 1.0);
    gl_Position = projectionMatrix * viewPos;

    // Project radius to pixels
    float dist = max(1e-4, -viewPos.z);
    // Assuming perspective with symmetric frustum; extract focal length from projectionMatrix
    float fy = 1.0 / projectionMatrix[1][1]; // tan(fovY/2) ~= 1/f[1][1]
    float halfScreenHeight = 1.0; // NDC scale; we’ll set gl_PointSize in pixels via viewport scale in app
    float worldToNDC = particleRadius / (fy * dist);
    // App should pass in viewport height (pixels) if desired. Simpler: compute on CPU; or set gl_PointSize in app.
    gl_PointSize = 0.0; // set in app via uniform if you prefer; or compute with viewport height.
}