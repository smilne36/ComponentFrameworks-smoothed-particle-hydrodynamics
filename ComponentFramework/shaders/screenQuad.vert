#version 450
out vec2 vTexCoord;

void main() {
    // Generates a fullscreen triangle covering NDC [-1,1]^2 from 3 vertices — no buffer needed
    vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
    vTexCoord = pos;
}
