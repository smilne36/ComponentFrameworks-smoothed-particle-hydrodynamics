#version 450
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNorm;

uniform mat4 projectionMatrix, viewMatrix;

out vec3 vWorldPos;
out vec3 vNorm;

void main() {
    vWorldPos = aPos;
    vNorm     = aNorm;
    gl_Position = projectionMatrix * viewMatrix * vec4(aPos, 1.0);
}
