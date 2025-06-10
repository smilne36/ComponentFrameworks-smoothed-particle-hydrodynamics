#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec3 vertexNormal;
layout(location = 2) in vec2 vertexUV;
layout(location = 3) in vec3 instancePos; // New
layout(location = 4) in vec3 instanceColor;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix; // This can be used for scaling

out vec3 fragColor;

void main() {
    mat4 model = modelMatrix;
    model[3].xyz += instancePos; // Translate by instance position
    gl_Position = projectionMatrix * viewMatrix * model * vec4(vertexPosition, 1.0);
    fragColor = instanceColor;
    // ... pass normal, uv, etc. as needed
}