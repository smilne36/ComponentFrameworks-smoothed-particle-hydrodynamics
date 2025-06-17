#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec3 vertexNormal;
layout(location = 2) in vec2 vertexUV;
layout(location = 4) in vec3 instanceColor;
layout(location = 5) in vec4 instancePos;
layout(location = 6) in vec3 instanceVel;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix; // Used for scaling

out vec3 fragColor;

void main() {
    mat4 model = modelMatrix;
    model[3].xyz += instancePos.xyz;
    gl_Position = projectionMatrix * viewMatrix * model * vec4(vertexPosition, 1.0);

    // --- Gradient by depth (y) ---
    float minY = -7.0; // Set to your box min y
    float maxY =  7.0; // Set to your box max y
    float t = clamp((instancePos.y - minY) / (maxY - minY), 0.0, 1.0);

    // Blue at bottom, cyan at top
    fragColor = mix(vec3(0.1, 0.4, 1.0), vec3(0.5, 1.0, 1.0), t);
}