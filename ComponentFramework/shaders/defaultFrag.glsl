#version 450
in vec3 fragColor;
out vec4 outColor;

void main() {
    outColor = vec4(fragColor, 0.8); // 0.8 = alpha for some transparency
}