#version 450
uniform vec3 uColor = vec3(0.8, 0.9, 1.0);
out vec4 outColor;
void main() { outColor = vec4(uColor, 1.0); }