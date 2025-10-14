#version 450
in vec3 fragColor;
flat in int fragGhost;
out vec4 outColor;
void main() {
    if (fragGhost == 1) discard;   // hide ghost particles
    outColor = vec4(fragColor, 1.0);
}
