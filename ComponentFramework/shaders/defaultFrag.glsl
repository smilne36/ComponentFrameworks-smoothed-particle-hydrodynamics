#version 450
flat in int fragGhost;
in vec3 vWorldPos;
in vec3 vViewPos;
in vec3 vNormal;
in vec3 vVel;
in float vPressure;
in float vDensity;
in vec3 vInstanceColor;

out vec4 outColor;

// ==== BEGIN SHARED PALETTE BLOCK (keep in sync: particleImpostor.frag / defaultFrag.glsl) ====
uniform mat4  viewMatrix;
uniform int   colorDrive;   // 0=Height 1=Speed 2=Pressure 3=Density 4=ViewDepth 5=VelocityDir 6=RadialDist 7=InstanceColor
uniform int   paletteId;    // 0=Classic 1=Turbo 2=Neon 3=Fire 4=Iridescent 5=Ice 6=Vaporwave 7=Toxic 8=Duotone
uniform vec2  vizRange;
uniform vec2  heightMinMax;
uniform vec3  boxCenter;
uniform vec3  duoColorA;
uniform vec3  duoColorB;
uniform float iridFreq;
uniform float iridShift;
uniform float hueShift;     // degrees
uniform float satMul;
uniform float brightMul;
uniform float contrastMul;
uniform int   invertColor;
uniform int   litSphere;
uniform vec3  sunDirWorld;
uniform vec3  sunColor;

float remap01(float v, float lo, float hi) {
    return clamp((v - lo) / max(1e-6, hi - lo), 0.0, 1.0);
}

float computeDrive(vec3 worldPos, vec3 viewPos, vec3 vel, float pressure, float density) {
    if (colorDrive == 0) return remap01(worldPos.y, heightMinMax.x, heightMinMax.y);
    if (colorDrive == 1) return remap01(length(vel), vizRange.x, vizRange.y);
    if (colorDrive == 2) return remap01(pressure,    vizRange.x, vizRange.y);
    if (colorDrive == 3) return remap01(density,     vizRange.x, vizRange.y);
    if (colorDrive == 4) return remap01(-viewPos.z,  vizRange.x, vizRange.y);
    if (colorDrive == 5) {
        if (dot(vel.xz, vel.xz) < 1e-12) return 0.0;
        return fract(atan(vel.z, vel.x) / 6.2831853 + 0.5);
    }
    return remap01(length(worldPos - boxCenter), vizRange.x, vizRange.y);
}

vec3 heightPalette(float t) {
    vec3 c1 = vec3(0.05, 0.15, 0.85);
    vec3 c2 = vec3(0.25, 0.60, 0.90);
    vec3 c3 = vec3(0.80, 0.30, 0.40);
    vec3 c4 = vec3(0.95, 0.10, 0.10);
    if (t < 0.33) {
        float u = t / 0.33;
        return mix(c1, c2, u);
    } else if (t < 0.66) {
        float u = (t - 0.33) / 0.33;
        return mix(c2, c3, u);
    } else {
        float u = (t - 0.66) / 0.34;
        return mix(c3, c4, u);
    }
}

vec3 turbo(float t) {
    t = clamp(t, 0.0, 1.0);
    return vec3(0.1357 + 4.0*t - 4.5*t*t,
                0.0000 + 2.0*t - 1.0*t*t,
                0.6667 - 1.5*t + 1.0*t*t);
}

vec3 iqPal(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * cos(6.2831853 * (c * t + d));
}

vec3 ramp4(float t, vec3 c1, vec3 c2, vec3 c3, vec3 c4) {
    if (t < 0.33) return mix(c1, c2, t / 0.33);
    if (t < 0.66) return mix(c2, c3, (t - 0.33) / 0.33);
    return mix(c3, c4, (t - 0.66) / 0.34);
}

vec3 applyPalette(float t, float facing) {
    if (paletteId == 0) return heightPalette(t);
    if (paletteId == 1) return turbo(t);
    if (paletteId == 2) return ramp4(t, vec3(0.05, 0.01, 0.18), vec3(0.45, 0.05, 0.65),
                                        vec3(1.00, 0.15, 0.55), vec3(0.15, 0.95, 1.00)); // Neon / Synthwave
    if (paletteId == 3) return ramp4(t, vec3(0.02, 0.00, 0.00), vec3(0.55, 0.05, 0.00),
                                        vec3(1.00, 0.45, 0.00), vec3(1.00, 0.95, 0.55)); // Fire / Lava
    if (paletteId == 4) return iqPal(t + iridFreq * (1.0 - facing) + iridShift,
                                     vec3(0.5), vec3(0.5), vec3(1.0), vec3(0.00, 0.33, 0.67)); // Iridescent / Oil slick
    if (paletteId == 5) return ramp4(t, vec3(0.02, 0.08, 0.20), vec3(0.15, 0.45, 0.75),
                                        vec3(0.55, 0.85, 0.95), vec3(0.95, 1.00, 1.00)); // Ice
    if (paletteId == 6) return ramp4(t, vec3(0.16, 0.06, 0.35), vec3(0.85, 0.35, 0.85),
                                        vec3(1.00, 0.55, 0.75), vec3(0.35, 0.95, 0.90)); // Vaporwave
    if (paletteId == 7) return ramp4(t, vec3(0.01, 0.03, 0.01), vec3(0.05, 0.35, 0.05),
                                        vec3(0.45, 0.95, 0.10), vec3(0.95, 1.00, 0.30)); // Toxic
    return mix(duoColorA, duoColorB, t); // 8 = Duotone
}

// Branchless RGB<->HSV (Sam Hocevar, public domain)
vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    return vec3(abs(q.z + (q.w - q.y) / (6.0*d + 1e-10)), d / (q.x + 1e-10), q.x);
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec3 applyColorAdjust(vec3 c) {
    vec3 hsv = rgb2hsv(clamp(c, 0.0, 1.0));
    hsv.x = fract(hsv.x + hueShift / 360.0);
    hsv.y = clamp(hsv.y * satMul, 0.0, 1.0);
    c = hsv2rgb(hsv) * brightMul;
    c = (c - 0.5) * contrastMul + 0.5;
    if (invertColor == 1) c = vec3(1.0) - c;
    return clamp(c, 0.0, 1.0);
}

vec3 shadeLit(vec3 col, vec3 N, vec3 V, float facing) {
    vec3 L = normalize(mat3(viewMatrix) * normalize(sunDirWorld));
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, normalize(L + V)), 0.0), 48.0);
    float rim  = pow(1.0 - facing, 3.0);
    return col * (0.35 + 0.65 * diff) + sunColor * spec * 0.6 + col * rim * 0.5;
}
// ==== END SHARED PALETTE BLOCK ====

void main() {
    if (fragGhost == 1) discard;   // hide ghost particles

    vec3 N = normalize(vNormal);
    vec3 V = normalize(-vViewPos);
    if (dot(N, V) < 0.0) N = -N;   // keep the normal camera-facing
    float facing = clamp(dot(N, V), 0.0, 1.0);

    vec3 col;
    if (colorDrive == 7) {
        col = vInstanceColor;
    } else {
        col = applyPalette(computeDrive(vWorldPos, vViewPos, vVel, vPressure, vDensity), facing);
    }
    if (litSphere == 1) col = shadeLit(col, N, V, facing);

    outColor = vec4(applyColorAdjust(col), 1.0);
}
