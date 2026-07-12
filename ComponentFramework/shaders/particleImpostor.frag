#version 450
in flat int vIsGhost;
in vec3 vWorldPos;
in vec3 vViewPos;
in vec3 vVel;
in float vPressure;
in float vDensity;

out vec4 outColor;

// ==== BEGIN SHARED PALETTE BLOCK (keep in sync: particleImpostor.frag / defaultFrag.glsl) ====
uniform mat4  viewMatrix;
uniform int   colorDrive;   // 0=Height 1=Speed 2=Pressure 3=Density 4=ViewDepth 5=VelocityDir 6=RadialDist 7=InstanceColor
uniform int   paletteId;    // 0=Classic 1=Turbo 2=Neon 3=Fire 4=Iridescent 5=Ice 6=Vaporwave 7=Toxic 8=Duotone
                             // 9=Galaxy 10=Plasma 11=Chrome 12=MoltenGold 13=AcidRings 14=Aurora
                             // 15=MarbleInk 16=LavaLamp 17=DiscoChecker 18=StainedGlass 19=PsychoSwirl 20=CandyStripes
                             // 21=Electric 22=Smoke 23=RGBPop
uniform vec2  vizRange;
uniform vec2  heightMinMax;
uniform vec3  boxCenter;
uniform vec3  duoColorA;
uniform vec3  duoColorB;
uniform float iridFreq;
uniform float iridShift;
uniform float animTime;      // seconds, for time-animated palettes
uniform float paletteFlow;   // scrolls ANY palette's gradient over time (0 = static)
uniform float patternScale;  // spatial frequency of the world-space pattern palettes
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

// Compact hash / value noise / fbm for the world-space pattern palettes
float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

float vnoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash13(i);
    float n100 = hash13(i + vec3(1, 0, 0));
    float n010 = hash13(i + vec3(0, 1, 0));
    float n110 = hash13(i + vec3(1, 1, 0));
    float n001 = hash13(i + vec3(0, 0, 1));
    float n101 = hash13(i + vec3(1, 0, 1));
    float n011 = hash13(i + vec3(0, 1, 1));
    float n111 = hash13(i + vec3(1, 1, 1));
    return mix(mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
               mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y), f.z);
}

float fbm(vec3 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 3; ++i) {
        v += a * vnoise(p);
        p *= 2.03;
        a *= 0.5;
    }
    return v;
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

vec3 applyPalette(float t, float facing, vec3 worldPos) {
    // Palette Flow scrolls any gradient over time (wrapped so ramps keep cycling)
    if (paletteFlow != 0.0) t = fract(t + paletteFlow * animTime);

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
    if (paletteId == 8) return mix(duoColorA, duoColorB, t); // Duotone
    if (paletteId == 9) return iqPal(t, vec3(0.20, 0.10, 0.35), vec3(0.35, 0.25, 0.55),
                                        vec3(1.00, 1.20, 0.70), vec3(0.10, 0.35, 0.65))
                              + vec3(0.10, 0.00, 0.25) * (1.0 - facing); // Galaxy / Nebula
    if (paletteId == 10) {                                              // Plasma
        float p = sin(t * 12.566 + facing * 6.2831853) * 0.5 + 0.5;
        float q = sin(t * 8.377  - facing * 9.4248)    * 0.5 + 0.5;
        return vec3(p, q, 1.0 - p * q);
    }
    if (paletteId == 11) {                                              // Chrome
        vec3 base = mix(vec3(0.05), vec3(0.85), t);
        return base + vec3(pow(1.0 - facing, 2.0));
    }
    if (paletteId == 12) {                                              // Molten Gold
        vec3 base = ramp4(t, vec3(0.10, 0.04, 0.00), vec3(0.55, 0.28, 0.02),
                              vec3(0.95, 0.65, 0.10), vec3(1.00, 0.92, 0.55));
        return base + vec3(1.00, 0.95, 0.80) * pow(1.0 - facing, 2.5) * 0.6;
    }
    if (paletteId == 13) return iqPal(t * 3.0 + iridFreq * (1.0 - facing) * 2.0 + iridShift,
                                      vec3(0.5), vec3(0.5), vec3(2.0, 3.0, 4.0),
                                      vec3(0.00, 0.15, 0.35)); // Acid Rings
    if (paletteId == 14) return iqPal(t + animTime * 0.15, vec3(0.15, 0.35, 0.35), vec3(0.25, 0.45, 0.45),
                                      vec3(0.80, 1.00, 1.20), vec3(0.25, 0.55, 0.85)); // Aurora

    // ---- World-space pattern palettes: the fluid swims THROUGH these ----
    vec3 wp = (worldPos - boxCenter) * patternScale;

    if (paletteId == 15) {                                              // Marble Ink
        float veins = sin((wp.x + wp.y * 0.7) * 1.8
                          + fbm(wp * 1.6 + vec3(0.0, animTime * 0.10, 0.0)) * 5.0);
        float v = smoothstep(-0.35, 0.35, veins);
        vec3 ink  = vec3(0.03, 0.05, 0.14);
        vec3 vein = mix(vec3(0.92, 0.90, 0.85), vec3(0.95, 0.75, 0.35), t);
        return mix(ink, vein, v);
    }
    if (paletteId == 16) {                                              // Lava Lamp
        float blob = fbm(wp * 0.55 + vec3(0.0, -animTime * 0.12, 0.0));
        float m = smoothstep(0.42, 0.58, blob);
        vec3 goo = iqPal(t * 0.4 + blob, vec3(0.70, 0.30, 0.10), vec3(0.35, 0.25, 0.10),
                         vec3(1.0), vec3(0.00, 0.10, 0.20));
        vec3 fluidBg = vec3(0.12, 0.02, 0.22);
        return mix(fluidBg, goo, m);
    }
    if (paletteId == 17) {                                              // Disco Checker
        vec3 cp = wp * 1.2 + vec3(animTime * 0.25);
        float checker = mod(floor(cp.x) + floor(cp.y) + floor(cp.z), 2.0);
        vec3 cA = hsv2rgb(vec3(fract(t + animTime * 0.05), 0.85, 1.0));
        vec3 cB = hsv2rgb(vec3(fract(t + animTime * 0.05 + 0.5), 0.85, 0.35));
        return mix(cA, cB, checker);
    }
    if (paletteId == 18) {                                              // Stained Glass
        vec3 cell = floor(wp * 1.1);
        vec3 g = fract(wp * 1.1) - 0.5;
        float edge = max(abs(g.x), max(abs(g.y), abs(g.z)));
        float grout = 1.0 - smoothstep(0.32, 0.5, edge);
        vec3 glass = hsv2rgb(vec3(hash13(cell), 0.75, 0.9));
        return glass * (0.15 + 0.85 * grout) * (0.6 + 0.4 * t);
    }
    if (paletteId == 19) {                                              // Psycho Swirl
        float ang = atan(wp.z, wp.x) / 6.2831853;
        float rad = length(wp.xz);
        float hue = fract(ang + rad * 0.20 + animTime * 0.08 + t * 0.30);
        return hsv2rgb(vec3(hue, 0.90, 0.95));
    }
    if (paletteId == 20) {                                              // Candy Stripes
        float s = sin(dot(wp, normalize(vec3(1.0, 0.35, 0.6))) * 5.0 + animTime * 0.8);
        float band = smoothstep(-0.25, 0.25, s);
        return mix(duoColorA, duoColorB, band) * (0.65 + 0.35 * t);
    }
    if (paletteId == 21) {                                              // Electric (hologram edge glow)
        vec3 body = vec3(0.02, 0.02, 0.05);
        vec3 glow = hsv2rgb(vec3(fract(0.50 + t * 0.35), 0.90, 1.0));
        float rim = pow(1.0 - facing, 1.5);
        return body + glow * (rim * 1.4 + 0.08);
    }
    if (paletteId == 22) {                                              // Smoke (moody monochrome)
        float n = fbm(wp * 0.8 + vec3(0.0, animTime * 0.05, 0.0));
        float v = clamp(0.15 + 0.85 * n * (0.4 + 0.6 * t), 0.0, 1.0);
        return vec3(v);
    }
    // 23 = RGB Pop: posterized rainbow bands (pop-art)
    float q = floor(fract(t) * 6.0) / 6.0;
    return hsv2rgb(vec3(q, 1.0, 1.0));
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
    if (vIsGhost == 1) discard;

    // gl_PointCoord Y increases downward; flip to get view-space Y (up = positive)
    vec2 disc = vec2(gl_PointCoord.x, 1.0 - gl_PointCoord.y) * 2.0 - 1.0;
    float r2 = dot(disc, disc);
    if (r2 > 1.0) discard;

    vec3 N = vec3(disc, sqrt(1.0 - r2));   // view-space fake sphere normal (billboard faces camera)
    vec3 V = normalize(-vViewPos);
    float facing = clamp(dot(N, V), 0.0, 1.0);

    vec3 col;
    if (colorDrive == 7) {
        col = vec3(1.0);   // instance color is not available in the impostor path
    } else {
        col = applyPalette(computeDrive(vWorldPos, vViewPos, vVel, vPressure, vDensity), facing, vWorldPos);
    }
    if (litSphere == 1) col = shadeLit(col, N, V, facing);

    outColor = vec4(applyColorAdjust(col), 1.0);
}
