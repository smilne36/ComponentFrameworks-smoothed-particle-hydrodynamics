#include <iostream>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <vector>
#include <SDL.h>
#include <MMath.h>
#include "stb_image_write.h"

#include "Scene0p.h"
#include "Debug.h"
#include "Mesh.h"
#include "Body.h"
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "SceneManager.h"
#include "SPHFluid3D.h"

using MATH::MMath;

static void MakeRotationMat3XYZ(float rxDeg, float ryDeg, float rzDeg, float outM[9]) {
    const float rx = rxDeg * float(M_PI / 180.0);
    const float ry = ryDeg * float(M_PI / 180.0);
    const float rz = rzDeg * float(M_PI / 180.0);
    const float cx = cosf(rx), sx = sinf(rx);
    const float cy = cosf(ry), sy = sinf(ry);
    const float cz = cosf(rz), sz = sinf(rz);
    const float Rz[9] = { cz, sz, 0, -sz, cz, 0, 0, 0, 1 };
    const float Ry[9] = { cy, 0, -sy, 0, 1, 0, sy, 0, cy };
    const float Rx[9] = { 1, 0, 0, 0, cx, sx, 0, -sx, cx };
    auto mul3 = [](const float A[9], const float B[9], float C[9]) {
        for (int c = 0; c < 3; ++c)
            for (int r = 0; r < 3; ++r)
                C[c * 3 + r] = A[0 * 3 + r] * B[c * 3 + 0] + A[1 * 3 + r] * B[c * 3 + 1] + A[2 * 3 + r] * B[c * 3 + 2];
        };
    float Rzy[9]; mul3(Rz, Ry, Rzy);
    mul3(Rzy, Rx, outM);
}

Scene0p::Scene0p() {}
Scene0p::~Scene0p() {}

bool Scene0p::OnCreate() {
    Debug::Info("Loading assets Scene0p", __FILE__, __LINE__);

    sphere = new Body(); sphere->OnCreate();
    sphere->SetPosition(Vec3(0.0f, -20.0f, 0.0f));

    mesh = new Mesh("meshes/Sphere.obj"); mesh->OnCreate();

    shader = new Shader("shaders/defaultVert.glsl", "shaders/defaultFrag.glsl");
    if (!shader->OnCreate()) { std::cerr << "default shader failed\n"; return false; }

    {
        GLuint prog = shader->GetProgram();
        GLuint block = glGetProgramResourceIndex(prog, GL_SHADER_STORAGE_BLOCK, "ParticleBuf");
        if (block != GL_INVALID_INDEX) glShaderStorageBlockBinding(prog, block, 0);
        else std::cerr << "SSBO block 'ParticleBuf' not found in program\n";
    }

    projectionMatrix = MMath::perspective(45.0f, 16.0f / 9.0f, 0.5f, 100.0f);
    viewMatrix = MMath::lookAt(cameraPos, cameraTarget, cameraUp);
    modelMatrix.loadIdentity();

    fluidGPU = new SPHFluidGPU(50000);
    mesh->BindInstanceBuffer(fluidGPU->GetFluidVBO(), static_cast<GLsizei>(sizeof(float) * 4));

    lineShader = new Shader("shaders/lineVert.glsl", "shaders/lineFrag.glsl");
    if (!lineShader->OnCreate()) { std::cerr << "line shader failed\n"; return false; }
    glGenVertexArrays(1, &boxVAO);
    glBindVertexArray(boxVAO);
    glGenBuffers(1, &boxVBO);
    glBindBuffer(GL_ARRAY_BUFFER, boxVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * 512, nullptr, GL_DYNAMIC_DRAW); // enough for sphere/cylinder wireframes
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, (void*)0);
    glBindVertexArray(0);

    impostorShader = new Shader("shaders/particleImpostor.vert", "shaders/particleImpostor.frag");
    if (!impostorShader->OnCreate()) { std::cerr << "impostor shader failed\n"; return false; }
    SetupImpostorVAO();

    // SSFR shaders
    ssfrDepthShader = new Shader("shaders/fluidDepth.vert", "shaders/fluidDepth.frag");
    if (!ssfrDepthShader->OnCreate()) { std::cerr << "ssfrDepth shader failed\n"; return false; }

    ssfrSmoothShader = new Shader("shaders/screenQuad.vert", "shaders/depthSmooth.frag");
    if (!ssfrSmoothShader->OnCreate()) { std::cerr << "depthSmooth shader failed\n"; return false; }

    ssfrThickShader = new Shader("shaders/fluidDepth.vert", "shaders/fluidThickness.frag");
    if (!ssfrThickShader->OnCreate()) { std::cerr << "fluidThickness shader failed\n"; return false; }

    ssfrCompositeShader = new Shader("shaders/screenQuad.vert", "shaders/fluidComposite.frag");
    if (!ssfrCompositeShader->OnCreate()) { std::cerr << "fluidComposite shader failed\n"; return false; }

    skyShader = new Shader("shaders/screenQuad.vert", "shaders/skyGradient.frag");
    if (!skyShader->OnCreate()) { std::cerr << "skyGradient shader failed\n"; return false; }

    glGenVertexArrays(1, &ssfrQuadVAO);
    glEnable(GL_PROGRAM_POINT_SIZE);

    terrainShader = new Shader("shaders/terrainVert.glsl", "shaders/terrainFrag.glsl");
    if (!terrainShader->OnCreate()) { std::cerr << "terrain shader failed\n"; return false; }
    glGenVertexArrays(1, &terrainVAO);
    glGenBuffers(1, &terrainVBO);
    glGenBuffers(1, &terrainEBO);

    {
        GLint vp[4] = {0,0,0,0};
        glGetIntegerv(GL_VIEWPORT, vp);
        if (vp[2] > 0 && vp[3] > 0)
            InitSSFRBuffers(vp[2], vp[3]);
    }

    UpdateContainerWireframe();
    lastBoxCenter = fluidGPU->param_boxCenter;
    lastBoxHalf = fluidGPU->param_boxHalf;
    lastBoxEuler = fluidGPU->param_boxEulerDeg;

    vizMode = 0;        // Default to height-based coloring
    vizRangeMin = 0.0f;
    vizRangeMax = 10.0f;

    audioReactive = new AudioReactive();   // capture thread starts only when enabled

    return true;
}

void Scene0p::OnDestroy() {
    // First: join the audio capture thread so it never outlives the scene.
    if (audioReactive) { audioReactive->Stop(); delete audioReactive; audioReactive = nullptr; }
    if (sphere) { sphere->OnDestroy(); delete sphere; sphere = nullptr; }
    if (mesh) { mesh->OnDestroy(); delete mesh; mesh = nullptr; }
    if (shader) { shader->OnDestroy(); delete shader; shader = nullptr; }
    if (impostorShader) { impostorShader->OnDestroy(); delete impostorShader; impostorShader = nullptr; }
    if (fluidGPU) { delete fluidGPU; fluidGPU = nullptr; }
    if (boxVBO)   glDeleteBuffers(1, &boxVBO);
    if (boxVAO)   glDeleteVertexArrays(1, &boxVAO);
    if (lineShader) { lineShader->OnDestroy(); delete lineShader; lineShader = nullptr; }
    if (impostorVAO) glDeleteVertexArrays(1, &impostorVAO);

    if (ssfrDepthShader)     { ssfrDepthShader->OnDestroy();     delete ssfrDepthShader;     ssfrDepthShader     = nullptr; }
    if (ssfrSmoothShader)    { ssfrSmoothShader->OnDestroy();    delete ssfrSmoothShader;    ssfrSmoothShader    = nullptr; }
    if (ssfrThickShader)     { ssfrThickShader->OnDestroy();     delete ssfrThickShader;     ssfrThickShader     = nullptr; }
    if (ssfrCompositeShader) { ssfrCompositeShader->OnDestroy(); delete ssfrCompositeShader; ssfrCompositeShader = nullptr; }
    if (skyShader)           { skyShader->OnDestroy();           delete skyShader;           skyShader           = nullptr; }
    if (ssfrQuadVAO)         glDeleteVertexArrays(1, &ssfrQuadVAO);
    DestroySSFRBuffers();
    if (terrainShader) { terrainShader->OnDestroy(); delete terrainShader; terrainShader = nullptr; }
    if (terrainVAO) glDeleteVertexArrays(1, &terrainVAO);
    if (terrainVBO) glDeleteBuffers(1, &terrainVBO);
    if (terrainEBO) glDeleteBuffers(1, &terrainEBO);
    if (riverBankVAO) glDeleteVertexArrays(1, &riverBankVAO);
    if (riverBankVBO) glDeleteBuffers(1, &riverBankVBO);
}

void Scene0p::HandleEvents(const SDL_Event& e) {
    // Let ImGui consume mouse events first
    bool imguiWantsMouse = ImGui::GetIO().WantCaptureMouse;

    switch (e.type) {
    case SDL_KEYDOWN:
        switch (e.key.keysym.scancode) {
        // Pan target with WASD/QE (unchanged)
        case SDL_SCANCODE_W: cameraTarget.y += 0.3f; break;
        case SDL_SCANCODE_S: cameraTarget.y -= 0.3f; break;
        case SDL_SCANCODE_A: cameraTarget.x -= 0.3f; break;
        case SDL_SCANCODE_D: cameraTarget.x += 0.3f; break;
        case SDL_SCANCODE_Q: cameraTarget.z += 0.3f; break;
        case SDL_SCANCODE_E: cameraTarget.z -= 0.3f; break;
        case SDL_SCANCODE_Z: drawInWireMode = !drawInWireMode; break;
        // P captures a screenshot (unless typing in the UI)
        case SDL_SCANCODE_P:
            if (!ImGui::GetIO().WantCaptureKeyboard) captureRequested = true;
            break;
        // R resets to default view
        case SDL_SCANCODE_R:
            cameraTarget  = Vec3(0, 0, 0);
            camDist       = 22.0f;
            camAzimuth    = 0.0f;
            camElevation  = 0.22f;
            break;
        default: break;
        }
        break;

    case SDL_MOUSEBUTTONDOWN:
        if (!imguiWantsMouse) {
            mouseButton = e.button.button;
            mouseX = e.button.x;
            mouseY = e.button.y;
            mouseDown = true;
        }
        break;

    case SDL_MOUSEBUTTONUP:
        mouseDown = false;
        mouseButton = -1;
        break;

    case SDL_MOUSEMOTION:
        if (mouseDown && !imguiWantsMouse) {
            int dx = e.motion.x - mouseX;
            int dy = e.motion.y - mouseY;
            mouseX = e.motion.x;
            mouseY = e.motion.y;

            if (mouseButton == SDL_BUTTON_LEFT) {
                // Left drag — orbit
                camAzimuth   -= dx * 0.005f;
                camElevation += dy * 0.005f;
                // Clamp elevation so camera doesn't flip over
                const float maxEl = 1.55f; // ~89 degrees
                if (camElevation >  maxEl) camElevation =  maxEl;
                if (camElevation < -maxEl) camElevation = -maxEl;
            } else if (mouseButton == SDL_BUTTON_RIGHT) {
                // Right drag — pan target in the camera's right/up plane
                float speed = camDist * 0.0015f;
                // Camera right vector
                float cosEl = std::cos(camElevation);
                Vec3 right(std::cos(camAzimuth), 0.0f, -std::sin(camAzimuth));
                Vec3 up(0.0f, 1.0f, 0.0f);
                cameraTarget = cameraTarget - right * (dx * speed) + up * (dy * speed);
            }
        }
        break;

    case SDL_MOUSEWHEEL:
        if (!imguiWantsMouse) {
            // Scroll to zoom: exponential feel
            camDist *= std::pow(0.90f, (float)e.wheel.y);
            if (camDist < 1.0f)  camDist = 1.0f;
            if (camDist > 120.0f) camDist = 120.0f;
        }
        break;

    default: break;
    }
}

void Scene0p::UpdateContainerWireframe() {
    if (!fluidGPU || !boxVBO) return;

    const Vec3 C = fluidGPU->param_boxCenter;
    const Vec3 H = fluidGPU->param_boxHalf;
    float Rm[9]; MakeRotationMat3XYZ(
        fluidGPU->param_boxEulerDeg.x,
        fluidGPU->param_boxEulerDeg.y,
        fluidGPU->param_boxEulerDeg.z, Rm);

    auto xform = [&](float x, float y, float z) -> Vec3 {
        return Vec3(
            Rm[0] * x + Rm[3] * y + Rm[6] * z + C.x,
            Rm[1] * x + Rm[4] * y + Rm[7] * z + C.y,
            Rm[2] * x + Rm[5] * y + Rm[8] * z + C.z);
        };

    std::vector<float> lines;
    lines.reserve(512 * 3);
    auto push = [&](const Vec3& p) { lines.push_back(p.x); lines.push_back(p.y); lines.push_back(p.z); };
    auto seg  = [&](const Vec3& a, const Vec3& b) { push(a); push(b); };

    const int   shape   = fluidGPU->param_shapeType;
    const float TWO_PI  = 6.28318530718f;
    const int   SEGS    = 48;

    if (shape == 1) {
        // Sphere: three orthogonal great circles
        const float r = H.x;
        for (int axis = 0; axis < 3; ++axis) {
            Vec3 prev;
            for (int s = 0; s <= SEGS; ++s) {
                float a = (float(s) / SEGS) * TWO_PI;
                float ca = std::cos(a) * r, sa = std::sin(a) * r;
                Vec3 p = (axis == 0) ? xform(0.0f, ca, sa)
                       : (axis == 1) ? xform(ca, 0.0f, sa)
                       :               xform(ca, sa, 0.0f);
                if (s > 0) seg(prev, p);
                prev = p;
            }
        }
    } else if (shape == 2) {
        // Cylinder: two cap circles + four vertical edges
        const float r = H.x, hh = H.y;
        for (int cap = 0; cap < 2; ++cap) {
            float y = cap ? hh : -hh;
            Vec3 prev;
            for (int s = 0; s <= SEGS; ++s) {
                float a = (float(s) / SEGS) * TWO_PI;
                Vec3 p = xform(std::cos(a) * r, y, std::sin(a) * r);
                if (s > 0) seg(prev, p);
                prev = p;
            }
        }
        for (int s = 0; s < 4; ++s) {
            float a = (float(s) / 4.0f) * TWO_PI;
            float cx = std::cos(a) * r, cz = std::sin(a) * r;
            seg(xform(cx, -hh, cz), xform(cx, hh, cz));
        }
    } else {
        // Box: 12 edges
        Vec3 c[8]; int i = 0;
        for (int sx = -1; sx <= 1; sx += 2)
            for (int sy = -1; sy <= 1; sy += 2)
                for (int sz = -1; sz <= 1; sz += 2)
                    c[i++] = xform(sx * H.x, sy * H.y, sz * H.z);
        int E[24] = { 0,1, 0,2, 0,4, 3,1, 3,2, 3,7, 5,1, 5,4, 5,7, 6,2, 6,4, 6,7 };
        for (int e = 0; e < 24; ++e) push(c[E[e]]);
    }

    containerWireVerts = int(lines.size() / 3);
    glBindBuffer(GL_ARRAY_BUFFER, boxVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, lines.size() * sizeof(float), lines.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Scene0p::Update(const float deltaTime) {
    ballAnimTime += deltaTime;
    float ballX = std::sin(ballAnimTime) * 3.0f;
    if (sphere) sphere->SetPosition(Vec3(ballX, 0.0f, 0.0f));

    // Rebuild camera position from spherical orbit parameters
    {
        float cosEl = std::cos(camElevation);
        float sinEl = std::sin(camElevation);
        cameraPos = cameraTarget + Vec3(
            std::sin(camAzimuth) * cosEl,
            sinEl,
            std::cos(camAzimuth) * cosEl) * camDist;
    }
    viewMatrix = MMath::lookAt(cameraPos, cameraTarget, cameraUp);

    // Resize SSFR FBOs if viewport changed; remember the on-screen size for captures
    {
        GLint vp[4] = {0,0,0,0};
        glGetIntegerv(GL_VIEWPORT, vp);
        if (vp[2] > 0 && vp[3] > 0) {
            windowW = vp[2];
            windowH = vp[3];
            if (vp[2] != ssfrW || vp[3] != ssfrH)
                InitSSFRBuffers(vp[2], vp[3]);
        }
    }

    if (!fluidGPU) return;

    ImGui::SetNextWindowSize(ImVec2(430.0f, 940.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("Fluid Controls");
    ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
    ImGui::Text("Particles: %zu (fluids: %zu)", fluidGPU->particles.size(), fluidGPU->GetNumFluids());
    ImGui::Checkbox("Pause simulation", &fluidGPU->param_pause);
    ImGui::SameLine();
    if (ImGui::Button("Reset Simulation")) pendingReset = true;
    ImGui::SameLine();
    if (ImGui::Button("Fit Camera")) {
        Vec3 minV(FLT_MAX, FLT_MAX, FLT_MAX);
        Vec3 maxV(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        size_t count = 0;
        for (const auto& p : fluidGPU->particles) {
            if (p.isGhost) continue;
            minV.x = std::min(minV.x, p.pos.x);
            minV.y = std::min(minV.y, p.pos.y);
            minV.z = std::min(minV.z, p.pos.z);
            maxV.x = std::max(maxV.x, p.pos.x);
            maxV.y = std::max(maxV.y, p.pos.y);
            maxV.z = std::max(maxV.z, p.pos.z);
            ++count;
        }
        if (count) {
            Vec3 center((minV.x + maxV.x) * 0.5f, (minV.y + maxV.y) * 0.5f, (minV.z + maxV.z) * 0.5f);
            float height = (maxV.y - minV.y);
            float width = (maxV.x - minV.x);
            const float fovY = 45.0f * (3.14159265359f / 180.0f);
            float halfH = std::max(height, width) * 0.5f;
            float dist = halfH / std::tan(fovY * 0.5f) * 1.35f;
            cameraTarget = center;
            cameraPos = Vec3(center.x, center.y, center.z + dist);
        }
    }
    ImGui::Separator();

    if (ImGui::CollapsingHeader("Presets")) {
        ImGui::PushID("Presets");
        if (ImGui::Button("Preset: Stable Water")) {
            fluidGPU->param_pause = false;
            fluidGPU->param_h = 0.28f;
            fluidGPU->param_restDensity = 1000.0f;
            fluidGPU->param_gasConstant = 2000.0f;
            fluidGPU->param_viscosity = 3.5f;
            fluidGPU->param_gravityY = -980.0f;
            fluidGPU->param_surfaceTension = 0.0f;
            fluidGPU->param_timeStep = 1.0e-3f;
            pendingReset = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Preset: Splashy Water")) {
            fluidGPU->param_pause = false;
            fluidGPU->param_h = 0.22f;
            fluidGPU->param_restDensity = 1000.0f;
            fluidGPU->param_gasConstant = 6000.0f;
            fluidGPU->param_viscosity = 1.2f;
            fluidGPU->param_gravityY = -980.0f;
            fluidGPU->param_surfaceTension = 0.12f;
            fluidGPU->param_timeStep = 5.0e-4f;
            fluidGPU->param_useJitter = false;
            fluidGPU->param_jitterAmp = 0.06f;
            fluidGPU->param_wallRestitution = 0.05f;
            fluidGPU->param_wallFriction = 0.05f;
            pendingReset = true;
        }
        ImGui::Separator();
        ImGui::TextDisabled("Art presets: physics + colors + audio, tuned for videos");
        if (ImGui::Button("Zero-G Nebula")) ApplyArtPreset(0);
        ImGui::SameLine();
        if (ImGui::Button("Dream Float"))   ApplyArtPreset(1);
        ImGui::SameLine();
        if (ImGui::Button("Acid Trip"))     ApplyArtPreset(2);
        if (ImGui::Button("Club Water"))    ApplyArtPreset(3);
        ImGui::SameLine();
        if (ImGui::Button("Molten Disco"))  ApplyArtPreset(4);
        ImGui::PopID();
    }

    if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushID("Simulation");
        ImGui::SliderFloat("h (smoothing)", &fluidGPU->param_h, 0.10f, 1.00f);
        ImGui::SliderFloat("mass", &fluidGPU->param_mass, 1.0f, 50.0f);
        ImGui::SliderFloat("restDensity", &fluidGPU->param_restDensity, 100.0f, 2000.0f);
        ImGui::SliderFloat("gasConstant", &fluidGPU->param_gasConstant, 100.0f, 30000.0f);
        ImGui::SliderFloat("viscosity", &fluidGPU->param_viscosity, 0.0f, 20.0f);
        ImGui::SliderFloat("gravityY", &fluidGPU->param_gravityY, -5000.0f, 0.0f);
        ImGui::SliderFloat("surfaceTension", &fluidGPU->param_surfaceTension, 0.0f, 1.0f);
        ImGui::SliderFloat("timeStep", &fluidGPU->param_timeStep, 1e-5f, 5e-3f, "%.6f", ImGuiSliderFlags_Logarithmic);
        ImGui::PopID();
    }

    if (ImGui::CollapsingHeader("Container")) {
        ImGui::PushID("ContainerBox");
        if (fluidGPU->riverMode) {
            fluidGPU->param_shapeType = 0;   // river terrain assumes the box
            ImGui::TextDisabled("River mode uses the Box container.");
        } else {
            ImGui::Combo("Shape", &fluidGPU->param_shapeType, "Box\0Sphere\0Cylinder\0");
        }
        ImGui::DragFloat3("Center", &fluidGPU->param_boxCenter.x, 0.05f);
        if (fluidGPU->param_shapeType == 1) {
            ImGui::DragFloat("Radius", &fluidGPU->param_boxHalf.x, 0.05f, 0.05f, 100.0f);
        } else if (fluidGPU->param_shapeType == 2) {
            ImGui::DragFloat("Radius", &fluidGPU->param_boxHalf.x, 0.05f, 0.05f, 100.0f);
            ImGui::DragFloat("Half Height", &fluidGPU->param_boxHalf.y, 0.05f, 0.05f, 100.0f);
        } else {
            ImGui::DragFloat3("Half Extents", &fluidGPU->param_boxHalf.x, 0.05f, 0.05f, 100.0f);
        }
        if (fluidGPU->param_shapeType != 1)   // rotation is meaningless for a sphere
            ImGui::DragFloat3("Euler XYZ", &fluidGPU->param_boxEulerDeg.x, 0.5f, -180.0f, 180.0f);
        ImGui::SliderFloat("Wall Restitution", &fluidGPU->param_wallRestitution, 0.0f, 1.0f);
        ImGui::SliderFloat("Wall Friction", &fluidGPU->param_wallFriction, 0.0f, 1.0f);
        ImGui::Separator();
        ImGui::Checkbox("Show Outline", &showContainerOutline);
        if (showContainerOutline)
            ImGui::ColorEdit3("Outline Color", containerOutlineColor);
        ImGui::TextDisabled("The sim grid follows container edits automatically;\nfluid is squeezed to stay inside as walls move.");
        ImGui::PopID();
    }

    if (ImGui::CollapsingHeader("Spawn Layout")) {
        ImGui::PushID("SpawnLayout");
        ImGui::Checkbox("Use jitter", &fluidGPU->param_useJitter); ImGui::SameLine();
        ImGui::SliderFloat("Jitter amp * spacing", &fluidGPU->param_jitterAmp, 0.0f, 0.5f, "%.2f"); ImGui::SameLine();
        if (ImGui::Button("Rebuild Layout")) { pendingReset = true; }
        ImGui::PopID();
    }

    if (ImGui::CollapsingHeader("Waves")) {
        ImGui::PushID("Waves");
        ImGui::SliderFloat("Amplitude", &waveAmplitude, 0.0f, 25.0f);
        ImGui::SliderFloat("Wavelength", &waveWavelength, 0.5f, 10.0f);
        ImGui::SliderFloat("Phase speed", &wavePhaseSpeed, 0.0f, 20.0f);
        ImGui::RadioButton("Dir X", &waveDirIdx, 0); ImGui::SameLine();
        ImGui::RadioButton("Dir Y", &waveDirIdx, 1); ImGui::SameLine();
        ImGui::RadioButton("Dir Z", &waveDirIdx, 2);
        ImGui::InputFloat("Band Y min", &yBandMin);
        ImGui::InputFloat("Band Y max", &yBandMax);
        ImGui::Checkbox("Continuous wave", &continuousWave);
        if (ImGui::Button("Impulse Now")) {
            Vec3 dir = (waveDirIdx == 0) ? Vec3(1, 0, 0) : (waveDirIdx == 1) ? Vec3(0, 1, 0) : Vec3(0, 0, 1);
            fluidGPU->ApplyWaveImpulse(waveAmplitude, waveWavelength, wavePhase, dir, yBandMin, yBandMax);
        }
        ImGui::PopID();
    }

    if (ImGui::CollapsingHeader("Audio Reactive")) {
        ImGui::PushID("AudioReactive");
        ImGui::Checkbox("Enable", &audioReactiveEnabled);   // thread start/stop happens in Update()
        ImGui::TextDisabled("%s", audioReactive->GetStatusText().c_str());
        ImGui::TextDisabled("Reacts to whatever your computer is playing.");

        if (ImGui::SliderFloat("Master Gain", &audioMasterGain, 0.0f, 4.0f))
            audioReactive->gain.store(audioMasterGain);

        ImGui::ProgressBar(std::min(1.0f, audioReactive->GetBass()),   ImVec2(-1.0f, 0.0f), "Bass");
        ImGui::ProgressBar(std::min(1.0f, audioReactive->GetMid()),    ImVec2(-1.0f, 0.0f), "Mid");
        ImGui::ProgressBar(std::min(1.0f, audioReactive->GetTreble()), ImVec2(-1.0f, 0.0f), "Treble");

        ImGui::Separator(); ImGui::Text("Physical (splashes)");
        ImGui::SliderFloat("Bass Force",       &audioBassForce,       0.0f, 30.0f);
        ImGui::SliderFloat("Bass Threshold",   &audioBassThreshold,   0.0f, 1.0f);
        ImGui::SliderFloat("Mid Force",        &audioMidForce,        0.0f, 30.0f);
        ImGui::SliderFloat("Mid Threshold",    &audioMidThreshold,    0.0f, 1.0f);
        ImGui::SliderFloat("Treble Force",     &audioTrebleForce,     0.0f, 30.0f);
        ImGui::SliderFloat("Treble Threshold", &audioTrebleThreshold, 0.0f, 1.0f);

        ImGui::Separator(); ImGui::Text("Visual (pulses)");
        ImGui::SliderFloat("Size Kick (bass)", &audioSizeKick,    0.0f, 2.0f);
        ImGui::SliderFloat("Shimmer (treble)", &audioShimmerKick, 0.0f, 2.0f);
        ImGui::SliderFloat("Foam Kick (mid)",  &audioFoamKick,    0.0f, 2.0f);

        if (ImGui::TreeNode("Advanced")) {
            float atk = audioReactive->attackMs.load();
            float rel = audioReactive->releaseMs.load();
            if (ImGui::SliderFloat("Attack (ms)",  &atk, 1.0f, 100.0f)) audioReactive->attackMs.store(atk);
            if (ImGui::SliderFloat("Release (ms)", &rel, 20.0f, 800.0f)) audioReactive->releaseMs.store(rel);
            ImGui::SliderFloat("Bass Wavelength",    &audioBassWavelength,   1.0f, 30.0f);
            ImGui::SliderFloat("Mid Wavelength",     &audioMidWavelength,    0.5f, 10.0f);
            ImGui::SliderFloat("Treble Wavelength",  &audioTrebleWavelength, 0.2f, 3.0f);
            ImGui::SliderFloat("Bass Phase Speed",   &audioBassPhaseSpeed,   0.0f, 10.0f);
            ImGui::SliderFloat("Mid Rotation Speed", &audioMidRotSpeed,      0.0f, 5.0f);
            ImGui::SliderFloat("Treble Phase Speed", &audioTreblePhaseSpeed, 0.0f, 30.0f);
            ImGui::TreePop();
        }
        ImGui::PopID();
    }

    if (ImGui::CollapsingHeader("Reels Export")) {
        ImGui::PushID("ReelsExport");
        if (reelExporting) {
            float prog = reelBands.frameCount > 0
                ? float(reelFrame) / float(reelBands.frameCount) : 0.0f;
            ImGui::ProgressBar(prog, ImVec2(-1.0f, 0.0f));
            ImGui::Text("Rendering frame %d / %d", reelFrame, reelBands.frameCount);
            if (ImGui::Button("Cancel")) FinishReelExport(false);
        } else {
            ImGui::InputText("Audio File", reelAudioPath, sizeof(reelAudioPath));
            ImGui::InputText("Output Folder", reelOutDir, sizeof(reelOutDir));
            ImGui::Combo("FPS", &reelFpsIdx, "30\0" "60\0");
            ImGui::Combo("Aspect", &reelResIdx,
                "1080 x 1920 (Reel)\0" "1080 x 1350 (4:5)\0" "1920 x 1080 (Wide)\0");
            ImGui::InputFloat("Max seconds (0 = full)", &reelMaxSeconds);
            if (ImGui::Button("Export Reel")) StartReelExport();
            if (!reelStatus.empty()) ImGui::TextWrapped("%s", reelStatus.c_str());
            ImGui::TextDisabled(
                "Uses your current live look + audio settings. Renders numbered\n"
                "PNGs to the folder, then double-click mux_reel.bat (needs ffmpeg)\n"
                "to make the vertical mp4 with your track.");
        }
        ImGui::PopID();
    }

    if (ImGui::CollapsingHeader("River / Stream Mode")) {
        ImGui::PushID("River");
        bool wasRiver = fluidGPU->riverMode;
        ImGui::Checkbox("Enable River Mode", &fluidGPU->riverMode);
        ImGui::SliderInt("River Seed", &riverSeed, 1, 999);
        if (ImGui::Button("Generate New River")) {
            if (!fluidGPU->riverMode) fluidGPU->riverMode = true;
            // Elongate box in Z for a longer channel
            fluidGPU->param_boxHalf = Vec3(7.0f, 8.0f, 10.0f);
            fluidGPU->param_boxCenter = Vec3(0.0f, 0.0f, 0.0f);
            fluidGPU->param_boxEulerDeg = Vec3(0, 0, 0);
            fluidGPU->GenerateRiverTerrain(riverSeed);
            BuildTerrainMesh();
            BuildRiverBankLines();
            pendingReset = true;
            // Auto-position camera for a clear river view
            cameraTarget  = Vec3(fluidGPU->param_boxCenter.x, 0.0f, fluidGPU->param_boxCenter.z);
            camDist       = 28.0f;
            camAzimuth    = 0.0f;
            camElevation  = 0.75f; // ~43 degrees — nice angled top-down
        }
        if (fluidGPU->riverMode) {
            ImGui::Text("Emitter: (%.1f, %.1f, %.1f)", fluidGPU->riverEmitterPos.x, fluidGPU->riverEmitterPos.y, fluidGPU->riverEmitterPos.z);
            ImGui::SliderFloat("Emitter vel Z", &fluidGPU->riverEmitterVel.z, 0.0f, 15.0f);
            ImGui::SliderFloat("Gravity Z (flow)", &fluidGPU->param_gravityZ, 0.0f, 400.0f);
            ImGui::SliderFloat("Gravity Y",         &fluidGPU->param_gravityY, -980.0f, -50.0f);
            ImGui::Checkbox("Show bank lines", &showRiverLines);
        }
        if (!fluidGPU->riverMode && wasRiver) {
            // Restore box to default when river mode is turned off
            fluidGPU->param_boxHalf = Vec3(7, 7, 7);
            fluidGPU->param_boxCenter = Vec3(0, 0, 0);
            fluidGPU->param_gravityY = -980.0f;
            fluidGPU->param_gravityZ =    0.0f;
            pendingReset = true;
        }
        ImGui::PopID();
    }

    if (ImGui::CollapsingHeader("Appearance", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushID("Appearance");
        int renderMode = useWaterRendering ? 0 : (useImpostors ? 1 : 2);
        if (ImGui::Combo("Render Mode", &renderMode, "Water Surface\0Point Impostors\0Mesh Spheres\0")) {
            useWaterRendering = (renderMode == 0);
            useImpostors      = (renderMode == 1);
        }
        ImGui::Separator(); ImGui::Text("Color");
        ImGui::Combo("Palette", &paletteId,
            "Classic Height\0Turbo\0Neon / Synthwave\0Fire / Lava\0Iridescent / Oil Slick\0Ice\0Vaporwave\0Toxic\0Duotone\0"
            "Galaxy / Nebula\0Plasma\0Chrome\0Molten Gold\0Acid Rings\0Aurora\0"
            "Marble Ink\0Lava Lamp\0Disco Checker\0Stained Glass\0Psycho Swirl\0Candy Stripes\0"
            "Electric\0Smoke\0RGB Pop\0");
        ImGui::SliderFloat("Palette Flow", &paletteFlow, -2.0f, 2.0f);
        if (paletteId >= 15)
            ImGui::SliderFloat("Pattern Scale", &patternScale, 0.1f, 5.0f);
        ImGui::Combo("Color Drive", &vizMode,
            "Height\0Speed\0Pressure\0Density\0View Depth\0Velocity Direction\0Distance from Center\0Instance Color\0");
        ImGui::DragFloat("Range Min", &vizRangeMin, 0.1f);
        ImGui::DragFloat("Range Max", &vizRangeMax, 0.1f);
        ImGui::TextDisabled("Height drive uses box Y extents, not Range. Palette & drive\ncolor the Impostor/Mesh modes; Adjustments grade every mode.");
        if (paletteId == 8 || paletteId == 20) {   // Duotone + Candy Stripes share the pickers
            ImGui::ColorEdit3("Duotone A", duoColorA);
            ImGui::ColorEdit3("Duotone B", duoColorB);
        }
        if (paletteId == 4 || paletteId == 13) {
            ImGui::SliderFloat("Irid Frequency", &iridFreq, 0.0f, 8.0f);
            ImGui::SliderFloat("Irid Shift",     &iridShift, 0.0f, 1.0f);
        }
        ImGui::Checkbox("Lit particles", &litParticles);

        ImGui::Separator(); ImGui::Text("Adjustments");
        ImGui::SliderFloat("Hue Shift",  &hueShiftDeg, -180.0f, 180.0f);
        ImGui::SliderFloat("Saturation", &satMul,      0.0f, 2.0f);
        ImGui::SliderFloat("Brightness", &brightMul,   0.0f, 2.0f);
        ImGui::SliderFloat("Contrast",   &contrastMul, 0.0f, 2.0f);
        ImGui::Checkbox("Invert", &invertColor);

        ImGui::Separator(); ImGui::Text("Background");
        ImGui::ColorEdit3("Background", bgColor);
        if (useWaterRendering) {
            ImGui::Checkbox("Sky Background", &showSkyBackground);
            ImGui::ColorEdit3("Sky Horizon", skyColor);
            ImGui::ColorEdit3("Sky Zenith", skyZenith);
            ImGui::ColorEdit3("Reflect Tint", envReflectColor);
            ImGui::TextDisabled("Sky colors always drive the water's reflections;\nthe checkbox only draws them as the backdrop.");
        }
        if (useWaterRendering && ImGui::TreeNode("Water Surface Detail")) {
            if (ImGui::Checkbox("Half-Res Fluid (faster)", &ssfrHalfRes) && windowW > 0)
                InitSSFRBuffers(windowW, windowH);
            ImGui::SliderInt("Smooth Iterations",  &smoothIterations,    0,    20);
            ImGui::SliderFloat("Smoothing Scale",   &worldFilterScale,   0.0f, 10.0f);
            ImGui::SliderFloat("Surface Merge",     &surfaceMerge,       0.5f, 8.0f);
            ImGui::SliderFloat("Render Radius",     &renderRadiusScale,  0.5f, 2.0f);
            ImGui::Separator();
            ImGui::ColorEdit3("Water Extinction",   waterExtinction);
            ImGui::SliderFloat("Thickness Scale",   &thicknessScale,      0.01f, 20.0f);
            ImGui::SliderFloat("Blob Strength",     &thicknessStrength,   0.005f, 0.3f, "%.3f");
            ImGui::SliderFloat("Blob Falloff",      &thicknessFalloff,    1.0f, 8.0f);
            ImGui::ColorEdit3("Deep Water Color",   deepWaterColor);
            ImGui::Separator();
            ImGui::SliderFloat3("Sun Dir (World)",  sunDirWorld,         -1.0f,  1.0f);
            ImGui::ColorEdit3("Sun Color",          sunColor);
            ImGui::SliderFloat("Specular Power",    &specularPower,       8.0f,  1024.0f);
            ImGui::SliderFloat("Specular Strength", &specularStrength,    0.0f,  3.0f);
            ImGui::SliderFloat("Refraction",        &refractionStrength,  0.0f,  0.2f);
            ImGui::SliderFloat("Fresnel Bias",      &fresnelBias,         0.0f,  0.3f);
            ImGui::Separator();
            ImGui::SliderFloat("Foam Generation",   &fluidGPU->param_foamGen, 0.0f, 2.0f);
            ImGui::SliderFloat("Foam Threshold",    &fluidGPU->param_foamVelRef, 1.0f, 30.0f);
            ImGui::SliderFloat("Foam Amount",       &foamAmount,          0.0f,  4.0f);
            ImGui::SliderFloat("Exposure",          &exposure,            0.25f, 4.0f);
            ImGui::TextDisabled("Lower Foam Threshold = foam appears at gentler motion.");
            ImGui::TreePop();
        }
        ImGui::PopID();
    }

    if (ImGui::CollapsingHeader("Performance")) {
        ImGui::PushID("Performance");
        ImGui::Checkbox("Render from SSBO (fast)", &renderFromSSBO);
        ImGui::Checkbox("Enable ghost boundaries", &fluidGPU->param_enableGhosts);
        ImGui::SameLine();
        ImGui::Checkbox("Grid sort (unused)", &fluidGPU->param_enableSort);
        ImGui::PopID();
    }

    if (ImGui::CollapsingHeader("Screenshot", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushID("Screenshot");
        ImGui::Combo("Resolution", &captureResIdx,
            "3000 x 3000 (SoundCloud)\0" "3840 x 2160 (4K)\0" "Window size\0");
        if (ImGui::Button("Capture Screenshot (P)")) captureRequested = true;
        ImGui::TextDisabled("Saves a PNG to screenshots/ in the working directory.\nThe UI is never included in the capture.");
        if (!lastScreenshotPath.empty()) ImGui::TextWrapped("Last: %s", lastScreenshotPath.c_str());
        ImGui::PopID();
    }
    ImGui::End();

    // While exporting a reel, drive the sim from the pre-analyzed track and
    // skip the live-audio + wall-clock stepping entirely (fully deterministic).
    if (reelExporting) {
        ReelExportStep();
        return;
    }

    if (lastBoxCenter.x != fluidGPU->param_boxCenter.x ||
        lastBoxCenter.y != fluidGPU->param_boxCenter.y ||
        lastBoxCenter.z != fluidGPU->param_boxCenter.z ||
        lastBoxHalf.x != fluidGPU->param_boxHalf.x ||
        lastBoxHalf.y != fluidGPU->param_boxHalf.y ||
        lastBoxHalf.z != fluidGPU->param_boxHalf.z ||
        lastBoxEuler.x != fluidGPU->param_boxEulerDeg.x ||
        lastBoxEuler.y != fluidGPU->param_boxEulerDeg.y ||
        lastBoxEuler.z != fluidGPU->param_boxEulerDeg.z ||
        lastShapeType != fluidGPU->param_shapeType) {
        UpdateContainerWireframe();
        lastBoxCenter = fluidGPU->param_boxCenter;
        lastBoxHalf = fluidGPU->param_boxHalf;
        lastBoxEuler = fluidGPU->param_boxEulerDeg;
        lastShapeType = fluidGPU->param_shapeType;
    }

    if (pendingReset) {
        fluidGPU->ResetSimulation();
        fluidGPU->param_pause = false;
        mesh->BindInstanceBuffer(fluidGPU->GetFluidVBO(), static_cast<GLsizei>(sizeof(float) * 4));
        pendingReset = false;
        dtAccumulator = 0.0f;
    }

    if (continuousWave) {
        wavePhase += wavePhaseSpeed * deltaTime;
        Vec3 dir = (waveDirIdx == 0) ? Vec3(1, 0, 0) : (waveDirIdx == 1) ? Vec3(0, 1, 0) : Vec3(0, 0, 1);
        fluidGPU->ApplyWaveImpulse(waveAmplitude, waveWavelength, wavePhase, dir, yBandMin, yBandMax);
    }

    // --- Audio Reactive: each band hits the fluid somewhere different ---
    if (audioReactiveEnabled && audioReactive) {
        if (!audioReactive->IsRunning()) audioReactive->Start();
        DriveAudioReaction(audioReactive->GetBass(), audioReactive->GetMid(),
                           audioReactive->GetTreble(), deltaTime);
    } else {
        if (audioReactive && audioReactive->IsRunning()) audioReactive->Stop();
        renderRadiusScaleLive = renderRadiusScale;
        brightMulLive         = brightMul;
        foamAmountLive        = foamAmount;
    }

    const float fixedDt = std::max(1e-6f, fluidGPU->param_timeStep);
    dtAccumulator += deltaTime;
    const int cap = (deltaTime > 0.033f) ? std::min(8, maxSubstepsPerFrame) : maxSubstepsPerFrame;

    int didSteps = 0;
    while (dtAccumulator >= fixedDt && didSteps < cap) {
        fluidGPU->DispatchCompute(fixedDt);
        dtAccumulator -= fixedDt;
        ++didSteps;
    }
    if (didSteps == cap && dtAccumulator > fixedDt) {
        dtAccumulator = fmodf(dtAccumulator, fixedDt);
    }

    // Service screenshot requests after the sim step; the capture renders
    // entirely offscreen so the on-screen frame that follows is unaffected.
    if (captureRequested) {
        captureRequested = false;
        DoCapture();
    }
}

void Scene0p::Render() const {
    // During a reel export the SSFR buffers are sized for the reel, not the
    // window, so skip the on-screen scene draw (the progress panel still shows).
    if (reelExporting) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, windowW, windowH);
        glClearColor(0.02f, 0.02f, 0.03f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        return;
    }
    RenderSceneTo(0, windowW, windowH, projectionMatrix);
}

// Renders the full scene (whichever render path is active) into targetFBO at
// outW x outH with the given projection. targetFBO 0 = the window; the
// screenshot capture passes its own FBO, size, and aspect-corrected projection.
void Scene0p::RenderSceneTo(GLuint targetFBO, int outW, int outH, const Matrix4& proj) const {
    if (outW <= 0 || outH <= 0) return;

    // Water surface rendering: all 5 passes handled inside RenderSSFR()
    if (useWaterRendering && ssfrW > 0 && ssfrDepthShader && fluidGPU) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        RenderSSFR(targetFBO, proj);
        return;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, targetFBO);
    glViewport(0, 0, outW, outH);
    glClearColor(bgColor[0], bgColor[1], bgColor[2], 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glPolygonMode(GL_FRONT_AND_BACK, drawInWireMode ? GL_LINE : GL_FILL);

    if (showContainerOutline && lineShader && boxVAO) {
        glUseProgram(lineShader->GetProgram());
        glUniformMatrix4fv(lineShader->GetUniformID("projectionMatrix"), 1, GL_FALSE, proj);
        glUniformMatrix4fv(lineShader->GetUniformID("viewMatrix"), 1, GL_FALSE, viewMatrix);
        if (GLint col = lineShader->GetUniformID("uColor"); col != -1)
            glUniform3fv(col, 1, containerOutlineColor);
        glBindVertexArray(boxVAO);
        glLineWidth(1.5f);
        glDrawArrays(GL_LINES, 0, containerWireVerts);
        glBindVertexArray(0);
        glUseProgram(0);
    }

    if (!fluidGPU) return;

    if (useImpostors && impostorShader) {
        DrawFluidImpostors(proj, outH);
        return;
    }

    if (!renderFromSSBO) {
        fluidGPU->UpdateFluidVBOFromGPU();
    }

    glUseProgram(shader->GetProgram());
    glUniformMatrix4fv(shader->GetUniformID("projectionMatrix"), 1, GL_FALSE, proj);
    glUniformMatrix4fv(shader->GetUniformID("viewMatrix"), 1, GL_FALSE, viewMatrix);

    if (GLint useLoc = shader->GetUniformID("useSSBO"); useLoc != -1)
        glUniform1i(useLoc, renderFromSSBO ? 1 : 0);
    SetColorUniforms(shader);

    float particleRadius = std::max(0.02f, 0.5f * fluidGPU->param_h) * renderRadiusScaleLive;
    Matrix4 scaleMat = MMath::scale(particleRadius, particleRadius, particleRadius);
    glUniformMatrix4fv(shader->GetUniformID("modelMatrix"), 1, GL_FALSE, scaleMat);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, fluidGPU->ssbo);
    mesh->RenderInstanced(GL_TRIANGLES, fluidGPU->GetNumFluids());
    glUseProgram(0);
}

void Scene0p::SetupImpostorVAO() {
    glGenVertexArrays(1, &impostorVAO);
}

int Scene0p::CurrentViewportHeight() const {
    GLint vp[4] = { 0,0,0,0 };
    glGetIntegerv(GL_VIEWPORT, vp);
    return vp[3] > 0 ? vp[3] : 1080;
}

// One-click looks: container + physics + palette + audio reaction in one go,
// tuned for audio-reactive videos. The floaty ones use weak gravity + high
// viscosity so the fluid drifts and reacts instead of falling.
void Scene0p::ApplyArtPreset(int which) {
    if (!fluidGPU) return;

    // Common canvas: black backdrop, neutral grade, no river
    fluidGPU->riverMode = false;
    showSkyBackground = false;
    bgColor[0] = bgColor[1] = bgColor[2] = 0.0f;
    hueShiftDeg = 0.0f; satMul = 1.0f; brightMul = 1.0f; contrastMul = 1.0f;
    invertColor = false;
    fluidGPU->param_boxCenter = Vec3(0, 0, 0);
    fluidGPU->param_boxEulerDeg = Vec3(0, 0, 0);
    fluidGPU->param_h = 0.28f;
    fluidGPU->param_restDensity = 1000.0f;
    fluidGPU->param_timeStep = 1.0e-3f;
    fluidGPU->param_pause = false;

    switch (which) {
    case 0: // Zero-G Nebula: drifting cloud in a sphere, galaxy colors
        fluidGPU->param_shapeType = 1;
        fluidGPU->param_boxHalf = Vec3(7, 7, 7);
        fluidGPU->param_gravityY = -15.0f;
        fluidGPU->param_viscosity = 6.0f;
        fluidGPU->param_gasConstant = 1500.0f;
        fluidGPU->param_surfaceTension = 0.05f;
        useWaterRendering = false; useImpostors = true; litParticles = true;
        renderRadiusScale = 1.2f;
        paletteId = 9; vizMode = 1; vizRangeMin = 0.0f; vizRangeMax = 8.0f;
        paletteFlow = 0.05f; patternScale = 1.0f;
        audioMasterGain = 1.5f;
        audioBassForce = 12.0f;  audioBassThreshold = 0.06f;
        audioMidForce  = 5.0f;   audioMidThreshold  = 0.06f;
        audioTrebleForce = 2.0f; audioTrebleThreshold = 0.05f;
        audioSizeKick = 0.5f; audioShimmerKick = 0.6f; audioFoamKick = 0.3f;
        break;
    case 1: // Dream Float: slow syrupy drift, aurora colors by depth
        fluidGPU->param_shapeType = 0;
        fluidGPU->param_boxHalf = Vec3(7, 7, 7);
        fluidGPU->param_gravityY = -35.0f;
        fluidGPU->param_viscosity = 8.0f;
        fluidGPU->param_gasConstant = 1200.0f;
        fluidGPU->param_surfaceTension = 0.08f;
        useWaterRendering = false; useImpostors = true; litParticles = true;
        renderRadiusScale = 1.5f;
        paletteId = 14; vizMode = 4; vizRangeMin = 8.0f; vizRangeMax = 40.0f;
        paletteFlow = 0.08f; patternScale = 1.0f;
        audioMasterGain = 1.2f;
        audioBassForce = 8.0f;   audioBassThreshold = 0.08f;
        audioMidForce  = 4.0f;   audioMidThreshold  = 0.08f;
        audioTrebleForce = 1.5f; audioTrebleThreshold = 0.06f;
        audioSizeKick = 0.35f; audioShimmerKick = 0.5f; audioFoamKick = 0.2f;
        break;
    case 2: // Acid Trip: floaty sphere, kaleidoscope rings, hard audio hits
        fluidGPU->param_shapeType = 1;
        fluidGPU->param_boxHalf = Vec3(7, 7, 7);
        fluidGPU->param_gravityY = -60.0f;
        fluidGPU->param_viscosity = 2.0f;
        fluidGPU->param_gasConstant = 3500.0f;
        fluidGPU->param_surfaceTension = 0.10f;
        useWaterRendering = false; useImpostors = true; litParticles = true;
        renderRadiusScale = 1.1f;
        paletteId = 13; iridFreq = 4.0f; iridShift = 0.0f;
        vizMode = 6; vizRangeMin = 0.0f; vizRangeMax = 7.0f;
        paletteFlow = 0.20f; patternScale = 1.0f;
        audioMasterGain = 1.8f;
        audioBassForce = 15.0f;  audioBassThreshold = 0.05f;
        audioMidForce  = 7.0f;   audioMidThreshold  = 0.06f;
        audioTrebleForce = 3.0f; audioTrebleThreshold = 0.04f;
        audioSizeKick = 0.6f; audioShimmerKick = 1.0f; audioFoamKick = 0.3f;
        break;
    case 3: // Club Water: real water surface, black backdrop, heavy bass splashes
        fluidGPU->param_shapeType = 0;
        fluidGPU->param_boxHalf = Vec3(7, 7, 7);
        fluidGPU->param_gravityY = -980.0f;
        fluidGPU->param_viscosity = 3.5f;
        fluidGPU->param_gasConstant = 2500.0f;
        fluidGPU->param_surfaceTension = 0.10f;
        useWaterRendering = true; useImpostors = false;
        fluidGPU->param_foamGen = 1.3f;
        foamAmount = 2.2f;
        audioMasterGain = 1.5f;
        audioBassForce = 18.0f;  audioBassThreshold = 0.08f;
        audioMidForce  = 8.0f;   audioMidThreshold  = 0.08f;
        audioTrebleForce = 4.0f; audioTrebleThreshold = 0.06f;
        audioSizeKick = 0.2f; audioShimmerKick = 0.4f; audioFoamKick = 1.2f;
        break;
    default: // 4, Molten Disco: gold metal sloshing in a cylinder
        fluidGPU->param_shapeType = 2;
        fluidGPU->param_boxHalf = Vec3(6, 5, 6);
        fluidGPU->param_gravityY = -200.0f;
        fluidGPU->param_viscosity = 4.0f;
        fluidGPU->param_gasConstant = 2000.0f;
        fluidGPU->param_surfaceTension = 0.10f;
        useWaterRendering = false; useImpostors = true; litParticles = true;
        renderRadiusScale = 1.25f;
        paletteId = 12; vizMode = 1; vizRangeMin = 0.0f; vizRangeMax = 12.0f;
        paletteFlow = 0.10f; patternScale = 1.0f;
        audioMasterGain = 1.4f;
        audioBassForce = 14.0f;  audioBassThreshold = 0.07f;
        audioMidForce  = 6.0f;   audioMidThreshold  = 0.07f;
        audioTrebleForce = 2.5f; audioTrebleThreshold = 0.05f;
        audioSizeKick = 0.45f; audioShimmerKick = 0.7f; audioFoamKick = 0.3f;
        break;
    }

    // Turn the audio reaction on with the new settings and respawn the fluid
    audioReactiveEnabled = true;
    if (audioReactive) audioReactive->gain.store(audioMasterGain);
    pendingReset = true;
}

// Uploads the shared-palette-block uniforms (see particleImpostor.frag / defaultFrag.glsl)
void Scene0p::SetColorUniforms(Shader* s) const {
    if (GLint loc = s->GetUniformID("colorDrive"); loc != -1) glUniform1i(loc, vizMode);
    if (GLint loc = s->GetUniformID("paletteId");  loc != -1) glUniform1i(loc, paletteId);
    if (GLint loc = s->GetUniformID("vizRange");   loc != -1) glUniform2f(loc, vizRangeMin, vizRangeMax);
    if (GLint loc = s->GetUniformID("heightMinMax"); loc != -1) {
        glUniform2f(loc,
            fluidGPU->param_boxCenter.y - fluidGPU->param_boxHalf.y,
            fluidGPU->param_boxCenter.y + fluidGPU->param_boxHalf.y);
    }
    if (GLint loc = s->GetUniformID("boxCenter"); loc != -1)
        glUniform3f(loc, fluidGPU->param_boxCenter.x, fluidGPU->param_boxCenter.y, fluidGPU->param_boxCenter.z);
    if (GLint loc = s->GetUniformID("duoColorA");   loc != -1) glUniform3fv(loc, 1, duoColorA);
    if (GLint loc = s->GetUniformID("duoColorB");   loc != -1) glUniform3fv(loc, 1, duoColorB);
    if (GLint loc = s->GetUniformID("iridFreq");    loc != -1) glUniform1f(loc, iridFreq);
    if (GLint loc = s->GetUniformID("iridShift");   loc != -1) glUniform1f(loc, iridShift);
    if (GLint loc = s->GetUniformID("animTime");    loc != -1) glUniform1f(loc, ballAnimTime);
    if (GLint loc = s->GetUniformID("paletteFlow");  loc != -1) glUniform1f(loc, paletteFlow);
    if (GLint loc = s->GetUniformID("patternScale"); loc != -1) glUniform1f(loc, patternScale);
    if (GLint loc = s->GetUniformID("litSphere");   loc != -1) glUniform1i(loc, litParticles ? 1 : 0);
    if (GLint loc = s->GetUniformID("sunDirWorld"); loc != -1) glUniform3fv(loc, 1, sunDirWorld);
    if (GLint loc = s->GetUniformID("sunColor");    loc != -1) glUniform3fv(loc, 1, sunColor);
    SetGradeUniforms(s);
}

// Uploads only the color-adjustment uniforms (shared with fluidComposite.frag)
void Scene0p::SetGradeUniforms(Shader* s) const {
    if (GLint loc = s->GetUniformID("hueShift");    loc != -1) glUniform1f(loc, hueShiftDeg);
    if (GLint loc = s->GetUniformID("satMul");      loc != -1) glUniform1f(loc, satMul);
    if (GLint loc = s->GetUniformID("brightMul");   loc != -1) glUniform1f(loc, brightMulLive);
    if (GLint loc = s->GetUniformID("contrastMul"); loc != -1) glUniform1f(loc, contrastMul);
    if (GLint loc = s->GetUniformID("invertColor"); loc != -1) glUniform1i(loc, invertColor ? 1 : 0);
}

void Scene0p::DrawFluidImpostors(const Matrix4& proj, int outH) const {
    glUseProgram(impostorShader->GetProgram());
    glUniformMatrix4fv(impostorShader->GetUniformID("projectionMatrix"), 1, GL_FALSE, proj);
    glUniformMatrix4fv(impostorShader->GetUniformID("viewMatrix"), 1, GL_FALSE, viewMatrix);

    SetColorUniforms(impostorShader);

    if (GLint r = impostorShader->GetUniformID("particleRadius"); r != -1)
        glUniform1f(r, std::max(0.02f, 0.5f * fluidGPU->param_h) * renderRadiusScaleLive);
    if (GLint r = impostorShader->GetUniformID("viewportH"); r != -1)
        glUniform1f(r, static_cast<float>(outH));

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, fluidGPU->ssbo);
    glBindVertexArray(impostorVAO);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(fluidGPU->GetNumFluids()));
    glBindVertexArray(0);
    glUseProgram(0);
}

// ---------------------------------------------------------------------------
// SSFR helpers
// ---------------------------------------------------------------------------

static GLuint MakeR32FFBO(GLuint& texOut, int w, int h) {
    glGenTextures(1, &texOut);
    glBindTexture(GL_TEXTURE_2D, texOut);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, h, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texOut, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return fbo;
}

void Scene0p::InitSSFRBuffers(int w, int h) {
    DestroySSFRBuffers();
    ssfrW = w; ssfrH = h;
    // Fluid passes (depth/smooth/thickness/foam) can run at half resolution;
    // the background and composite always stay full-res.
    const int fw = ssfrHalfRes ? std::max(1, w / 2) : w;
    const int fh = ssfrHalfRes ? std::max(1, h / 2) : h;
    ssfrFluidW = fw; ssfrFluidH = fh;

    // --- Pass 1: depth FBO (R32F color + depth renderbuffer) ---
    glGenTextures(1, &ssfrDepthTex);
    glBindTexture(GL_TEXTURE_2D, ssfrDepthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, fw, fh, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenRenderbuffers(1, &ssfrDepthRBO);
    glBindRenderbuffer(GL_RENDERBUFFER, ssfrDepthRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, fw, fh);

    glGenFramebuffers(1, &ssfrDepthFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, ssfrDepthFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ssfrDepthTex, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, ssfrDepthRBO);

    // --- Pass 2: ping-pong smooth FBOs (R32F) ---
    for (int i = 0; i < 2; ++i)
        ssfrSmoothFBO[i] = MakeR32FFBO(ssfrSmoothTex[i], fw, fh);

    // --- Pass 3: thickness FBO (R32F, additive) ---
    ssfrThickFBO = MakeR32FFBO(ssfrThickTex, fw, fh);
    // Second attachment: foam accumulation (same additive blend, MRT)
    glGenTextures(1, &ssfrFoamTex);
    glBindTexture(GL_TEXTURE_2D, ssfrFoamTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, fw, fh, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindFramebuffer(GL_FRAMEBUFFER, ssfrThickFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, ssfrFoamTex, 0);
    const GLenum thickBufs[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glDrawBuffers(2, thickBufs);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // --- Pass 4: background FBO (RGBA8 + depth) ---
    glGenTextures(1, &ssfrBgTex);
    glBindTexture(GL_TEXTURE_2D, ssfrBgTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenRenderbuffers(1, &ssfrBgRBO);
    glBindRenderbuffer(GL_RENDERBUFFER, ssfrBgRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);

    glGenFramebuffers(1, &ssfrBgFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, ssfrBgFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ssfrBgTex, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, ssfrBgRBO);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
}

void Scene0p::DestroySSFRBuffers() {
    if (ssfrDepthFBO)    { glDeleteFramebuffers(1, &ssfrDepthFBO);    ssfrDepthFBO = 0; }
    if (ssfrDepthTex)    { glDeleteTextures(1, &ssfrDepthTex);        ssfrDepthTex = 0; }
    if (ssfrDepthRBO)    { glDeleteRenderbuffers(1, &ssfrDepthRBO);   ssfrDepthRBO = 0; }
    for (int i = 0; i < 2; ++i) {
        if (ssfrSmoothFBO[i]) { glDeleteFramebuffers(1, &ssfrSmoothFBO[i]); ssfrSmoothFBO[i] = 0; }
        if (ssfrSmoothTex[i]) { glDeleteTextures(1, &ssfrSmoothTex[i]);     ssfrSmoothTex[i] = 0; }
    }
    if (ssfrThickFBO)    { glDeleteFramebuffers(1, &ssfrThickFBO);    ssfrThickFBO = 0; }
    if (ssfrThickTex)    { glDeleteTextures(1, &ssfrThickTex);        ssfrThickTex = 0; }
    if (ssfrFoamTex)     { glDeleteTextures(1, &ssfrFoamTex);         ssfrFoamTex = 0; }
    if (ssfrBgFBO)       { glDeleteFramebuffers(1, &ssfrBgFBO);       ssfrBgFBO = 0; }
    if (ssfrBgTex)       { glDeleteTextures(1, &ssfrBgTex);           ssfrBgTex = 0; }
    if (ssfrBgRBO)       { glDeleteRenderbuffers(1, &ssfrBgRBO);      ssfrBgRBO = 0; }
    ssfrW = ssfrH = 0;
}

void Scene0p::RenderSSFR(GLuint targetFBO, const Matrix4& proj) const {
    if (!fluidGPU || ssfrW <= 0 || ssfrH <= 0) return;

    const float radius = std::max(0.02f, 0.5f * fluidGPU->param_h) * renderRadiusScaleLive;
    const auto  nFluid = static_cast<GLsizei>(fluidGPU->GetNumFluids());

    glEnable(GL_PROGRAM_POINT_SIZE);

    // -----------------------------------------------------------------------
    // Pass 1 — Sphere depth: render spherical impostors, write view-space Z
    // -----------------------------------------------------------------------
    glBindFramebuffer(GL_FRAMEBUFFER, ssfrDepthFBO);
    glViewport(0, 0, ssfrFluidW, ssfrFluidH);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDisable(GL_BLEND);

    glUseProgram(ssfrDepthShader->GetProgram());
    glUniformMatrix4fv(ssfrDepthShader->GetUniformID("projectionMatrix"), 1, GL_FALSE, proj);
    glUniformMatrix4fv(ssfrDepthShader->GetUniformID("viewMatrix"),       1, GL_FALSE, viewMatrix);
    if (GLint r = ssfrDepthShader->GetUniformID("particleRadius"); r != -1)
        glUniform1f(r, radius);
    if (GLint r = ssfrDepthShader->GetUniformID("viewportH"); r != -1)
        glUniform1f(r, static_cast<float>(ssfrFluidH));

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, fluidGPU->ssbo);
    glBindVertexArray(impostorVAO);
    glDrawArrays(GL_POINTS, 0, nFluid);

    // -----------------------------------------------------------------------
    // Pass 2 — Bilateral depth smoothing (ping-pong, N iterations)
    // -----------------------------------------------------------------------
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    // Initial source is the raw depth texture
    GLuint srcTex = ssfrDepthTex;
    int    pingPong = 0;

    glUseProgram(ssfrSmoothShader->GetProgram());
    glUniform2f(ssfrSmoothShader->GetUniformID("screenSize"),    static_cast<float>(ssfrFluidW), static_cast<float>(ssfrFluidH));
    glUniform1f(ssfrSmoothShader->GetUniformID("particleRadius"),   radius);
    glUniform1f(ssfrSmoothShader->GetUniformID("worldFilterScale"), worldFilterScale);
    glUniform1f(ssfrSmoothShader->GetUniformID("surfaceMerge"),     surfaceMerge);
    // World-size-to-pixels projection factor; using the target height keeps
    // the smoothing consistent between the window and high-res captures.
    const float* pm = proj;
    glUniform1f(ssfrSmoothShader->GetUniformID("projScaleY"), pm[5] * static_cast<float>(ssfrFluidH) * 0.5f);

    glBindVertexArray(ssfrQuadVAO);

    for (int iter = 0; iter < smoothIterations; ++iter) {
        // Horizontal pass
        glBindFramebuffer(GL_FRAMEBUFFER, ssfrSmoothFBO[pingPong]);
        glClear(GL_COLOR_BUFFER_BIT);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, srcTex);
        glUniform1i(ssfrSmoothShader->GetUniformID("depthTex"), 0);
        glUniform2f(ssfrSmoothShader->GetUniformID("filterDir"), 1.0f, 0.0f);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        srcTex   = ssfrSmoothTex[pingPong];
        pingPong = 1 - pingPong;

        // Vertical pass
        glBindFramebuffer(GL_FRAMEBUFFER, ssfrSmoothFBO[pingPong]);
        glClear(GL_COLOR_BUFFER_BIT);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, srcTex);
        glUniform1i(ssfrSmoothShader->GetUniformID("depthTex"), 0);
        glUniform2f(ssfrSmoothShader->GetUniformID("filterDir"), 0.0f, 1.0f);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        srcTex   = ssfrSmoothTex[pingPong];
        pingPong = 1 - pingPong;
    }

    // After all iterations, srcTex holds the final smoothed depth
    GLuint finalSmooth = srcTex;

    // -----------------------------------------------------------------------
    // Pass 3 — Thickness accumulation (additive blending, no depth test)
    // -----------------------------------------------------------------------
    glBindFramebuffer(GL_FRAMEBUFFER, ssfrThickFBO);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE); // additive

    glUseProgram(ssfrThickShader->GetProgram());
    glUniformMatrix4fv(ssfrThickShader->GetUniformID("projectionMatrix"), 1, GL_FALSE, proj);
    glUniformMatrix4fv(ssfrThickShader->GetUniformID("viewMatrix"),       1, GL_FALSE, viewMatrix);
    if (GLint r = ssfrThickShader->GetUniformID("particleRadius"); r != -1)
        glUniform1f(r, radius);
    if (GLint r = ssfrThickShader->GetUniformID("viewportH"); r != -1)
        glUniform1f(r, static_cast<float>(ssfrFluidH));
    if (GLint r = ssfrThickShader->GetUniformID("thicknessStrength"); r != -1)
        glUniform1f(r, thicknessStrength);
    if (GLint r = ssfrThickShader->GetUniformID("thicknessFalloff"); r != -1)
        glUniform1f(r, thicknessFalloff);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, fluidGPU->ssbo);
    glBindVertexArray(impostorVAO);
    glDrawArrays(GL_POINTS, 0, nFluid);

    // -----------------------------------------------------------------------
    // Pass 4 — Background scene (terrain + box wireframe on sky)
    // -----------------------------------------------------------------------
    glBindFramebuffer(GL_FRAMEBUFFER, ssfrBgFBO);
    glViewport(0, 0, ssfrW, ssfrH);
    // Backdrop: flat color by default (fluid reads much better on black);
    // the sky gradient stays available for reflections either way.
    if (showSkyBackground)
        glClearColor(skyColor[0], skyColor[1], skyColor[2], 1.0f);
    else
        glClearColor(bgColor[0], bgColor[1], bgColor[2], 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_BLEND);

    // Procedural sky gradient behind everything (no depth writes)
    if (showSkyBackground && skyShader) {
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glUseProgram(skyShader->GetProgram());
        glUniformMatrix4fv(skyShader->GetUniformID("projectionMatrix"), 1, GL_FALSE, proj);
        glUniformMatrix4fv(skyShader->GetUniformID("viewMatrix"),       1, GL_FALSE, viewMatrix);
        glUniform3fv(skyShader->GetUniformID("skyHorizonColor"), 1, skyColor);
        glUniform3fv(skyShader->GetUniformID("skyZenithColor"),  1, skyZenith);
        glUniform3fv(skyShader->GetUniformID("sunDirWorld"),     1, sunDirWorld);
        glUniform3fv(skyShader->GetUniformID("sunColor"),        1, sunColor);
        glBindVertexArray(ssfrQuadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindVertexArray(0);
        glDepthMask(GL_TRUE);
    }

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    // Terrain mesh (when river mode is active and terrain has been built)
    if (terrainShader && terrainVAO && terrainIndexCount > 0 &&
        fluidGPU && fluidGPU->riverMode) {
        glUseProgram(terrainShader->GetProgram());
        glUniformMatrix4fv(terrainShader->GetUniformID("projectionMatrix"), 1, GL_FALSE, proj);
        glUniformMatrix4fv(terrainShader->GetUniformID("viewMatrix"),       1, GL_FALSE, viewMatrix);
        glUniform3fv(terrainShader->GetUniformID("sunDirWorld"), 1, sunDirWorld);
        glUniform3fv(terrainShader->GetUniformID("sunColor"),    1, sunColor);
        glBindVertexArray(terrainVAO);
        glDrawElements(GL_TRIANGLES, terrainIndexCount, GL_UNSIGNED_INT, nullptr);
        glBindVertexArray(0);
    }

    // River bank lines (channel edges + centerline on terrain surface)
    if (showRiverLines && fluidGPU && fluidGPU->riverMode)
        DrawRiverBankLines(proj);

    // Box wireframe (skip in river mode to reduce visual clutter)
    if (showContainerOutline && lineShader && boxVAO && fluidGPU && !fluidGPU->riverMode) {
        glUseProgram(lineShader->GetProgram());
        glUniformMatrix4fv(lineShader->GetUniformID("projectionMatrix"), 1, GL_FALSE, proj);
        glUniformMatrix4fv(lineShader->GetUniformID("viewMatrix"),       1, GL_FALSE, viewMatrix);
        if (GLint c = lineShader->GetUniformID("uColor"); c != -1)
            glUniform3fv(c, 1, containerOutlineColor);
        glBindVertexArray(boxVAO);
        glLineWidth(1.5f);
        glDrawArrays(GL_LINES, 0, containerWireVerts);
        glBindVertexArray(0);
    }

    // -----------------------------------------------------------------------
    // Pass 5 — Composite: normals + Fresnel + specular + refraction + tint
    // -----------------------------------------------------------------------
    glBindFramebuffer(GL_FRAMEBUFFER, targetFBO);
    glViewport(0, 0, ssfrW, ssfrH);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    glUseProgram(ssfrCompositeShader->GetProgram());

    // Bind textures
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, finalSmooth);
    glUniform1i(ssfrCompositeShader->GetUniformID("smoothDepthTex"), 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, ssfrThickTex);
    glUniform1i(ssfrCompositeShader->GetUniformID("thicknessTex"), 1);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, ssfrBgTex);
    glUniform1i(ssfrCompositeShader->GetUniformID("backgroundTex"), 2);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, ssfrFoamTex);
    glUniform1i(ssfrCompositeShader->GetUniformID("foamTex"), 3);

    glUniformMatrix4fv(ssfrCompositeShader->GetUniformID("projectionMatrix"), 1, GL_FALSE, proj);
    glUniformMatrix4fv(ssfrCompositeShader->GetUniformID("viewMatrix"),       1, GL_FALSE, viewMatrix);
    glUniform2f(ssfrCompositeShader->GetUniformID("screenSize"),
                static_cast<float>(ssfrFluidW), static_cast<float>(ssfrFluidH));

    // Lighting & water appearance
    glUniform3fv(ssfrCompositeShader->GetUniformID("sunDirWorld"),     1, sunDirWorld);
    glUniform3fv(ssfrCompositeShader->GetUniformID("sunColor"),        1, sunColor);
    glUniform1f(ssfrCompositeShader->GetUniformID("specularPower"),    specularPower);
    glUniform1f(ssfrCompositeShader->GetUniformID("specularStrength"), specularStrength);
    glUniform3fv(ssfrCompositeShader->GetUniformID("extinction"),      1, waterExtinction);
    glUniform1f(ssfrCompositeShader->GetUniformID("thicknessScale"),   thicknessScale);
    glUniform1f(ssfrCompositeShader->GetUniformID("refractionStrength"), refractionStrength);
    glUniform1f(ssfrCompositeShader->GetUniformID("fresnelBias"),      fresnelBias);
    glUniform3fv(ssfrCompositeShader->GetUniformID("deepWaterColor"),  1, deepWaterColor);
    glUniform3fv(ssfrCompositeShader->GetUniformID("envReflectColor"), 1, envReflectColor);
    glUniform3fv(ssfrCompositeShader->GetUniformID("skyHorizonColor"), 1, skyColor);
    glUniform3fv(ssfrCompositeShader->GetUniformID("skyZenithColor"),  1, skyZenith);
    glUniform1f(ssfrCompositeShader->GetUniformID("foamAmount"), foamAmountLive);
    glUniform1f(ssfrCompositeShader->GetUniformID("exposure"),   exposure);
    SetGradeUniforms(ssfrCompositeShader);

    glBindVertexArray(ssfrQuadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
    glUseProgram(0);

    // Restore state
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glActiveTexture(GL_TEXTURE0);
}

void Scene0p::BuildTerrainMesh() {
    if (!fluidGPU || fluidGPU->terrainHeights.empty()) return;

    const int W = fluidGPU->terrainW;
    const int H = fluidGPU->terrainH;
    const float xMin  = fluidGPU->terrainWorldMinX;
    const float zMin  = fluidGPU->terrainWorldMinZ;
    const float xSize = fluidGPU->terrainWorldSizeX;
    const float zSize = fluidGPU->terrainWorldSizeZ;
    const auto& ht    = fluidGPU->terrainHeights;
    const float yBase = fluidGPU->param_boxCenter.y - fluidGPU->param_boxHalf.y;

    float dx = xSize / float(W - 1);
    float dz = zSize / float(H - 1);

    // Extend the visual mesh N_ext columns on each X side, fading to below the box floor.
    // Physics terrain stays box-exact (full 64×64 resolution for normal quality).
    const int N_ext = 20;
    const int RW    = W + 2 * N_ext;
    const float rxMin = xMin - N_ext * dx;

    // Height at a render column index (rix) and terrain row (iz)
    auto renderY = [&](int rix, int iz) -> float {
        if (rix >= N_ext && rix < N_ext + W) {
            return ht[(rix - N_ext) + iz * W];     // physics terrain
        }
        // Extended left or right: blend from box-edge height to below floor
        float hEdge = (rix < N_ext) ? ht[0 + iz * W] : ht[(W - 1) + iz * W];
        float steps  = (rix < N_ext) ? float(N_ext - rix) : float(rix - (N_ext + W - 1));
        float t      = std::min(1.0f, steps / float(N_ext));
        float fade   = t * t * (3.0f - 2.0f * t); // smoothstep
        return hEdge * (1.0f - fade) + (yBase - 1.0f) * fade;
    };

    struct TV { float x, y, z, nx, ny, nz; };
    std::vector<TV>       verts(RW * H);
    std::vector<uint32_t> idx;
    idx.reserve((RW - 1) * (H - 1) * 6);

    for (int iz = 0; iz < H; ++iz) {
        for (int rix = 0; rix < RW; ++rix) {
            float wx = rxMin + rix * dx;
            float wz = zMin  + iz  * dz;
            float wy = renderY(rix, iz);

            // Finite-difference normal using renderY neighbours
            float hR = renderY(std::min(rix + 1, RW - 1), iz);
            float hL = renderY(std::max(rix - 1, 0),       iz);
            float hF = (iz < H - 1) ? renderY(rix, iz + 1) : wy;
            float hB = (iz > 0)     ? renderY(rix, iz - 1) : wy;
            float nx = (hL - hR) / (2.0f * dx);
            float ny = 1.0f;
            float nz = (hB - hF) / (2.0f * dz);
            float len = std::sqrt(nx*nx + ny*ny + nz*nz);
            if (len > 1e-6f) { nx/=len; ny/=len; nz/=len; }

            verts[iz * RW + rix] = {wx, wy, wz, nx, ny, nz};
        }
    }

    for (int iz = 0; iz < H - 1; ++iz) {
        for (int rix = 0; rix < RW - 1; ++rix) {
            uint32_t base = uint32_t(iz * RW + rix);
            idx.push_back(base);
            idx.push_back(base + 1);
            idx.push_back(base + RW);
            idx.push_back(base + 1);
            idx.push_back(base + RW + 1);
            idx.push_back(base + RW);
        }
    }
    terrainIndexCount = int(idx.size());

    glBindVertexArray(terrainVAO);

    glBindBuffer(GL_ARRAY_BUFFER, terrainVBO);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(TV), verts.data(), GL_STATIC_DRAW);

    // position (loc=0), normal (loc=1)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(TV), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(TV), (void*)(3*sizeof(float)));

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, terrainEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(uint32_t), idx.data(), GL_STATIC_DRAW);

    glBindVertexArray(0);
}

void Scene0p::BuildRiverBankLines() {
    if (!fluidGPU || fluidGPU->terrainHeights.empty()) return;

    const int N     = 300;
    const float zMin  = fluidGPU->terrainWorldMinZ;
    const float zSize = fluidGPU->terrainWorldSizeZ;
    const float xMin  = fluidGPU->terrainWorldMinX;
    const float xSize = fluidGPU->terrainWorldSizeX;
    const int   TW    = fluidGPU->terrainW;
    const int   TH    = fluidGPU->terrainH;

    // Bilinear sample of the physics terrain (box footprint, full resolution)
    auto sampleH = [&](float wx, float wz) -> float {
        float u = (wx - xMin) / xSize * float(TW - 1);
        float v = (wz - zMin) / zSize * float(TH - 1);
        u = std::max(0.0f, std::min(float(TW - 2), u));
        v = std::max(0.0f, std::min(float(TH - 2), v));
        int ix = int(u), iz = int(v);
        float fx = u - ix, fz = v - iz;
        float h00 = fluidGPU->terrainHeights[ ix      +  iz      * TW];
        float h10 = fluidGPU->terrainHeights[(ix + 1) +  iz      * TW];
        float h01 = fluidGPU->terrainHeights[ ix      + (iz + 1) * TW];
        float h11 = fluidGPU->terrainHeights[(ix + 1) + (iz + 1) * TW];
        return h00*(1-fx)*(1-fz) + h10*fx*(1-fz) + h01*(1-fx)*fz + h11*fx*fz;
    };

    // 3 strips: 0=left bank, 1=right bank, 2=centerline
    std::vector<float> verts;
    verts.reserve(N * 3 * 3);
    for (int strip = 0; strip < 3; ++strip) {
        for (int i = 0; i < N; ++i) {
            float wz = zMin + (float(i) / float(N - 1)) * zSize;
            float cx = fluidGPU->param_boxCenter.x
                     + fluidGPU->riverAmp * std::sinf(fluidGPU->riverFreq * wz + fluidGPU->riverPhase);

            float wx, wy;
            if (strip == 0 || strip == 1) {
                // Sample terrain 10% OUTSIDE the channel edge so we're on the plateau surface.
                // plateau = channelEdge + 3.0, so this gives the terrain height on the bank top.
                const float sample_mult = 1.10f;
                wx = (strip == 0) ? cx - fluidGPU->riverChannelWidth
                                  : cx + fluidGPU->riverChannelWidth;
                float sampleX = (strip == 0) ? cx - fluidGPU->riverChannelWidth * sample_mult
                                             : cx + fluidGPU->riverChannelWidth * sample_mult;
                wy = sampleH(sampleX, wz) + 0.18f; // sit on top of the bank surface
            } else {
                // Centerline: sample at channel center, height = riverFloor ≈ terrain at cx
                wx = cx;
                wy = sampleH(cx, wz) + 0.15f;
            }

            verts.push_back(wx);
            verts.push_back(wy);
            verts.push_back(wz);
        }
    }

    riverBankN = N;
    if (!riverBankVAO) { glGenVertexArrays(1, &riverBankVAO); glGenBuffers(1, &riverBankVBO); }
    glBindVertexArray(riverBankVAO);
    glBindBuffer(GL_ARRAY_BUFFER, riverBankVBO);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glBindVertexArray(0);
}

void Scene0p::DrawRiverBankLines(const Matrix4& proj) const {
    if (!lineShader || !riverBankVAO || riverBankN == 0) return;

    glUseProgram(lineShader->GetProgram());
    glUniformMatrix4fv(lineShader->GetUniformID("projectionMatrix"), 1, GL_FALSE, proj);
    glUniformMatrix4fv(lineShader->GetUniformID("viewMatrix"),       1, GL_FALSE, viewMatrix);
    glBindVertexArray(riverBankVAO);
    glLineWidth(2.5f);

    GLint colorLoc = lineShader->GetUniformID("uColor");

    // Left bank — red
    glUniform3f(colorLoc, 1.0f, 0.15f, 0.15f);
    glDrawArrays(GL_LINE_STRIP, 0, riverBankN);

    // Right bank — red
    glDrawArrays(GL_LINE_STRIP, riverBankN, riverBankN);

    // Centerline — warm orange (flow direction guide)
    glUniform3f(colorLoc, 1.0f, 0.55f, 0.05f);
    glDrawArrays(GL_LINE_STRIP, 2 * riverBankN, riverBankN);

    glLineWidth(1.0f);
    glBindVertexArray(0);
}

// ---------------------------------------------------------------------------
// Screenshot capture
// ---------------------------------------------------------------------------

// One frame of audio reaction: three directional wave impulses (bass heave,
// rotating mid push, treble surface ripple) plus the three visual "Live"
// values. Shared by the live reactor and the offline reels render so the
// export matches what you tuned live.
void Scene0p::DriveAudioReaction(float bass, float mid, float treble, float dt) {
    if (!fluidGPU) return;

    const Vec3  half      = fluidGPU->EffectiveHalf();
    const float boxBottom = fluidGPU->param_boxCenter.y - half.y;
    const float boxSpanY  = 2.0f * half.y;

    audioBassPhase   += audioBassPhaseSpeed   * dt;
    audioMidPhase    += audioMidRotSpeed      * dt;   // doubles as rotation angle
    audioTreblePhase += audioTreblePhaseSpeed * dt;

    if (bass > audioBassThreshold) {
        fluidGPU->ApplyWaveImpulse(audioBassForce * bass, audioBassWavelength, audioBassPhase,
            Vec3(0, 1, 0), boxBottom, boxBottom + boxSpanY * 0.4f);
    }
    if (mid > audioMidThreshold) {
        Vec3 dir(std::cos(audioMidPhase), 0.0f, std::sin(audioMidPhase));
        fluidGPU->ApplyWaveImpulse(audioMidForce * mid, audioMidWavelength, audioMidPhase, dir,
            boxBottom + boxSpanY * 0.3f, boxBottom + boxSpanY * 0.7f);
    }
    if (treble > audioTrebleThreshold) {
        fluidGPU->ApplyWaveImpulse(audioTrebleForce * treble, audioTrebleWavelength, audioTreblePhase,
            Vec3(0, 1, 0), boxBottom + boxSpanY * 0.6f, boxBottom + boxSpanY);
    }

    renderRadiusScaleLive = renderRadiusScale * (1.0f + audioSizeKick    * bass);
    brightMulLive         = brightMul         * (1.0f + audioShimmerKick * treble);
    foamAmountLive        = foamAmount        * (1.0f + audioFoamKick    * mid);
}

// ---------------------------------------------------------------------------
// Reels Export — offline, frame-accurate, music-synced render
// ---------------------------------------------------------------------------

void Scene0p::StartReelExport() {
    if (reelExporting || !fluidGPU) return;

    const int fps = (reelFpsIdx == 1) ? 60 : 30;
    switch (reelResIdx) {
        case 1:  reelW = 1080; reelH = 1350; break;
        case 2:  reelW = 1920; reelH = 1080; break;
        default: reelW = 1080; reelH = 1920; break;
    }

    reelBands = AnalyzeTrack(reelAudioPath, fps, reelMaxSeconds);
    if (!reelBands.error.empty()) {
        reelStatus = "FAILED: " + reelBands.error;
        return;
    }

    // Deterministic start: fresh fluid + zeroed reaction phases.
    audioBassPhase = audioMidPhase = audioTreblePhase = 0.0f;
    fluidGPU->ResetSimulation();
    fluidGPU->param_pause = false;
    mesh->BindInstanceBuffer(fluidGPU->GetFluidVBO(), static_cast<GLsizei>(sizeof(float) * 4));
    dtAccumulator = 0.0f;

    // Reel render target (RGBA8 + depth), sized once for the whole export.
    glGenTextures(1, &reelTex);
    glBindTexture(GL_TEXTURE_2D, reelTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, reelW, reelH, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glGenRenderbuffers(1, &reelRBO);
    glBindRenderbuffer(GL_RENDERBUFFER, reelRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, reelW, reelH);
    glGenFramebuffers(1, &reelFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, reelFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, reelTex, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, reelRBO);
    const bool fboOK = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    if (!fboOK) {
        if (reelFBO) glDeleteFramebuffers(1, &reelFBO);
        if (reelTex) glDeleteTextures(1, &reelTex);
        if (reelRBO) glDeleteRenderbuffers(1, &reelRBO);
        reelFBO = reelTex = reelRBO = 0;
        reelStatus = "FAILED: could not create reel framebuffer";
        return;
    }
    if (useWaterRendering) InitSSFRBuffers(reelW, reelH);

    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(reelOutDir) / "frames", ec);

    reelFrame = 0;
    reelExporting = true;
    reelStatus = "Exporting...";
}

void Scene0p::ReelExportStep() {
    if (!reelExporting) return;

    const int fps = (reelFpsIdx == 1) ? 60 : 30;
    const float frameDt = 1.0f / float(fps);
    const Matrix4 reelProj = MMath::perspective(
        45.0f, static_cast<float>(reelW) / static_cast<float>(reelH), 0.5f, 100.0f);

    // Deterministic substeps: never larger than the sim's own timestep.
    const float ts   = std::max(1e-6f, fluidGPU->param_timeStep);
    const int   nSub = std::max(1, int(std::ceil(frameDt / ts)));
    const float subDt = frameDt / float(nSub);

    std::vector<unsigned char> pixels(static_cast<size_t>(reelW) * static_cast<size_t>(reelH) * 3);

    // Render a small batch of frames per Update() so the UI stays responsive.
    const int batch = 3;
    for (int b = 0; b < batch && reelFrame < reelBands.frameCount; ++b) {
        DriveAudioReaction(reelBands.bass[reelFrame], reelBands.mid[reelFrame],
                           reelBands.treble[reelFrame], frameDt);
        for (int i = 0; i < nSub; ++i) fluidGPU->DispatchCompute(subDt);

        RenderSceneTo(reelFBO, reelW, reelH, reelProj);

        glBindFramebuffer(GL_FRAMEBUFFER, reelFBO);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, reelW, reelH, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
        glPixelStorei(GL_PACK_ALIGNMENT, 4);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        char name[640];
        snprintf(name, sizeof(name), "%s/frames/f_%05d.png", reelOutDir, reelFrame);
        stbi_flip_vertically_on_write(1);   // GL rows are bottom-up
        stbi_write_png(name, reelW, reelH, 3, pixels.data(), reelW * 3);

        ++reelFrame;
    }

    if (reelFrame >= reelBands.frameCount) FinishReelExport(true);
}

void Scene0p::FinishReelExport(bool wroteBat) {
    if (wroteBat) {
        const int fps = (reelFpsIdx == 1) ? 60 : 30;
        std::error_code ec;
        std::filesystem::path abs = std::filesystem::absolute(reelAudioPath, ec);
        std::string audioAbs = ec ? std::string(reelAudioPath) : abs.string();

        std::filesystem::path batPath = std::filesystem::path(reelOutDir) / "mux_reel.bat";
        if (FILE* f = fopen(batPath.string().c_str(), "wb")) {
            fprintf(f, "@echo off\r\n");
            fprintf(f, "REM Needs ffmpeg on PATH (https://ffmpeg.org). Makes the reel from the rendered frames + your track.\r\n");
            fprintf(f, "ffmpeg -y -framerate %d -i \"frames\\f_%%05d.png\" -i \"%s\" "
                       "-c:v libx264 -pix_fmt yuv420p -crf 18 -c:a aac -shortest \"reel.mp4\"\r\n",
                    fps, audioAbs.c_str());
            fprintf(f, "pause\r\n");
            fclose(f);
        }
        reelStatus = "Done: " + std::to_string(reelFrame) + " frames. Run "
                   + batPath.string() + " to make reel.mp4";
    } else {
        reelStatus = "Cancelled at frame " + std::to_string(reelFrame);
    }

    if (reelFBO) glDeleteFramebuffers(1, &reelFBO);
    if (reelTex) glDeleteTextures(1, &reelTex);
    if (reelRBO) glDeleteRenderbuffers(1, &reelRBO);
    reelFBO = reelTex = reelRBO = 0;

    if (useWaterRendering && windowW > 0 && windowH > 0)
        InitSSFRBuffers(windowW, windowH);
    reelExporting = false;
}

void Scene0p::DoCapture() {
    int capW = 3000, capH = 3000;
    switch (captureResIdx) {
        case 0:  capW = 3000; capH = 3000; break;   // SoundCloud cover art
        case 1:  capW = 3840; capH = 2160; break;   // 4K UHD
        default: capW = windowW; capH = windowH; break;
    }
    if (capW <= 0 || capH <= 0) {
        lastScreenshotPath = "FAILED: window size unknown";
        return;
    }

    GLint maxTex = 0, maxRb = 0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTex);
    glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE, &maxRb);
    const GLint maxSide = std::min(maxTex, maxRb);
    if (capW > maxSide || capH > maxSide) {
        lastScreenshotPath = "FAILED: resolution exceeds GPU limits";
        return;
    }

    // Capture target: RGBA8 color texture + 24-bit depth renderbuffer
    GLuint capTex = 0, capRBO = 0, capFBO = 0;
    glGenTextures(1, &capTex);
    glBindTexture(GL_TEXTURE_2D, capTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, capW, capH, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenRenderbuffers(1, &capRBO);
    glBindRenderbuffer(GL_RENDERBUFFER, capRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, capW, capH);

    glGenFramebuffers(1, &capFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, capFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, capTex, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, capRBO);
    const bool fboOK = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    if (!fboOK) {
        glDeleteFramebuffers(1, &capFBO);
        glDeleteTextures(1, &capTex);
        glDeleteRenderbuffers(1, &capRBO);
        lastScreenshotPath = "FAILED: could not create capture framebuffer";
        return;
    }

    // Render one frame at capture resolution with a matching aspect ratio.
    // The SSFR intermediate buffers must match the target size, so resize
    // them for the capture and restore afterwards.
    if (useWaterRendering) InitSSFRBuffers(capW, capH);
    const Matrix4 capProj = MMath::perspective(
        45.0f, static_cast<float>(capW) / static_cast<float>(capH), 0.5f, 100.0f);
    RenderSceneTo(capFBO, capW, capH, capProj);

    std::vector<unsigned char> pixels(static_cast<size_t>(capW) * static_cast<size_t>(capH) * 3);
    glBindFramebuffer(GL_FRAMEBUFFER, capFBO);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, capW, capH, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glDeleteFramebuffers(1, &capFBO);
    glDeleteTextures(1, &capTex);
    glDeleteRenderbuffers(1, &capRBO);

    // Restore the on-screen render state
    if (useWaterRendering) InitSSFRBuffers(windowW, windowH);
    glViewport(0, 0, windowW, windowH);

    std::error_code ec;
    std::filesystem::create_directories("screenshots", ec);

    time_t now = time(nullptr);
    tm local{};
    localtime_s(&local, &now);
    char name[128];
    snprintf(name, sizeof(name), "screenshots/sph_%04d%02d%02d_%02d%02d%02d_%dx%d.png",
             local.tm_year + 1900, local.tm_mon + 1, local.tm_mday,
             local.tm_hour, local.tm_min, local.tm_sec, capW, capH);

    stbi_flip_vertically_on_write(1);   // GL reads rows bottom-up
    if (stbi_write_png(name, capW, capH, 3, pixels.data(), capW * 3))
        lastScreenshotPath = name;
    else
        lastScreenshotPath = "FAILED: could not write PNG";
}