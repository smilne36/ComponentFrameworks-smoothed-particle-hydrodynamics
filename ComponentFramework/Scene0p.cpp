#include <iostream>
#include <limits>
#include <algorithm>
#include <cmath>
#include <SDL.h>
#include <MMath.h>

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

    // Instanced sphere shader (your original)
    shader = new Shader("shaders/defaultVert.glsl", "shaders/defaultFrag.glsl");
    if (!shader->OnCreate()) { std::cerr << "default shader failed\n"; return false; }

    // Make sure SSBO block is at binding=0 for vertex fetch
    {
        GLuint prog = shader->GetProgram();
        GLuint block = glGetProgramResourceIndex(prog, GL_SHADER_STORAGE_BLOCK, "ParticleBuf");
        if (block != GL_INVALID_INDEX) glShaderStorageBlockBinding(prog, block, 0);
        else std::cerr << "SSBO block 'ParticleBuf' not found in program\n";
    }

    projectionMatrix = MMath::perspective(45.0f, 16.0f / 9.0f, 0.5f, 100.0f);
    viewMatrix = MMath::lookAt(cameraPos, cameraTarget, cameraUp);
    modelMatrix.loadIdentity();

    // Fluid sim (allocs SSBOs/compute)
    fluidGPU = new SPHFluidGPU(30000);

    // One-time instancing hookup (attribute 5 for instance pos)
    mesh->BindInstanceBuffer(fluidGPU->GetFluidVBO(), static_cast<GLsizei>(sizeof(float) * 4));

    // Wireframe box shader + VBO
    lineShader = new Shader("shaders/lineVert.glsl", "shaders/lineFrag.glsl");
    if (!lineShader->OnCreate()) { std::cerr << "line shader failed\n"; return false; }
    glGenVertexArrays(1, &boxVAO);
    glBindVertexArray(boxVAO);
    glGenBuffers(1, &boxVBO);
    glBindBuffer(GL_ARRAY_BUFFER, boxVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * 24, nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, (void*)0);
    glBindVertexArray(0);

    // Point-impostor shader & dummy VAO (gl_VertexID path)
    impostorShader = new Shader("shaders/particleImpostor.vert", "shaders/particleImpostor.frag");
    if (!impostorShader->OnCreate()) { std::cerr << "impostor shader failed\n"; return false; }
    SetupImpostorVAO();

    // Initialize wire once
    UpdateBoxWireframe();
    lastBoxCenter = fluidGPU->param_boxCenter;
    lastBoxHalf = fluidGPU->param_boxHalf;
    lastBoxEuler = fluidGPU->param_boxEulerDeg;

    return true;
}

void Scene0p::OnDestroy() {
    if (sphere) { sphere->OnDestroy(); delete sphere; sphere = nullptr; }
    if (mesh) { mesh->OnDestroy(); delete mesh; mesh = nullptr; }
    if (shader) { shader->OnDestroy(); delete shader; shader = nullptr; }
    if (impostorShader) { impostorShader->OnDestroy(); delete impostorShader; impostorShader = nullptr; }
    if (fluidGPU) { delete fluidGPU; fluidGPU = nullptr; }
    if (boxVBO)   glDeleteBuffers(1, &boxVBO);
    if (boxVAO)   glDeleteVertexArrays(1, &boxVAO);
    if (lineShader) { lineShader->OnDestroy(); delete lineShader; lineShader = nullptr; }
    if (impostorVAO) glDeleteVertexArrays(1, &impostorVAO);
}

void Scene0p::HandleEvents(const SDL_Event& e) {
    switch (e.type) {
    case SDL_KEYDOWN:
        switch (e.key.keysym.scancode) {
        case SDL_SCANCODE_W: cameraPos.y += 0.2f; break;
        case SDL_SCANCODE_S: cameraPos.y -= 0.2f; break;
        case SDL_SCANCODE_A: cameraPos.x -= 0.2f; break;
        case SDL_SCANCODE_D: cameraPos.x += 0.2f; break;
        case SDL_SCANCODE_R: cameraPos.z += 0.2f; break;
        case SDL_SCANCODE_E: cameraPos.z -= 0.2f; break;
        case SDL_SCANCODE_Z: drawInWireMode = !drawInWireMode; break;
        default: break;
        }
        break;
    default: break;
    }
}

void Scene0p::UpdateBoxWireframe() {
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

    Vec3 c[8]; int i = 0;
    for (int sx = -1; sx <= 1; sx += 2)
        for (int sy = -1; sy <= 1; sy += 2)
            for (int sz = -1; sz <= 1; sz += 2)
                c[i++] = xform(sx * H.x, sy * H.y, sz * H.z);

    int E[24] = { 0,1, 0,2, 0,4, 3,1, 3,2, 3,7, 5,1, 5,4, 5,7, 6,2, 6,4, 6,7 };
    float lines[24 * 3];
    for (int e = 0; e < 24; ++e) {
        Vec3 p = c[E[e]];
        lines[e * 3 + 0] = p.x; lines[e * 3 + 1] = p.y; lines[e * 3 + 2] = p.z;
    }
    glBindBuffer(GL_ARRAY_BUFFER, boxVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(lines), lines);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Scene0p::Update(const float deltaTime) {
    // simple anim so you can see something moving
    ballAnimTime += deltaTime;
    float ballX = std::sin(ballAnimTime) * 3.0f;
    if (sphere) sphere->SetPosition(Vec3(ballX, 0.0f, 0.0f));
    viewMatrix = MMath::lookAt(cameraPos, cameraTarget, cameraUp);

    if (!fluidGPU) return;

    // --- ImGui (kept intact + small adds) ---
    ImGui::Begin("Fluid Controls");
    ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
    ImGui::Text("Particles: %zu (fluids: %zu)", fluidGPU->particles.size(), fluidGPU->GetNumFluids());
    ImGui::Separator();

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
    if (ImGui::Button("Preset: Splashy Water")) {
        fluidGPU->param_pause = false;
        fluidGPU->param_h = 0.22f;
        fluidGPU->param_restDensity = 1000.0f;
        fluidGPU->param_gasConstant = 6000.0f;
        fluidGPU->param_viscosity = 1.2f;
        fluidGPU->param_gravityY = -980.0f;
        fluidGPU->param_surfaceTension = 0.12f;
        fluidGPU->param_timeStep = 5.0e-4f;
        fluidGPU->param_useJitter = true;
        fluidGPU->param_jitterAmp = 0.06f;
        fluidGPU->param_wallRestitution = 0.05f;
        fluidGPU->param_wallFriction = 0.05f;
        pendingReset = true;
    }
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

    // Sim params
    ImGui::Checkbox("Pause simulation", &fluidGPU->param_pause);
    ImGui::SliderFloat("h (smoothing)", &fluidGPU->param_h, 0.10f, 1.00f);
    ImGui::SliderFloat("mass", &fluidGPU->param_mass, 1.0f, 50.0f);
    ImGui::SliderFloat("restDensity", &fluidGPU->param_restDensity, 100.0f, 2000.0f);
    ImGui::SliderFloat("gasConstant", &fluidGPU->param_gasConstant, 100.0f, 30000.0f);
    ImGui::SliderFloat("viscosity", &fluidGPU->param_viscosity, 0.0f, 20.0f);
    ImGui::SliderFloat("gravityY", &fluidGPU->param_gravityY, -5000.0f, 0.0f);
    ImGui::SliderFloat("surfaceTension", &fluidGPU->param_surfaceTension, 0.0f, 1.0f);
    ImGui::SliderFloat("timeStep", &fluidGPU->param_timeStep, 1e-5f, 5e-3f, "%.6f", ImGuiSliderFlags_Logarithmic);

    // Performance toggles (deduped)
    ImGui::Separator(); ImGui::Text("Performance");
    ImGui::Checkbox("Render from SSBO (fast)", &renderFromSSBO);
    ImGui::Checkbox("Enable ghost boundaries (slower)", &fluidGPU->param_enableGhosts);
    ImGui::SameLine();
    ImGui::Checkbox("Grid sort (unused)", &fluidGPU->param_enableSort);
    ImGui::Checkbox("Use point impostors", &useImpostors);

    // Box transform
    ImGui::Separator(); ImGui::Text("Box Transform (OBB)");
    ImGui::DragFloat3("Center", &fluidGPU->param_boxCenter.x, 0.05f);
    ImGui::DragFloat3("Half Extents", &fluidGPU->param_boxHalf.x, 0.05f, 0.05f, 100.0f);
    ImGui::DragFloat3("Euler XYZ", &fluidGPU->param_boxEulerDeg.x, 0.5f, -180.0f, 180.0f);
    ImGui::SliderFloat("Wall Restitution", &fluidGPU->param_wallRestitution, 0.0f, 1.0f);
    ImGui::SliderFloat("Wall Friction", &fluidGPU->param_wallFriction, 0.0f, 1.0f);
    if (ImGui::Button("Rebuild Grid for Box")) {
        fluidGPU->RecreateGridForBox();
        UpdateBoxWireframe();
    }

    // Spawn
    ImGui::Separator(); ImGui::Text("Spawn Layout");
    ImGui::Checkbox("Use jitter", &fluidGPU->param_useJitter); ImGui::SameLine();
    ImGui::SliderFloat("Jitter amp * spacing", &fluidGPU->param_jitterAmp, 0.0f, 0.5f, "%.2f"); ImGui::SameLine();
    if (ImGui::Button("Rebuild Layout")) { pendingReset = true; }

    // Waves (now GPU-only in SPHFluidGPU)
    ImGui::Separator(); ImGui::Text("Waves");
    static float waveAmplitude = 1.5f;
    static float waveWavelength = 3.0f;
    static float wavePhaseSpeed = 4.0f;
    static int   waveDirIdx = 1; // 0:X 1:Y 2:Z
    static float yBandMin = -std::numeric_limits<float>::infinity();
    static float yBandMax = std::numeric_limits<float>::infinity();
    static bool  continuousWave = false;
    static float wavePhase = 0.0f;

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
    ImGui::SameLine();
    if (ImGui::Button("Reset Simulation")) pendingReset = true;

    // Visualization
    ImGui::Separator(); ImGui::Text("Visualization");
    static int   vizMode = 1;
    static float rangeMin = 0.0f, rangeMax = 10.0f;
    ImGui::Combo("Color by", &vizMode, "Depth\0Speed\0Pressure\0Density\0\0");
    ImGui::DragFloat("Range Min", &rangeMin, 0.1f);
    ImGui::DragFloat("Range Max", &rangeMax, 0.1f);
    ImGui::TextDisabled("Tip: SSBO mode avoids CPU readbacks and boosts FPS.");
    ImGui::End();
    // --- end ImGui ---

    // Update wireframe only when box params changed
    if (lastBoxCenter.x != fluidGPU->param_boxCenter.x ||
        lastBoxCenter.y != fluidGPU->param_boxCenter.y ||
        lastBoxCenter.z != fluidGPU->param_boxCenter.z ||
        lastBoxHalf.x != fluidGPU->param_boxHalf.x ||
        lastBoxHalf.y != fluidGPU->param_boxHalf.y ||
        lastBoxHalf.z != fluidGPU->param_boxHalf.z ||
        lastBoxEuler.x != fluidGPU->param_boxEulerDeg.x ||
        lastBoxEuler.y != fluidGPU->param_boxEulerDeg.y ||
        lastBoxEuler.z != fluidGPU->param_boxEulerDeg.z) {
        UpdateBoxWireframe();
        lastBoxCenter = fluidGPU->param_boxCenter;
        lastBoxHalf = fluidGPU->param_boxHalf;
        lastBoxEuler = fluidGPU->param_boxEulerDeg;
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

    // Fixed-step sim loop (member accumulator + lower cap)
    const float fixedDt = std::max(1e-6f, fluidGPU->param_timeStep);
    dtAccumulator += deltaTime;

    // simple adaptive safety: if frame is very slow, shrink the cap from 16 -> 8
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
}

void Scene0p::Render() const {
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (drawInWireMode) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    else                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    // 1) Box wireframe
    if (lineShader && boxVAO) {
        glUseProgram(lineShader->GetProgram());
        glUniformMatrix4fv(lineShader->GetUniformID("projectionMatrix"), 1, GL_FALSE, projectionMatrix);
        glUniformMatrix4fv(lineShader->GetUniformID("viewMatrix"), 1, GL_FALSE, viewMatrix);
        if (GLint col = lineShader->GetUniformID("uColor"); col != -1) glUniform3f(col, 0.85f, 0.95f, 1.0f);
        glBindVertexArray(boxVAO);
        glLineWidth(1.5f);
        glDrawArrays(GL_LINES, 0, 24);
        glBindVertexArray(0);
        glUseProgram(0);
    }

    if (!fluidGPU) return;

    // 2) Particles
    if (useImpostors && impostorShader) {
        DrawFluidImpostors();
        return;
    }

    // Instanced spheres path
    if (!renderFromSSBO) {
        // If the VS reads instance pos from a VBO, refresh it from GPU (slow path)
        fluidGPU->UpdateFluidVBOFromGPU();
    }

    glUseProgram(shader->GetProgram());
    glUniformMatrix4fv(shader->GetUniformID("projectionMatrix"), 1, GL_FALSE, projectionMatrix);
    glUniformMatrix4fv(shader->GetUniformID("viewMatrix"), 1, GL_FALSE, viewMatrix);

    // Height band (for your heat map)
    if (GLint hLoc = shader->GetUniformID("heightMinMax"); hLoc != -1) {
        glUniform2f(hLoc,
            fluidGPU->param_boxCenter.y - fluidGPU->param_boxHalf.y,
            fluidGPU->param_boxCenter.y + fluidGPU->param_boxHalf.y);
    }

    if (GLint useLoc = shader->GetUniformID("useSSBO"); useLoc != -1)
        glUniform1i(useLoc, renderFromSSBO ? 1 : 0);
    if (GLint cmLoc = shader->GetUniformID("colorMode"); cmLoc != -1)
        glUniform1i(cmLoc, 1); // Speed
    if (GLint vrLoc = shader->GetUniformID("vizRange"); vrLoc != -1)
        glUniform2f(vrLoc, 0.0f, 10.0f);

    float particleRadius = std::max(0.02f, 0.5f * fluidGPU->param_h);
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
    return vp[3] > 0 ? vp[3] : 1080; // fallback
}

void Scene0p::DrawFluidImpostors() const {
    glUseProgram(impostorShader->GetProgram());
    glUniformMatrix4fv(impostorShader->GetUniformID("projectionMatrix"), 1, GL_FALSE, projectionMatrix);
    glUniformMatrix4fv(impostorShader->GetUniformID("viewMatrix"), 1, GL_FALSE, viewMatrix);

    // screen-space point size scale (many impostor VS expect this)
    if (GLint s = impostorShader->GetUniformID("uPointScale"); s != -1) {
        const float fovY = 45.0f * (3.14159265359f / 180.0f);
        float pointScale = float(CurrentViewportHeight()) / (2.0f * tanf(fovY * 0.5f));
        glUniform1f(s, pointScale);
    }

    if (GLint hLoc = impostorShader->GetUniformID("heightMinMax"); hLoc != -1) {
        glUniform2f(hLoc,
            fluidGPU->param_boxCenter.y - fluidGPU->param_boxHalf.y,
            fluidGPU->param_boxCenter.y + fluidGPU->param_boxHalf.y);
    }

    if (GLint cmLoc = impostorShader->GetUniformID("colorMode"); cmLoc != -1)
        glUniform1i(cmLoc, 1); // speed
    if (GLint vrLoc = impostorShader->GetUniformID("vizRange"); vrLoc != -1)
        glUniform2f(vrLoc, 0.0f, 10.0f);

    // Particle radius (if shader needs it)
    if (GLint r = impostorShader->GetUniformID("particleRadius"); r != -1)
        glUniform1f(r, std::max(0.02f, 0.5f * fluidGPU->param_h));

    // SSBO with all particles (VS uses gl_VertexID to index)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, fluidGPU->ssbo);

    glBindVertexArray(impostorVAO);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(fluidGPU->GetNumFluids()));
    glBindVertexArray(0);
    glUseProgram(0);
}
