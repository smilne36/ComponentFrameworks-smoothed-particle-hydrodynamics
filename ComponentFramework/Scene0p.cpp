#include <iostream>
#include <SDL.h>
#include "Scene0p.h"
#include <MMath.h>
#include "Debug.h"
#include "Mesh.h"
#include "Shader.h"
#include "Body.h"
#include <cmath>
#include "SPHFluid3D.h"
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "SceneManager.h"
#include <limits>
#include <algorithm>

Scene0p::Scene0p() :
    sphere{ nullptr }, shader{ nullptr }, mesh{ nullptr },
    drawInWireMode{ true }, mouseX{ 0 }, mouseY{ 0 }, mouseDown{ false }, ballAnimTime{ 0.0f },
    fluidGPU{ nullptr }
{
    Debug::Info("Created Scene0: ", __FILE__, __LINE__);
}

Scene0p::~Scene0p() {
    Debug::Info("Deleted Scene0: ", __FILE__, __LINE__);
}


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
        for (int c = 0; c < 3; c++)
            for (int r = 0; r < 3; r++)
                C[c * 3 + r] = A[0 * 3 + r] * B[c * 3 + 0] + A[1 * 3 + r] * B[c * 3 + 1] + A[2 * 3 + r] * B[c * 3 + 2];
        };
    float Rzy[9]; mul3(Rz, Ry, Rzy);
    mul3(Rzy, Rx, outM);
}



bool Scene0p::OnCreate() {
    Debug::Info("Loading assets Scene0: ", __FILE__, __LINE__);

    sphere = new Body();
    sphere->OnCreate();
    sphere->SetPosition(Vec3(0.0f, -20.0f, 0.0f));

    mesh = new Mesh("meshes/Sphere.obj");
    mesh->OnCreate();

    shader = new Shader("shaders/defaultVert.glsl", "shaders/defaultFrag.glsl");
    if (!shader->OnCreate()) {
        std::cerr << "Shader failed to load\n";
        return false;
    }

    // Ensure the shader storage block used by the vertex shader is bound to binding = 0
    {
        GLuint prog = shader->GetProgram();
        GLuint block = glGetProgramResourceIndex(prog, GL_SHADER_STORAGE_BLOCK, "ParticleBuf");
        if (block != GL_INVALID_INDEX) {
            glShaderStorageBlockBinding(prog, block, 0);
        }
        else {
            std::cerr << "SSBO block 'ParticleBuf' not found in program\n";
        }
    }

    projectionMatrix = MMath::perspective(45.0f, 16.0f / 9.0f, 0.5f, 100.0f);
    viewMatrix = MMath::lookAt(cameraPos, cameraTarget, cameraUp);
    modelMatrix.loadIdentity();

    // Create fluid system (allocates SSBOs and compute pipelines)
    fluidGPU = new SPHFluidGPU(30000);

    // One-time VAO setup for instancing: bind instance attribute 5 to fluid VBO, disable 4/6
    mesh->BindInstanceBuffer(fluidGPU->GetFluidVBO(), static_cast<GLsizei>(sizeof(float) * 4));

    lineShader = new Shader("shaders/lineVert.glsl", "shaders/lineFrag.glsl");
    if (!lineShader->OnCreate()) {
        std::cerr << "Line shader failed to load\n";
        return false;
    }
    glGenVertexArrays(1, &boxVAO);
    glBindVertexArray(boxVAO);
    glGenBuffers(1, &boxVBO);
    glBindBuffer(GL_ARRAY_BUFFER, boxVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * 24, nullptr, GL_DYNAMIC_DRAW); // 12 edges * 2 verts
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, (void*)0);
    glBindVertexArray(0);

    UpdateBoxWireframe();

    return true;
}

void Scene0p::OnDestroy() {
    Debug::Info("Deleting assets Scene0: ", __FILE__, __LINE__);
    if (sphere) {
        sphere->OnDestroy();
        delete sphere;
    }

    if (mesh) {
        mesh->OnDestroy();
        delete mesh;
    }

    if (shader) {
        shader->OnDestroy();
        delete shader;
    }

    if (fluidGPU) {
        delete fluidGPU;
    }
    if (boxVBO) glDeleteBuffers(1, &boxVBO);
    if (boxVAO) glDeleteVertexArrays(1, &boxVAO);
    if (lineShader) { lineShader->OnDestroy(); delete lineShader; }
}

void Scene0p::HandleEvents(const SDL_Event& sdlEvent) {
    switch (sdlEvent.type) {
    case SDL_KEYDOWN:
        switch (sdlEvent.key.keysym.scancode) {
        case SDL_SCANCODE_W: cameraPos.y += 0.2f; break;
        case SDL_SCANCODE_S: cameraPos.y -= 0.2f; break;
        case SDL_SCANCODE_A: cameraPos.x -= 0.2f; break;
        case SDL_SCANCODE_D: cameraPos.x += 0.2f; break;
        case SDL_SCANCODE_R: cameraPos.z += 0.2f; break;
        case SDL_SCANCODE_E: cameraPos.z -= 0.2f; break;
        case SDL_SCANCODE_Z: drawInWireMode = !drawInWireMode; break;
        }
        break;
    }
}
void Scene0p::UpdateBoxWireframe() {
    if (!fluidGPU || !boxVBO) return;

    const Vec3 C = fluidGPU->param_boxCenter;
    const Vec3 H = fluidGPU->param_boxHalf;
    float Rm[9]; MakeRotationMat3XYZ(fluidGPU->param_boxEulerDeg.x,
        fluidGPU->param_boxEulerDeg.y,
        fluidGPU->param_boxEulerDeg.z, Rm);
    auto xform = [&](float x, float y, float z) -> Vec3 {
        // rotate then translate
        return Vec3(
            Rm[0] * x + Rm[3] * y + Rm[6] * z + C.x,
            Rm[1] * x + Rm[4] * y + Rm[7] * z + C.y,
            Rm[2] * x + Rm[5] * y + Rm[8] * z + C.z
        );
        };

    // 8 corners
    Vec3 c[8];
    int i = 0;
    for (int sx = -1; sx <= 1; sx += 2)
        for (int sy = -1; sy <= 1; sy += 2)
            for (int sz = -1; sz <= 1; sz += 2)
                c[i++] = xform(sx * H.x, sy * H.y, sz * H.z);

    // edges (12) as pairs of indices into c[]
    int E[24] = {
        0,1, 0,2, 0,4,
        3,1, 3,2, 3,7,
        5,1, 5,4, 5,7,
        6,2, 6,4, 6,7
    };
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
    ballAnimTime += deltaTime;
    float ballX = std::sin(ballAnimTime) * 3.0f;
    if (sphere) { sphere->SetPosition(Vec3(ballX, 0.0f, 0.0f)); }
    viewMatrix = MMath::lookAt(cameraPos, cameraTarget, cameraUp);

    if (fluidGPU) {
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
            fluidGPU->param_gravityY = -980.0f;     // cm/s^2 (was -9.81f)
            fluidGPU->param_surfaceTension = 0.0f;
            fluidGPU->param_timeStep = 1.0e-3f;     // was 2e-4f (too small)
            pendingReset = true;                    // ensure buffers rebuilt when h or layout changes
        }

        if (ImGui::Button("Preset: Splashy Water")) {
            fluidGPU->param_pause = false;
            fluidGPU->param_h = 0.22f;
            fluidGPU->param_restDensity = 1000.0f;
            fluidGPU->param_gasConstant = 6000.0f;
            fluidGPU->param_viscosity = 1.2f;
            fluidGPU->param_gravityY = -980.0f;        // cm/s^2 (was -9.81f)
            fluidGPU->param_surfaceTension = 0.12f;
            fluidGPU->param_timeStep = 5.0e-4f;        // was 1e-4f
            fluidGPU->param_useJitter = true;
            fluidGPU->param_jitterAmp = 0.06f;

            fluidGPU->param_wallRestitution = 0.05f;
            fluidGPU->param_wallFriction = 0.05f;

            pendingReset = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Fit Camera")) {
            Vec3 minV(std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max());
            Vec3 maxV(-std::numeric_limits<float>::max(),
                -std::numeric_limits<float>::max(),
                -std::numeric_limits<float>::max());
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
            if (count > 0) {
                Vec3 center((minV.x + maxV.x) * 0.5f,
                    (minV.y + maxV.y) * 0.5f,
                    (minV.z + maxV.z) * 0.5f);
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

        // Performance toggles
        static bool useSSBO = true; // fast path (render)
        ImGui::Separator();
        ImGui::Text("Performance");
        ImGui::Checkbox("Render from SSBO (fast)", &useSSBO);
        ImGui::Checkbox("Enable ghost boundaries (slower)", &fluidGPU->param_enableGhosts);
        ImGui::Checkbox("Render from SSBO (fast)", &renderFromSSBO);
        ImGui::SameLine();
        ImGui::Checkbox("Enable grid sort (unused)", &fluidGPU->param_enableSort);

        // Box transform (move/resize/rotate)
        ImGui::Separator();
        ImGui::Text("Box Transform (OBB)");
        ImGui::DragFloat3("Center", &fluidGPU->param_boxCenter.x, 0.05f);
        ImGui::DragFloat3("Half Extents", &fluidGPU->param_boxHalf.x, 0.05f, 0.05f, 100.0f);
        ImGui::DragFloat3("Euler XYZ (deg)", &fluidGPU->param_boxEulerDeg.x, 0.5f, -180.0f, 180.0f);
        ImGui::SliderFloat("Wall Restitution", &fluidGPU->param_wallRestitution, 0.0f, 1.0f);
        ImGui::SliderFloat("Wall Friction", &fluidGPU->param_wallFriction, 0.0f, 1.0f);
        if (ImGui::Button("Rebuild Grid for Box")) {
            fluidGPU->RecreateGridForBox();

        }

        // Spawn layout
        ImGui::Separator();
        ImGui::Text("Spawn Layout");
        ImGui::Checkbox("Use jitter", &fluidGPU->param_useJitter);
        ImGui::SameLine();
        ImGui::SliderFloat("Jitter amp * spacing", &fluidGPU->param_jitterAmp, 0.0f, 0.5f, "%.2f");
        ImGui::SameLine();
        if (ImGui::Button("Rebuild Layout")) {
            pendingReset = true;

        }

        // Waves
        ImGui::Separator();
        ImGui::Text("Waves");
        static float waveAmplitude = 1.5f;
        static float waveWavelength = 3.0f;
        static float wavePhaseSpeed = 4.0f;
        static int waveDirIdx = 1;
        static float yBandMin = -std::numeric_limits<float>::infinity();
        static float yBandMax = std::numeric_limits<float>::infinity();
        static bool continuousWave = false;
        static float wavePhase = 0.0f;

        ImGui::SliderFloat("Amplitude", &waveAmplitude, 0.0f, 25.0f);
        ImGui::SliderFloat("Wavelength", &waveWavelength, 0.5f, 10.0f);
        ImGui::SliderFloat("Phase speed", &wavePhaseSpeed, 0.0f, 20.0f);
        ImGui::RadioButton("Dir X", &waveDirIdx, 0); ImGui::SameLine();
        ImGui::RadioButton("Dir Y", &waveDirIdx, 1); ImGui::SameLine();
        ImGui::RadioButton("Dir Z", &waveDirIdx, 2);
        ImGui::InputFloat("Band Y min", &yBandMin);
        ImGui::InputFloat("Band Y max", &yBandMax);
        ImGui::Checkbox("Continuous wave (CPU sync)", &continuousWave);
        if (ImGui::Button("Impulse Now")) {
            Vec3 dir = (waveDirIdx == 0) ? Vec3(1, 0, 0) : (waveDirIdx == 1) ? Vec3(0, 1, 0) : Vec3(0, 0, 1);
            fluidGPU->ApplyWaveImpulse(waveAmplitude, waveWavelength, wavePhase, dir, yBandMin, yBandMax);
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset Simulation")) {
            pendingReset = true;
        }

        // Visualization
        ImGui::Separator();
        ImGui::Text("Visualization");
        static int vizMode = 1;
        static float rangeMin = 0.0f, rangeMax = 10.0f;
        ImGui::Combo("Color by", &vizMode, "Depth\0Speed\0Pressure\0Density\0\0");
        ImGui::DragFloat("Range Min", &rangeMin, 0.1f);
        ImGui::DragFloat("Range Max", &rangeMax, 0.1f);
        ImGui::TextDisabled("Tip: SSBO mode avoids CPU readbacks and boosts FPS.");

        ImGui::End();


		UpdateBoxWireframe(); // in case box was changed

        if (pendingReset) {
            fluidGPU->ResetSimulation();
            fluidGPU->param_pause = false;
            pendingReset = false;
            // clear sim time accumulator to avoid large catch-up
            dtAccumulator = 0.0f;
        }

        if (continuousWave) {
            wavePhase += wavePhaseSpeed * deltaTime;
            Vec3 dir = (waveDirIdx == 0) ? Vec3(1, 0, 0) : (waveDirIdx == 1) ? Vec3(0, 1, 0) : Vec3(0, 0, 1);
            fluidGPU->ApplyWaveImpulse(waveAmplitude, waveWavelength, wavePhase, dir, yBandMin, yBandMax);
        }

        // Sim step
       // Sim step (fixed step accumulator; never exceed param_timeStep)
        static float dtAccumulator = 0.0f;
        const float fixedDt = std::max(1e-6f, fluidGPU->param_timeStep);
        const int maxSubstepsPerFrame = 64; // cap to avoid long frames

        // If we just reset, drop leftovers to avoid a huge burst
        if (pendingReset == false) {
            dtAccumulator += deltaTime;
        }


        if (pendingReset) {
            fluidGPU->ResetSimulation();
            fluidGPU->param_pause = false;

            // Rebind instance attribute 5 to the new fluid VBO
            mesh->BindInstanceBuffer(fluidGPU->GetFluidVBO(), static_cast<GLsizei>(sizeof(float) * 4));

            pendingReset = false;
            // clear sim time accumulator to avoid large catch-up
            dtAccumulator = 0.0f;
        }


        int didSteps = 0;
        while (dtAccumulator >= fixedDt && didSteps < maxSubstepsPerFrame) {
            fluidGPU->DispatchCompute(fixedDt);
            dtAccumulator -= fixedDt;
            ++didSteps;
        }

        // Optional: if we hit the cap, discard excess to prevent spiral-of-death
        if (didSteps == maxSubstepsPerFrame && dtAccumulator > fixedDt) {
            dtAccumulator = fmodf(dtAccumulator, fixedDt);
        }

        // Show what happened this frame
        ImGui::Text("Substeps: %d  dt: %.6f  accum: %.6f", didSteps, fixedDt, dtAccumulator);

        if (!useSSBO) {
            fluidGPU->UpdateFluidVBOFromGPU();
        }
    }
}

void Scene0p::Render() const {
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (drawInWireMode) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    else glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    // 1) Draw wireframe box
    if (lineShader && boxVAO) {
        glUseProgram(lineShader->GetProgram());
        glUniformMatrix4fv(lineShader->GetUniformID("projectionMatrix"), 1, GL_FALSE, projectionMatrix);
        glUniformMatrix4fv(lineShader->GetUniformID("viewMatrix"), 1, GL_FALSE, viewMatrix);
        GLint colLoc = lineShader->GetUniformID("uColor");
        if (colLoc != (GLint)-1) glUniform3f(colLoc, 0.85f, 0.95f, 1.0f);
        glBindVertexArray(boxVAO);
        glLineWidth(1.5f);
        glDrawArrays(GL_LINES, 0, 24);
        glBindVertexArray(0);
        glUseProgram(0);
    }

    // 2) Draw particles (instanced)
    if (fluidGPU && mesh) {
        // Make sure instance positions are fresh (copies pos from SSBO -> fluidVBO)
        if (!renderFromSSBO) {
            fluidGPU->UpdateFluidVBOFromGPU(); // only when using instance VBO
        }

        glUseProgram(shader->GetProgram());

        glUniformMatrix4fv(shader->GetUniformID("projectionMatrix"), 1, GL_FALSE, projectionMatrix);
        glUniformMatrix4fv(shader->GetUniformID("viewMatrix"), 1, GL_FALSE, viewMatrix);

        // Height band uniforms used by shader for height coloring
        GLint hLoc = shader->GetUniformID("heightMinMax");
        if (hLoc != -1) {
            glUniform2f(hLoc,
                fluidGPU->param_boxCenter.y - fluidGPU->param_boxHalf.y,
                fluidGPU->param_boxCenter.y + fluidGPU->param_boxHalf.y);
        }

        // Visualization controls
        if (GLint useLoc = shader->GetUniformID("useSSBO"); useLoc != -1)
            glUniform1i(useLoc, renderFromSSBO ? 1 : 0);
        GLint cmLoc = shader->GetUniformID("colorMode");
        if (cmLoc != -1) glUniform1i(cmLoc, 1); // Speed
        GLint vrLoc = shader->GetUniformID("vizRange");
        if (vrLoc != -1) glUniform2f(vrLoc, 0.0f, 10.0f);

        // Scale each sphere by particle radius
        float particleRadius = std::max(0.02f, 0.5f * fluidGPU->param_h);
        Matrix4 scaleMat = MMath::scale(particleRadius, particleRadius, particleRadius);
        glUniformMatrix4fv(shader->GetUniformID("modelMatrix"), 1, GL_FALSE, scaleMat);

        // Bind particle SSBO for speed/pressure/density reads in VS
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, fluidGPU->ssbo);
        mesh->RenderInstanced(GL_TRIANGLES, fluidGPU->GetNumFluids());
    

        glUseProgram(0);
    }
}