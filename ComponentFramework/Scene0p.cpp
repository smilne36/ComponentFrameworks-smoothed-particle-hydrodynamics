#include "Scene0p.h"
#include "Debug.h"
#include <cstring>

using namespace MATH;

Scene0p::Scene0p() {}
Scene0p::~Scene0p() {}

void Scene0p::BoxWire::init() {
    if (vao) return;
    const float v[] = {
        // bottom square
        -1,-1,-1,  1,-1,-1,   1,-1,-1,  1,-1, 1,
         1,-1, 1, -1,-1, 1,  -1,-1, 1, -1,-1,-1,
         // top square
         -1, 1,-1,  1, 1,-1,   1, 1,-1,  1, 1, 1,
          1, 1, 1, -1, 1, 1,  -1, 1, 1, -1, 1,-1,
          // verticals
          -1,-1,-1, -1, 1,-1,   1,-1,-1,  1, 1,-1,
           1,-1, 1,  1, 1, 1,  -1,-1, 1, -1, 1, 1
    };
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, (void*)0);
    glBindVertexArray(0);
}
void Scene0p::BoxWire::draw(Shader& lineShader, const Matrix4& P, const Matrix4& V,
    const Matrix4& M, const Vec3& color) {
    glUseProgram(lineShader.GetProgram());
    glUniformMatrix4fv(lineShader.GetUniformID("projectionMatrix"), 1, GL_FALSE, P);
    glUniformMatrix4fv(lineShader.GetUniformID("viewMatrix"), 1, GL_FALSE, V);
    glUniformMatrix4fv(lineShader.GetUniformID("modelMatrix"), 1, GL_FALSE, M);
    GLint lc = glGetUniformLocation(lineShader.GetProgram(), "lineColor");
    if (lc >= 0) glUniform3f(lc, color.x, color.y, color.z);
    glBindVertexArray(vao);
    glDrawArrays(GL_LINES, 0, 24);
    glBindVertexArray(0);
    glUseProgram(0);
}
void Scene0p::BoxWire::destroy() {
    if (vbo) { glDeleteBuffers(1, &vbo); vbo = 0; }
    if (vao) { glDeleteVertexArrays(1, &vao); vao = 0; }
}

bool Scene0p::OnCreate() {
    Debug::Info("Scene0p OnCreate", __FILE__, __LINE__);

    // Camera
    projectionMatrix = MMath::perspective(45.0f, 16.0f / 9.0f, 0.2f, 200.0f);
    viewMatrix = MMath::lookAt(Vec3(0, 0, 18), Vec3(0, 0, 0), Vec3(0, 1, 0));
    modelMatrix.loadIdentity();

    // Fluid
    fluidGPU = new SPHFluidGPU(30000); // choose your N
    // (Grid build & SPH uniforms in DispatchCompute() match your existing codepath) :contentReference[oaicite:3]{index=3}

    // Shaders
    impostorShader = new Shader("shaders/particleImpostor.vert", "shaders/particleImpostor.frag");
    if (!impostorShader->OnCreate()) { Debug::FatalError("Impostor shader create failed", __FILE__, __LINE__); }

    lineShader = new Shader("shaders/lineVert.glsl", "shaders/lineFrag.glsl");
    if (!lineShader->OnCreate()) { Debug::FatalError("Line shader create failed", __FILE__, __LINE__); }

    // GPU state
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // nice for impostors (enable/disable as you like)

    // Geometry for impostors and wireframe
    SetupImpostorVAO();
    boxWire.init();
    UpdateBoxWireframe();   // build once
    lastBoxCenter = fluidGPU->param_boxCenter;
    lastBoxHalf = fluidGPU->param_boxHalf;
    lastBoxEuler = fluidGPU->param_boxEulerDeg;

    return true;
}

void Scene0p::OnDestroy() {
    boxWire.destroy();
    if (impostorVAO) { glDeleteVertexArrays(1, &impostorVAO); impostorVAO = 0; }
    if (lineShader) { lineShader->OnDestroy(); delete lineShader; lineShader = nullptr; }
    if (impostorShader) { impostorShader->OnDestroy(); delete impostorShader; impostorShader = nullptr; }
    if (fluidGPU) { delete fluidGPU; fluidGPU = nullptr; }
}

void Scene0p::HandleEvents(const SDL_Event& e) {
    if (e.type == SDL_KEYDOWN) {
        switch (e.key.keysym.scancode) {
        case SDL_SCANCODE_SPACE: // quick splash on space as a demo
            fluidGPU->ApplyWaveImpulseGPU(2.0f, 3.5f, SDL_GetTicks() * 0.002f, Vec3(0, 1, 0), -1.0f, 2.0f);
            break;
        }
    }
}

void Scene0p::Update(float deltaTime) {
    // Single member dtAccumulator; no duplicate toggle
    // Tiny adaptive substep cap: if frame slow, reduce to 8 else 16
    maxSubstepsPerFrame = (deltaTime > (1.0f / 50.0f)) ? 8 : 16;

    dtAccumulator += deltaTime;
    int did = 0;
    while (dtAccumulator >= fixedDt && did < maxSubstepsPerFrame) {
        fluidGPU->DispatchCompute();      // grid -> SPH -> OBB, minimal barriers
        dtAccumulator -= fixedDt;
        ++did;
    }
    // clamp runaway
    if (dtAccumulator > fixedDt * 4.0f) dtAccumulator = fixedDt * 4.0f;

    // If box params changed (maybe via UI), rebuild wireframe once
    const Vec3 c = fluidGPU->param_boxCenter;
    const Vec3 h = fluidGPU->param_boxHalf;
    const Vec3 e = fluidGPU->param_boxEulerDeg;
    if (c.x != lastBoxCenter.x || c.y != lastBoxCenter.y || c.z != lastBoxCenter.z ||
        h.x != lastBoxHalf.x || h.y != lastBoxHalf.y || h.z != lastBoxHalf.z ||
        e.x != lastBoxEuler.x || e.y != lastBoxEuler.y || e.z != lastBoxEuler.z) {
        UpdateBoxWireframe();
        lastBoxCenter = c; lastBoxHalf = h; lastBoxEuler = e;
    }
}

void Scene0p::Render() {
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Particles (impostors)
    DrawFluidImpostors();

    // Box outline
    Matrix4 Mbox = MMath::translate(fluidGPU->param_boxCenter) *
        MMath::scale(fluidGPU->param_boxHalf);
    boxWire.draw(*lineShader, projectionMatrix, viewMatrix, Mbox, Vec3(0.9f, 0.9f, 0.9f));
}

void Scene0p::SetupImpostorVAO() {
    if (impostorVAO) return;
    glGenVertexArrays(1, &impostorVAO);
    glBindVertexArray(impostorVAO);

    // per-instance position from fluid VBO (vec4)
    glBindBuffer(GL_ARRAY_BUFFER, fluidGPU->GetFluidVBO());
    glEnableVertexAttribArray(5);
    glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 4, (void*)0);
    glVertexAttribDivisor(5, 1);

    glBindVertexArray(0);
}

int Scene0p::CurrentViewportHeight() const {
    GLint vp[4]; glGetIntegerv(GL_VIEWPORT, vp);
    return vp[3] > 0 ? vp[3] : 1080;
}

void Scene0p::DrawFluidImpostors() {
    // refresh instance positions (maps SSBO -> VBO for fluids only)
    if (renderFromSSBO) {
        fluidGPU->UpdateFluidVBOFromGPU();  // keeps draw path simple/fast  :contentReference[oaicite:4]{index=4}
    }

    glUseProgram(impostorShader->GetProgram());
    glUniformMatrix4fv(impostorShader->GetUniformID("projectionMatrix"), 1, GL_FALSE, projectionMatrix);
    glUniformMatrix4fv(impostorShader->GetUniformID("viewMatrix"), 1, GL_FALSE, viewMatrix);
    glUniformMatrix4fv(impostorShader->GetUniformID("modelMatrix"), 1, GL_FALSE, modelMatrix);

    // Heat-map: speed in [0..150] by default
    glUniform1i(glGetUniformLocation(impostorShader->GetProgram(), "colorMode"), 1);
    glUniform2f(glGetUniformLocation(impostorShader->GetProgram(), "vizRange"), 0.0f, 150.0f);

    // Point size based on world radius + viewport height
    glUniform1f(glGetUniformLocation(impostorShader->GetProgram(), "particleRadius"), 0.08f);
    glUniform1f(glGetUniformLocation(impostorShader->GetProgram(), "viewportHeight"), (float)CurrentViewportHeight());

    // Bind particle SSBO for the vertex shader (heat-map fetch)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, fluidGPU->ssbo); // graphics path needs its own bind  :contentReference[oaicite:5]{index=5}

    glBindVertexArray(impostorVAO);
    glDrawArraysInstanced(GL_POINTS, 0, 1, (GLsizei)fluidGPU->GetNumFluids());
    glBindVertexArray(0);

    glUseProgram(0);
}

void Scene0p::UpdateBoxWireframe() {
    // nothing to rebuild here except you may update a cached buffer;
    // BoxWire uses unit cube and we scale/translate via model matrix in draw().
    // Left as hook if you later prebake a rotated OBB frame.
}
