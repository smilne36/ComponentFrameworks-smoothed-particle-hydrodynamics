#include "SPHFluid3D.h"
#include "Debug.h"
#include <fstream>
#include <sstream>
#include <random>
#include <vector>
#include <iostream>
#include <cmath>
#include <SDL_stdinc.h>
#include <algorithm>

static void Mul3x3(const float A[9], const float B[9], float C[9]) {
    for (int c = 0; c < 3; c++)
        for (int r = 0; r < 3; r++)
            C[c * 3 + r] = A[0 * 3 + r] * B[c * 3 + 0] + A[1 * 3 + r] * B[c * 3 + 1] + A[2 * 3 + r] * B[c * 3 + 2];
}

void SPHFluid3D::MakeRotationMat3XYZ(float rxDeg, float ryDeg, float rzDeg, float outM[9]) {
    const float rx = rxDeg * float(M_PI / 180.0);
    const float ry = ryDeg * float(M_PI / 180.0);
    const float rz = rzDeg * float(M_PI / 180.0);
    const float cx = cosf(rx), sx = sinf(rx);
    const float cy = cosf(ry), sy = sinf(ry);
    const float cz = cosf(rz), sz = sinf(rz);
    const float Rz[9] = { cz, sz, 0, -sz, cz, 0, 0, 0, 1 };
    const float Ry[9] = { cy, 0, -sy, 0, 1, 0, sy, 0, cy };
    const float Rx[9] = { 1, 0, 0, 0, cx, sx, 0, -sx, cx };
    float Rzy[9]; Mul3x3(Rz, Ry, Rzy);
    Mul3x3(Rzy, Rx, outM);
}

SPHFluidGPU::SPHFluidGPU(size_t numParticles_)
    : numParticles(numParticles_), fluidVBO(0), vboPtr(nullptr), numFluids(0)
{
    clearGridShader = LoadComputeShader("shaders/ClearGrid.comp");
    buildGridShader = LoadComputeShader("shaders/BuildGrid.comp");
    sphGridShader = LoadComputeShader("shaders/SPHFluid.comp");
    radixSortShader = LoadComputeShader("shaders/RadixSort.comp");
    obbConstraintShader = LoadComputeShader("shaders/OBBConstraints.comp");
    waveImpulseShader = LoadComputeShader("shaders/WaveImpulse.comp"); // NEW

    // 1) Build particles (also sets 'box')
    InitializeParticles();

    // 2) Now size grid/aux SSBOs using the actual particle count
    InitializeGridAndBuffers();
    InitializeSortBuffers();

    // 3) GPU resources
    UploadDataToGPU();
    InitializeFluidVBO();

    // Cache OBB rotation once
    cachedBoxCenter = param_boxCenter;
    cachedBoxHalf = param_boxHalf;
    cachedBoxEuler = param_boxEulerDeg;
    MakeRotationMat3XYZ(cachedBoxEuler.x, cachedBoxEuler.y, cachedBoxEuler.z, cachedRot3x3);
}

SPHFluidGPU::~SPHFluidGPU() {
    if (fluidVBO) { glBindBuffer(GL_ARRAY_BUFFER, 0); glDeleteBuffers(1, &fluidVBO); }
    glDeleteBuffers(1, &ssbo);
    glDeleteBuffers(1, &cellHeadSSBO);
    glDeleteBuffers(1, &particleNextSSBO);
    glDeleteBuffers(1, &particleCellSSBO);
    glDeleteBuffers(1, &cellKeySSBO);
    glDeleteBuffers(1, &sortIdxSSBO);
    glDeleteBuffers(1, &sortTmpSSBO);
    glDeleteProgram(clearGridShader);
    glDeleteProgram(buildGridShader);
    glDeleteProgram(sphGridShader);
    glDeleteProgram(radixSortShader);
    glDeleteProgram(obbConstraintShader);
    glDeleteProgram(waveImpulseShader);
}

void SPHFluidGPU::InitializeParticles() {
    this->box = std::max(param_boxHalf.x, std::max(param_boxHalf.y, param_boxHalf.z));

    float h = param_h;
    float spacing = h * 0.85f;
    float box = this->box;

    particles.clear();
    int count = 0;

    const float fillFraction = 0.4f;
    const float fluidHeight = (2.0f * box) * fillFraction;
    const int fluidLayersY = int(fluidHeight / spacing);
    const int fluidSide = int((box * 1.7f) / spacing);

    std::default_random_engine rng(static_cast<unsigned>(time(nullptr)));
    std::uniform_real_distribution<float> jitterDist(-spacing * param_jitterAmp,
        +spacing * param_jitterAmp);
    auto j = [&]() -> float { return param_useJitter ? jitterDist(rng) : 0.0f; };

    // Fill a block at the bottom of the box
    for (int x = 0; x < fluidSide && count < (int)numParticles; ++x) {
        for (int y = 0; y < fluidLayersY && count < (int)numParticles; ++y) {
            for (int z = 0; z < fluidSide && count < (int)numParticles; ++z) {
                SPHParticle p{};
                p.pos = Vec4(-box * 0.7f + x * spacing + j(),
                    -box + spacing + y * spacing + j(),
                    -box * 0.7f + z * spacing + j(), 0.0f);
                p.vel = Vec4(0, 0, 0, 0);
                p.acc = Vec4(0, 0, 0, 0);
                p.density = 0.0f; p.pressure = 0.0f;
                p.isGhost = 0; p.isActive = 0;
                particles.push_back(p);
                ++count;
            }
        }
    }

    // Boundary ghosts
    auto add_ghost = [&](float x, float y, float z) {
        SPHParticle p{};
        p.pos = Vec4(x, y, z, 0); p.vel = Vec4(0, 0, 0, 0); p.acc = Vec4(0, 0, 0, 0);
        p.density = 0; p.pressure = 0; p.isGhost = 1; p.isActive = 0;
        particles.push_back(p);
        };
    for (float y = -box; y <= box; y += spacing)
        for (float z = -box; z <= box; z += spacing) { add_ghost(-box, y, z); add_ghost(box, y, z); }
    for (float x = -box; x <= box; x += spacing)
        for (float z = -box; z <= box; z += spacing) { add_ghost(x, -box, z); add_ghost(x, box, z); }
    for (float x = -box; x <= box; x += spacing)
        for (float y = -box; y <= box; y += spacing) { add_ghost(x, y, -box); add_ghost(x, y, box); }

    std::cout << "Fluid particles: " << count
        << ", ghosts: " << (particles.size() - count) << std::endl;

    BuildGhostGrids();
}

void SPHFluidGPU::InitializeFluidVBO() {
    numFluids = 0;
    for (const auto& p : particles) if (p.isGhost == 0) ++numFluids;

    if (fluidVBO) glDeleteBuffers(1, &fluidVBO);
    glGenBuffers(1, &fluidVBO);
    glBindBuffer(GL_ARRAY_BUFFER, fluidVBO);
    glBufferStorage(GL_ARRAY_BUFFER, sizeof(float) * 4 * numFluids, nullptr,
        GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);

    vboPtr = (float*)glMapBufferRange(GL_ARRAY_BUFFER, 0, sizeof(float) * 4 * numFluids,
        GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void SPHFluidGPU::UpdateFluidVBOFromGPU() {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    const SPHParticle* gpuParticles = (const SPHParticle*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    if (gpuParticles && vboPtr) {
        size_t v = 0;
        for (size_t i = 0; i < particles.size(); ++i) {
            if (gpuParticles[i].isGhost == 0) {
                vboPtr[4 * v + 0] = gpuParticles[i].pos.x;
                vboPtr[4 * v + 1] = gpuParticles[i].pos.y;
                vboPtr[4 * v + 2] = gpuParticles[i].pos.z;
                vboPtr[4 * v + 3] = 1.0f;
                ++v;
            }
        }
    }
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void SPHFluidGPU::InitializeGridAndBuffers() {
    float h = param_h;
    cellSize = h;

    Vec3 half = param_boxHalf;
    gridSizeX = std::max(1, int(ceil((2.0f * half.x) / cellSize)));
    gridSizeY = std::max(1, int(ceil((2.0f * half.y) / cellSize)));
    gridSizeZ = std::max(1, int(ceil((2.0f * half.z) / cellSize)));
    numCells = gridSizeX * gridSizeY * gridSizeZ;

    box = std::max(half.x, std::max(half.y, half.z));

    const size_t N = std::max<size_t>(particles.size(), 1);

    if (cellHeadSSBO) glDeleteBuffers(1, &cellHeadSSBO);
    if (particleNextSSBO) glDeleteBuffers(1, &particleNextSSBO);
    if (particleCellSSBO) glDeleteBuffers(1, &particleCellSSBO);

    glGenBuffers(1, &cellHeadSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cellHeadSSBO);
    std::vector<int> cellHeadInit(numCells, -1);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int) * numCells, cellHeadInit.data(), GL_DYNAMIC_COPY);

    glGenBuffers(1, &particleNextSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, particleNextSSBO);
    std::vector<int> nextInit(N, -1);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int) * N, nextInit.data(), GL_DYNAMIC_COPY);

    glGenBuffers(1, &particleCellSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, particleCellSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int) * N, nullptr, GL_DYNAMIC_COPY);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cellHeadSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, particleNextSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, particleCellSSBO);
}

void SPHFluidGPU::RecreateGridForBox() { InitializeGridAndBuffers(); }

void SPHFluidGPU::InitializeSortBuffers() {
    size_t N = particles.size();

    glGenBuffers(1, &cellKeySSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cellKeySSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int) * N, nullptr, GL_DYNAMIC_COPY);

    glGenBuffers(1, &sortIdxSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sortIdxSSBO);
    std::vector<int> idxInit(N); for (size_t i = 0; i < N; ++i) idxInit[i] = int(i);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int) * N, idxInit.data(), GL_DYNAMIC_COPY);

    glGenBuffers(1, &sortTmpSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sortTmpSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int) * N, nullptr, GL_DYNAMIC_COPY);
}

void SPHFluidGPU::RadixSortByCell() {
    size_t N = particles.size();
    for (int pass = 0; pass < 8; ++pass) {
        glUseProgram(radixSortShader);
        glUniform1i(glGetUniformLocation(radixSortShader, "N"), int(N));
        glUniform1i(glGetUniformLocation(radixSortShader, "pass"), pass);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, cellKeySSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, (pass % 2 == 0) ? sortIdxSSBO : sortTmpSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, (pass % 2 == 0) ? sortTmpSSBO : sortIdxSSBO);
        glDispatchCompute((N + 255) / 256, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
}

void SPHFluidGPU::UploadDataToGPU() {
    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    const GLsizeiptr sz = sizeof(SPHParticle) * particles.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, sz, particles.data(), GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void SPHFluidGPU::DownloadDataFromGPU() {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    const SPHParticle* gpuParticles = (const SPHParticle*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    if (gpuParticles) {
        memcpy(particles.data(), gpuParticles, sizeof(SPHParticle) * particles.size());
    }
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void SPHFluidGPU::UpdateCachedBoxIfNeeded() {
    if (cachedBoxCenter.x != param_boxCenter.x || cachedBoxCenter.y != param_boxCenter.y || cachedBoxCenter.z != param_boxCenter.z ||
        cachedBoxHalf.x != param_boxHalf.x || cachedBoxHalf.y != param_boxHalf.y || cachedBoxHalf.z != param_boxHalf.z ||
        cachedBoxEuler.x != param_boxEulerDeg.x || cachedBoxEuler.y != param_boxEulerDeg.y || cachedBoxEuler.z != param_boxEulerDeg.z) {
        cachedBoxCenter = param_boxCenter;
        cachedBoxHalf = param_boxHalf;
        cachedBoxEuler = param_boxEulerDeg;
        MakeRotationMat3XYZ(cachedBoxEuler.x, cachedBoxEuler.y, cachedBoxEuler.z, cachedRot3x3);
    }
}

void SPHFluidGPU::DispatchCompute() {
    if (param_pause) return;

    // Live params
    float h = param_h;
    float mass = param_mass;
    float restDensity = param_restDensity;
    float gasConstant = param_gasConstant;
    float viscosity = param_viscosity;
    Vec3  gravity = Vec3(0, param_gravityY, 0);
    float surfaceTension = param_surfaceTension;
    float timeStep = param_timeStep;

    // No CPU ghost activation in fast path
    if (param_enableGhosts) {
        DownloadDataFromGPU();
        UpdateGhostParticlesDynamic(h);
        UploadGhostActivityToGPU();
    }

    // 1) Clear grid
    glUseProgram(clearGridShader);
    glUniform1i(glGetUniformLocation(clearGridShader, "numCells"), numCells);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cellHeadSSBO);
    glDispatchCompute((numCells + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // 2) Build grid
    glUseProgram(buildGridShader);
    glUniform3i(glGetUniformLocation(buildGridShader, "gridSize"), gridSizeX, gridSizeY, gridSizeZ);
    glUniform1f(glGetUniformLocation(buildGridShader, "cellSize"), cellSize);
    const Vec3 gridMin = param_boxCenter - param_boxHalf;
    glUniform3f(glGetUniformLocation(buildGridShader, "gridMin"), gridMin.x, gridMin.y, gridMin.z);
    glUniform1i(glGetUniformLocation(buildGridShader, "numParticles"), int(particles.size()));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cellHeadSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, particleNextSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, particleCellSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, cellKeySSBO);
    glDispatchCompute((particles.size() + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    if (param_enableSort) RadixSortByCell();

    // 3) SPH step
    glUseProgram(sphGridShader);
    glUniform3i(glGetUniformLocation(sphGridShader, "gridSize"), gridSizeX, gridSizeY, gridSizeZ);
    glUniform1f(glGetUniformLocation(sphGridShader, "cellSize"), cellSize);
    glUniform1i(glGetUniformLocation(sphGridShader, "numParticles"), int(particles.size()));
    glUniform1f(glGetUniformLocation(sphGridShader, "timeStep"), timeStep);
    glUniform1f(glGetUniformLocation(sphGridShader, "h"), h);
    glUniform1f(glGetUniformLocation(sphGridShader, "mass"), mass);
    glUniform1f(glGetUniformLocation(sphGridShader, "restDensity"), restDensity);
    glUniform1f(glGetUniformLocation(sphGridShader, "gasConstant"), gasConstant);
    glUniform1f(glGetUniformLocation(sphGridShader, "viscosity"), viscosity);
    glUniform3f(glGetUniformLocation(sphGridShader, "gravity"), gravity.x, gravity.y, gravity.z);
    glUniform1f(glGetUniformLocation(sphGridShader, "surfaceTension"), surfaceTension);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cellHeadSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, particleNextSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, particleCellSSBO);
    glDispatchCompute((particles.size() + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // 4) OBB constraints (use cached rotation)
    UpdateCachedBoxIfNeeded();
    glUseProgram(obbConstraintShader);
    glUniformMatrix3fv(glGetUniformLocation(obbConstraintShader, "uBoxRot"), 1, GL_FALSE, cachedRot3x3);
    glUniform3f(glGetUniformLocation(obbConstraintShader, "uBoxCenter"),
        param_boxCenter.x, param_boxCenter.y, param_boxCenter.z);
    glUniform3f(glGetUniformLocation(obbConstraintShader, "uBoxHalf"),
        param_boxHalf.x, param_boxHalf.y, param_boxHalf.z);
    glUniform1f(glGetUniformLocation(obbConstraintShader, "uRestitution"), param_wallRestitution);
    glUniform1f(glGetUniformLocation(obbConstraintShader, "uFriction"), param_wallFriction);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glDispatchCompute((particles.size() + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glUseProgram(0);
}

void SPHFluidGPU::ApplyWaveImpulseGPU(float amplitude, float wavelength, float phase,
    const Vec3& dir, float yMin, float yMax)
{
    if (amplitude == 0.0f || wavelength <= 1e-6f) return;
    glUseProgram(waveImpulseShader);
    glUniform1f(glGetUniformLocation(waveImpulseShader, "amplitude"), amplitude);
    glUniform1f(glGetUniformLocation(waveImpulseShader, "wavelength"), wavelength);
    glUniform1f(glGetUniformLocation(waveImpulseShader, "phase"), phase);
    glUniform3f(glGetUniformLocation(waveImpulseShader, "dir"), dir.x, dir.y, dir.z);
    glUniform1f(glGetUniformLocation(waveImpulseShader, "yMin"), yMin);
    glUniform1f(glGetUniformLocation(waveImpulseShader, "yMax"), yMax);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glDispatchCompute((particles.size() + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT); // velocities used next SPH step
    glUseProgram(0);
}

GLuint SPHFluidGPU::GetFluidVBO()  const { return fluidVBO; }
size_t SPHFluidGPU::GetNumFluids() const { return numFluids; }

GLuint SPHFluidGPU::LoadComputeShader(const char* filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) { Debug::FatalError("Failed to open compute shader file", __FILE__, __LINE__); }
    std::stringstream ss; ss << file.rdbuf();
    std::string sourceStr = ss.str();
    const char* source = sourceStr.c_str();

    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint ok = 0; glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint maxLen = 0; glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLen);
        std::string log(maxLen, ' '); glGetShaderInfoLog(shader, maxLen, &maxLen, &log[0]);
        glDeleteShader(shader);
        Debug::FatalError("Compute shader compilation failed:\n" + log, __FILE__, __LINE__);
    }
    GLuint prog = glCreateProgram();
    glAttachShader(prog, shader);
    glLinkProgram(prog);
    glDeleteShader(shader);
    GLint linked = 0; glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    if (!linked) {
        GLint maxLen = 0; glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &maxLen);
        std::string log(maxLen, ' '); glGetProgramInfoLog(prog, maxLen, &maxLen, &log[0]);
        glDeleteProgram(prog);
        Debug::FatalError("Compute shader link failed:\n" + log, __FILE__, __LINE__);
    }
    return prog;
}

void SPHFluidGPU::BuildGhostGrids() {
    float spacing = param_h * 0.85f;
    int cells = int(ceil((2.0f * box) / spacing)) + 1;

    ghostXNeg.init(-box, -box, spacing, cells, cells);
    ghostXPos.init(-box, -box, spacing, cells, cells);
    ghostYNeg.init(-box, -box, spacing, cells, cells);
    ghostYPos.init(-box, -box, spacing, cells, cells);
    ghostZNeg.init(-box, -box, spacing, cells, cells);
    ghostZPos.init(-box, -box, spacing, cells, cells);

    for (size_t i = 0; i < particles.size(); ++i) {
        const SPHParticle& p = particles[i];
        if (!p.isGhost) continue;
        if (fabs(p.pos.x + box) < 1e-4f) ghostXNeg.addGhost(p.pos.y, p.pos.z, i);
        if (fabs(p.pos.x - box) < 1e-4f) ghostXPos.addGhost(p.pos.y, p.pos.z, i);
        if (fabs(p.pos.y + box) < 1e-4f) ghostYNeg.addGhost(p.pos.x, p.pos.z, i);
        if (fabs(p.pos.y - box) < 1e-4f) ghostYPos.addGhost(p.pos.x, p.pos.z, i);
        if (fabs(p.pos.z + box) < 1e-4f) ghostZNeg.addGhost(p.pos.x, p.pos.y, i);
        if (fabs(p.pos.z - box) < 1e-4f) ghostZPos.addGhost(p.pos.x, p.pos.y, i);
    }
}

// --- kept for completeness (not used when param_enableGhosts=false)
void SPHFluidGPU::ActivateClosestGhost(float, float, float) {}
void SPHFluidGPU::UpdateGhostParticlesDynamic(float) {}
void SPHFluidGPU::UploadGhostActivityToGPU() {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(SPHParticle) * particles.size(), particles.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}
