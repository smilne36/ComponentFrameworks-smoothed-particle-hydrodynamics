#include "SPHFluid3D.h"
#include "Debug.h"
#include <fstream>
#include <sstream>
#include <random>
#include <vector>
#include <iostream>
#include <cmath>
#include <SDL_stdinc.h>



static void MakeRotationMat3XYZ(float rxDeg, float ryDeg, float rzDeg, float outM[9]) {
    const float rx = rxDeg * float(M_PI / 180.0);
    const float ry = ryDeg * float(M_PI / 180.0);
    const float rz = rzDeg * float(M_PI / 180.0);
    const float cx = cosf(rx), sx = sinf(rx);
    const float cy = cosf(ry), sy = sinf(ry);
    const float cz = cosf(rz), sz = sinf(rz);
    // Column-major mat3 of Rz*Ry*Rx
    const float Rz[9] = { cz, sz, 0, -sz, cz, 0, 0, 0, 1 };
    const float Ry[9] = { cy, 0, -sy, 0, 1, 0, sy, 0, cy };
    const float Rx[9] = { 1, 0, 0, 0, cx, sx, 0, -sx, cx };
    auto mul3 = [](const float A[9], const float B[9], float C[9]) {
        // C = A*B (column-major)
        for (int c = 0; c < 3; c++) {
            for (int r = 0; r < 3; r++) {
                C[c * 3 + r] = A[0 * 3 + r] * B[c * 3 + 0] + A[1 * 3 + r] * B[c * 3 + 1] + A[2 * 3 + r] * B[c * 3 + 2];
            }
        }
        };
    float Rzy[9]; mul3(Rz, Ry, Rzy);
    mul3(Rzy, Rx, outM);
}



SPHFluidGPU::SPHFluidGPU(size_t numParticles_)
    : numParticles(numParticles_), fluidVBO(0), vboPtr(nullptr), numFluids(0)
{
    clearGridShader = LoadComputeShader("shaders/ClearGrid.comp");
    buildGridShader = LoadComputeShader("shaders/BuildGrid.comp");
    sphGridShader = LoadComputeShader("shaders/SPHFluid.comp");
    radixSortShader = LoadComputeShader("shaders/RadixSort.comp");
    obbConstraintShader = LoadComputeShader("shaders/OBBConstraints.comp");

    // 1) Build particles (also sets 'box')
    InitializeParticles();

    // 2) Now size grid/aux SSBOs using the actual particle count
    InitializeGridAndBuffers();

    // 3) GPU resources
    UploadDataToGPU();
    InitializeSortBuffers();
    InitializeFluidVBO();
}

SPHFluidGPU::~SPHFluidGPU() {
    if (fluidVBO) {
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDeleteBuffers(1, &fluidVBO);
    }
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
}

void SPHFluidGPU::InitializeParticles() {

    this->box = std::max(param_boxHalf.x, std::max(param_boxHalf.y, param_boxHalf.z));

    float h = param_h;                      // use current smoothing length
    float spacing = h * 0.85f;
    float box = this->box;

    param_mass = param_restDensity * spacing * spacing * spacing;

    particles.clear();
    int count = 0;

    float fillFraction = 0.4f;
    float fluidHeight = (2.0f * box) * fillFraction;
    int fluidLayersY = int(fluidHeight / spacing);
    int fluidSide = int((box * 1.7f) / spacing);

    std::default_random_engine rng(static_cast<unsigned>(time(nullptr)));
    std::uniform_real_distribution<float> jitterDist(-spacing * param_jitterAmp,
        +spacing * param_jitterAmp);

    auto j = [&]() -> float { return param_useJitter ? jitterDist(rng) : 0.0f; };

    // Fill a block at the bottom of the box
    for (int x = 0; x < fluidSide && count < numParticles; ++x) {
        for (int y = 0; y < fluidLayersY && count < numParticles; ++y) {
            for (int z = 0; z < fluidSide && count < numParticles; ++z) {
                SPHParticle p;
                p.pos = Vec4(
                    -box * 0.7f + x * spacing + j(),
                    -box + spacing + y * spacing + j(), // Start at bottom
                    -box * 0.7f + z * spacing + j(),
                    0.0f
                );
                p.vel = Vec4(0, 0, 0, 0);
                p.acc = Vec4(0, 0, 0, 0);
                p.density = 0.0f;
                p.pressure = 0.0f;
                p.isGhost = 0;
                p.isActive = 0;
                p.pad0 = 0.0f;
                particles.push_back(p);
                ++count;
            }
        }
    }

    // Ghosts on all boundaries (use spacing from param_h)
    auto add_ghost = [&](float x, float y, float z) {
        SPHParticle p;
        p.pos = Vec4(x, y, z, 0.0f);
        p.vel = Vec4(0, 0, 0, 0);
        p.acc = Vec4(0, 0, 0, 0);
        p.density = 0.0f;
        p.pressure = 0.0f;
        p.isGhost = 1;
        p.isActive = 0;
        p.pad0 = 0.0f;
        particles.push_back(p);
        };
    for (float y = -box; y <= box; y += spacing)
        for (float z = -box; z <= box; z += spacing) { add_ghost(-box, y, z); add_ghost(box, y, z); }
    for (float x = -box; x <= box; x += spacing)
        for (float z = -box; z <= box; z += spacing) { add_ghost(x, -box, z); add_ghost(x, box, z); }
    for (float x = -box; x <= box; x += spacing)
        for (float y = -box; y <= box; y += spacing) { add_ghost(x, y, -box); add_ghost(x, y, box); }

    std::cout << "Fluid particles: " << count << ", ghosts: " << (particles.size() - count) << std::endl;
    BuildGhostGrids();
}

void SPHFluidGPU::InitializeFluidVBO() {
    // Count fluids
    numFluids = 0;
    for (const auto& p : particles)
        if (p.isGhost == 0) ++numFluids;

    if (fluidVBO) glDeleteBuffers(1, &fluidVBO);

    glGenBuffers(1, &fluidVBO);
    glBindBuffer(GL_ARRAY_BUFFER, fluidVBO);
    glBufferStorage(GL_ARRAY_BUFFER, sizeof(float) * 4 * numFluids, nullptr,
        GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);

    vboPtr = (float*)glMapBufferRange(
        GL_ARRAY_BUFFER, 0, sizeof(float) * 4 * numFluids,
        GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT
    );
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void SPHFluidGPU::UpdateFluidVBOFromGPU() {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    const SPHParticle* gpuParticles = (const SPHParticle*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);

    if (gpuParticles && vboPtr) {
        size_t vboIndex = 0;
        for (size_t i = 0; i < particles.size(); ++i) {
            if (gpuParticles[i].isGhost == 0) {
                vboPtr[4 * vboIndex + 0] = gpuParticles[i].pos.x;
                vboPtr[4 * vboIndex + 1] = gpuParticles[i].pos.y;
                vboPtr[4 * vboIndex + 2] = gpuParticles[i].pos.z;
                vboPtr[4 * vboIndex + 3] = 1.0f;
                ++vboIndex;
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
    gridSizeX = std::max(1, int(std::ceil((2.0f * half.x) / cellSize)));
    gridSizeY = std::max(1, int(std::ceil((2.0f * half.y) / cellSize)));
    gridSizeZ = std::max(1, int(std::ceil((2.0f * half.z) / cellSize)));
    numCells = std::max(1, gridSizeX * gridSizeY * gridSizeZ);

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


void SPHFluidGPU::RecreateGridForBox() {
    InitializeGridAndBuffers();
}


// --- NEW: Initialize sorting buffers
void SPHFluidGPU::InitializeSortBuffers() {
    size_t N = particles.size();

    glGenBuffers(1, &cellKeySSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cellKeySSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int) * N, nullptr, GL_DYNAMIC_COPY);

    glGenBuffers(1, &sortIdxSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sortIdxSSBO);
    std::vector<int> idxInit(N);
    for (size_t i = 0; i < N; ++i) idxInit[i] = int(i);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int) * N, idxInit.data(), GL_DYNAMIC_COPY);

    glGenBuffers(1, &sortTmpSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sortTmpSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int) * N, nullptr, GL_DYNAMIC_COPY);
}

void SPHFluidGPU::RadixSortByCell() {
    size_t N = particles.size();
    // Assume cellKeys[] has been filled on GPU by BuildGrid.comp
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

void SPHFluidGPU::ActivateClosestGhost(float x, float y, float z) {
    // Determine which face is closest  
    float dists[6] = { fabs(x + box), fabs(x - box), fabs(y + box), fabs(y - box), fabs(z + box), fabs(z - box) };
    int face = int(std::min_element(dists, dists + 6) - dists);

    const std::vector<size_t>* candidates = nullptr;
    float a = 0, b = 0;
    switch (face) {
    case 0: candidates = &ghostXNeg.getCell(y, z); a = y; b = z; break;
    case 1: candidates = &ghostXPos.getCell(y, z); a = y; b = z; break;
    case 2: candidates = &ghostYNeg.getCell(x, z); a = x; b = z; break;
    case 3: candidates = &ghostYPos.getCell(x, z); a = x; b = z; break;
    case 4: candidates = &ghostZNeg.getCell(x, y); a = x; b = y; break;
    case 5: candidates = &ghostZPos.getCell(x, y); a = x; b = y; break;
    }

    if (candidates == nullptr || candidates->empty()) {
        return;
    }

    float minDist2 = std::numeric_limits<float>::max();
    size_t minIdx = size_t(-1);
    for (size_t idx : *candidates) {
        const SPHParticle& ghost = particles[idx];
        float dx = ghost.pos.x - x;
        float dy = ghost.pos.y - y;
        float dz = ghost.pos.z - z;
        float dist2 = dx * dx + dy * dy + dz * dz;
        if (dist2 < minDist2) {
            minDist2 = dist2;
            minIdx = idx;
        }
    }
    if (minIdx != size_t(-1)) {
        particles[minIdx].isActive = 1;
    }
}

void SPHFluidGPU::UpdateGhostParticlesDynamic(float h) {
    // Deactivate all ghosts
    for (auto& p : particles) {
        if (p.isGhost) p.isActive = 0;
    }

    // For each fluid, check proximity to wall; activate nearest ghost(s) as needed
    for (const auto& fluid : particles) {
        if (fluid.isGhost) continue;

        // x-faces
        if (fluid.pos.x < -box + h) ActivateClosestGhost(-box, fluid.pos.y, fluid.pos.z);
        if (fluid.pos.x > box - h)  ActivateClosestGhost(box, fluid.pos.y, fluid.pos.z);

        // y-faces
        if (fluid.pos.y < -box + h) ActivateClosestGhost(fluid.pos.x, -box, fluid.pos.z);
        if (fluid.pos.y > box - h)  ActivateClosestGhost(fluid.pos.x, box, fluid.pos.z);

        // z-faces
        if (fluid.pos.z < -box + h) ActivateClosestGhost(fluid.pos.x, fluid.pos.y, -box);
        if (fluid.pos.z > box - h)  ActivateClosestGhost(fluid.pos.x, fluid.pos.y, box);
    }
}

void SPHFluidGPU::UploadGhostActivityToGPU() {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
        sizeof(SPHParticle) * particles.size(), particles.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void SPHFluidGPU::UploadDataToGPU() {
    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    const GLsizeiptr sz = sizeof(SPHParticle) * particles.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, sz, particles.data(), GL_DYNAMIC_COPY);

    GLint64 gpuSize = 0;
    glGetBufferParameteri64v(GL_SHADER_STORAGE_BUFFER, GL_BUFFER_SIZE, &gpuSize);
    if (gpuSize != sz) std::cerr << "SSBO size mismatch: " << gpuSize << " vs " << sz << "\n";

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    std::cout << "sizeof(SPHParticle)=" << sizeof(SPHParticle) << " bytes\n"; // should be 80
}

void SPHFluidGPU::DownloadDataFromGPU() {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    const SPHParticle* gpuParticles =
        (const SPHParticle*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    if (gpuParticles) {
        memcpy(particles.data(), gpuParticles,
            sizeof(SPHParticle) * particles.size());
    }
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void SPHFluidGPU::DispatchCompute(float overrideDt) {
    if (param_pause) return;

    // Live parameters
    float h = param_h;
    float mass = param_mass;
    float restDensity = param_restDensity;
    float gasConstant = param_gasConstant;
    float viscosity = param_viscosity;
    MATH::Vec3 gravity = MATH::Vec3(0, param_gravityY, 0);
    float surfaceTension = param_surfaceTension;

    // Use override dt if provided, else the configured param_timeStep
    float timeStep = (overrideDt > 0.0f ? overrideDt : param_timeStep);

    // Optional ghosts sync
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

    if (param_enableSort) {
        RadixSortByCell();
    }

    // 4) SPH step
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
    glUniform1f(glGetUniformLocation(sphGridShader, "box"), box);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cellHeadSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, particleNextSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, particleCellSSBO);
    glDispatchCompute((particles.size() + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // 5) OBB constraints
    glUseProgram(obbConstraintShader);
    float R[9]; MakeRotationMat3XYZ(param_boxEulerDeg.x, param_boxEulerDeg.y, param_boxEulerDeg.z, R);
    glUniformMatrix3fv(glGetUniformLocation(obbConstraintShader, "uBoxRot"), 1, GL_FALSE, R);
    glUniform3f(glGetUniformLocation(obbConstraintShader, "uBoxCenter"), param_boxCenter.x, param_boxCenter.y, param_boxCenter.z);
    glUniform3f(glGetUniformLocation(obbConstraintShader, "uBoxHalf"), param_boxHalf.x, param_boxHalf.y, param_boxHalf.z);
    glUniform1f(glGetUniformLocation(obbConstraintShader, "uRestitution"), param_wallRestitution);
    glUniform1f(glGetUniformLocation(obbConstraintShader, "uFriction"), param_wallFriction);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glDispatchCompute((particles.size() + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glUseProgram(0);
}

GLuint SPHFluidGPU::GetFluidVBO() const { return fluidVBO; }
size_t SPHFluidGPU::GetNumFluids() const { return numFluids; }

GLuint SPHFluidGPU::LoadComputeShader(const char* filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        Debug::FatalError("Failed to open compute shader file", __FILE__, __LINE__);
    }
    std::stringstream ss;
    ss << file.rdbuf();
    std::string sourceStr = ss.str();
    const char* source = sourceStr.c_str();

    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint isCompiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled);
    if (!isCompiled) {
        GLint maxLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);
        std::string errorLog(maxLength, ' ');
        glGetShaderInfoLog(shader, maxLength, &maxLength, &errorLog[0]);
        glDeleteShader(shader);
        Debug::FatalError("Compute shader compilation failed:\n" + errorLog, __FILE__, __LINE__);
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    glDeleteShader(shader);

    GLint isLinked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &isLinked);
    if (!isLinked) {
        GLint maxLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);
        std::string errorLog(maxLength, ' ');
        glGetProgramInfoLog(program, maxLength, &maxLength, &errorLog[0]);
        glDeleteProgram(program);
        Debug::FatalError("Compute shader link failed:\n" + errorLog, __FILE__, __LINE__);
    }

    return program;
}

void SPHFluidGPU::BuildGhostGrids() {
    float spacing = std::max(1e-6f, param_h * 0.85f);
    int cells = std::max(1, int(std::ceil((2.0f * box) / spacing)) + 1);

    // X faces: grid in (y, z)
    ghostXNeg.init(-box, -box, spacing, cells, cells);
    ghostXPos.init(-box, -box, spacing, cells, cells);
    // Y faces: grid in (x, z)
    ghostYNeg.init(-box, -box, spacing, cells, cells);
    ghostYPos.init(-box, -box, spacing, cells, cells);
    // Z faces: grid in (x, y)
    ghostZNeg.init(-box, -box, spacing, cells, cells);
    ghostZPos.init(-box, -box, spacing, cells, cells);

    for (size_t i = 0; i < particles.size(); ++i) {
        const SPHParticle& p = particles[i];
        if (!p.isGhost) continue;
        if (fabsf(p.pos.x + box) < 1e-4f) ghostXNeg.addGhost(p.pos.y, p.pos.z, i);
        if (fabsf(p.pos.x - box) < 1e-4f) ghostXPos.addGhost(p.pos.y, p.pos.z, i);
        if (fabsf(p.pos.y + box) < 1e-4f) ghostYNeg.addGhost(p.pos.x, p.pos.z, i);
        if (fabsf(p.pos.y - box) < 1e-4f) ghostYPos.addGhost(p.pos.x, p.pos.z, i);
        if (fabsf(p.pos.z + box) < 1e-4f) ghostZNeg.addGhost(p.pos.x, p.pos.y, i);
        if (fabsf(p.pos.z - box) < 1e-4f) ghostZPos.addGhost(p.pos.x, p.pos.y, i);
    }
}

// NEW: sine-wave velocity kick
void SPHFluidGPU::ApplyWaveImpulse(float amplitude, float wavelength, float phase, const Vec3& dir,
    float yMin, float yMax)
{
    if (amplitude == 0.0f || wavelength <= 1e-6f) return;

    // Bring CPU side in sync, modify CPU buffer, push back to GPU.
    DownloadDataFromGPU();

    const float k = 2.0f * float(M_PI) / wavelength;
    Vec3 nDir = dir;
    float len = sqrtf(nDir.x * nDir.x + nDir.y * nDir.y + nDir.z * nDir.z);
    if (len > 0.0f) { nDir /= len; }
    else { nDir = Vec3(0, 1, 0); }

    for (auto& p : particles) {
        if (p.isGhost) continue;
        if (p.pos.y < yMin || p.pos.y > yMax) continue;

        // Phase using x+z to create ripples across the surface
        float theta = k * (p.pos.x + p.pos.z) + phase;
        float kick = amplitude * sinf(theta);

        p.vel.x += nDir.x * kick;
        p.vel.y += nDir.y * kick;
        p.vel.z += nDir.z * kick;
    }

    // Upload updated particle velocities
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(SPHParticle) * particles.size(), particles.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

// NEW: full reset of particle set and buffers
void SPHFluidGPU::ResetSimulation()
{
    // Destroy GPU buffers we own
    if (fluidVBO) { glBindBuffer(GL_ARRAY_BUFFER, 0); glDeleteBuffers(1, &fluidVBO); fluidVBO = 0; vboPtr = nullptr; }
    if (ssbo) { glDeleteBuffers(1, &ssbo); ssbo = 0; }
    if (cellHeadSSBO) { glDeleteBuffers(1, &cellHeadSSBO); cellHeadSSBO = 0; }
    if (particleNextSSBO) { glDeleteBuffers(1, &particleNextSSBO); particleNextSSBO = 0; }
    if (particleCellSSBO) { glDeleteBuffers(1, &particleCellSSBO); particleCellSSBO = 0; }
    if (cellKeySSBO) { glDeleteBuffers(1, &cellKeySSBO); cellKeySSBO = 0; }
    if (sortIdxSSBO) { glDeleteBuffers(1, &sortIdxSSBO); sortIdxSSBO = 0; }
    if (sortTmpSSBO) { glDeleteBuffers(1, &sortTmpSSBO); sortTmpSSBO = 0; }

    // Rebuild in the same order as the ctor to keep sizes consistent
    InitializeParticles();          // fills 'particles', sets 'box'
    InitializeGridAndBuffers();     // sizes per-cell/per-particle SSBOs from particles.size()
    UploadDataToGPU();              // alloc+upload main SSBO (binding=0)
    InitializeSortBuffers();        // alloc sort buffers using particles.size()
    InitializeFluidVBO();           // alloc fluid-only VBO (numFluids)

    // Optional: print sizes for quick sanity
    std::cout << "Reset: particles=" << particles.size()
        << " fluids=" << numFluids
        << " grid=" << gridSizeX << "x" << gridSizeY << "x" << gridSizeZ
        << " cells=" << numCells << std::endl;
}