#include "SPHFluid3D.h"
#include "Debug.h"
#include <fstream>
#include <sstream>
#include <random>
#include <vector>
#include <iostream>
#include <cmath>
#include <limits>
#include <SDL_stdinc.h>
#include <algorithm>

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

SPHFluidGPU::SPHFluidGPU(size_t numParticles_)
    : numParticles(numParticles_), fluidVBO(0), vboPtr(nullptr), numFluids(0)
{
    clearGridShader = LoadComputeShader("shaders/ClearGrid.comp");
    buildGridShader = LoadComputeShader("shaders/BuildGrid.comp");
    sphGridShader = LoadComputeShader("shaders/SPHFluid.comp");
    radixSortShader = LoadComputeShader("shaders/RadixSort.comp");
    obbConstraintShader = LoadComputeShader("shaders/OBBConstraints.comp");
    waveImpulseShader = LoadComputeShader("shaders/WaveImpulse.comp");
    terrainConstraintShader  = LoadComputeShader("shaders/TerrainConstraints.comp");
    streamEmitShader         = LoadComputeShader("shaders/StreamEmit.comp");
    channelConstraintShader  = LoadComputeShader("shaders/ChannelConstraint.comp");

    // 1) Build particle array (also sets 'box' from OBB half)
    InitializeParticles();

    // 2) Size grid/aux SSBOs from particles.size()
    InitializeGridAndBuffers();

    // 3) Upload particles SSBO + sort buffers + VBO
    UploadDataToGPU();
    InitializeSortBuffers();
    InitializeFluidVBO();
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
    if (terrainConstraintShader)  glDeleteProgram(terrainConstraintShader);
    if (streamEmitShader)         glDeleteProgram(streamEmitShader);
    if (channelConstraintShader)  glDeleteProgram(channelConstraintShader);
    if (terrainSSBO)              glDeleteBuffers(1, &terrainSSBO);
}

void SPHFluidGPU::InitializeParticles() {
    this->box = std::max(param_boxHalf.x, std::max(param_boxHalf.y, param_boxHalf.z));

    float h = param_h;
    float spacing = h * 0.85f;
    float box = this->box;

    param_mass = param_restDensity * spacing * spacing * spacing;

    particles.clear();
    int count = 0;

    float fillFraction = 0.4f;
    float fluidHeight = (2.0f * box) * fillFraction;
    int   fluidLayersY = int(fluidHeight / spacing);
    int   fluidSide = int((box * 1.7f) / spacing);

    std::default_random_engine rng(static_cast<unsigned>(time(nullptr)));
    std::uniform_real_distribution<float> jitterDist(-spacing * param_jitterAmp,
        +spacing * param_jitterAmp);
    auto j = [&]() -> float { return param_useJitter ? jitterDist(rng) : 0.0f; };

    if (riverMode && !terrainHeights.empty()) {
        // Distribute particles along the river channel
        float xMin  = terrainWorldMinX;
        float zMin  = terrainWorldMinZ;
        float xSize = terrainWorldSizeX;
        float zSize = terrainWorldSizeZ;
        float yBase = param_boxCenter.y - param_boxHalf.y;

        // Sample terrain height at (wx, wz)
        auto sampleH = [&](float wx, float wz) -> float {
            float u = (wx - xMin) / xSize * float(terrainW - 1);
            float v = (wz - zMin) / zSize * float(terrainH - 1);
            u = std::max(0.0f, std::min(float(terrainW - 2), u));
            v = std::max(0.0f, std::min(float(terrainH - 2), v));
            int ix = int(u), iz = int(v);
            float fx = u - ix, fz = v - iz;
            int W = terrainW;
            float h00 = terrainHeights[ ix      +  iz      * W];
            float h10 = terrainHeights[(ix + 1) +  iz      * W];
            float h01 = terrainHeights[ ix      + (iz + 1) * W];
            float h11 = terrainHeights[(ix + 1) + (iz + 1) * W];
            return h00*(1-fx)*(1-fz) + h10*fx*(1-fz) + h01*(1-fx)*fz + h11*fx*fz;
        };

        // Walk along the channel and pack particles in
        for (float wz = zMin + spacing; wz < zMin + zSize - spacing && count < (int)numParticles; wz += spacing) {
            float centerX = param_boxCenter.x + riverAmp * std::sinf(riverFreq * wz + riverPhase);
            for (float wx = centerX - riverChannelWidth; wx <= centerX + riverChannelWidth && count < (int)numParticles; wx += spacing) {
                float ty = sampleH(wx, wz);
                for (float wy = ty + spacing; wy <= ty + 2.5f && count < (int)numParticles; wy += spacing) {
                    SPHParticle p{};
                    p.pos = Vec4(wx + j(), wy + j(), wz + j(), 0.0f);
                    p.vel = Vec4(0, 0, 0.5f, 0); // channel constraint drives flow
                    p.acc = Vec4(0, 0, 0, 0);
                    p.density = p.pressure = 0.0f;
                    p.isGhost = 0; p.isActive = 0; p.pad0 = 0;
                    particles.push_back(p); ++count;
                }
            }
        }
        // Fill remaining particle slots at emitter if channel wasn't enough
        while (count < (int)numParticles) {
            std::uniform_real_distribution<float> rx(-riverChannelWidth * 0.5f, riverChannelWidth * 0.5f);
            std::uniform_real_distribution<float> ry(0.0f, 1.5f);
            float wx = riverEmitterPos.x + rx(rng);
            float wz = riverEmitterPos.z + rx(rng);
            float ty = sampleH(wx, wz);
            SPHParticle p{};
            p.pos = Vec4(wx, ty + ry(rng), wz, 0.0f);
            p.vel = Vec4(0, 0, 2.0f, 0);
            p.acc = Vec4(0, 0, 0, 0);
            p.density = p.pressure = 0.0f;
            p.isGhost = 0; p.isActive = 0; p.pad0 = 0;
            particles.push_back(p); ++count;
        }
    } else {
        // Standard box fill
        for (int x = 0; x < fluidSide && count < (int)numParticles; ++x)
            for (int y = 0; y < fluidLayersY && count < (int)numParticles; ++y)
                for (int z = 0; z < fluidSide && count < (int)numParticles; ++z) {
                    SPHParticle p{};
                    p.pos = Vec4(-box * 0.7f + x * spacing + j(),
                        -box + spacing + y * spacing + j(),
                        -box * 0.7f + z * spacing + j(), 0.0f);
                    p.vel = Vec4(0, 0, 0, 0);
                    p.acc = Vec4(0, 0, 0, 0);
                    p.density = p.pressure = 0.0f;
                    p.isGhost = 0; p.isActive = 0; p.pad0 = 0;
                    particles.push_back(p); ++count;
                }
    }

    // Create axis-aligned ghost shell (used if param_enableGhosts is true)
    auto add_ghost = [&](float x, float y, float z) {
        SPHParticle p{};
        p.pos = Vec4(x, y, z, 0); p.vel = Vec4(0, 0, 0, 0); p.acc = Vec4(0, 0, 0, 0);
        p.density = p.pressure = 0.0f; p.isGhost = 1; p.isActive = 0; p.pad0 = 0;
        particles.push_back(p);
        };
    for (float y = -box; y <= box; y += spacing) for (float z = -box; z <= box; z += spacing) { add_ghost(-box, y, z); add_ghost(box, y, z); }
    for (float x = -box; x <= box; x += spacing) for (float z = -box; z <= box; z += spacing) { add_ghost(x, -box, z); add_ghost(x, box, z); }
    for (float x = -box; x <= box; x += spacing) for (float y = -box; y <= box; y += spacing) { add_ghost(x, y, -box); add_ghost(x, y, box); }

    std::cout << "Fluid particles: " << count
        << ", ghosts: " << (particles.size() - count) << std::endl;
    BuildGhostGrids(); 

    
}

void SPHFluidGPU::BuildGhostGrids() {
    // Build 2D uniform grids on each AABB face (-X,+X,-Y,+Y,-Z,+Z)
    // Faces are still axis-aligned even if an OBB is used for constraints (ghost shell made AABB).
    const float hx = param_boxHalf.x;
    const float hy = param_boxHalf.y;
    const float hz = param_boxHalf.z;
    const float faceCell = param_h; // use smoothing length for ghost grid resolution

    auto makeDim = [&](float extent) -> int {
        int c = int(std::ceil((2.0f * extent) / faceCell));
        return std::max(1, c);
        };

    int cellsY = makeDim(hy);
    int cellsZ = makeDim(hz);
    int cellsX = makeDim(hx);

    ghostXNeg.init(-hy, -hz, faceCell, cellsY, cellsZ); // a=y, b=z
    ghostXPos.init(-hy, -hz, faceCell, cellsY, cellsZ);
    ghostYNeg.init(-hx, -hz, faceCell, cellsX, cellsZ); // a=x, b=z
    ghostYPos.init(-hx, -hz, faceCell, cellsX, cellsZ);
    ghostZNeg.init(-hx, -hy, faceCell, cellsX, cellsY); // a=x, b=y
    ghostZPos.init(-hx, -hy, faceCell, cellsX, cellsY);

    const float eps = faceCell * 0.25f;

    for (size_t i = 0; i < particles.size(); ++i) {
        const SPHParticle& p = particles[i];
        if (!p.isGhost) continue;

        const float x = p.pos.x;
        const float y = p.pos.y;
        const float z = p.pos.z;

        // Face tests (AABB shell built with 'box' == max half but we rely on param_boxHalf per axis)
        if (std::fabs(x + hx) < eps)      ghostXNeg.addGhost(y, z, i);
        else if (std::fabs(x - hx) < eps) ghostXPos.addGhost(y, z, i);

        if (std::fabs(y + hy) < eps)      ghostYNeg.addGhost(x, z, i);
        else if (std::fabs(y - hy) < eps) ghostYPos.addGhost(x, z, i);

        if (std::fabs(z + hz) < eps)      ghostZNeg.addGhost(x, y, i);
        else if (std::fabs(z - hz) < eps) ghostZPos.addGhost(x, y, i);
    }

    std::cout << "Ghost grids built: "
        << " X(-/+) cells=" << ghostXNeg.cellsA << "x" << ghostXNeg.cellsB
        << " Y(-/+) cells=" << ghostYNeg.cellsA << "x" << ghostYNeg.cellsB
        << " Z(-/+) cells=" << ghostZNeg.cellsA << "x" << ghostZNeg.cellsB
        << std::endl;
}

/* Optional simple usage helpers (keep lightweight) */

void SPHFluidGPU::ActivateClosestGhost(float x, float y, float z) {
    // Pick nearest face to (x,y,z), then search that face grid cell for closest ghost
    float dxNeg = std::fabs(x + param_boxHalf.x);
    float dxPos = std::fabs(x - param_boxHalf.x);
    float dyNeg = std::fabs(y + param_boxHalf.y);
    float dyPos = std::fabs(y - param_boxHalf.y);
    float dzNeg = std::fabs(z + param_boxHalf.z);
    float dzPos = std::fabs(z - param_boxHalf.z);

    enum Face { XN, XP, YN, YP, ZN, ZP };
    float dArr[6] = { dxNeg, dxPos, dyNeg, dyPos, dzNeg, dzPos };
    Face best = XN;
    float bestD = dArr[0];
    for (int i = 1; i < 6; ++i) if (dArr[i] < bestD) { bestD = dArr[i]; best = (Face)i; }

    const std::vector<size_t>* candidates = nullptr;
    switch (best) {
    case XN: candidates = &ghostXNeg.getCell(y, z); break;
    case XP: candidates = &ghostXPos.getCell(y, z); break;
    case YN: candidates = &ghostYNeg.getCell(x, z); break;
    case YP: candidates = &ghostYPos.getCell(x, z); break;
    case ZN: candidates = &ghostZNeg.getCell(x, y); break;
    case ZP: candidates = &ghostZPos.getCell(x, y); break;
    }
    if (!candidates || candidates->empty()) return;

    size_t closestIdx = (*candidates)[0];
    float closestDist2 = FLT_MAX;
    for (size_t idx : *candidates) {
        const SPHParticle& gp = particles[idx];
        float dx = gp.pos.x - x;
        float dy = gp.pos.y - y;
        float dz = gp.pos.z - z;
        float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < closestDist2) { closestDist2 = d2; closestIdx = idx; }
    }
    particles[closestIdx].isActive = 1;
}

void SPHFluidGPU::UpdateGhostParticlesDynamic(float h) {
    // Simple decay: deactivate ghosts set active last frame (placeholder)
    for (auto& p : particles) if (p.isGhost) p.isActive = 0;
    (void)h;
}

void SPHFluidGPU::UploadGhostActivityToGPU() {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    SPHParticle* gpuParticles = (SPHParticle*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_WRITE);
    if (gpuParticles) {
        for (size_t i = 0; i < particles.size(); ++i)
            gpuParticles[i].isActive = particles[i].isActive;
    }
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
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

void SPHFluidGPU::RecreateGridForBox() { InitializeGridAndBuffers(); }

// --- sorting buffers
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

void SPHFluidGPU::UploadDataToGPU() {
    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    const GLsizeiptr sz = sizeof(SPHParticle) * particles.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, sz, particles.data(), GL_DYNAMIC_COPY);

    GLint64 gpuSize = 0; glGetBufferParameteri64v(GL_SHADER_STORAGE_BUFFER, GL_BUFFER_SIZE, &gpuSize);
    if (gpuSize != sz) std::cerr << "SSBO size mismatch: " << gpuSize << " vs " << sz << "\n";

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    std::cout << "sizeof(SPHParticle)=" << sizeof(SPHParticle) << " bytes\n";
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

void SPHFluidGPU::DispatchCompute(float overrideDt) {
    if (param_pause) return;

    const float timeStep = (overrideDt > 0.0f ? overrideDt : param_timeStep);

    // Optional ghost syncing (leave off for perf)
    if (param_enableGhosts) {
        DownloadDataFromGPU();
        UpdateGhostParticlesDynamic(param_h);
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
    glUniform1f(glGetUniformLocation(sphGridShader, "h"), param_h);
    glUniform1f(glGetUniformLocation(sphGridShader, "mass"), param_mass);
    glUniform1f(glGetUniformLocation(sphGridShader, "restDensity"), param_restDensity);
    glUniform1f(glGetUniformLocation(sphGridShader, "gasConstant"), param_gasConstant);
    glUniform1f(glGetUniformLocation(sphGridShader, "viscosity"), param_viscosity);
    glUniform3f(glGetUniformLocation(sphGridShader, "gravity"), param_gravityX, param_gravityY, param_gravityZ);
    glUniform1f(glGetUniformLocation(sphGridShader, "surfaceTension"), param_surfaceTension);
    glUniform1f(glGetUniformLocation(sphGridShader, "box"), box);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cellHeadSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, particleNextSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, particleCellSSBO);
    glDispatchCompute((particles.size() + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // 4) OBB constraints
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

    // 5) River mode: terrain collision, channel confinement, particle recycling
    if (riverMode && !terrainHeights.empty()) {
        DispatchTerrainConstraints();
        DispatchChannelConstraint();
        DispatchStreamEmit();
    }

    glUseProgram(0);
}

void SPHFluidGPU::DispatchTerrainConstraints() {
    if (!terrainConstraintShader || !terrainSSBO) return;

    glUseProgram(terrainConstraintShader);
    glUniform1i(glGetUniformLocation(terrainConstraintShader, "numParticles"), int(particles.size()));
    glUniform2i(glGetUniformLocation(terrainConstraintShader, "terrainGrid"),  terrainW, terrainH);
    glUniform2f(glGetUniformLocation(terrainConstraintShader, "terrainMin"),   terrainWorldMinX, terrainWorldMinZ);
    glUniform2f(glGetUniformLocation(terrainConstraintShader, "terrainSize"),  terrainWorldSizeX, terrainWorldSizeZ);
    glUniform1f(glGetUniformLocation(terrainConstraintShader, "terrainRestitution"), 0.02f);
    glUniform1f(glGetUniformLocation(terrainConstraintShader, "terrainFriction"),    0.05f);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, terrainSSBO);
    glDispatchCompute((particles.size() + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void SPHFluidGPU::DispatchChannelConstraint() {
    if (!channelConstraintShader) return;

    glUseProgram(channelConstraintShader);
    glUniform1i(glGetUniformLocation(channelConstraintShader, "numParticles"),  int(particles.size()));
    glUniform1f(glGetUniformLocation(channelConstraintShader, "boxCenterX"),    param_boxCenter.x);
    glUniform1f(glGetUniformLocation(channelConstraintShader, "riverAmp"),      riverAmp);
    glUniform1f(glGetUniformLocation(channelConstraintShader, "riverFreq"),     riverFreq);
    glUniform1f(glGetUniformLocation(channelConstraintShader, "riverPhase"),    riverPhase);
    glUniform1f(glGetUniformLocation(channelConstraintShader, "channelWidth"),  riverChannelWidth);
    glUniform1f(glGetUniformLocation(channelConstraintShader, "flowGravity"),   80.0f);
    glUniform1f(glGetUniformLocation(channelConstraintShader, "timeStep"),      param_timeStep);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glDispatchCompute((particles.size() + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void SPHFluidGPU::DispatchStreamEmit() {
    if (!streamEmitShader) return;

    glUseProgram(streamEmitShader);
    glUniform1i(glGetUniformLocation(streamEmitShader, "numParticles"),  int(particles.size()));
    glUniform1f(glGetUniformLocation(streamEmitShader, "sinkY"),         riverSinkY);
    glUniform1f(glGetUniformLocation(streamEmitShader, "sinkZMax"),      riverSinkZMax);
    glUniform3f(glGetUniformLocation(streamEmitShader, "emitterPos"),    riverEmitterPos.x, riverEmitterPos.y, riverEmitterPos.z);
    glUniform3f(glGetUniformLocation(streamEmitShader, "emitterVel"),    riverEmitterVel.x, riverEmitterVel.y, riverEmitterVel.z);
    glUniform1f(glGetUniformLocation(streamEmitShader, "emitterRadius"),  riverEmitterRadius);
    glUniform1f(glGetUniformLocation(streamEmitShader, "emitterSpreadZ"), riverSinkZMax - riverEmitterPos.z);
    glUniform1f(glGetUniformLocation(streamEmitShader, "restDensity"),    param_restDensity);
    glUniform1f(glGetUniformLocation(streamEmitShader, "boxCenterX"),     param_boxCenter.x);
    glUniform1f(glGetUniformLocation(streamEmitShader, "riverAmp"),       riverAmp);
    glUniform1f(glGetUniformLocation(streamEmitShader, "riverFreq"),      riverFreq);
    glUniform1f(glGetUniformLocation(streamEmitShader, "riverPhase"),     riverPhase);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glDispatchCompute((particles.size() + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

GLuint SPHFluidGPU::GetFluidVBO() const { return fluidVBO; }
size_t SPHFluidGPU::GetNumFluids() const { return numFluids; }

// --- GPU-only WaveImpulse (replaces CPU readback path)
void SPHFluidGPU::ApplyWaveImpulse(float amplitude, float wavelength, float phase, const Vec3& dir,
    float yMin, float yMax)
{
    if (amplitude == 0.0f || wavelength <= 1e-6f) return;

    glUseProgram(waveImpulseShader);
    glUniform1i(glGetUniformLocation(waveImpulseShader, "N"), int(particles.size()));
    glUniform1f(glGetUniformLocation(waveImpulseShader, "amplitude"), amplitude);
    glUniform1f(glGetUniformLocation(waveImpulseShader, "wavelength"), wavelength);
    glUniform1f(glGetUniformLocation(waveImpulseShader, "phase"), phase);
    // normalize dir on GPU (shader can do it), but pass anyway
    glUniform3f(glGetUniformLocation(waveImpulseShader, "dir"), dir.x, dir.y, dir.z);
    glUniform1f(glGetUniformLocation(waveImpulseShader, "yMin"), yMin);
    glUniform1f(glGetUniformLocation(waveImpulseShader, "yMax"), yMax);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glDispatchCompute((particles.size() + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glUseProgram(0);
}

// Full reset (unchanged logic, grouped)
void SPHFluidGPU::ResetSimulation() {
    if (fluidVBO) { glBindBuffer(GL_ARRAY_BUFFER, 0); glDeleteBuffers(1, &fluidVBO); fluidVBO = 0; vboPtr = nullptr; }
    if (ssbo) { glDeleteBuffers(1, &ssbo); ssbo = 0; }
    if (cellHeadSSBO) { glDeleteBuffers(1, &cellHeadSSBO);     cellHeadSSBO = 0; }
    if (particleNextSSBO) { glDeleteBuffers(1, &particleNextSSBO); particleNextSSBO = 0; }
    if (particleCellSSBO) { glDeleteBuffers(1, &particleCellSSBO); particleCellSSBO = 0; }
    if (cellKeySSBO) { glDeleteBuffers(1, &cellKeySSBO);      cellKeySSBO = 0; }
    if (sortIdxSSBO) { glDeleteBuffers(1, &sortIdxSSBO);      sortIdxSSBO = 0; }
    if (sortTmpSSBO) { glDeleteBuffers(1, &sortTmpSSBO);      sortTmpSSBO = 0; }

    InitializeParticles();
    InitializeGridAndBuffers();
    UploadDataToGPU();
    InitializeSortBuffers();
    InitializeFluidVBO();

    std::cout << "Reset: particles=" << particles.size()
        << " fluids=" << numFluids
        << " grid=" << gridSizeX << "x" << gridSizeY << "x" << gridSizeZ
        << " cells=" << numCells << std::endl;
}

void SPHFluidGPU::RadixSortByCell() {
    // Map buffers
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cellKeySSBO);
    const int* keys = (const int*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    if (!keys) { glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); return; }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sortIdxSSBO);
    int* indices = (int*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_WRITE);
    if (!indices) {
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER); // keys
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        return;
    }

    size_t N = particles.size();
    // Build vector of (key,index) then stable sort
    std::vector<std::pair<int, int>> kv;
    kv.reserve(N);
    for (size_t i = 0; i < N; ++i) kv.emplace_back(keys[i], indices[i]);

    std::stable_sort(kv.begin(), kv.end(),
        [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
            return a.first < b.first;
        });

    // Write back sorted indices
    for (size_t i = 0; i < N; ++i) indices[i] = kv[i].second;

    // Unmap
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER); // sortIdxSSBO
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cellKeySSBO);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER); // cellKeySSBO
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

// Simple helper (currently just returns param_boxHalf; extend if you need fitted logic)
Vec3 SPHFluidGPU::ComputeAABBFittedHalf() const {
    return param_boxHalf;
}
GLuint SPHFluidGPU::LoadComputeShader(const char* filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) Debug::FatalError(std::string("Failed to open compute shader: ") + filePath, __FILE__, __LINE__);
    std::stringstream ss; ss << file.rdbuf(); std::string src = ss.str();
    const char* source = src.c_str();

    GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(sh, 1, &source, nullptr);
    glCompileShader(sh);
    GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0; glGetShaderiv(sh, GL_INFO_LOG_LENGTH, &len);
        std::string log(len, ' '); glGetShaderInfoLog(sh, len, &len, log.data());
        glDeleteShader(sh);
        Debug::FatalError("Compute compile failed:\n" + log, __FILE__, __LINE__);
    }
    GLuint prog = glCreateProgram();
    glAttachShader(prog, sh);
    glLinkProgram(prog);
    glDeleteShader(sh);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0; glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
        std::string log(len, ' '); glGetProgramInfoLog(prog, len, &len, log.data());
        glDeleteProgram(prog);
        Debug::FatalError("Compute link failed:\n" + log, __FILE__, __LINE__);
    }
    return prog;
}

// ---------------------------------------------------------------------------
// River / terrain generation
// ---------------------------------------------------------------------------

void SPHFluidGPU::GenerateRiverTerrain(int seed) {
    std::srand(static_cast<unsigned>(seed));
    auto frand = []() { return std::rand() / float(RAND_MAX); };

    // Randomised river channel parameters
    riverAmp          = 1.5f + frand() * 2.5f;     // meander amplitude
    riverFreq         = 0.15f + frand() * 0.20f;   // meander frequency
    riverPhase        = frand() * 6.2831f;
    riverChannelWidth = 2.5f + frand() * 2.0f;
    float channelDepth = 3.0f + frand() * 1.0f;    // deep channel so walls have real height
    float slopeDrop    = 0.6f + frand() * 0.8f;    // terrain drop: keep inside box bounds

    // Noise phase seeds for the surrounding terrain
    float ph[8];
    for (int k = 0; k < 8; ++k) ph[k] = frand() * 6.2831f;

    // Match terrain footprint to simulation box (elongated for river)
    terrainWorldMinX  = param_boxCenter.x - param_boxHalf.x;
    terrainWorldMinZ  = param_boxCenter.z - param_boxHalf.z;
    terrainWorldSizeX = 2.0f * param_boxHalf.x;
    terrainWorldSizeZ = 2.0f * param_boxHalf.z;

    float xMin   = terrainWorldMinX;
    float zMin   = terrainWorldMinZ;
    float xSize  = terrainWorldSizeX;
    float zSize  = terrainWorldSizeZ;
    float yBase  = param_boxCenter.y - param_boxHalf.y; // box floor in world Y

    terrainHeights.resize(terrainW * terrainH);

    for (int iz = 0; iz < terrainH; ++iz) {
        for (int ix = 0; ix < terrainW; ++ix) {
            float wx = xMin + (float(ix) / float(terrainW - 1)) * xSize;
            float wz = zMin + (float(iz) / float(terrainH - 1)) * zSize;

            // Normalized downstream position [0,1]
            float tFlow = (wz - zMin) / zSize;

            // River centerline sinusoidal path (in X)
            float centerX = param_boxCenter.x + riverAmp * std::sinf(riverFreq * wz + riverPhase);
            float distToRiver = std::fabsf(wx - centerX);

            // Background terrain from stacked sine noise
            float h = yBase + 2.5f;
            h += 1.2f * std::sinf(wx * 0.40f + ph[0]) * std::cosf(wz * 0.30f + ph[1]);
            h += 0.70f * std::sinf(wx * 0.80f + ph[2]) * std::sinf(wz * 0.70f + ph[3]);
            h += 0.35f * std::sinf(wx * 1.50f + ph[4]) * std::cosf(wz * 1.30f + ph[5]);
            h += 0.18f * std::sinf(wx * 2.60f + ph[6]) * std::sinf(wz * 2.20f + ph[7]);

            // River floor slopes downhill (tFlow=0 → upstream, tFlow=1 → downstream)
            float riverFloor = yBase + 1.0f - tFlow * slopeDrop;

            // Carve parabolic channel cross-section
            float channelEdge = riverFloor + channelDepth; // height at bank top
            if (distToRiver < riverChannelWidth) {
                float u = distToRiver / riverChannelWidth; // 0=centre, 1=bank
                h = riverFloor + channelDepth * u * u;     // set exactly, don't min
            } else {
                // Outside channel: raise terrain to create a genuine physical wall.
                // Without this, noise dips can make the bank lower than the channel
                // edge and the terrain constraint provides zero lateral containment.
                float wallBase = channelEdge + 1.5f;
                float pastBank = std::min((distToRiver - riverChannelWidth) * 1.5f, 2.0f);
                h = std::max(h, wallBase + pastBank);
            }

            // Don't punch through the box floor
            h = std::max(h, yBase - 0.3f);

            terrainHeights[iz * terrainW + ix] = h;
        }
    }

    // Position emitter at the upstream mouth of the channel
    float startX = param_boxCenter.x + riverAmp * std::sinf(riverPhase);
    riverEmitterPos    = Vec3(startX, yBase + 2.5f, zMin + 0.4f);
    riverEmitterVel    = Vec3(0.0f, -0.5f, 0.5f);   // channel constraint steers flow
    riverEmitterRadius = riverChannelWidth * 0.35f;
    riverSinkY         = yBase + 0.3f;               // just above box floor — recycled when they hit bottom
    riverSinkZMax      = param_boxCenter.z + param_boxHalf.z - 0.5f; // at downstream edge

    // Channel constraint handles flow along the meander; no straight Z gravity needed
    param_gravityY = -120.0f;
    param_gravityZ =   0.0f;

    // Upload heightfield to GPU
    if (terrainSSBO == 0) glGenBuffers(1, &terrainSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, terrainSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 terrainHeights.size() * sizeof(float),
                 terrainHeights.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    std::cout << "[River] seed=" << seed
              << " amp=" << riverAmp << " freq=" << riverFreq
              << " width=" << riverChannelWidth << " slope=" << slopeDrop << "\n";
}
