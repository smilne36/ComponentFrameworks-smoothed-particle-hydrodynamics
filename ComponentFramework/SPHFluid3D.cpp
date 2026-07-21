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
    obbConstraintShader = LoadComputeShader("shaders/OBBConstraints.comp");
    waveImpulseShader = LoadComputeShader("shaders/WaveImpulse.comp");
    vortexImpulseShader = LoadComputeShader("shaders/VortexImpulse.comp");
    attractorImpulseShader = LoadComputeShader("shaders/AttractorImpulse.comp");
    fountainShader = LoadComputeShader("shaders/FountainRecycle.comp");
    curlFlowShader = LoadComputeShader("shaders/CurlFlow.comp");
    stencilShader = LoadComputeShader("shaders/StencilAttract.comp");
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
    glDeleteProgram(clearGridShader);
    glDeleteProgram(buildGridShader);
    glDeleteProgram(sphGridShader);
    glDeleteProgram(obbConstraintShader);
    glDeleteProgram(waveImpulseShader);
    if (vortexImpulseShader)      glDeleteProgram(vortexImpulseShader);
    if (attractorImpulseShader)   glDeleteProgram(attractorImpulseShader);
    if (fountainShader)           glDeleteProgram(fountainShader);
    if (curlFlowShader)           glDeleteProgram(curlFlowShader);
    if (stencilShader)            glDeleteProgram(stencilShader);
    if (stencilSSBO)              glDeleteBuffers(1, &stencilSSBO);
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
                    p.isGhost = 0; p.isActive = 0; p.padC = count & 1; p.pad0 = 0;
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
            p.isGhost = 0; p.isActive = 0; p.padC = count & 1; p.pad0 = 0;
            particles.push_back(p); ++count;
        }
    } else {
        // Standard fill: a bottom-anchored block centered on the container,
        // sized per axis, with lattice points outside the shape rejected.
        const Vec3 c  = param_boxCenter;
        const Vec3 hf = EffectiveHalf();
        const float margin = spacing * 0.5f;
        // Offsets are relative to the container center (rotation is ignored at
        // spawn, as before; the constraint pass settles particles afterwards).
        auto insideShape = [&](float lx, float ly, float lz) -> bool {
            switch (param_shapeType) {
            case 1: {
                float r = hf.x - margin;
                return lx * lx + ly * ly + lz * lz <= r * r;
            }
            case 2: {
                float r = hf.x - margin;
                return lx * lx + lz * lz <= r * r && std::fabs(ly) <= hf.y - margin;
            }
            case 3: {   // torus: distance from the ring circle <= tube radius
                float R = param_boxHalf.x, r = param_boxHalf.y - margin;
                float dr = std::sqrt(lx * lx + lz * lz) - R;
                return r > 0.0f && (dr * dr + ly * ly) <= r * r;
            }
            case 4: {   // capsule: distance from the Y core segment <= radius
                float r = param_boxHalf.x - margin, H = param_boxHalf.y;
                float dy = ly - std::clamp(ly, -H, H);
                return (lx * lx + lz * lz + dy * dy) <= r * r;
            }
            case 5: {   // hourglass: radius limit grows from neck (y=0) to bases
                float baseR = param_boxHalf.x, H = std::max(param_boxHalf.y, 1e-6f);
                float neckR = std::min(param_boxHalf.z, baseR);
                if (std::fabs(ly) > H - margin) return false;
                float rMax = neckR + (baseR - neckR) * std::fabs(ly) / H - margin;
                return rMax > 0.0f && (lx * lx + lz * lz) <= rMax * rMax;
            }
            case 6: {   // egg (ellipsoid), margin-shrunk semi-axes
                float a = std::max(param_boxHalf.x - margin, 1e-4f);
                float b = std::max(param_boxHalf.y - margin, 1e-4f);
                float u = lx / a, v = ly / b, w = lz / a;
                return (u * u + v * v + w * w) <= 1.0f;
            }
            case 7: {   // star prism: wall radius oscillates with angle
                float R     = param_boxHalf.x;
                float H     = param_boxHalf.y;
                float pts   = std::max(3.0f, param_shapeAux.x);
                float depth = std::clamp(param_shapeAux.y, 0.0f, 0.9f);
                if (std::fabs(ly) > H - margin) return false;
                float ang  = std::atan2(lz, lx);
                float rMax = R * (1.0f - depth * (0.5f + 0.5f * std::cos(pts * ang))) - margin;
                return rMax > 0.0f && (lx * lx + lz * lz) <= rMax * rMax;
            }
            case 8: {   // superellipsoid |x/a|^n + |y/b|^n + |z/a|^n <= 1
                float a = std::max(param_boxHalf.x - margin, 1e-4f);
                float b = std::max(param_boxHalf.y - margin, 1e-4f);
                float n = std::clamp(param_shapeAux.z, 0.6f, 8.0f);
                float F = std::pow(std::fabs(lx) / a, n) + std::pow(std::fabs(ly) / b, n)
                        + std::pow(std::fabs(lz) / a, n);
                return F <= 1.0f;
            }
            case 9: {   // trefoil knot tube: within tube radius of the curve
                float S = param_boxHalf.x;
                float r = param_boxHalf.y - margin;
                if (r <= 0.0f) return false;
                float bestD2 = 1e30f;
                for (int k = 0; k < 48; ++k) {
                    float t  = 6.2831853f * float(k) / 48.0f;
                    float cx = S * (std::sin(t) + 2.0f * std::sin(2.0f * t));
                    float cy = S * 0.35f * (-std::sin(3.0f * t));
                    float cz = S * (std::cos(t) - 2.0f * std::cos(2.0f * t));
                    float dx = lx - cx, dy = ly - cy, dz = lz - cz;
                    bestD2 = std::min(bestD2, dx * dx + dy * dy + dz * dz);
                }
                return bestD2 <= r * r;
            }
            default: return true;
            }
        };
        int   layersY = std::max(1, int((2.0f * hf.y * fillFraction) / spacing));
        int   sideX   = std::max(1, int((hf.x * 1.7f) / spacing));
        int   sideZ   = std::max(1, int((hf.z * 1.7f) / spacing));
        for (int x = 0; x < sideX && count < (int)numParticles; ++x)
            for (int y = 0; y < layersY && count < (int)numParticles; ++y)
                for (int z = 0; z < sideZ && count < (int)numParticles; ++z) {
                    float lx = -hf.x * 0.85f + x * spacing + j();
                    float ly = -hf.y + spacing + y * spacing + j();
                    float lz = -hf.z * 0.85f + z * spacing + j();
                    if (!insideShape(lx, ly, lz)) continue;
                    SPHParticle p{};
                    p.pos = Vec4(c.x + lx, c.y + ly, c.z + lz, 0.0f);
                    p.vel = Vec4(0, 0, 0, 0);
                    p.acc = Vec4(0, 0, 0, 0);
                    p.density = p.pressure = 0.0f;
                    p.isGhost = 0; p.isActive = 0; p.pad0 = 0;
                    // Color-group tag for Two-Color mode (read as flags.z in shaders)
                    switch (param_mixPattern) {
                        case 1:  p.padC = (x + y + z) & 1; break;         // alternating lattice
                        case 2:  p.padC = int(rng() & 1u); break;         // random
                        default: p.padC = (lx < 0.0f) ? 0 : 1; break;     // split halves along X
                    }
                    particles.push_back(p); ++count;
                }
    }

    std::cout << "Fluid particles: " << count << std::endl;
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

// Sizes the spatial hash grid to the world-space AABB of the (possibly
// rotated) container, with one cell of padding per side. Cheap enough to run
// every dispatch; buffers are only reallocated when numCells actually changes.
void SPHFluidGPU::ComputeGridExtents() {
    cellSize = param_h;

    float R[9];
    MakeRotationMat3XYZ(param_boxEulerDeg.x, param_boxEulerDeg.y, param_boxEulerDeg.z, R);
    const Vec3 half = EffectiveHalf();
    // World AABB extent of the rotated container: ext_i = sum_j |R[i][j]| * half_j
    // (R is column-major world_from_box, so world axis i reads R[i], R[3+i], R[6+i].)
    Vec3 ext(
        std::fabs(R[0]) * half.x + std::fabs(R[3]) * half.y + std::fabs(R[6]) * half.z,
        std::fabs(R[1]) * half.x + std::fabs(R[4]) * half.y + std::fabs(R[7]) * half.z,
        std::fabs(R[2]) * half.x + std::fabs(R[5]) * half.y + std::fabs(R[8]) * half.z);
    ext = Vec3(ext.x + cellSize, ext.y + cellSize, ext.z + cellSize);

    gridMinV = param_boxCenter - ext;
    auto dim = [&](float e) {
        return std::min(160, std::max(1, int(std::ceil((2.0f * e) / cellSize))));
    };
    gridSizeX = dim(ext.x);
    gridSizeY = dim(ext.y);
    gridSizeZ = dim(ext.z);
    numCells = std::max(1, gridSizeX * gridSizeY * gridSizeZ);
}

void SPHFluidGPU::InitializeGridAndBuffers() {
    ComputeGridExtents();
    box = std::max(param_boxHalf.x, std::max(param_boxHalf.y, param_boxHalf.z));

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

    allocatedCells = numCells;
}

// Cell-key buffer written by BuildGrid.comp (binding 4)
void SPHFluidGPU::InitializeSortBuffers() {
    size_t N = particles.size();
    glGenBuffers(1, &cellKeySSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cellKeySSBO);
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

void SPHFluidGPU::DispatchCompute(float overrideDt) {
    if (param_pause) return;

    const float timeStep = (overrideDt > 0.0f ? overrideDt : param_timeStep);

    // Track live container edits: refresh the grid origin/dims every step and
    // reallocate the cell-head buffer only when the cell count changed.
    // (ClearGrid initializes the buffer contents each step, so no CPU init.)
    ComputeGridExtents();
    if (numCells != allocatedCells) {
        if (cellHeadSSBO) glDeleteBuffers(1, &cellHeadSSBO);
        glGenBuffers(1, &cellHeadSSBO);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cellHeadSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int) * numCells, nullptr, GL_DYNAMIC_COPY);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cellHeadSSBO);
        allocatedCells = numCells;
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
    glUniform3f(glGetUniformLocation(buildGridShader, "gridMin"), gridMinV.x, gridMinV.y, gridMinV.z);
    glUniform1i(glGetUniformLocation(buildGridShader, "numParticles"), int(particles.size()));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cellHeadSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, particleNextSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, particleCellSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, cellKeySSBO);
    glDispatchCompute((particles.size() + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

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
    glUniform3f(glGetUniformLocation(sphGridShader, "gridMin"), gridMinV.x, gridMinV.y, gridMinV.z);
    glUniform1f(glGetUniformLocation(sphGridShader, "foamGen"), param_foamGen);
    glUniform1f(glGetUniformLocation(sphGridShader, "foamVelRef"), param_foamVelRef);
    // CFL-style cap: a particle may not cross more than ~0.4 smoothing lengths
    // per substep. Kills the integration spikes that read as jitter.
    glUniform1f(glGetUniformLocation(sphGridShader, "maxSpeed"), 0.4f * param_h / std::max(timeStep, 1e-6f));
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
    glUniform1i(glGetUniformLocation(obbConstraintShader, "uShapeType"), param_shapeType);
    glUniform3f(glGetUniformLocation(obbConstraintShader, "uShapeAux"),
        param_shapeAux.x, param_shapeAux.y, param_shapeAux.z);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glDispatchCompute((particles.size() + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // 5) River mode: terrain collision, channel confinement, particle recycling
    if (riverMode && !terrainHeights.empty()) {
        DispatchTerrainConstraints();
        DispatchChannelConstraint();
        DispatchStreamEmit();
    }

    // 6) Fountain: recycle pooled bottom water back into the upward jet
    if (fountainMode && !riverMode) DispatchFountainRecycle(timeStep);

    glUseProgram(0);
}

// Recycles pooled water below the drain plane back to the nozzle as a jet.
// Relies on SSBO binding 0 set by the constraint pass just above.
void SPHFluidGPU::DispatchFountainRecycle(float dt) {
    if (!fountainShader) return;
    const Vec3 half = EffectiveHalf();
    const Vec3 emit = param_boxCenter + fountainOffset;
    glUseProgram(fountainShader);
    glUniform1i(glGetUniformLocation(fountainShader, "numParticles"), int(particles.size()));
    glUniform3f(glGetUniformLocation(fountainShader, "uEmitterPos"), emit.x, emit.y, emit.z);
    glUniform1f(glGetUniformLocation(fountainShader, "uEmitterRadius"), fountainRadius);
    glUniform1f(glGetUniformLocation(fountainShader, "uJetSpeed"), fountainJetSpeedLive);
    glUniform1f(glGetUniformLocation(fountainShader, "uJetSpread"), fountainSpread);
    glUniform1f(glGetUniformLocation(fountainShader, "uDrainY"),
        (param_boxCenter.y - half.y) + fountainDrainLevel);
    glUniform1f(glGetUniformLocation(fountainShader, "uDrainChance"),
        std::min(1.0f, fountainDrainPerSec * dt));
    glUniform1f(glGetUniformLocation(fountainShader, "uRestDensity"), param_restDensity);
    glUniform1ui(glGetUniformLocation(fountainShader, "uSeed"), fountainSeed++);
    glDispatchCompute((particles.size() + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
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

// Whirlpool around the container's local Y axis. Kicks are velocity deltas
// (callers pre-multiply by dt), so the swirl is framerate-independent.
void SPHFluidGPU::ApplyVortexImpulse(float tangentKick, float inwardKick) {
    if (std::fabs(tangentKick) < 1e-6f && std::fabs(inwardKick) < 1e-6f) return;

    float R[9]; MakeRotationMat3XYZ(param_boxEulerDeg.x, param_boxEulerDeg.y, param_boxEulerDeg.z, R);
    // column-major world_from_box: local +Y = column 1 = (R[3], R[4], R[5])
    glUseProgram(vortexImpulseShader);
    glUniform1i(glGetUniformLocation(vortexImpulseShader, "N"), int(particles.size()));
    glUniform3f(glGetUniformLocation(vortexImpulseShader, "uCenter"),
        param_boxCenter.x, param_boxCenter.y, param_boxCenter.z);
    glUniform3f(glGetUniformLocation(vortexImpulseShader, "uAxis"), R[3], R[4], R[5]);
    glUniform1f(glGetUniformLocation(vortexImpulseShader, "uTangent"), tangentKick);
    glUniform1f(glGetUniformLocation(vortexImpulseShader, "uInward"), inwardKick);
    const Vec3 half = EffectiveHalf();
    glUniform1f(glGetUniformLocation(vortexImpulseShader, "uRadius"), std::max(half.x, half.z));

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glDispatchCompute((particles.size() + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glUseProgram(0);
}

// Movable gravity well: softened inverse-distance pull toward a point.
// pullKick is a velocity delta (callers pre-multiply by dt).
void SPHFluidGPU::ApplyAttractorImpulse(const Vec3& point, float pullKick, float radius) {
    if (std::fabs(pullKick) < 1e-6f) return;

    glUseProgram(attractorImpulseShader);
    glUniform1i(glGetUniformLocation(attractorImpulseShader, "N"), int(particles.size()));
    glUniform3f(glGetUniformLocation(attractorImpulseShader, "uPoint"), point.x, point.y, point.z);
    glUniform1f(glGetUniformLocation(attractorImpulseShader, "uPull"), pullKick);
    glUniform1f(glGetUniformLocation(attractorImpulseShader, "uRadius"), std::max(radius, 0.1f));
    glUniform1f(glGetUniformLocation(attractorImpulseShader, "uSoften"), std::max(0.15f * radius, 0.2f));

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glDispatchCompute((particles.size() + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glUseProgram(0);
}

// Silk Flow: curl-noise drift (divergence-free, so it swirls instead of
// clumping). kick is a velocity delta (callers pre-multiply by dt).
void SPHFluidGPU::ApplyCurlFlow(float kick, float scale, float time) {
    if (std::fabs(kick) < 1e-6f) return;

    glUseProgram(curlFlowShader);
    glUniform1i(glGetUniformLocation(curlFlowShader, "N"), int(particles.size()));
    glUniform1f(glGetUniformLocation(curlFlowShader, "uKick"), kick);
    glUniform1f(glGetUniformLocation(curlFlowShader, "uScale"), std::max(scale, 1e-3f));
    glUniform1f(glGetUniformLocation(curlFlowShader, "uTime"), time);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glDispatchCompute((particles.size() + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glUseProgram(0);
}

// Liquid Logo: upload the stencil's target points (world space, w unused)
void SPHFluidGPU::SetStencilTargets(const std::vector<Vec4>& points) {
    stencilCount = int(points.size());
    if (stencilSSBO) { glDeleteBuffers(1, &stencilSSBO); stencilSSBO = 0; }
    if (points.empty()) return;
    glGenBuffers(1, &stencilSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, stencilSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(Vec4) * points.size(),
                 points.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void SPHFluidGPU::ApplyStencilAttract(float pullKick, float dampKick) {
    if (stencilCount <= 0 || !stencilSSBO) return;
    if (std::fabs(pullKick) < 1e-6f && dampKick < 1e-6f) return;

    glUseProgram(stencilShader);
    glUniform1i(glGetUniformLocation(stencilShader, "N"), int(particles.size()));
    glUniform1i(glGetUniformLocation(stencilShader, "uNumTargets"), stencilCount);
    glUniform1f(glGetUniformLocation(stencilShader, "uPull"), pullKick);
    glUniform1f(glGetUniformLocation(stencilShader, "uDamp"), std::min(dampKick, 0.5f));

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, stencilSSBO);
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
    riverAmp          = 0.5f + frand() * 1.5f;     // moderate meander — stays inside box
    riverFreq         = 0.18f + frand() * 0.18f;
    riverPhase        = frand() * 6.2831f;
    riverChannelWidth = 1.8f + frand() * 1.2f;     // 1.8-3.0 unit half-width
    riverChannelDepth = 3.5f + frand() * 1.0f;
    riverSlopeDrop    = 0.3f + frand() * 0.5f;
    float channelDepth = riverChannelDepth;
    float slopeDrop    = riverSlopeDrop;

    // Noise phase seeds for the surrounding terrain
    float ph[8];
    for (int k = 0; k < 8; ++k) ph[k] = frand() * 6.2831f;

    // Terrain footprint: exact box extent — keep 64×64 resolution concentrated inside the
    // simulation volume so terrain normals at channel walls are sharp (good lateral containment)
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

            // River floor slopes gently downstream
            float riverFloor = yBase + 1.0f - tFlow * slopeDrop;
            float channelEdge = riverFloor + channelDepth;

            // Plateau: always 3 units above channel rim so the carved gorge is clearly visible.
            // With channelDepth ~4 and 3 units of bank, total relief from box floor to
            // plateau top is ~8 units — looks like a proper river canyon.
            float plateau = channelEdge + 3.0f;
            float h = plateau;
            h += 0.5f * std::sinf(wx * 0.35f + ph[0]) * std::cosf(wz * 0.28f + ph[1]);
            h += 0.25f * std::sinf(wx * 0.70f + ph[2]) * std::sinf(wz * 0.60f + ph[3]);
            h += 0.12f * std::sinf(wx * 1.40f + ph[4]) * std::cosf(wz * 1.20f + ph[5]);

            if (distToRiver < riverChannelWidth) {
                // Trapezoidal channel: flat floor (inner 50%) + parabolic walls (outer 50%)
                // Flat floor makes depth visible; steep walls give strong lateral containment
                float u = distToRiver / riverChannelWidth;
                const float floorFrac = 0.50f;
                if (u < floorFrac) {
                    h = riverFloor;
                } else {
                    float uw = (u - floorFrac) / (1.0f - floorFrac);
                    h = riverFloor + channelDepth * uw * uw;
                }
            } else {
                // Outside channel: keep plateau, ensure it's always above channel edge
                h = std::max(h, channelEdge + 0.3f);
            }

            // Don't punch through the box floor
            h = std::max(h, yBase - 0.3f);

            terrainHeights[iz * terrainW + ix] = h;
        }
    }

    // Position emitter at the upstream mouth of the channel, above the floor
    float emitterZ  = zMin + 0.5f;
    float startX    = param_boxCenter.x + riverAmp * std::sinf(riverFreq * emitterZ + riverPhase);
    float floorUp   = yBase + 1.0f; // upstream floor
    riverEmitterPos = Vec3(startX, floorUp + channelDepth * 0.5f, emitterZ);
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
