#include "SPHFluid3D.h"
#include "Debug.h"
#include <fstream>
#include <sstream>
#include <random>
#include <vector>
#include <iostream>

SPHFluidGPU::SPHFluidGPU(size_t numParticles_)
    : numParticles(numParticles_), fluidVBO(0), vboPtr(nullptr), numFluids(0)
{
    clearGridShader = LoadComputeShader("shaders/ClearGrid.comp");
    buildGridShader = LoadComputeShader("shaders/BuildGrid.comp");
    sphGridShader = LoadComputeShader("shaders/SPHFluid.comp");
    radixSortShader = LoadComputeShader("shaders/RadixSort.comp"); // NEW

    InitializeParticles();
    InitializeFluidVBO();
    InitializeGridAndBuffers();
    UploadDataToGPU();
    InitializeSortBuffers(); // NEW
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
    glDeleteBuffers(1, &cellKeySSBO);   // NEW
    glDeleteBuffers(1, &sortIdxSSBO);   // NEW
    glDeleteBuffers(1, &sortTmpSSBO);   // NEW
    glDeleteProgram(clearGridShader);
    glDeleteProgram(buildGridShader);
    glDeleteProgram(sphGridShader);
    glDeleteProgram(radixSortShader);   // NEW
}

void SPHFluidGPU::InitializeParticles() {
    float h = 0.28f;
    float spacing = h * 0.85f;
    float box = this->box;

    particles.clear();
    int count = 0;

    // Set the fluid fill height as a fraction of the box (e.g., 60% full)
    float fillFraction = 0.4f;
    float fluidHeight = (2.0f * box) * fillFraction;

    // Calculate number of layers and side count
    int fluidLayersY = int(fluidHeight / spacing);
    int fluidSide = int((box * 1.7f) / spacing);

    std::default_random_engine rng(static_cast<unsigned>(time(nullptr)));
    std::uniform_real_distribution<float> jitter(-spacing * 0.2f, spacing * 0.2f);

    // Fill a block at the bottom of the box
    for (int x = 0; x < fluidSide; ++x) {
        for (int y = 0; y < fluidLayersY; ++y) {
            for (int z = 0; z < fluidSide; ++z) {
                if (count >= numParticles) break;
                SPHParticle p;
                p.pos = Vec4(
                    -box * 0.7f + x * spacing + jitter(rng),
                    -box + spacing + y * spacing + jitter(rng), // Start at bottom
                    -box * 0.7f + z * spacing + jitter(rng),
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
            if (count >= numParticles) break;
        }
        if (count >= numParticles) break;
    }

    // Add ghost particles on all boundaries (unchanged)
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
        for (float z = -box; z <= box; z += spacing) {
            add_ghost(-box, y, z);
            add_ghost(box, y, z);
        }
    for (float x = -box; x <= box; x += spacing)
        for (float z = -box; z <= box; z += spacing) {
            add_ghost(x, -box, z);
            add_ghost(x, box, z);
        }
    for (float x = -box; x <= box; x += spacing)
        for (float y = -box; y <= box; y += spacing) {
            add_ghost(x, y, -box);
            add_ghost(x, y, box);
        }

    std::cout << "Fluid particles: " << count << ", ghosts: " << (particles.size() - count) << std::endl;
    BuildGhostGrids(); // <-- Add this line
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
    float h = 0.28f;
    cellSize = h;
    gridSizeX = gridSizeY = gridSizeZ = int(ceil((2.0f * box) / cellSize));
    numCells = gridSizeX * gridSizeY * gridSizeZ;

    glGenBuffers(1, &cellHeadSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cellHeadSSBO);
    std::vector<int> cellHeadInit(numCells, -1);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int) * numCells, cellHeadInit.data(), GL_DYNAMIC_COPY);

    glGenBuffers(1, &particleNextSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, particleNextSSBO);
    std::vector<int> nextInit(particles.size(), -1);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int) * particles.size(), nextInit.data(), GL_DYNAMIC_COPY);

    glGenBuffers(1, &particleCellSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, particleCellSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int) * particles.size(), nullptr, GL_DYNAMIC_COPY);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cellHeadSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, particleNextSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, particleCellSSBO);
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

// --- Call this before SPH step, after cell indices are filled
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

    // Ensure candidates is not null before dereferencing  
    if (candidates == nullptr || candidates->empty()) {  
        return; // No valid candidates, exit early  
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
// -- Upload full particle buffer to GPU
void SPHFluidGPU::UploadDataToGPU() {
    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(SPHParticle) * particles.size(), particles.data(), GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);

    GLenum err = glGetError();
    if (err != GL_NO_ERROR) std::cout << "OpenGL error after buffer alloc: " << err << std::endl;
    std::cout << "sizeof(SPHParticle): " << sizeof(SPHParticle) << std::endl;

}

// ... (DownloadDataFromGPU & VBO update same as before)

void SPHFluidGPU::DispatchCompute() {
    float h = 0.28f;
    float spacing = h * 0.85f;
    float mass = 5.0f;
    float restDensity = mass / (spacing * spacing * spacing); // ≈ 370     // mass / (spacing^3)
    float gasConstant = 15000.0f;
    float viscosity = 3.0f;
    MATH::Vec3 gravity = MATH::Vec3(0, -98000.0f, 0);
    float surfaceTension = 0.5f;
    float timeStep = 0.002f;

    UpdateGhostParticlesDynamic(h);
    UploadGhostActivityToGPU();

    // 1. Clear grid
    glUseProgram(clearGridShader);
    glUniform1i(glGetUniformLocation(clearGridShader, "numCells"), numCells);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cellHeadSSBO);
    glDispatchCompute((numCells + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // 2. Build grid (write cellKeySSBO, etc)
    glUseProgram(buildGridShader);
    glUniform3i(glGetUniformLocation(buildGridShader, "gridSize"), gridSizeX, gridSizeY, gridSizeZ);
    glUniform1f(glGetUniformLocation(buildGridShader, "cellSize"), cellSize);
    glUniform1f(glGetUniformLocation(buildGridShader, "box"), box);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cellHeadSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, particleNextSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, particleCellSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, cellKeySSBO); // <<== NEW
    glDispatchCompute((particles.size() + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // 3. Sort indices by cell index!
    RadixSortByCell();

    // 4. SPH step: now neighbor loops can use sorted indices!
    glUseProgram(sphGridShader);
    glUniform3i(glGetUniformLocation(sphGridShader, "gridSize"), gridSizeX, gridSizeY, gridSizeZ);
    glUniform1f(glGetUniformLocation(sphGridShader, "cellSize"), cellSize);
    glUniform1f(glGetUniformLocation(sphGridShader, "box"), box);
    glUniform1f(glGetUniformLocation(sphGridShader, "timeStep"), timeStep);
    glUniform1f(glGetUniformLocation(sphGridShader, "h"), h);
    glUniform1f(glGetUniformLocation(sphGridShader, "mass"), mass);
    glUniform1f(glGetUniformLocation(sphGridShader, "restDensity"), restDensity);
    glUniform1f(glGetUniformLocation(sphGridShader, "gasConstant"), gasConstant);
    glUniform1f(glGetUniformLocation(sphGridShader, "viscosity"), viscosity);
    glUniform3f(glGetUniformLocation(sphGridShader, "gravity"), gravity.x, gravity.y, gravity.z);
    glUniform1f(glGetUniformLocation(sphGridShader, "surfaceTension"), 2.0f); // Tune as needed
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cellHeadSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, particleNextSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, particleCellSSBO);
    // Optionally pass sorted indices to your SPH kernel here
    glDispatchCompute((particles.size() + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glUseProgram(0);
}

GLuint SPHFluidGPU::GetFluidVBO() const { return fluidVBO; }
size_t SPHFluidGPU::GetNumFluids() const { return numFluids; }
// Standard shader loading
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
    float spacing = 0.28f * 0.85f;
    int cells = int(ceil((2.0f * box) / spacing)) + 1;

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
        if (fabs(p.pos.x + box) < 1e-4f) ghostXNeg.addGhost(p.pos.y, p.pos.z, i);
        if (fabs(p.pos.x - box) < 1e-4f) ghostXPos.addGhost(p.pos.y, p.pos.z, i);
        if (fabs(p.pos.y + box) < 1e-4f) ghostYNeg.addGhost(p.pos.x, p.pos.z, i);
        if (fabs(p.pos.y - box) < 1e-4f) ghostYPos.addGhost(p.pos.x, p.pos.z, i);
        if (fabs(p.pos.z + box) < 1e-4f) ghostZNeg.addGhost(p.pos.x, p.pos.y, i);
        if (fabs(p.pos.z - box) < 1e-4f) ghostZPos.addGhost(p.pos.x, p.pos.y, i);
    }
}
