#pragma once
#include <vector>
#include <Vector.h>
#include <glad.h>
#include <tuple>
#include <cfloat>

using namespace MATH;

struct SPHParticle {
    Vec4 pos;        // 16
    Vec4 vel;        // 16
    Vec4 acc;        // 16
    float density;   // 4
    float pressure;  // 4
    float padA;      // 4
    float padB;      // 4
    int   isGhost;   // 4
    int   isActive;  // 4
    int   padC;      // 4
    int   pad0;      // 4
    // 64 bytes
};

class SPHFluidGPU {
public:
    SPHFluidGPU(size_t numParticles_);
    ~SPHFluidGPU();

    // Simulation
    void InitializeParticles();
    void InitializeGridAndBuffers();
    void InitializeSortBuffers();
    void RecreateGridForBox();
    void InitializeFluidVBO();
    void UploadDataToGPU();
    void DownloadDataFromGPU();          // available, but not used in fast path
    void UpdateFluidVBOFromGPU();
    void DispatchCompute();
    void RadixSortByCell();

    // GPU-only wave impulse (replaces CPU readback version)
    void ApplyWaveImpulseGPU(float amplitude, float wavelength, float phase,
        const Vec3& dir, float yMin, float yMax);

    // Rendering helpers
    GLuint GetFluidVBO()  const;
    size_t GetNumFluids() const;

    // Public so render code can bind them quickly
    GLuint ssbo = 0;

    // Tunables (same defaults as before)
    float param_h = 0.28f;
    float param_mass = 13.8f;
    float param_restDensity = 1000.0f;
    float param_gasConstant = 2000.0f;
    float param_viscosity = 3.5f;
    float param_gravityY = -980.0f;      // cm/s^2
    float param_surfaceTension = 0.0728f;
    float param_timeStep = 0.001f;
    bool  param_pause = false;

    bool  param_useJitter = true;
    float param_jitterAmp = 0.20f;

    bool  param_enableGhosts = false;  // keep false to avoid CPU↔GPU sync
    bool  param_enableSort = false;

    // Oriented box (OBB)
    Vec3  param_boxCenter = Vec3(0, 0, 0);
    Vec3  param_boxHalf = Vec3(7, 7, 7);
    Vec3  param_boxEulerDeg = Vec3(0, 0, 0);
    float param_wallRestitution = 0.15f;
    float param_wallFriction = 0.02f;

    // Grid
    float box = 7.0f; // legacy half-size (kept for compatibility)
    float cellSize = 0.0f;
    int gridSizeX = 0, gridSizeY = 0, gridSizeZ = 0;
    int numCells = 0;

    // Particle buffer (fluid + ghosts)
    std::vector<SPHParticle> particles;
    size_t numParticles;

    // SSBOs
    GLuint cellHeadSSBO = 0;
    GLuint particleNextSSBO = 0;
    GLuint particleCellSSBO = 0;

    // Sorting
    GLuint cellKeySSBO = 0;
    GLuint sortIdxSSBO = 0;
    GLuint sortTmpSSBO = 0;
    GLuint radixSortShader = 0;

    // Compute shaders
    GLuint clearGridShader = 0;
    GLuint buildGridShader = 0;
    GLuint sphGridShader = 0;
    GLuint obbConstraintShader = 0;
    GLuint waveImpulseShader = 0;   // NEW

    // Instance rendering (fluids only)
    GLuint fluidVBO = 0;
    float* vboPtr = nullptr;
    size_t numFluids = 0;

    // Ghost helper grids (unchanged API)
    struct GhostGrid2D {
        float minA, minB, cellSize;
        int cellsA, cellsB;
        std::vector<std::vector<std::vector<size_t>>> grid;
        GhostGrid2D() : minA(0), minB(0), cellSize(1), cellsA(0), cellsB(0) {}
        void init(float minA_, float minB_, float cellSize_, int cellsA_, int cellsB_) {
            minA = minA_; minB = minB_; cellSize = cellSize_; cellsA = cellsA_; cellsB = cellsB_;
            grid.resize(cellsA, std::vector<std::vector<size_t>>(cellsB));
        }
        void addGhost(float a, float b, size_t idx) {
            int ia = int((a - minA) / cellSize);
            int ib = int((b - minB) / cellSize);
            if (ia >= 0 && ia < cellsA && ib >= 0 && ib < cellsB) grid[ia][ib].push_back(idx);
        }
        const std::vector<size_t>& getCell(float a, float b) const {
            static const std::vector<size_t> empty;
            int ia = int((a - minA) / cellSize);
            int ib = int((b - minB) / cellSize);
            if (ia >= 0 && ia < cellsA && ib >= 0 && ib < cellsB) return grid[ia][ib];
            return empty;
        }
    };
    GhostGrid2D ghostXNeg, ghostXPos, ghostYNeg, ghostYPos, ghostZNeg, ghostZPos;
    void BuildGhostGrids();

private:
    GLuint LoadComputeShader(const char* filePath);

    // Cached OBB rotation to avoid sin/cos every substep
    float cachedRot3x3[9] = { 1,0,0, 0,1,0, 0,0,1 };
    Vec3  cachedBoxCenter{}, cachedBoxHalf{}, cachedBoxEuler{};
    void  UpdateCachedBoxIfNeeded();
    static void MakeRotationMat3XYZ(float rxDeg, float ryDeg, float rzDeg, float outM[9]);
    void   ActivateClosestGhost(float x, float y, float z); // kept for completeness
    void   UpdateGhostParticlesDynamic(float h);
    void   UploadGhostActivityToGPU();
};
