#pragma once
#include <vector>
#include <Vector.h>
#include <glad.h>
#include <unordered_map>
#include <tuple>
#include <cfloat> // add

using namespace MATH;
struct SPHParticle {
    Vec4 pos;        // 16
    Vec4 vel;        // 16
    Vec4 acc;        // 16
    float density;   // 4
    float pressure;  // 4
    float padA;      // 4
    float padB;      // 4
    int isGhost;     // 4
    int isActive;    // 4
    int padC;        // 4
    int pad0;        // 4
    // 64 bytes
};

class SPHFluidGPU {
public:
    SPHFluidGPU(size_t numParticles_);
    ~SPHFluidGPU();

    void InitializeParticles();
    void InitializeFluidVBO();
    void InitializeGridAndBuffers();
    void InitializeSortBuffers();
    void RadixSortByCell();
    void ActivateClosestGhost(float x, float y, float z);
    void UpdateGhostParticlesDynamic(float h);
    void UploadGhostActivityToGPU();
    void UploadDataToGPU();
    void DownloadDataFromGPU();

    void UpdateFluidVBOFromGPU();
    void DispatchCompute();
    GLuint GetFluidVBO() const;
    size_t GetNumFluids() const;
    std::vector<Vec3> GetPositions() const;

    // wave + reset helpers
    void ApplyWaveImpulse(float amplitude, float wavelength, float phase, const Vec3& dir,
        float yMin = -FLT_MAX, float yMax = FLT_MAX);
    void ResetSimulation();

    // NEW: rebuild grid only (when box/h changes), doesn’t touch particles
    void RecreateGridForBox();

    // Simulation/box/grid properties
    // Legacy 'box' kept for compatibility (half-size max extent), used as fallback
    float box = 7.0f; // size of bucket ([-box, box]) legacy
    float cellSize;
    int gridSizeX, gridSizeY, gridSizeZ;
    int numCells;

    // Particle buffer (fluid + ghosts)
    std::vector<SPHParticle> particles;
    size_t numParticles;

    // SSBOs
    GLuint ssbo = 0;
    GLuint cellHeadSSBO = 0;
    GLuint particleNextSSBO = 0;
    GLuint particleCellSSBO = 0;

    // Compute shaders
    GLuint clearGridShader = 0;
    GLuint buildGridShader = 0;
    GLuint sphGridShader = 0;
    // NEW: OBB constraint compute
    GLuint obbConstraintShader = 0;

    // Instance rendering (fluids only)
    GLuint fluidVBO = 0;
    float* vboPtr = nullptr;
    size_t numFluids = 0;

    // Sorting
    GLuint cellKeySSBO = 0;
    GLuint sortIdxSSBO = 0;
    GLuint sortTmpSSBO = 0;
    GLuint radixSortShader = 0;

    // Tunable simulation parameters
    float param_h = 0.28f;
    float param_mass = 13.8f;
    float param_restDensity = 1000.0f;
    float param_gasConstant = 2000.0f;
    float param_viscosity = 3.5f;
    float param_gravityY = -980.0f;      // cm/s^2
    float param_surfaceTension = 0.0728f;
    float param_timeStep = 0.001f;
    bool  param_pause = false;

    bool  param_useJitter = true;     // spawn jitter
    float param_jitterAmp = 0.20f;

    // Performance toggles
    bool  param_enableGhosts = false; // disable to remove CPU↔GPU sync
    bool  param_enableSort = false;   // not used by current SPH path

    // NEW: Oriented Box (OBB) controls
    Vec3  param_boxCenter = Vec3(0.0f, 0.0f, 0.0f);
    Vec3  param_boxHalf = Vec3(7.0f, 7.0f, 7.0f);
    Vec3  param_boxEulerDeg = Vec3(0.0f, 0.0f, 0.0f); // XYZ degrees
    float param_wallRestitution = 0.15f; // bounce
    float param_wallFriction = 0.02f; // tangential damping

    // Ghost grid structures (unchanged, but note ghosts are axis-aligned)
    struct GhostGrid2D {
        float minA, minB, cellSize;
        int cellsA, cellsB;
        std::vector<std::vector<std::vector<size_t>>> grid;

        GhostGrid2D() : minA(0), minB(0), cellSize(1), cellsA(0), cellsB(0) {}
        void init(float minA_, float minB_, float cellSize_, int cellsA_, int cellsB_) {
            minA = minA_; minB = minB_; cellSize = cellSize_;
            cellsA = cellsA_; cellsB = cellsB_;
            grid.resize(cellsA, std::vector<std::vector<size_t>>(cellsB));
        }
        void addGhost(float a, float b, size_t idx) {
            int ia = int((a - minA) / cellSize);
            int ib = int((b - minB) / cellSize);
            if (ia >= 0 && ia < cellsA && ib >= 0 && ib < cellsB)
                grid[ia][ib].push_back(idx);
        }
        const std::vector<size_t>& getCell(float a, float b) const {
            static const std::vector<size_t> empty;
            int ia = int((a - minA) / cellSize);
            int ib = int((b - minB) / cellSize);
            if (ia >= 0 && ia < cellsA && ib >= 0 && ib < cellsB)
                return grid[ia][ib];
            return empty;
        }
    };

    GhostGrid2D ghostXNeg, ghostXPos, ghostYNeg, ghostYPos, ghostZNeg, ghostZPos;
    void BuildGhostGrids();


private:
    GLuint LoadComputeShader(const char* filePath);
};