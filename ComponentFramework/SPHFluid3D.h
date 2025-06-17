#pragma once
#include <vector>
#include <Vector.h>
#include <glad.h>
#include <unordered_map>
#include <tuple>

using namespace MATH;
struct SPHParticle {
    Vec4 pos;        // 16
    Vec4 vel;        // 16
    Vec4 acc;        // 16
    float density;   // 4
    float pressure;  // 4
    float padA;      // 4
    float padB;      // 4  // <-- Floats up to 16
    int isGhost;     // 4
    int isActive;    // 4
    int padC;        // 4
    int pad0;        // 4  // <-- Ints up to 16
    // 16*3 + 16 + 16 = 64
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
    void UploadDataToGPU();
    void DownloadDataFromGPU();
  
    void UpdateFluidVBOFromGPU();
    void DispatchCompute();
    GLuint GetFluidVBO() const;
    size_t GetNumFluids() const;
    std::vector<Vec3> GetPositions() const;

    // Simulation/box/grid properties
    float box = 7.0f; // size of bucket ([-box, box])
    float cellSize;
    int gridSizeX, gridSizeY, gridSizeZ;
    int numCells;

    // Particle buffer (fluid + ghosts)
    std::vector<SPHParticle> particles;
    size_t numParticles;

    // SSBOs
    GLuint ssbo = 0;             // Particle buffer
    GLuint cellHeadSSBO = 0;     // Cell head buffer
    GLuint particleNextSSBO = 0; // Particle next buffer
    GLuint particleCellSSBO = 0; // Particle cell index buffer

    // Compute shaders
    GLuint clearGridShader = 0;
    GLuint buildGridShader = 0;
    GLuint sphGridShader = 0;

    GLuint fluidVBO = 0;
    float* vboPtr = nullptr;
    size_t numFluids = 0;

    GLuint cellKeySSBO = 0;      // Stores the grid cell index for each particle (int per particle)
    GLuint sortIdxSSBO = 0;      // Stores sorted indices for particles (int per particle, ping-pong)
    GLuint sortTmpSSBO = 0;      // Temporary buffer for radix sort (ping-pong)

    GLuint radixSortShader = 0;  // GPU radix sort compute shade

    // --- Ghost grid structures for each face ---
    struct GhostGrid2D {
        float minA, minB, cellSize;
        int cellsA, cellsB;
        // Each cell holds a vector of indices into the particles vector
        std::vector<std::vector<std::vector<size_t>>> grid; // [a][b][ghost indices]

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

    void BuildGhostGrids(); // new

private:
    GLuint LoadComputeShader(const char* filePath);
};