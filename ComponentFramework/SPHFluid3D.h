#pragma once
#include <vector>
#include <Vector.h>
#include <glad.h>
#include <unordered_map>
#include <tuple>
#include <cfloat>
#include <algorithm>

using namespace MATH;

struct SPHParticle {
    Vec4  pos;
    Vec4  vel;
    Vec4  acc;
    float density;
    float pressure;
    float padA;
    float padB;
    int   isGhost;
    int   isActive;
    int   padC;
    int   pad0;
};

class SPHFluidGPU {
public:
    SPHFluidGPU(size_t numParticles_);
    ~SPHFluidGPU();

    void   InitializeParticles();
    void   InitializeFluidVBO();
    void   InitializeGridAndBuffers();
    void   InitializeSortBuffers();
    void   RadixSortByCell();

    void   ActivateClosestGhost(float x, float y, float z);
    void   UpdateGhostParticlesDynamic(float h);
    void   UploadGhostActivityToGPU();
    void   UploadDataToGPU();
    void   DownloadDataFromGPU();

    void   UpdateFluidVBOFromGPU();
    void   DispatchCompute(float overrideDt = -1.0f);
    GLuint GetFluidVBO() const;
    size_t GetNumFluids() const;
    std::vector<Vec3> GetPositions() const;

    void   ApplyWaveImpulse(float amplitude, float wavelength, float phase, const Vec3& dir,
        float yMin = -FLT_MAX, float yMax = FLT_MAX);
    void   ResetSimulation();
    void   RecreateGridForBox();

    Vec3   ComputeAABBFittedHalf() const;

    float box = 7.0f;
    float cellSize = 0.0f;
    int   gridSizeX = 1, gridSizeY = 1, gridSizeZ = 1;
    int   numCells = 1;

    std::vector<SPHParticle> particles;
    size_t numParticles;

    GLuint ssbo = 0;
    GLuint cellHeadSSBO = 0;
    GLuint particleNextSSBO = 0;
    GLuint particleCellSSBO = 0;

    GLuint clearGridShader = 0;
    GLuint buildGridShader = 0;
    GLuint sphGridShader = 0;
    GLuint obbConstraintShader = 0;
    GLuint radixSortShader = 0;
    GLuint waveImpulseShader = 0;

    GLuint fluidVBO = 0;
    float* vboPtr = nullptr;
    size_t numFluids = 0;

    GLuint cellKeySSBO = 0;
    GLuint sortIdxSSBO = 0;
    GLuint sortTmpSSBO = 0;

    float param_h = 0.28f;
    float param_mass = 13.8f;
    float param_restDensity = 1000.0f;
    float param_gasConstant = 2000.0f;
    float param_viscosity = 3.5f;
    float param_gravityY = -980.0f;
    float param_surfaceTension = 0.0728f;
    float param_timeStep = 0.001f;
    bool  param_pause = false;

    bool  param_useJitter = true;
    float param_jitterAmp = 0.20f;

    bool  param_enableGhosts = false;
    bool  param_enableSort = false;

    Vec3  param_boxCenter = Vec3(0, 0, 0);
    Vec3  param_boxHalf = Vec3(7, 7, 7);
    Vec3  param_boxEulerDeg = Vec3(0, 0, 0);
    float param_wallRestitution = 0.15f;
    float param_wallFriction = 0.02f;

private:
    GLuint LoadComputeShader(const char* filePath);

    /* -------- Ghost face 2D acceleration grids --------
       Each axis-aligned face gets a 2D grid storing indices of the ghost particles
       lying on that face.  Used for quick neighbor / activation queries. */
    struct GhostGrid2D {
        float minA = 0, minB = 0, cellSize = 1;
        int   cellsA = 0, cellsB = 0;
        // Flattened (b * cellsA + a) -> vector of particle indices
        std::vector<std::vector<size_t>> buckets;

        void init(float minA_, float minB_, float cellSz_, int cA_, int cB_) {
            minA = minA_; minB = minB_; cellSize = cellSz_; cellsA = cA_; cellsB = cB_;
            buckets.clear();
            buckets.resize(std::max(1, cellsA * cellsB));
        }
        inline int toIndex(int ia, int ib) const { return ib * cellsA + ia; }

        void addGhost(float a, float b, size_t idx) {
            int ia = int((a - minA) / cellSize);
            int ib = int((b - minB) / cellSize);
            ia = std::clamp(ia, 0, cellsA - 1);
            ib = std::clamp(ib, 0, cellsB - 1);
            buckets[toIndex(ia, ib)].push_back(idx);
        }

        const std::vector<size_t>& getCell(float a, float b) const {
            static const std::vector<size_t> empty;
            if (cellsA == 0 || cellsB == 0) return empty;
            int ia = int((a - minA) / cellSize);
            int ib = int((b - minB) / cellSize);
            if (ia < 0 || ib < 0 || ia >= cellsA || ib >= cellsB) return empty;
            return buckets[toIndex(ia, ib)];
        }
    };

    GhostGrid2D ghostXNeg, ghostXPos;
    GhostGrid2D ghostYNeg, ghostYPos;
    GhostGrid2D ghostZNeg, ghostZPos;

    void BuildGhostGrids();

    // Cached OBB rotation
    float cachedRot3x3[9] = { 1,0,0, 0,1,0, 0,0,1 };
    void  UpdateCachedBoxIfNeeded();
};