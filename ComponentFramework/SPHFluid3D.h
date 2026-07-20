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
    void   UploadDataToGPU();
    void   DispatchCompute(float overrideDt = -1.0f);
    GLuint GetFluidVBO() const;
    size_t GetNumFluids() const;
    std::vector<Vec3> GetPositions() const;

    void   ApplyWaveImpulse(float amplitude, float wavelength, float phase, const Vec3& dir,
        float yMin = -FLT_MAX, float yMax = FLT_MAX);
    // Whirlpool: tangential kick around the container's local Y axis, plus an
    // optional inward pull. Pass kicks pre-multiplied by dt (velocity change).
    void   ApplyVortexImpulse(float tangentKick, float inwardKick);
    // Movable gravity well: softened inverse-distance pull toward a point,
    // fading out by radius. Pass the kick pre-multiplied by dt.
    void   ApplyAttractorImpulse(const Vec3& point, float pullKick, float radius);
    void   ResetSimulation();
    void   ComputeGridExtents();

    Vec3   ComputeAABBFittedHalf() const;

    float box = 7.0f;              // legacy scalar extent (ghost shell + spawn helpers)
    float cellSize = 0.0f;
    int   gridSizeX = 1, gridSizeY = 1, gridSizeZ = 1;
    int   numCells = 1;
    Vec3  gridMinV = Vec3(-7, -7, -7);  // world-space grid origin (rotation-aware AABB)
    int   allocatedCells = 0;           // cellHeadSSBO capacity, for auto-rebuild

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
    GLuint waveImpulseShader = 0;
    GLuint vortexImpulseShader = 0;
    GLuint attractorImpulseShader = 0;
    GLuint fountainShader = 0;

    GLuint fluidVBO = 0;
    float* vboPtr = nullptr;
    size_t numFluids = 0;

    GLuint cellKeySSBO = 0;   // written by BuildGrid.comp (binding 4)

    float param_h = 0.28f;
    float param_mass = 13.8f;
    float param_restDensity = 1000.0f;
    float param_gasConstant = 2000.0f;
    float param_viscosity = 3.5f;
    float param_gravityY = -980.0f;
    float param_gravityX = 0.0f;
    float param_gravityZ = 0.0f;
    float param_surfaceTension = 0.0728f;
    float param_timeStep = 0.001f;
    bool  param_pause = false;

    bool  param_useJitter = true;
    float param_jitterAmp = 0.20f;

    float param_foamGen = 1.0f;      // foam generation scale (0 disables)
    float param_foamVelRef = 8.0f;   // speed where foam generation saturates

    Vec3  param_boxCenter = Vec3(0, 0, 0);
    Vec3  param_boxHalf = Vec3(7, 7, 7);   // box: halves | sphere: x=radius | cylinder: x=radius, y=half height
                                           // torus: x=ring radius, y=tube radius | capsule: x=radius, y=core half length
                                           // hourglass: x=base radius, y=half height, z=neck radius | egg: x=XZ semi-axis, y=Y semi-axis
    Vec3  param_boxEulerDeg = Vec3(0, 0, 0);
    int   param_shapeType = 0;             // 0=Box, 1=Sphere, 2=Cylinder, 3=Torus, 4=Capsule, 5=Hourglass, 6=Egg
    int   param_mixPattern = 0;            // color-group tagging at spawn: 0=split-X, 1=alternating, 2=random
    float param_wallRestitution = 0.15f;
    float param_wallFriction = 0.02f;

    // Container half extents seen by the grid/spawn code, per shape
    Vec3  EffectiveHalf() const {
        switch (param_shapeType) {
        case 1:  return Vec3(param_boxHalf.x, param_boxHalf.x, param_boxHalf.x);
        case 2:  return Vec3(param_boxHalf.x, param_boxHalf.y, param_boxHalf.x);
        case 3:  return Vec3(param_boxHalf.x + param_boxHalf.y, param_boxHalf.y, param_boxHalf.x + param_boxHalf.y);
        case 4:  return Vec3(param_boxHalf.x, param_boxHalf.y + param_boxHalf.x, param_boxHalf.x);
        case 5:  return Vec3(param_boxHalf.x, param_boxHalf.y, param_boxHalf.x);
        case 6:  return Vec3(param_boxHalf.x, param_boxHalf.y, param_boxHalf.x);
        default: return param_boxHalf;
        }
    }

    // --- Fountain mode (jet from a nozzle; pooled water recycles) ---
    bool  fountainMode = false;
    Vec3  fountainOffset = Vec3(0.0f, -5.0f, 0.0f);   // nozzle, container-relative
    float fountainRadius = 1.0f;
    float fountainSpread = 0.25f;
    float fountainJetSpeedLive = 25.0f;   // written per frame (audio-kicked)
    float fountainDrainLevel = 1.0f;      // height above container bottom that drains
    float fountainDrainPerSec = 2.0f;
    unsigned fountainSeed = 0;
    void  DispatchFountainRecycle(float dt);

    // --- River / Stream mode ---
    bool  riverMode = false;

    // Terrain heightfield (CPU copy; GPU SSBO at binding 7)
    std::vector<float> terrainHeights;
    int   terrainW          = 64;
    int   terrainH          = 64;
    float terrainWorldMinX  = -7.0f;
    float terrainWorldMinZ  = -10.0f;
    float terrainWorldSizeX = 14.0f;
    float terrainWorldSizeZ = 20.0f;

    // Emitter / sink parameters (set by GenerateRiverTerrain)
    Vec3  riverEmitterPos    = Vec3(0,  3.0f, -9.0f);
    Vec3  riverEmitterVel    = Vec3(0, -0.5f,  4.0f);
    float riverEmitterRadius = 1.5f;
    float riverSinkY         = -8.5f;
    float riverSinkZMax      =  9.0f;

    // Stored per-generation values (used during InitializeParticles)
    float riverAmp          = 2.0f;
    float riverFreq         = 0.25f;
    float riverPhase        = 0.0f;
    float riverChannelWidth = 3.0f;
    float riverChannelDepth = 3.5f;
    float riverSlopeDrop    = 0.3f;

    GLuint terrainSSBO             = 0;
    GLuint terrainConstraintShader = 0;
    GLuint streamEmitShader        = 0;
    GLuint channelConstraintShader = 0;

    void GenerateRiverTerrain(int seed);
    void DispatchTerrainConstraints();
    void DispatchStreamEmit();
    void DispatchChannelConstraint();

private:
    GLuint LoadComputeShader(const char* filePath);
};