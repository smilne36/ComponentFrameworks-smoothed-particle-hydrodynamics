#ifndef SCENE0P_H
#define SCENE0P_H
#include "Scene.h"
#include "Vector.h"
#include <Matrix.h>
#include "SPHFluid3D.h"
#include <SDL.h>
#include "window.h" 
#include "Body.h"
#include "Mesh.h"
#include "Shader.h"
#include <glad.h>
#include <limits>
using namespace MATH;

union SDL_Event;

class Scene0p : public Scene {
private:
    Body* sphere = nullptr;
    Shader* shader = nullptr;
    Mesh* mesh = nullptr;
    Matrix4 projectionMatrix;
    Matrix4 viewMatrix;
    Matrix4 modelMatrix;
    bool    drawInWireMode = true;
    bool    mouseDown = false;
    int     mouseX = 0, mouseY = 0;
    int     mouseButton = -1;       // -1=none, 1=left, 3=right
    Vec3    cameraPos = Vec3(0.0f, 5.0f, 22.0f);
    Vec3    cameraTarget = Vec3(0.0f, 0.0f, 0.0f);
    Vec3    cameraUp = Vec3(0.0f, 1.0f, 0.0f);
    float   camDist      = 22.0f;
    float   camAzimuth   = 0.0f;   // radians, rotation around Y
    float   camElevation = 0.22f;  // radians, pitch above horizon

    Shader* lineShader = nullptr;
    GLuint  boxVAO = 0, boxVBO = 0;
    int     containerWireVerts = 24;   // vertex count currently in boxVBO
    bool    showContainerOutline = true;
    float   containerOutlineColor[3] = {0.85f, 0.95f, 1.0f};

    bool    pendingReset = false;
    float   ballAnimTime = 0.0f;

    float   dtAccumulator = 0.0f;
    int     maxSubstepsPerFrame = 16;

    bool    renderFromSSBO = true;

    Vec3    lastBoxCenter{};
    Vec3    lastBoxHalf{};
    Vec3    lastBoxEuler{};
    int     lastShapeType = -1;

    bool    useImpostors = false;
    Shader* impostorShader = nullptr;
    GLuint  impostorVAO = 0;

    // Visualization state
    int     vizMode = 0;          // color drive: 0=Height,1=Speed,2=Pressure,3=Density,4=ViewDepth,5=VelocityDir,6=RadialDist,7=InstanceColor
    float   vizRangeMin = 0.0f;
    float   vizRangeMax = 10.0f;

    // Artistic color state (palette + adjustments, see shared palette block in the frag shaders)
    int     paletteId    = 0;     // 0=Classic,1=Turbo,2=Neon,3=Fire,4=Iridescent,5=Ice,6=Vaporwave,7=Toxic,8=Duotone
    float   hueShiftDeg  = 0.0f;
    float   satMul       = 1.0f;
    float   brightMul    = 1.0f;
    float   contrastMul  = 1.0f;
    bool    invertColor  = false;
    bool    litParticles = true;
    float   iridFreq     = 3.0f;
    float   iridShift    = 0.0f;
    float   duoColorA[3] = {0.05f, 0.02f, 0.10f};
    float   duoColorB[3] = {1.00f, 0.35f, 0.75f};
    bool    showSkyBackground = false;                      // false = flat bgColor backdrop (water pops on black)
    float   bgColor[3]   = {0.0f, 0.0f, 0.0f};              // backdrop clear color (all render paths)
    float   skyColor[3]  = {0.40f, 0.55f, 0.65f};           // sky horizon color (reflections + optional backdrop)
    float   skyZenith[3] = {0.15f, 0.28f, 0.50f};           // sky zenith color
    float   envReflectColor[3] = {0.90f, 0.95f, 1.00f};     // tint on the reflected sky
    float   foamAmount   = 1.5f;
    float   exposure     = 1.0f;

    void    SetColorUniforms(Shader* s) const;
    void    SetGradeUniforms(Shader* s) const;

    // Screenshot capture state
    int     windowW = 0, windowH = 0;   // last known on-screen viewport size
    bool    captureRequested = false;
    int     captureResIdx = 0;          // 0=3000x3000, 1=3840x2160, 2=window size
    std::string lastScreenshotPath;

    void    RenderSceneTo(GLuint targetFBO, int outW, int outH, const Matrix4& proj) const;
    void    DoCapture();

    // Wave injection state (UI)
    float   waveAmplitude  = 1.5f;
    float   waveWavelength = 3.0f;
    float   wavePhaseSpeed = 4.0f;
    int     waveDirIdx     = 1;
    float   yBandMin       = -std::numeric_limits<float>::infinity();
    float   yBandMax       =  std::numeric_limits<float>::infinity();
    bool    continuousWave = false;
    float   wavePhase      = 0.0f;

    void    UpdateContainerWireframe();
    void    SetupImpostorVAO();
    void    DrawFluidImpostors(const Matrix4& proj, int outH) const;
    int     CurrentViewportHeight() const;

    // --- Screen-Space Fluid Rendering ---
    Shader* ssfrDepthShader     = nullptr;
    Shader* ssfrSmoothShader    = nullptr;
    Shader* ssfrThickShader     = nullptr;
    Shader* ssfrCompositeShader = nullptr;
    Shader* skyShader           = nullptr;
    GLuint  ssfrQuadVAO         = 0;

    GLuint  ssfrDepthFBO        = 0;
    GLuint  ssfrDepthTex        = 0;
    GLuint  ssfrDepthRBO        = 0;

    GLuint  ssfrSmoothFBO[2]    = {0, 0};
    GLuint  ssfrSmoothTex[2]    = {0, 0};

    GLuint  ssfrThickFBO        = 0;
    GLuint  ssfrThickTex        = 0;
    GLuint  ssfrFoamTex         = 0;   // second attachment of ssfrThickFBO

    GLuint  ssfrBgFBO           = 0;
    GLuint  ssfrBgTex           = 0;
    GLuint  ssfrBgRBO           = 0;

    int     ssfrW               = 0;   // full-res target size (background + composite)
    int     ssfrH               = 0;
    int     ssfrFluidW          = 0;   // fluid pass size (depth/smooth/thickness/foam)
    int     ssfrFluidH          = 0;
    bool    ssfrHalfRes         = false;   // render fluid passes at half resolution (~4x faster)

    bool    useWaterRendering   = true;
    int     smoothIterations    = 5;
    float   worldFilterScale    = 6.0f;   // smoothing kernel width, in particle radii
    float   surfaceMerge        = 3.0f;   // narrow-range band, in particle radii
    float   thicknessStrength   = 0.05f;
    float   thicknessFalloff    = 4.0f;
    float   renderRadiusScale   = 1.3f;   // visual particle size multiplier (physics untouched)
    float   waterExtinction[3]  = {0.45f, 0.15f, 0.05f};
    float   thicknessScale      = 1.0f;
    float   sunDirWorld[3]      = {0.4f, 1.0f, 0.5f};
    float   sunColor[3]         = {1.0f, 0.97f, 0.9f};
    float   deepWaterColor[3]   = {0.02f, 0.08f, 0.25f};
    float   specularPower       = 256.0f;
    float   specularStrength    = 0.8f;
    float   refractionStrength  = 0.04f;
    float   fresnelBias         = 0.02f;

    void    InitSSFRBuffers(int w, int h);
    void    RenderSSFR(GLuint targetFBO, const Matrix4& proj) const;
    void    DestroySSFRBuffers();

    // --- Terrain mesh ---
    Shader* terrainShader     = nullptr;
    GLuint  terrainVAO        = 0;
    GLuint  terrainVBO        = 0;
    GLuint  terrainEBO        = 0;
    int     terrainIndexCount = 0;
    void    BuildTerrainMesh();

    // River generation UI state
    int     riverSeed         = 1;
    bool    pendingRiverRegen = false;

    // --- River bank / flow lines ---
    GLuint  riverBankVAO      = 0;
    GLuint  riverBankVBO      = 0;
    int     riverBankN        = 0; // vertices per strip (3 strips: left, right, center)
    bool    showRiverLines    = true;
    void    BuildRiverBankLines();
    void    DrawRiverBankLines(const Matrix4& proj) const;

public:
    explicit Scene0p();
    ~Scene0p() override;

    bool OnCreate() override;
    void OnDestroy() override;
    void HandleEvents(const SDL_Event& sdlEvent) override;
    void Update(const float deltaTime) override;
    void Render() const override;

    SPHFluidGPU* fluidGPU = nullptr;
};

#endif // SCENE0P_H