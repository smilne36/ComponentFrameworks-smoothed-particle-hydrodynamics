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

    bool    pendingReset = false;
    float   ballAnimTime = 0.0f;

    float   dtAccumulator = 0.0f;
    int     maxSubstepsPerFrame = 16;

    bool    renderFromSSBO = true;

    Vec3    lastBoxCenter{};
    Vec3    lastBoxHalf{};
    Vec3    lastBoxEuler{};

    bool    useImpostors = false;
    Shader* impostorShader = nullptr;
    GLuint  impostorVAO = 0;

    // Visualization state
    int     vizMode = 0;          // 0=Height,1=Speed,2=Pressure,3=Density,4=InstanceColor
    float   vizRangeMin = 0.0f;
    float   vizRangeMax = 10.0f;

    void    UpdateBoxWireframe();
    void    SetupImpostorVAO();
    void    DrawFluidImpostors() const;
    int     CurrentViewportHeight() const;

    // --- Screen-Space Fluid Rendering ---
    Shader* ssfrDepthShader     = nullptr;
    Shader* ssfrSmoothShader    = nullptr;
    Shader* ssfrThickShader     = nullptr;
    Shader* ssfrCompositeShader = nullptr;
    GLuint  ssfrQuadVAO         = 0;

    GLuint  ssfrDepthFBO        = 0;
    GLuint  ssfrDepthTex        = 0;
    GLuint  ssfrDepthRBO        = 0;

    GLuint  ssfrSmoothFBO[2]    = {0, 0};
    GLuint  ssfrSmoothTex[2]    = {0, 0};

    GLuint  ssfrThickFBO        = 0;
    GLuint  ssfrThickTex        = 0;

    GLuint  ssfrBgFBO           = 0;
    GLuint  ssfrBgTex           = 0;
    GLuint  ssfrBgRBO           = 0;

    int     ssfrW               = 0;
    int     ssfrH               = 0;

    bool    useWaterRendering   = true;
    int     smoothIterations    = 5;
    float   smoothFilterRadius  = 10.0f;
    float   smoothDepthFalloff  = 0.1f;
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
    void    RenderSSFR() const;
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
    void    DrawRiverBankLines() const;

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