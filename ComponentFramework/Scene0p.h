#ifndef SCENE0P_H
#define SCENE0P_H
#include "Scene.h"
#include "Vector.h"
#include <Matrix.h>
#include "SPHFluid3D.h"
#include <SDL.h>
#include "window.h" 
#include "Mesh.h"
#include "Shader.h"
#include "AudioReactive.h"
#include "ReelExport.h"
#include <glad.h>
#include <limits>
using namespace MATH;

union SDL_Event;

class Scene0p : public Scene {
private:
    Shader* shader = nullptr;
    Mesh* mesh = nullptr;
    Matrix4 projectionMatrix;
    Matrix4 viewMatrix;
    Matrix4 modelMatrix;
    bool    mouseDown = false;
    int     mouseX = 0, mouseY = 0;
    int     mouseButton = -1;       // -1=none, 1=left, 3=right
    Vec3    cameraPos = Vec3(0.0f, 5.0f, 22.0f);
    Vec3    cameraTarget = Vec3(0.0f, 0.0f, 0.0f);
    Vec3    cameraUp = Vec3(0.0f, 1.0f, 0.0f);
    float   camDist      = 22.0f;
    float   camAzimuth   = 0.0f;   // radians, rotation around Y
    float   camElevation = 0.22f;  // radians, pitch above horizon
    float   viewFarPlane = 300.0f; // camera draw distance; raise for huge containers

    Shader* lineShader = nullptr;
    GLuint  boxVAO = 0, boxVBO = 0;
    int     containerWireVerts = 24;   // vertex count currently in boxVBO
    bool    showContainerOutline = true;
    float   containerOutlineColor[3] = {0.85f, 0.95f, 1.0f};

    bool    pendingReset = false;
    float   ballAnimTime = 0.0f;

    float   dtAccumulator = 0.0f;
    int     maxSubstepsPerFrame = 16;

    int     uiParticleCount = 50000;   // Performance panel slider; applied on reset

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
    float   paletteFlow  = 0.0f;   // scrolls any palette over time (0 = static)
    float   patternScale = 1.0f;   // spatial frequency of world-space pattern palettes
    float   duoColorA[3] = {0.05f, 0.02f, 0.10f};
    float   duoColorB[3] = {1.00f, 0.35f, 0.75f};
    bool    showSkyBackground = false;                      // false = flat bgColor backdrop (water pops on black)
    float   bgColor[3]   = {0.0f, 0.0f, 0.0f};              // backdrop clear color (all render paths)
    float   skyColor[3]  = {0.40f, 0.55f, 0.65f};           // sky horizon color (reflections + optional backdrop)
    float   skyZenith[3] = {0.15f, 0.28f, 0.50f};           // sky zenith color
    float   envReflectColor[3] = {0.90f, 0.95f, 1.00f};     // tint on the reflected sky
    float   foamAmount   = 1.5f;
    float   exposure     = 1.0f;

    void    ApplyArtPreset(int which);
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

    // --- Audio Reactive (own phase accumulators, independent of manual Waves) ---
    AudioReactive* audioReactive = nullptr;
    bool    audioReactiveEnabled = false;
    float   audioMasterGain      = 1.0f;

    float   audioBassForce   = 8.0f,  audioBassThreshold   = 0.05f;
    float   audioBassWavelength = 10.0f, audioBassPhaseSpeed = 1.5f,  audioBassPhase   = 0.0f;
    float   audioMidForce    = 4.0f,  audioMidThreshold    = 0.05f;
    float   audioMidWavelength  = 3.0f,  audioMidRotSpeed    = 1.2f,  audioMidPhase    = 0.0f;
    float   audioTrebleForce = 1.5f,  audioTrebleThreshold = 0.05f;
    float   audioTrebleWavelength = 1.0f, audioTreblePhaseSpeed = 14.0f, audioTreblePhase = 0.0f;

    float   audioSizeKick    = 0.3f;   // bass -> particle render size
    float   audioShimmerKick = 0.5f;   // treble -> brightness pulse
    float   audioFoamKick    = 0.6f;   // mid -> foam boost
    float   audioHueKickDeg  = 0.0f;   // bass -> hue shift (beat color pulse), degrees at full envelope
    float   audioFlashKick   = 0.0f;   // bass -> brightness flash (multiplies with the treble shimmer)

    // --- Auto camera orbit (slow cinematic spin; works live and in reel export) ---
    bool    autoOrbitEnabled  = false;
    float   autoOrbitSpeedDeg = 8.0f;  // deg/sec; sign flips direction
    float   audioOrbitKick    = 0.0f;  // bass -> orbit speed multiplier kick

    // --- Vortex swirl (whirlpool around the container's Y axis) ---
    float   vortexBaseSwirl   = 0.0f;  // constant tangential accel; works without audio
    float   audioVortexForce  = 0.0f;  // mid -> extra swirl
    float   vortexInwardPull  = 0.0f;  // radial pull toward the axis

    // "Live" values fed to the renderer each frame. The base members stay the
    // user's pure slider settings; these are recomputed fresh every Update()
    // (base, or base*(1+kick*envelope) when audio-reactive is on), so there is
    // no compounding drift and no fighting the sliders.
    float   renderRadiusScaleLive = 1.3f;
    float   brightMulLive         = 1.0f;
    float   foamAmountLive        = 1.5f;
    float   hueShiftDegLive       = 0.0f;
    float   orbitSpeedDegLive     = 0.0f;

    // Applies one frame of audio reaction (wave impulses + vortex + the *Live
    // values), shared by the live reactor and the offline Reels render. Also
    // called with all-zero bands when audio is off, so every render mode runs
    // the exact same code path (base swirl etc. still apply).
    void    DriveAudioReaction(float bass, float mid, float treble, float dt);

    // Rebuilds cameraPos + viewMatrix from the spherical orbit parameters
    // (camAzimuth/camElevation/camDist around cameraTarget).
    void    RebuildOrbitCamera();

    // --- Reels Export (offline, frame-accurate, music-synced render) ---
    bool    reelExporting = false;
    int     reelFrame     = 0;
    int     reelFpsIdx    = 0;          // 0=30, 1=60
    int     reelResIdx    = 0;          // 0=1080x1920, 1=1080x1350, 2=1920x1080
    int     reelW = 1080, reelH = 1920;
    float   reelMaxSeconds = 0.0f;      // 0 = whole track
    int     reelSubstepCap = 0;         // 0 = accurate (full substeps); >0 caps for speed
    bool    reelSupersample = false;    // render frames at 2x + downsample (crisp, ~4x cost)
    char    reelAudioPath[512] = {0};
    char    reelOutDir[512]    = "reels";
    std::string  reelStatus;
    ReelAnalysis reelBands;
    GLuint  reelFBO = 0, reelTex = 0, reelRBO = 0;
    // Supersample render target (2x reelW/H); blitted down into reelFBO per frame
    GLuint  reelSSFBO = 0, reelSSTex = 0, reelSSRBO = 0;
    int     reelRenderW = 0, reelRenderH = 0;   // actual render size (reelW/H or 2x)
    bool    reelPrevHalfRes = false;            // ssfrHalfRes to restore after export
    unsigned reelStartMs = 0;           // SDL ticks at export start, for ETA
    void    StartReelExport();
    void    ReelExportStep();
    void    FinishReelExport(bool wroteBat);

    // --- Reel Preview (frame the live view as a 9:16 reel for OBS capture) ---
    // Renders the live sim into a portrait offscreen target at the reel aspect
    // (reelW:reelH), then blits it letterboxed to the window. Sim + audio still
    // run live; only the framing changes. For quick OBS grabs; the offline Reels
    // export stays the path for longer, higher-quality renders.
    bool    reelPreview = false;
    GLuint  previewFBO = 0, previewTex = 0, previewRBO = 0;
    int     previewW = 0, previewH = 0;   // portrait target size (fits the window)
    void    EnsurePreviewTarget();        // (re)build target when window/aspect changes
    void    DestroyPreviewTarget();

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