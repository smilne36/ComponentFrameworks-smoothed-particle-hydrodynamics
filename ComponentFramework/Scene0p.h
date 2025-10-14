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

/// Forward declarations 
union SDL_Event;

class Scene0p : public Scene {
private:
    // Your existing members
    Body* sphere = nullptr;
    Shader* shader = nullptr;         // instanced sphere shader (defaultVert/defaultFrag)
    Mesh* mesh = nullptr;
    Matrix4 projectionMatrix;
    Matrix4 viewMatrix;
    Matrix4 modelMatrix;
    bool    drawInWireMode = true;
    bool    mouseDown = false;
    int     mouseX = 0, mouseY = 0;
    Vec3    cameraPos = Vec3(0.0f, 0.0f, 10.0f);
    Vec3    cameraTarget = Vec3(0.0f, 0.0f, 0.0f);
    Vec3    cameraUp = Vec3(0.0f, 1.0f, 0.0f);

    // Wireframe box
    Shader* lineShader = nullptr;
    GLuint  boxVAO = 0, boxVBO = 0;

    // Performance / sim controls
    bool    pendingReset = false;
    float   ballAnimTime = 0.0f;

    // Fixed-step accumulator (member, not local static)
    float   dtAccumulator = 0.0f;
    int     maxSubstepsPerFrame = 16;        // lowered cap

    // Render fast path (single toggle)
    bool    renderFromSSBO = true;

    // Track last box values to update wireframe only when something changed
    Vec3    lastBoxCenter{};
    Vec3    lastBoxHalf{};
    Vec3    lastBoxEuler{};

    // Point-impostor path
    bool    useImpostors = false;
    Shader* impostorShader = nullptr;        // shaders/particleImpostor.vert/frag
    GLuint  impostorVAO = 0;                 // empty VAO for gl_VertexID impostors

    // Helpers
    void    UpdateBoxWireframe();
    void    SetupImpostorVAO();
    void    DrawFluidImpostors() const;
    int     CurrentViewportHeight() const;

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
