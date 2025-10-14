#pragma once
#include <glad.h>
#include <SDL.h>
#include <MMath.h>
#include "Shader.h"
#include "SPHFluid3D.h"
#include "Scene.h"

class Scene0p : public Scene {
public:
    Scene0p();
    ~Scene0p();

    bool OnCreate();
    void OnDestroy();
    void HandleEvents(const SDL_Event& sdlEvent);
    void Update(float deltaTime);
    void Render();

private:
    // Shaders
    Shader* impostorShader = nullptr;
    Shader* lineShader = nullptr;

    // Impostor VAO (GL_POINTS)
    GLuint impostorVAO = 0;

    // Wireframe box
    struct BoxWire {
        GLuint vao = 0, vbo = 0;
        void init();
        void draw(Shader& lineShader, const MATH::Matrix4& P, const MATH::Matrix4& V,
            const MATH::Matrix4& M, const MATH::Vec3& color);
        void destroy();
    } boxWire;

    // Fluid
    SPHFluidGPU* fluidGPU = nullptr;

    // Camera
    MATH::Matrix4 projectionMatrix, viewMatrix, modelMatrix;

    // Timing
    float dtAccumulator = 0.0f;   // member (no local static!)
    const float fixedDt = 1.0f / 120.0f;
    int   maxSubstepsPerFrame = 16;  // adaptive between 8 and 16

    // Render toggles
    bool renderFromSSBO = true;   // single toggle

    // Track last box to rebuild wireframe only on change
    MATH::Vec3 lastBoxCenter{}, lastBoxHalf{}, lastBoxEuler{};

    // Helpers
    void SetupImpostorVAO();
    void UpdateBoxWireframe();
    void DrawFluidImpostors();

    // Utility
    int CurrentViewportHeight() const;
};
