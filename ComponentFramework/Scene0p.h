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
    Vec3    cameraPos = Vec3(0.0f, 0.0f, 10.0f);
    Vec3    cameraTarget = Vec3(0.0f, 0.0f, 0.0f);
    Vec3    cameraUp = Vec3(0.0f, 1.0f, 0.0f);

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