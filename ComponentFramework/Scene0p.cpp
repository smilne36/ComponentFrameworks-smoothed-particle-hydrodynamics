#include <iostream>
#include <SDL.h>
#include "Scene0p.h"
#include <MMath.h>
#include "Debug.h"
#include "Mesh.h"
#include "Shader.h"
#include "Body.h"
#include <cmath>
#include "SPHFluid3D.h"
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "SceneManager.h"


Scene0p::Scene0p() :
    sphere{ nullptr }, shader{ nullptr }, mesh{ nullptr },
    drawInWireMode{ true }, mouseX{ 0 }, mouseY{ 0 }, mouseDown{ false }, ballAnimTime{ 0.0f },
    fluidGPU{ nullptr }
{
    Debug::Info("Created Scene0: ", __FILE__, __LINE__);
}

Scene0p::~Scene0p() {
    Debug::Info("Deleted Scene0: ", __FILE__, __LINE__);
}

bool Scene0p::OnCreate() {
    Debug::Info("Loading assets Scene0: ", __FILE__, __LINE__);

    sphere = new Body();
    sphere->OnCreate();
    sphere->SetPosition(Vec3(0.0f, -20.0f, 0.0f));

    mesh = new Mesh("meshes/Sphere.obj");
    mesh->OnCreate();

    shader = new Shader("shaders/defaultVert.glsl", "shaders/defaultFrag.glsl");
    if (!shader->OnCreate()) {
        std::cerr << "Shader failed to load\n";
        return false;
    }

    projectionMatrix = MMath::perspective(45.0f, 16.0f / 9.0f, 0.5f, 100.0f);
    viewMatrix = MMath::lookAt(cameraPos, cameraTarget, cameraUp);
    modelMatrix.loadIdentity();

    fluidGPU = new SPHFluidGPU(50000); // ? You can increase this!

    return true;
}

void Scene0p::OnDestroy() {
    Debug::Info("Deleting assets Scene0: ", __FILE__, __LINE__);
    if (sphere) {
        sphere->OnDestroy();
        delete sphere;
    }

    if (mesh) {
        mesh->OnDestroy();
        delete mesh;
    }

    if (shader) {
        shader->OnDestroy();
        delete shader;
    }

    if (fluidGPU) {
        delete fluidGPU;
    }
}

void Scene0p::HandleEvents(const SDL_Event& sdlEvent) {
    switch (sdlEvent.type) {
    case SDL_KEYDOWN:
        switch (sdlEvent.key.keysym.scancode) {
        case SDL_SCANCODE_W: cameraPos.y += 0.2f; break;
        case SDL_SCANCODE_S: cameraPos.y -= 0.2f; break;
        case SDL_SCANCODE_A: cameraPos.x -= 0.2f; break;
        case SDL_SCANCODE_D: cameraPos.x += 0.2f; break;
        case SDL_SCANCODE_R: cameraPos.z += 0.2f; break;
        case SDL_SCANCODE_E: cameraPos.z -= 0.2f; break;
        case SDL_SCANCODE_Z: drawInWireMode = !drawInWireMode; break;
        }
        break;
    }
}

void Scene0p::Update(const float deltaTime) {
    ballAnimTime += deltaTime;
    float ballX = std::sin(ballAnimTime) * 3.0f;
    if (sphere) {
        sphere->SetPosition(Vec3(ballX, 0.0f, 0.0f));
    }

    viewMatrix = MMath::lookAt(cameraPos, cameraTarget, cameraUp);

    if (fluidGPU) {
        fluidGPU->DispatchCompute();
        fluidGPU->UpdateFluidVBOFromGPU();

        std::vector<Vec3> velocities;
        velocities.reserve(fluidGPU->particles.size());
        for (const auto& p : fluidGPU->particles) {
            if (p.isGhost == 0) {
                velocities.emplace_back(p.vel.x, p.vel.y, p.vel.z);
            }
        }
        mesh->SetInstanceVelocities(velocities);
    }
}

void Scene0p::Render() const {
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (drawInWireMode) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    }
    else {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    glUseProgram(shader->GetProgram());
    glUniformMatrix4fv(shader->GetUniformID("projectionMatrix"), 1, GL_FALSE, projectionMatrix);
    glUniformMatrix4fv(shader->GetUniformID("viewMatrix"), 1, GL_FALSE, viewMatrix);

    if (fluidGPU && mesh) {
        // 1. Get the mapped VBO (must hold only fluid particle positions)
        GLuint vbo = fluidGPU->GetFluidVBO();       // returns VBO for instance positions
        size_t numInstances = fluidGPU->GetNumFluids(); // returns fluid particle count

        // 2. Bind the instance buffer for position (vec4 per instance)
        mesh->BindInstanceBuffer(vbo, sizeof(Vec4)); // or 4 * sizeof(float)

        // 3. Set scale/model matrix as usual
        Matrix4 scaleMat = MMath::scale(0.1f, 0.1f, 0.1f);
        glUniformMatrix4fv(shader->GetUniformID("modelMatrix"), 1, GL_FALSE, scaleMat);

        // 4. Draw fluid instances
        mesh->RenderInstanced(GL_TRIANGLES, numInstances); // Only fluids!
    }
   
    glUseProgram(0);
}