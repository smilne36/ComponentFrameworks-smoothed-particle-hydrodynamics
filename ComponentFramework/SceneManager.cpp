#include <SDL.h>
#include "SceneManager.h"
#include "Timer.h"
#include "Window.h"
#include "Scene0p.h"

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#define IMGUI_IMPL_OPENGL_LOADER_GLAD
#include "imgui_impl_opengl3.h"

SceneManager::SceneManager() :
    currentScene{ nullptr }, window{ nullptr }, timer{ nullptr },
    fps(60), isRunning{ false }, fullScreen{ false } {
    Debug::Info("Starting the SceneManager", __FILE__, __LINE__);
}

SceneManager::~SceneManager() {
    Debug::Info("Deleting the SceneManager", __FILE__, __LINE__);

    // ImGui shutdown (SDL2 + OpenGL3)
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    if (currentScene) {
        currentScene->OnDestroy();
        delete currentScene;
        currentScene = nullptr;
    }
    if (timer) {
        delete timer;
        timer = nullptr;
    }
    if (window) {
        delete window;
        window = nullptr;
    }
}

bool SceneManager::Initialize(std::string name_, int width_, int height_) {
    window = new Window();
    if (!window->OnCreate(name_, width_, height_)) {
        Debug::FatalError("Failed to initialize Window object", __FILE__, __LINE__);
        return false;
    }

    timer = new Timer();
    if (timer == nullptr) {
        Debug::FatalError("Failed to initialize Timer object", __FILE__, __LINE__);
        return false;
    }

    // ImGui init (SDL2 + OpenGL3)
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplSDL2_InitForOpenGL(window->getWindow(), SDL_GL_GetCurrentContext());
    ImGui_ImplOpenGL3_Init("#version 130"); // works on GL 3.x+ (and GL 4.x core)

    // Default first scene
    BuildNewScene(SCENE_NUMBER::SCENE0p);
    return true;
}

void SceneManager::Run() {
    timer->Start();
    isRunning = true;
    while (isRunning) {
        HandleEvents();

        // Begin ImGui frame BEFORE Scene Update
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        // Optional: quick sanity UI
        static bool show_demo_window = true;
        if (show_demo_window) ImGui::ShowDemoWindow(&show_demo_window);
        ImGui::Begin("Stats");
        ImGui::Text("Delta: %.3f ms (%.1f FPS)", timer->GetDeltaTime() * 1000.0f, 1.0f / timer->GetDeltaTime());
        ImGui::Checkbox("Show Demo", &show_demo_window);
        ImGui::End();

        timer->UpdateFrameTicks();
        if (currentScene) {
            currentScene->Update(timer->GetDeltaTime()); // Scene0p builds its own ImGui here
            currentScene->Render();
        }

        // Render ImGui AFTER scene render
        ImGui::Render();

        // Ensure UI is drawn solid (scene may have left GL in wireframe)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        SDL_GL_SwapWindow(window->getWindow());
        SDL_Delay(timer->GetSleepTime(fps));
    }
}

void SceneManager::HandleEvents() {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        ImGui_ImplSDL2_ProcessEvent(&e); // feed SDL2 events to ImGui

        if (e.type == SDL_QUIT) {
            isRunning = false;
            return;
        }

        if (e.type == SDL_KEYDOWN) {
            switch (e.key.keysym.scancode) {
            case SDL_SCANCODE_ESCAPE:
            case SDL_SCANCODE_Q:
                isRunning = false; return;
            case SDL_SCANCODE_F1:
            case SDL_SCANCODE_F2:
            case SDL_SCANCODE_F3:
            case SDL_SCANCODE_F4:
            case SDL_SCANCODE_F5:
                BuildNewScene(SCENE_NUMBER::SCENE0p);
                break;
            default: break;
            }
        }

        if (currentScene == nullptr) {
            Debug::FatalError("No currentScene", __FILE__, __LINE__);
            isRunning = false;
            return;
        }
        currentScene->HandleEvents(e);
    }
}

bool SceneManager::BuildNewScene(SCENE_NUMBER scene) {
    bool status = false;

    if (currentScene != nullptr) {
        currentScene->OnDestroy();
        delete currentScene;
        currentScene = nullptr;
    }

    switch (scene) {
    case SCENE_NUMBER::SCENE0p:
        currentScene = new Scene0p();
        status = currentScene->OnCreate();
        break;
    default:
        Debug::Error("Incorrect scene number assigned in the manager", __FILE__, __LINE__);
        return false;
    }
    return status;
}