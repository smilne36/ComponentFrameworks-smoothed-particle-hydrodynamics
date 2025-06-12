#ifndef SCENE0P_H
#define SCENE0P_H
#include "Scene.h"
#include "Vector.h"
#include <Matrix.h>
#include "SPHFluid3D.h"
#include <SDL.h>
#include "window.h" 
#include "Body.h"

using namespace MATH;

/// Forward declarations 
union SDL_Event;
class Body;
class Mesh;
class Shader;

class Scene0p : public Scene {
private:
	Body* sphere;
	Shader* shader;
	Mesh* mesh;
	Matrix4 projectionMatrix;
	Matrix4 viewMatrix;
	Matrix4 modelMatrix;
	bool drawInWireMode;
	bool mouseDown = false;
	int mouseX = 0, mouseY = 0;
	Vec3 cameraPos = Vec3(0.0f, 0.0f, 10.0f);
	Vec3 cameraTarget = Vec3(0.0f, 0.0f, 0.0f);
	Vec3 cameraUp = Vec3(0.0f, 1.0f, 0.0f);
	GLuint posSSBO = 0, velSSBO = 0, accSSBO = 0, densitySSBO = 0, pressureSSBO = 0;
	GLuint computeShaderID = 0;
	float ballAnimTime = 0.0f;
	int pos;
	bool ballMoving = false;
	bool simulationRunning = false;

public:
	explicit Scene0p();
	virtual ~Scene0p();
	void SetPosition(const Vec3& position) { pos = static_cast<int>(position.x); }
	virtual bool OnCreate() override;
	virtual void OnDestroy() override;
	virtual void Update(const float deltaTime) override;
	virtual void Render() const override;
	virtual void HandleEvents(const SDL_Event& sdlEvent) override;
	SPHFluidGPU* fluidGPU;
};

#endif // SCENE0P_H