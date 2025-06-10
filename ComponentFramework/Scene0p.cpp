#include <glew.h>
#include <iostream>
#include <SDL.h>
#include "Scene0p.h"
#include <MMath.h>
#include "Debug.h"
#include "Mesh.h"
#include "Shader.h"
#include "Body.h"
#include <cmath> 


Scene0p::Scene0p() :sphere{nullptr}, shader{nullptr}, mesh{nullptr} ,
drawInWireMode{true}, mouseX{0}, mouseY{0}, mouseDown{false}, ballAnimTime{0.0f} {
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
	ballMoving = false;
	
	mesh = new Mesh("meshes/Sphere.obj");
	mesh->OnCreate();

	shader = new Shader("shaders/defaultVert.glsl", "shaders/defaultFrag.glsl");
	if (shader->OnCreate() == false) {
		std::cout << "Shader failed ... we have a problem\n";
	}

	projectionMatrix = MMath::perspective(45.0f, (16.0f / 9.0f), 0.5f, 100.0f);
	viewMatrix = MMath::lookAt(Vec3(0.0f, 0.0f, 10.0f), Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));
	modelMatrix.loadIdentity();

	
	fluid = new SPHFluid3D(500);

	return true;
}

void Scene0p::OnDestroy() {
	Debug::Info("Deleting assets Scene0: ", __FILE__, __LINE__);
	sphere->OnDestroy();
	delete sphere;

	mesh->OnDestroy();
	delete mesh;

	shader->OnDestroy();
	delete shader;

	
}

void Scene0p::HandleEvents(const SDL_Event &sdlEvent) {
    switch (sdlEvent.type) {
    case SDL_KEYDOWN:
        switch (sdlEvent.key.keysym.scancode) {
            case SDL_SCANCODE_W: cameraPos.y += 0.2f; break;
            case SDL_SCANCODE_S: cameraPos.y -= 0.2f; break;
            case SDL_SCANCODE_A: cameraPos.x -= 0.2f; break;
            case SDL_SCANCODE_D: cameraPos.x += 0.2f; break;
            case SDL_SCANCODE_R: cameraPos.z += 0.2f; break;
            case SDL_SCANCODE_E: cameraPos.z -= 0.2f; break;
            case SDL_SCANCODE_Z: // toggle wireframe
                drawInWireMode = !drawInWireMode;
                break;
            case SDL_SCANCODE_T:
                ballMoving = true;
                break;
        }
        break;
    case SDL_MOUSEMOTION:
        mouseX = sdlEvent.motion.x;
        mouseY = sdlEvent.motion.y;
        break;
    case SDL_MOUSEBUTTONDOWN:
        if (sdlEvent.button.button == SDL_BUTTON_LEFT)
            mouseDown = true;
        break;
    case SDL_MOUSEBUTTONUP:
        if (sdlEvent.button.button == SDL_BUTTON_LEFT)
            mouseDown = false;
        break;
    case SDL_SCANCODE_SPACE:
        if (true)
        {
            simulationRunning = !simulationRunning;
        }
        
        break;
    default:
        break;
    }
}

void Scene0p::Update(const float deltaTime) {
    static int lastMouseX = mouseX, lastMouseY = mouseY;
    static Vec3 lastMouseWorld;

    // Animate the ball side-to-side
    ballAnimTime += deltaTime;
    float ballX = std::sin(ballAnimTime) * 3.0f; // amplitude 3 units
    float ballY = 0.0f;
    float ballZ = 0.0f;
    if (sphere) {
        sphere->SetPosition(Vec3(ballX, ballY, ballZ));
    }

    if (mouseDown && fluid) {
        // Convert mouse (mouseX, mouseY) to world coordinates
        float ndcX = (2.0f * mouseX) / 2560 - 1.0f;
        float ndcY = 1.0f - (2.0f * mouseY) / 1440;
        Vec3 mouseWorld;
        mouseWorld.x = ndcX * 8.0f;
        mouseWorld.y = ndcY * 4.5f;
        mouseWorld.z = 0.0f;

        // Calculate mouse movement in world space
        Vec3 mouseDelta = mouseWorld - lastMouseWorld;

        float radius = 1.0f; // Influence radius
        float sigma2 = radius * radius * 0.5f; // For Gaussian falloff

        for (auto& p : fluid->particles) {
            Vec3 diff = p.pos - mouseWorld;
            float dist2 = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
            float dist = std::sqrt(dist2);

            float falloff = std::exp(-dist2 / sigma2);
            Vec3 force = mouseDelta * 20.0f * falloff * deltaTime;
            p.vel += force;
            p.lastMouseForce = force; // Store for visualization
        }

        lastMouseWorld = mouseWorld;
        lastMouseX = mouseX;
        lastMouseY = mouseY;
    }
    else {
        // Reset lastMouseWorld when not dragging
        float ndcX = (2.0f * mouseX) / 2560 - 1.0f;
        float ndcY = 1.0f - (2.0f * mouseY) / 1440;
        lastMouseWorld.x = ndcX * 8.0f;
        lastMouseWorld.y = ndcY * 4.5f;
        lastMouseWorld.z = 0.0f;
    }

    

    if (sphere) {
       
        if (sphere && fluid) {
            Vec3 ballPos = sphere->GetPosition();
            float ballRadius = 0.5f; // Or use sphere->GetRadius() if you want

            for (auto& p : fluid->particles) {
                Vec3 toParticle = p.pos - ballPos;
                float dist = std::sqrt(toParticle.x * toParticle.x + toParticle.y * toParticle.y + toParticle.z * toParticle.z);
                float minDist = ballRadius + 0.05f; // 0.05f: particle radius (adjust as needed)
                if (dist < minDist) {
                    // Move particle to the surface of the ball
                    Vec3 normal = toParticle / (dist + 1e-6f);
                    p.pos = ballPos + normal * minDist;
                    // Reflect velocity (simple elastic collision)
                    float vDotN = p.vel.x * normal.x + p.vel.y * normal.y + p.vel.z * normal.z;
                    p.vel = p.vel - normal * (2.0f * vDotN);
                    // Optional: dampen velocity
                    p.vel *= 0.8f;
                }
            }
        }
        viewMatrix = MMath::lookAt(cameraPos, cameraTarget, cameraUp);
       
        fluid->step();

    }
}
void Scene0p::Render() const {
	/// Set the background color then clear the screen
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if(drawInWireMode){
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	}else{
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
	glUseProgram(shader->GetProgram());
	glUniformMatrix4fv(shader->GetUniformID("projectionMatrix"), 1, GL_FALSE, projectionMatrix);
	glUniformMatrix4fv(shader->GetUniformID("viewMatrix"), 1, GL_FALSE, viewMatrix);
	glUniformMatrix4fv(shader->GetUniformID("modelMatrix"), 1, GL_FALSE, modelMatrix);
	//mesh->Render(GL_TRIANGLES);

	// --- Render SPH particles as points ---
	if (fluid && mesh) {
		std::vector<Vec3> positions, colors;
		for (const auto& p : fluid->particles) {
			positions.push_back(p.pos);
			float densityNorm = (p.density - fluid->restDensity) / fluid->restDensity;
			densityNorm = std::max(0.0f, std::min(1.0f, densityNorm));
			colors.push_back(Vec3(0.2f, 0.4f + densityNorm, 1.0f - densityNorm));
		}
		mesh->SetInstanceData(positions);
		mesh->SetInstanceColors(colors);

		Matrix4 scaleMat = MMath::scale(0.1f, 0.1f, 0.1f);
		glUniformMatrix4fv(shader->GetUniformID("modelMatrix"), 1, GL_FALSE, scaleMat);
		mesh->RenderInstanced(GL_TRIANGLES, positions.size());
	}

	// --- Render the moving ball ---
	//if (sphere && mesh && shader) {
	//	glUseProgram(shader->GetProgram());
	//	Matrix4 ballModelMat = MMath::translate(sphere->GetPosition()) * MMath::scale(0.5f, 0.5f, 0.5f);
	//	glUniformMatrix4fv(shader->GetUniformID("modelMatrix"), 1, GL_FALSE, ballModelMat);
	//	mesh->Render(GL_TRIANGLES);
	//	glUseProgram(0);
	//}

	// --- Render force vectors as lines ---
	glUseProgram(0); // Use fixed-function pipeline for lines
	glColor3f(1.0f, 0.2f, 0.2f); // Red color for force vectors
	glBegin(GL_LINES);
	for (const auto& p : fluid->particles) {
		if (p.lastMouseForce.x != 0.0f || p.lastMouseForce.y != 0.0f || p.lastMouseForce.z != 0.0f) {
			glVertex3f(p.pos.x, p.pos.y, p.pos.z);
			// Scale the force vector for visibility
			glVertex3f(p.pos.x + p.lastMouseForce.x * 10.0f,
					   p.pos.y + p.lastMouseForce.y * 10.0f,
					   p.pos.z + p.lastMouseForce.z * 10.0f);
		}
	}
	glEnd();



	glUseProgram(0);
}
