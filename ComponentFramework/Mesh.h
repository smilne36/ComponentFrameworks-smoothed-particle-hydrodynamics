#pragma once

#include "glad.h"
#include <vector>
#include <Vector.h>
using namespace MATH;

class Mesh {
	Mesh(const Mesh&) = delete;
	Mesh(Mesh&&) = delete;
	Mesh& operator = (const Mesh&) = delete;
	Mesh& operator = (Mesh&&) = delete;

private:
	const char* filename;
	std::vector<Vec3> vertices;
	std::vector<Vec3> normals;
	std::vector<Vec2> uvCoords;
	size_t dateLength;
	GLenum drawmode;

	/// Private helper methods
	void LoadModel(const char* filename);
	void StoreMeshData(GLenum drawmode_);
	GLuint vao, vbo;
	GLuint instanceVBO = 0;
	GLuint instanceColorVBO = 0;
	size_t instanceCount = 0;
	GLuint instanceVelVBO = 0;

public:
	
	Mesh(const char* filename_);
	~Mesh();
	bool OnCreate();
	void OnDestroy();
	void Update(const float deltaTime);
	void Render() const;
	void Render(GLenum drawmode) const;
	void BindInstanceBuffer(GLuint vbo, GLsizei stride);
	void SetInstanceData(const std::vector<Vec3>& positions);
	void SetInstanceColors(const std::vector<Vec3>& colors);
	void SetInstanceVelocities(const std::vector<Vec3>& velocities);
	void RenderInstanced(GLenum drawmode, size_t count) const;
	

};

