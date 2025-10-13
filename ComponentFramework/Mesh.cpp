#include "Mesh.h"


#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

Mesh::Mesh(const char* filename_):dateLength{0}, drawmode{0}, 
vao{0}, vbo{0}, instanceVBO{0}, instanceColorVBO{0}, instanceVelVBO{0}, instanceCount{0} {
	filename = filename_;
}

Mesh::~Mesh() {}

bool Mesh::OnCreate() {
	LoadModel(filename);
    StoreMeshData(GL_TRIANGLES);
    return true;
}

void Mesh::LoadModel(const char* filename) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    vertices.clear();
    normals.clear();
	uvCoords.clear();

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename)) {
        throw std::runtime_error(warn + err);
    }
    for (const auto& shape : shapes) {
        for (const auto& index : shape.mesh.indices) {
            Vec3 vertex{};
            vertex.x = attrib.vertices[3 * index.vertex_index + 0];
            vertex.y = attrib.vertices[3 * index.vertex_index + 1];
            vertex.z = attrib.vertices[3 * index.vertex_index + 2];
            
            Vec3 normal{};
            normal.x = attrib.normals[3 * index.normal_index + 0];
            normal.y = attrib.normals[3 * index.normal_index + 1];
            normal.z = attrib.normals[3 * index.normal_index + 2];

            Vec2 uvCoord{};
            uvCoord.x = attrib.texcoords[2 * index.texcoord_index + 0];
            uvCoord.y = attrib.texcoords[2 * index.texcoord_index + 1];

            vertices.push_back(vertex);
            normals.push_back(normal);
            uvCoords.push_back(uvCoord);
        }
    } 
}

void Mesh::StoreMeshData(GLenum drawmode_) {
    drawmode = drawmode_;
/// These just make the code easier for me to read
#define VERTEX_LENGTH 	(vertices.size() * (sizeof(Vec3)))
#define NORMAL_LENGTH 	(normals.size() * (sizeof(Vec3)))
#define TEXCOORD_LENGTH (uvCoords.size() * (sizeof(Vec2)))

	const int verticiesLayoutLocation = 0;
	const int normalsLayoutLocation = 1;
	const int uvCoordsLayoutLocation = 2;

	/// create and bind the VOA
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	/// Create and initialize vertex buffer object VBO
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, VERTEX_LENGTH + NORMAL_LENGTH + TEXCOORD_LENGTH, NULL, GL_STATIC_DRAW);

	/// assigns the addr of "points" to be the beginning of the array buffer "sizeof(points)" in length
	glBufferSubData(GL_ARRAY_BUFFER, 0, VERTEX_LENGTH, &vertices[0]);
	/// assigns the addr of "normals" to be "sizeof(points)" offset from the beginning and "sizeof(normals)" in length  
	glBufferSubData(GL_ARRAY_BUFFER, VERTEX_LENGTH, NORMAL_LENGTH, &normals[0]);
	/// assigns the addr of "texCoords" to be "sizeof(points) + sizeof(normals)" offset from the beginning and "sizeof(texCoords)" in length  
	glBufferSubData(GL_ARRAY_BUFFER, VERTEX_LENGTH + NORMAL_LENGTH, TEXCOORD_LENGTH, &uvCoords[0]);

	glEnableVertexAttribArray(verticiesLayoutLocation);
	glVertexAttribPointer(verticiesLayoutLocation, 3, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<void*>(0));
	glEnableVertexAttribArray(normalsLayoutLocation);
	glVertexAttribPointer(normalsLayoutLocation, 3, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<void*>(VERTEX_LENGTH));
	glEnableVertexAttribArray(uvCoordsLayoutLocation);
	glVertexAttribPointer(uvCoordsLayoutLocation, 2, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<void*>(VERTEX_LENGTH + NORMAL_LENGTH));

    dateLength = vertices.size();

    /// give back the memory used in these vectors. The data is safely stored in the GPU now
    vertices.clear();
    normals.clear();
    uvCoords.clear();

    /// Don't need these defines sticking around anymore si undefine them. 
#undef VERTEX_LENGTH
#undef NORMAL_LENGTH
#undef TEXCOORD_LENGTH

}

void Mesh::Render() const {
    glBindVertexArray(vao);
	glDrawArrays(drawmode, 0, dateLength);
	glBindVertexArray(0); // Unbind the VAO
}

void Mesh::Render(GLenum drawmode_) const {
    glBindVertexArray(vao);
	glDrawArrays(drawmode_, 0, dateLength);
	glBindVertexArray(0); // Unbind the VAO
}

void Mesh::RenderInstanced(GLenum drawmode_, size_t count) const {
    glBindVertexArray(vao);
    glDrawArraysInstanced(drawmode_, 0, dateLength, static_cast<GLsizei>(count));
    GLenum err = glGetError();
    if (err) std::cerr << "glDrawArraysInstanced error: 0x" << std::hex << err << std::dec << "\n";
    glBindVertexArray(0);
}

void Mesh::BindInstanceBuffer(GLuint vbo, GLsizei stride) {
    glBindVertexArray(this->vao);   // Your mesh's VAO
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(5);   // Choose location 4, for example
    glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, stride, (void*)0);
    glVertexAttribDivisor(5, 1);    // Advance once per instance
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
	//std::cout << "Instance buffer bound to VAO: " << vao << " with VBO: " << vbo << std::endl;
} 
void Mesh::SetInstanceData(const std::vector<Vec3>& positions) {
    if (instanceVBO == 0) {
        glGenBuffers(1, &instanceVBO);
    }
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(Vec3), positions.data(), GL_DYNAMIC_DRAW);

    // Match defaultVert.glsl (instancePos is location = 5)
    glEnableVertexAttribArray(5);
    glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3), (void*)0);
    glVertexAttribDivisor(5, 1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    instanceCount = positions.size();
}

void Mesh::SetInstanceColors(const std::vector<Vec3>& colors) {
    if (instanceColorVBO == 0) {
        glGenBuffers(1, &instanceColorVBO);
    }
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, instanceColorVBO);
    glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(Vec3), colors.data(), GL_DYNAMIC_DRAW);

    // Attribute 4: instance color
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3), (void*)0);
    glVertexAttribDivisor(4, 1); // Advance per instance

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void Mesh::SetInstanceVelocities(const std::vector<Vec3>& velocities) {
    if (instanceVelVBO == 0) {
        glGenBuffers(1, &instanceVelVBO);
    }
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVelVBO);
    glBufferData(GL_ARRAY_BUFFER, velocities.size() * sizeof(Vec3), velocities.data(), GL_DYNAMIC_DRAW);

    // Attribute 6: instance velocity
    glEnableVertexAttribArray(6);
    glVertexAttribPointer(6, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3), (void*)0);
    glVertexAttribDivisor(6, 1); // Advance per instance

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
	//std::cout << "Instance velocities set for " << velocities.size() << " instances." << std::endl;
}

void Mesh::OnDestroy() {
    glDeleteBuffers(1, &vbo);
    if (instanceVBO != 0) {
        glDeleteBuffers(1, &instanceVBO);

    }
    if (instanceColorVBO != 0) {
        glDeleteBuffers(1, &instanceColorVBO);
    }
    if (instanceVelVBO != 0) {
        glDeleteBuffers(1, &instanceVelVBO);
    }
	glDeleteVertexArrays(1, &vao);
}

/// Currently unused.
void Mesh::Update(const float deltaTime) {}