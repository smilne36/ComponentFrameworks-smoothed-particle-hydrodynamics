#include "SPHFluid3D.h"
#include "Debug.h"
#include <fstream>
#include <sstream>
#include <random> 

SPHFluidGPU::SPHFluidGPU(size_t numParticles_) : numParticles(numParticles_) {
    computeShader = LoadComputeShader("shaders/SPHFluid.comp");
    InitializeParticles();
    UploadDataToGPU();
}

SPHFluidGPU::~SPHFluidGPU() {
    glDeleteBuffers(1, &ssbo);
    glDeleteProgram(computeShader);
}

void SPHFluidGPU::InitializeParticles() {
    particles.resize(numParticles);
    int count = 0;

    int cubeSize = static_cast<int>(std::round(std::cbrt(numParticles)));
    float spacing = 0.3f;
    float half = (cubeSize - 1) * spacing * 0.5f;

    // Create random jitter generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> jitterDist(-0.05f, 0.05f); // small random offset

    for (int x = 0; x < cubeSize && count < numParticles; x++) {
        for (int y = 0; y < cubeSize / 3 && count < numParticles; y++) {
            for (int z = 0; z < cubeSize && count < numParticles; z++) {
                SPHParticle& p = particles[count++];
                p.pos = Vec4(
                    x * spacing - half + jitterDist(gen),
                    y * spacing - half + jitterDist(gen),
                    z * spacing - half + jitterDist(gen),
                    0.0f
                );
                p.vel = Vec4(0.0f, 0.0f, 0.0f, 0.0f);
                p.acc = Vec4(0.0f, 0.0f, 0.0f, 0.0f);
                p.density = 0.0f;
                p.pressure = 0.0f;
            }
        }
    }
    std::cout << "Total particles initialized: " << count << std::endl;
}

void SPHFluidGPU::UploadDataToGPU() {
    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(SPHParticle) * particles.size(), particles.data(), GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
}

void SPHFluidGPU::DispatchCompute() {
    glUseProgram(computeShader);
    float timeStep = 0.0001f;
    float h = 0.8f;
    float mass = 0.02f;
    float restDensity = 1000.0f;
    float gasConstant = 750.0f;
    float viscosity = 3.0f;
    MATH::Vec3 gravity = MATH::Vec3(0, -98.0, 0);   // More dramatic



    glUniform1f(glGetUniformLocation(computeShader, "timeStep"), timeStep);
    glUniform1f(glGetUniformLocation(computeShader, "h"), h);
    glUniform1f(glGetUniformLocation(computeShader, "mass"), mass);
    glUniform1f(glGetUniformLocation(computeShader, "restDensity"), restDensity);
    glUniform1f(glGetUniformLocation(computeShader, "gasConstant"), gasConstant);
    glUniform1f(glGetUniformLocation(computeShader, "viscosity"), viscosity);
    glUniform3f(glGetUniformLocation(computeShader, "gravity"), gravity.x, gravity.y, gravity.z);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glDispatchCompute((GLuint)((numParticles + 255) / 256), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glUseProgram(0);
}

void SPHFluidGPU::DownloadDataFromGPU() {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    void* ptr = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    if (ptr) {
        memcpy(particles.data(), ptr, sizeof(SPHParticle) * particles.size());
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
       
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
   
}

std::vector<Vec3> SPHFluidGPU::GetPositions() const {
    std::vector<Vec3> result;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    const SPHParticle* gpuData = (const SPHParticle*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    if (gpuData) {
        for (size_t i = 0; i < numParticles; ++i) {
            result.push_back(Vec3(gpuData[i].pos.x, gpuData[i].pos.y, gpuData[i].pos.z));
        }
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        /*for (int i = 0; i < 20 && i < numParticles; ++i) {
            std::cout << "P" << i << ": "
                << gpuData[i].pos.x << ", "
                << gpuData[i].pos.y << ", "
                << gpuData[i].pos.z << std::endl;
        }*/
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return result;
}

GLuint SPHFluidGPU::LoadComputeShader(const char* filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        Debug::FatalError("Failed to open compute shader file", __FILE__, __LINE__);
    }

    std::stringstream ss;
    ss << file.rdbuf();
    std::string sourceStr = ss.str();
    const char* source = sourceStr.c_str();

    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint isCompiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled);
    if (!isCompiled) {
        GLint maxLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);

        std::string errorLog(maxLength, ' ');
        glGetShaderInfoLog(shader, maxLength, &maxLength, &errorLog[0]);

        glDeleteShader(shader);
        Debug::FatalError("Compute shader compilation failed:\n" + errorLog, __FILE__, __LINE__);
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    glDeleteShader(shader);

    GLint isLinked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &isLinked);
    if (!isLinked) {
        GLint maxLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);
        std::string errorLog(maxLength, ' ');
        glGetProgramInfoLog(program, maxLength, &maxLength, &errorLog[0]);

        glDeleteProgram(program);
        Debug::FatalError("Compute shader link failed:\n" + errorLog, __FILE__, __LINE__);
    }

    return program;
}
