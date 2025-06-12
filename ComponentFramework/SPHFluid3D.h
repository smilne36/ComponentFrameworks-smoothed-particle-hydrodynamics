#pragma once
#include <vector>
#include <Vector.h>
#include <glad.h>

using namespace MATH;

class SPHFluidGPU {
public:
    SPHFluidGPU(size_t numParticles);
    ~SPHFluidGPU();

    void DispatchCompute();
    std::vector<Vec3> GetPositions() const;

private:
    struct SPHParticle {
        Vec4 pos;      // xyz = position, w = unused
        Vec4 vel;      // xyz = velocity, w = unused
        Vec4 acc;      // xyz = acceleration, w = unused
        float density;
        float pressure;
        float pad[2];
    };

 

    GLuint computeShader = 0;
    GLuint ssbo = 0;
    size_t numParticles = 0;
    std::vector<SPHParticle> particles;

    

    GLuint LoadComputeShader(const char* filePath);
    void InitializeParticles();
    void UploadDataToGPU();
    void DownloadDataFromGPU();
};
