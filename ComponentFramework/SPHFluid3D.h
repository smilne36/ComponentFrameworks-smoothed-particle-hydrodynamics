#pragma once
#include <vector>
#include <cmath>
#include <unordered_map>
#include "Vector.h" // For MATH::Vec3

struct SPHParticle {
    MATH::Vec3 pos, vel, acc;
    float density, pressure;
    MATH::Vec3 lastMouseForce = MATH::Vec3(0,0,0); // Add this line
};

struct GridCell {  
    std::vector<int> particleIndices;  
};

class SPHFluid3D {
public:
    std::vector<SPHParticle> particles;
    float restDensity = 1000.0f;
    float gasConstant = 461.52f;
    float viscosity = 1.0002f;
    float timeStep = 0.0005f;
    float mass = 18.01528f;
    float h = 1.0f; // Smoothing radius
    MATH::Vec3 gravity = MATH::Vec3(0.0f, -9.8f, 0.0f);

    SPHFluid3D(int numParticles);

    void step();
    void findNeighbors(int pi, std::vector<int>& neighbors) const;
private:
    float poly6(float r2) const;
    MATH::Vec3 spikyGrad(const MATH::Vec3& rij) const;
    float viscosityLaplacian(float r) const;

    float cellSize;
    std::unordered_map<long long, GridCell> grid;

    long long computeCellHash(const MATH::Vec3& pos) const;
    void buildGrid();
    void clearGrid();

};