#include "SPHFluid3D.h"
#include <SDL_stdinc.h>
#include  < omp.h >
SPHFluid3D::SPHFluid3D(int numParticles) {
    int n = static_cast<int>(std::cbrt(numParticles));
    float spacing = h * 0.5f;
    for (int x = 0; x < n; ++x)
        for (int y = 0; y < n; ++y)
            for (int z = 0; z < n; ++z) {
                SPHParticle p;
                p.pos = MATH::Vec3(x * spacing, y * spacing, z * spacing);
                p.vel = MATH::Vec3(0, 0, 0);
                p.acc = MATH::Vec3(0, 0, 0);
                p.density = restDensity;
                p.pressure = 0;
                particles.push_back(p);
            }
}

float SPHFluid3D::poly6(float r2) const {
    float h2 = h * h;
    if (r2 >= 0 && r2 <= h2) {
        float coeff = 315.0f / (64.0f * M_PI * std::pow(h, 9));
        return coeff * std::pow(h2 - r2, 3);
    }
    return 0.0f;
}

MATH::Vec3 SPHFluid3D::spikyGrad(const MATH::Vec3& rij) const {
    float r = std::sqrt(rij.x * rij.x + rij.y * rij.y + rij.z * rij.z);
    if (r > 0 && r <= h) {
        float coeff = -45.0f / (M_PI * std::pow(h, 6));
        return rij * (coeff * std::pow(h - r, 2) / r);
    }
    return MATH::Vec3(0, 0, 0);
}

float SPHFluid3D::viscosityLaplacian(float r) const {
    if (r >= 0 && r <= h) {
        float coeff = 45.0f / (M_PI * std::pow(h, 6));
        return coeff * (h - r);
    }
    return 0.0f;
}

// Helper to compute a unique hash for a 3D cell
long long SPHFluid3D::computeCellHash(const MATH::Vec3& pos) const {
    int xi = static_cast<int>(std::floor(pos.x / cellSize));
    int yi = static_cast<int>(std::floor(pos.y / cellSize));
    int zi = static_cast<int>(std::floor(pos.z / cellSize));
    // Combine into a unique key (use a better hash in production)
    return (static_cast<long long>(xi) << 40) | (static_cast<long long>(yi) << 20) | zi;
}

void SPHFluid3D::clearGrid() {
    grid.clear();
}

void SPHFluid3D::buildGrid() {
    clearGrid();
    for (int i = 0; i < particles.size(); ++i) {
        long long hash = computeCellHash(particles[i].pos);
        grid[hash].particleIndices.push_back(i);
    }
}

void SPHFluid3D::findNeighbors(int pi, std::vector<int>& neighbors) const {
    neighbors.clear();
    const MATH::Vec3& pos = particles[pi].pos;
    int xi = static_cast<int>(std::floor(pos.x / cellSize));
    int yi = static_cast<int>(std::floor(pos.y / cellSize));
    int zi = static_cast<int>(std::floor(pos.z / cellSize));
    for (int dx = -1; dx <= 1; ++dx)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dz = -1; dz <= 1; ++dz) {
        long long hash = (static_cast<long long>(xi+dx) << 40) | (static_cast<long long>(yi+dy) << 20) | (zi+dz);
        auto it = grid.find(hash);
        if (it != grid.end()) {
            for (int pj : it->second.particleIndices) {
                if (pj != pi) neighbors.push_back(pj);
            }
        }
    }
}

void SPHFluid3D::step() {
    cellSize = h;
    buildGrid();

    std::vector<int> neighbors;

    // 1. Compute density and pressure (parallel)
    #pragma omp parallel for private(neighbors)
    for (int i = 0; i < particles.size(); ++i) {
        SPHParticle& pi = particles[i];
        pi.density = 0.0f;
        findNeighbors(i, neighbors);
        for (int j : neighbors) {
            const SPHParticle& pj = particles[j];
            MATH::Vec3 rij = pi.pos - pj.pos;
            float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
            if (r2 < h * h) {
                pi.density += mass * poly6(r2);
            }
        }
        // Self contribution
        pi.density += mass * poly6(0.0f);
        pi.pressure = gasConstant * (pi.density - restDensity);
    }

    // 3. Compute forces
    for (int i = 0; i < particles.size(); ++i) {
        SPHParticle& pi = particles[i];
        MATH::Vec3 fPressure(0, 0, 0), fViscosity(0, 0, 0);
        findNeighbors(i, neighbors);
        for (int j : neighbors) {
            const SPHParticle& pj = particles[j];
            if (i == j) continue;
            MATH::Vec3 rij = pi.pos - pj.pos;
            float r = std::sqrt(rij.x * rij.x + rij.y * rij.y + rij.z * rij.z);
            if (r < h) {
                // Pressure force
                fPressure += spikyGrad(rij) * (-mass * (pi.pressure + pj.pressure) / (2.0f * pj.density));
                // Viscosity force
                fViscosity += (pj.vel - pi.vel) * (viscosityLaplacian(r) * mass / pj.density);
            }
        }
        MATH::Vec3 fGravity = gravity * pi.density;
        pi.acc = (fPressure + viscosity * fViscosity + fGravity) / pi.density;
    }

    // 4. Integrate
    for (auto& p : particles) {
        p.vel += p.acc * timeStep;
        p.vel *= 0.99f; // Damping
        p.pos += p.vel * timeStep;
        // Simple boundary (cube)
        for (int i = 0; i < 3; ++i) {
            if (p.pos[i] < -5.0f) { p.pos[i] = -5.0f; p.vel[i] *= -0.3f; }
            if (p.pos[i] > 5.0f)  { p.pos[i] = 5.0f;  p.vel[i] *= -0.3f; }
        }
    }
}