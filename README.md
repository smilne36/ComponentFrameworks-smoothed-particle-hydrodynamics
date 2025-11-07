What it does

Simulates fluid-like behavior using the SPH approach: nearby particles influence each other using smoothing kernels.
Provides a C++ simulation core (particle integration, pressure/viscosity, neighbor searches).
Optionally renders particles using OpenGL/GLSL for real-time visualization.
Includes tuning parameters to explore different fluid behaviors and stability limits.
High level features

Particle-based fluid simulation
Neighbor search (grid, spatial hashing, or naive N^2 — document which one you use)
Pressure solver, viscosity model, boundary handling
Optional visualization with GLSL shaders for point rendering / splatting
