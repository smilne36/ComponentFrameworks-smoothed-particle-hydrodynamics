# ComponentFramework SPH Fluid Simulation

## Overview
Real-time Smoothed Particle Hydrodynamics (SPH) fluid simulation implemented in C++17 using OpenGL 4.x compute shaders. Particles reside in Shader Storage Buffer Objects (SSBOs) and are updated fully on the GPU in staged compute passes (clear grid, build grid, SPH force integration, OBB constraints, optional wave impulse). Rendering supports instanced mesh spheres or point impostors with multiple visualization modes.

## Core Features
- GPU SPH solver (density, pressure, viscosity, gravity, surface tension).
- Uniform 3D spatial hash/grid built each substep for neighbor traversal.
- Rotated Oriented Bounding Box (OBB) collision with restitution + friction.
- Optional ghost boundary particles and activation helper grids per face.
- Optional radix sort pass for experimentation (currently not required).
- Wave impulse compute shader injects directional velocity waves.
- Persistent mapped VBO for fast instanced rendering without CPU copies.
- ImGui real-time parameter editing (presets, box transform, waves, visualization).
- Multiple rendering paths:
  - Instanced mesh (defaultVert/defaultFrag).
  - Point impostors (particleImpostor shaders).
  - Wireframe OBB outline (line shader).
- Visualization modes: Height, Speed, Pressure, Density, InstanceColor.

## Folder / Module Summary
- Scene0p.*: Main scene tying UI, simulation loop, rendering paths.
- SPHFluid3D.*: Simulation class (particle data, buffers, compute dispatch).
- shaders/*.comp / *.glsl: Compute + render shader stages.
- Mesh.*: OBJ loader (tinyobjloader), VAO/VBO setup, instancing.
- Body, SceneManager, etc.: Framework utilities.

## GPU Pipeline (per substep)
1. ClearGrid.comp: Reset cell heads.
2. BuildGrid.comp: Insert particles into linked lists (cellHeadSSBO + particleNext).
3. (Optional) RadixSort.comp path (CPU sort currently).
4. SPHFluid.comp: Density/pressure, force accumulation and integration.
5. OBBConstraints.comp: Collision/response in rotated box.
6. WaveImpulse.comp (on demand / continuous toggle).

Barriers (`glMemoryBarrier`) inserted between stages to ensure write visibility.

## Key Buffers
- ssbo: SPHParticle array (position, velocity, acceleration, density, pressure, flags).
- cellHeadSSBO / particleNextSSBO / particleCellSSBO: Grid + per-particle cell links.
- cellKeySSBO / sortIdxSSBO / sortTmpSSBO: Optional sorting support.
- fluidVBO (persistent mapped): Packed fluid-only positions for instanced draw.

## Build & Dependencies
- C++17 compiler.
- OpenGL loader: glad.
- SDL2 for window/events.
- ImGui + backend (imgui_impl_sdl2 / imgui_impl_opengl3).
- tinyobjloader for mesh loading.
- Ensure shader files under `shaders/` are deployed relative to working directory.

## Configuration / Parameters (ImGui)
- Physical: h, mass, restDensity, gasConstant, viscosity, gravityY, surfaceTension, timeStep.
- Box: center, half extents, Euler rotation, wall restitution, friction.
- Rendering: color mode, range, impostor toggle, wireframe toggle.
- Performance toggles: SSBO direct render, ghost particles, grid sort (experimental).
- Wave injection: amplitude, wavelength, phase speed, direction, Y band filter.

## Controls (Keyboard)
- WASD / RE: Move camera position axes.
- Z: Toggle wireframe.
- Buttons in UI: presets, reset, wave impulse, camera fit.

## Simulation Loop (Scene0p::Update)
- Accumulate frame time.
- Perform fixed-time substeps (`param_timeStep`).
- Apply optional continuous wave impulse.
- Rebuild wireframe if OBB parameters change.
- Handle pending full simulation reset (recreate particles + buffers).

## Data Structures
Ghost grids: lightweight 2D bucket arrays per face to accelerate ghost activation and queries.

## Shaders (Selected Uniforms)
- SPHFluid.comp: gridSize, cellSize, h, mass, restDensity, gasConstant, viscosity, gravity, surfaceTension.
- OBBConstraints.comp: uBoxRot (3x3), uBoxCenter, uBoxHalf, uRestitution, uFriction.
- particleImpostor.vert/frag & defaultVert/Frag: projectionMatrix, viewMatrix, modelMatrix, colorMode, vizRange, heightMinMax, particleRadius.

## Performance Notes
- Persistent mapped VBO eliminates per-frame CPU copies in SSBO render path.
- Adjustable max substeps per frame to cap update cost.
- Ghost particle updates optional (readback cost).
- Sorting pass currently CPU-side after key readback; disabled by default.

## Extensibility Ideas
- Replace CPU radix sort with full GPU parallel radix sort.
- Add adaptive time stepping based on CFL condition.
- Introduce multi-fluid or phase change behavior.
- Expand visualization (vorticity, pressure gradients).
- Emit real splashes via particle classification (surface detection).

## Reset Path
`ResetSimulation()` fully reinitializes particles, grid buffers, SSBO, sort buffers, and remaps VBO.

## Error Handling
- Shader compilation/linking uses `Debug::FatalError`.
- glGetError check after instanced draw.
- Bounds clamp in ghost grid insertion.

## License / Attribution
- Uses tinyobjloader (MIT).
- Uses ImGui (MIT).
- stb_* modified headers inside project (public domain / MIT dual).

## Quick Start
1. Initialize SDL + OpenGL context.
2. Create Scene0p, call OnCreate().
3. Enter main loop: poll events, ImGui frame, scene Update(delta), scene Render(), ImGui render.
4. Clean up via OnDestroy().

## Minimal Pseudocode
