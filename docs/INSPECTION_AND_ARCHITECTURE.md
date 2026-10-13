# KLUSTOR × KRPO: Phase 0 Inspection & Integration Architecture Report

## 1. Actual Repository & Git Layout

### Repositories
- **KLUSTOR (Parent Game):** `c:\Users\Ekaansh\OneDrive\Desktop\AB\projects\thePhotoNeverLies`
  - Git Root: `c:\Users\Ekaansh\OneDrive\Desktop\AB\projects\thePhotoNeverLies\.git`
  - Technology: React 18, Vite 5.4, TypeScript 5.5, Three.js 0.186, React Three Fiber 8.18, Zustand 4.5.
  - Role: The single source of truth for the playable 3D arcade racing game and livery design studio.
- **KRPO (Nested Optimizer):** `c:\Users\Ekaansh\OneDrive\Desktop\AB\projects\thePhotoNeverLies\KRPO`
  - Git Root: `c:\Users\Ekaansh\OneDrive\Desktop\AB\projects\thePhotoNeverLies\KRPO\.git`
  - Cloned from: `https://github.com/swaekaa/KRPO` (empty repository).
  - Role: Independent racing trajectory and policy optimizer.
  - Git Safety: All optimizer code resides strictly in `KRPO/`. Neither `.git` directory is reinitialized, reset, or modified.

---

## 2. Actual Track Representation in KLUSTOR

### Authoritative Geometry (`src/game/data/viceCoastCircuit.ts`)
- **Centerline Curve:** `THREE.CatmullRomCurve3(WAYPOINTS, true, 'catmullrom', 0.5)`
  - Control Points: 29 waypoints in 3D (planar XZ, $y=0$).
  - Sampling: `TRACK_SAMPLES = 500` samples along the closed loop.
  - Scale: 1 unit $\approx$ 1 meter. Car forward = local $+Z$.
  - Circuit Length: $\approx 3,380$ m ($3.4$ km).
- **Track Dimensions & Boundaries:**
  - Track Width: `ROAD_WIDTH = 12.0` m (constant across circuit).
  - Left Edge: $\mathbf{p} - \hat{\mathbf{n}} \cdot 6.0$
  - Right Edge: $\mathbf{p} + \hat{\mathbf{n}} \cdot 6.0$
  - Normal Vector: $\hat{\mathbf{n}} = (-\hat{\mathbf{t}}_z, 0, \hat{\mathbf{t}}_x)$ (points rightward relative to travel direction).
- **Checkpoints:**
  - 10 checkpoints evenly spaced along the 500 samples.
  - Checkpoint 10 (`cp-finish`) is the finish line.
- **Visual & Collision Layers (`src/game/components/Track.tsx`):**
  - Road mesh width: 12m.
  - Curbs: Red/white curbs on both sides, width $1.2$m, height $0.08$m.
  - Guardrails on outer side: concrete base + beam + posts at offset $+1.25$m outside road.
  - Beach/lake on inner side.

---

## 3. Actual Vehicle Dynamics & Physics

### Arcade Kinematic Physics (`src/game/hooks/useCarPhysics.ts`)
- **Vehicle Dimensions:**
  - Half-width: `CAR_HALF_WIDTH = 1.1` m.
  - Legal Lateral Corridor: $|w| \le (12/2 - 1.1) = 4.9$ m from centerline.
  - Collision Penalty: Lateral clamping to $\pm 4.9$ m with a $15\%$ speed penalty (`speed *= 0.85`).
- **Kinematic & Dynamic Parameters:**
  - Base Max Speed: $22.0$ m/s ($79.2$ km/h). Scales up to $\approx 39.6$ m/s ($142.5$ km/h) based on livery design score.
  - Base Acceleration: $7.0$ m/s$^2$.
  - Base Friction: $5.0$ m/s$^2$.
  - Braking: $22.0$ m/s$^2$.
  - Base Steering Strength: $2.0$ rad/s.
  - Steer Factor: $\min(1.0, |v| / 5.0)$.
- **Governing Physics Limits:**
  - Yaw rate $\omega = \frac{d\theta}{dt} \le \text{BASE\_STEERING} = 2.0$ rad/s.
  - Since curvature $\kappa = \frac{d\theta}{ds} = \frac{\omega}{v}$, maximum achievable speed through a turn of curvature $\kappa$ is:
    $$v_{\text{steer}}(\kappa) = \frac{\omega_{\text{max}}}{|\kappa|} = \frac{2.0}{|\kappa|}$$
  - Combined with engine top speed and braking limits, this provides an exact, deterministic model for speed profiling and lap time evaluation.

---

## 4. Proposed KRPO Architecture

```text
KLUSTOR (Authoritative Game)
  ├── src/game/data/viceCoastCircuit.ts (29 waypoints, CatmullRom spline)
  └── src/game/components/RacingLineOverlay.tsx (3D Visualizer & HUD Toggle)
                    │
           Track Geometry Export
                    ▼
KRPO (Optimization Engine in KRPO/)
  ├── src/
  │    ├── types.ts             # Profile & Track Data schemas
  │    ├── track/
  │    │    └── TrackExtractor.ts # Extracts track data from KLUSTOR waypoints
  │    ├── physics/
  │    │    └── VehicleDynamics.ts # Deterministic model of KLUSTOR vehicle
  │    ├── profiler/
  │    │    └── SpeedProfiler.ts   # Forward-backward dynamic programming profiler
  │    ├── optimizers/
  │    │    ├── BaselineOptimizer.ts   # Centerline reference trajectory
  │    │    ├── GeometricOptimizer.ts  # Minimum Curvature Path (MCP) optimizer
  │    │    └── PolicyOptimizer.ts     # Trajectory search & policy evaluator
  │    ├── evaluation/
  │    │    └── LapEvaluator.ts        # Lap simulation & validator
  │    └── profiles/
  │         └── vice_coast_profile.json # Exported racing line profile
  └── tests/
       └── run_tests.ts         # Validation suite
```

---

## 5. Deliverable Plan

1. **Deliverable 1:** Inspection & Architecture (Completed).
2. **Deliverable 2:** Track Extraction Adapter & Schema (`krpo-track.schema.json` & `TrackExtractor.ts`).
3. **Deliverable 3:** Baseline Optimizer & Speed Profiling (Centerline benchmark with forward-backward passes).
4. **Deliverable 4:** Minimum Curvature Path & Policy Optimizer (Curvature gradient descent, apex finding, braking zones).
5. **Deliverable 5:** Seamless KLUSTOR Integration (`RacingLineOverlay.tsx` with dynamic line, apexes, braking markers, and toggle HUD).
6. **Deliverable 6:** Evaluation, Benchmark Comparisons & Documentation.
