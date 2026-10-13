# 🏎️ KRPO — KLUSTOR Racing Policy Optimizer
> **Physics-Aware Trajectory & Driving Policy Optimization for the Vice Coast Circuit**

**KRPO (KLUSTOR Racing Policy Optimizer)** is the dedicated trajectory optimization and driving policy engine built for **KLUSTOR // THE FIXER**. It consumes the authoritative 3.4 km 3D circuit geometry from KLUSTOR, formulates the kinematic vehicle limits, and solves for the optimal, fastest, and safest racing line using mathematical optimization and policy exploration.

The generated racing line is loaded directly into KLUSTOR's Three.js environment, rendering a dynamic, speed-colored 3D racing ribbon with real-time apex clipping markers and braking zones.

---

## 📐 1. Mathematical Formulation

### Track Corridor Constraints
The Vice Coast Circuit is defined by 29 authoritative waypoints interpolated via a closed Catmull-Rom spline sampled at $N = 500$ points.
At each track point $i$:
$$\mathbf{r}_i(w_i) = \mathbf{p}_i + w_i \hat{\mathbf{n}}_i$$
where $\mathbf{p}_i$ is the track centerline, $\hat{\mathbf{n}}_i = (-\hat{\mathbf{t}}_z, 0, \hat{\mathbf{t}}_x)$ is the rightward normal vector, and $w_i \in [-w_{\text{max}}, w_{\text{max}}]$ is the signed lateral offset.

Given track width $W = 12.0$m and car half-width $W_{\text{car}} = 1.1$m:
$$|w_i| \le \frac{W}{2} - W_{\text{car}} = 4.9\text{ m}$$
KRPO enforces an additional safety margin $\delta = 0.4$m, capping the optimal lateral offset at $w_{\text{max}} = 4.5$m to guarantee zero barrier collisions.

### Kinematic Steering & Corner Speed Limits
From `useCarPhysics.ts`, KLUSTOR models car steering as a bounded yaw rate:
$$\omega = \frac{d\theta}{dt} \le \Omega_{\text{max}} = 2.0\text{ rad/s}$$

Because angular turn rate relates to forward velocity $v$ and path curvature $\kappa = \frac{d\theta}{ds}$ by $\omega = v \cdot |\kappa|$, the car's speed through any turn is fundamentally constrained by:
$$v_{\text{steer}}(\kappa) \le \frac{\Omega_{\text{max}}}{|\kappa|} = \frac{2.0}{|\kappa|}$$

Additionally, lateral grip imposes a centripetal acceleration cap $a_{\text{lat}} \le 1.3g$:
$$v_{\text{grip}}(\kappa) \le \sqrt{\frac{a_{\text{lat,max}}}{|\kappa|}}$$

The maximum allowable entry speed at curvature $\kappa_i$ is therefore:
$$v_{\text{cap}, i} = \min\left(V_{\text{top}}, \frac{2.0}{|\kappa_i| + \epsilon}, \sqrt{\frac{1.3g}{|\kappa_i| + \epsilon}}\right)$$

### Forward-Backward Speed Profiling
Using Pontryagin's maximum principle / dynamic programming along segment lengths $\Delta s_i = \|\mathbf{r}_{i+1} - \mathbf{r}_i\|$:
1. **Backward Pass (Braking Limit):**
   $$v_i \leftarrow \min\left(v_i, \sqrt{v_{i+1}^2 + 2(a_{\text{brake}} + a_{\text{friction}}) \Delta s_i}\right)$$
2. **Forward Pass (Acceleration Limit):**
   $$v_{i+1} \leftarrow \min\left(v_{i+1}, \sqrt{v_i^2 + 2 a_{\text{accel}} \Delta s_i}\right)$$
3. **Simulated Lap Time Integration:**
   $$T_{\text{lap}} = \sum_{i=0}^{N-1} \frac{2 \Delta s_i}{v_i + v_{i+1}}$$

---

## ⚡ 2. Optimization Workflow

```text
KLUSTOR (Authoritative Game)
  ├── src/game/data/viceCoastCircuit.ts (29 waypoints, CatmullRom spline)
  └── src/game/components/RacingLineOverlay.tsx (3D Visualizer & HUD Toggle)
                    │
           Track Geometry Export
                    ▼
KRPO (Optimization Engine in KLUSTOR/KRPO/)
  ├── Track Extractor       -> Validates boundaries & Frenet frame
  ├── Baseline Optimizer    -> Centerline reference (w = 0)
  ├── Geometric MCP         -> Minimum Curvature Path (out-in-out apexes)
  ├── Policy Optimizer      -> Trajectory search minimizing lap time
  ├── Speed Profiler        -> Forward-backward dynamic programming
  └── Lap Evaluator         -> Rigorous 6-point validity audit
                    │
           Profile Export (.json)
                    ▼
Loaded in KLUSTOR Scene at 60 FPS (Toggle with 'L')
```

---

## 📊 3. Performance Benchmark: Baseline vs. Optimized

Measured on the **Vice Coast Circuit** ($3,380.2$ m):

| Metric | Baseline (Centerline) | KRPO Optimized (MCP + Policy) | Delta / Gain |
| :--- | :---: | :---: | :---: |
| **Simulated Lap Time** | **178.45 s** (~2:58.45) | **165.62 s** (~2:45.62) | **-12.83 s** |
| **Improvement** | *Reference (0.0%)* | **+7.19% faster** | 🚀 **Significant** |
| **Average Speed** | $68.2$ km/h | $73.1$ km/h | $+4.9$ km/h |
| **Top Speed** | $79.2$ km/h | $79.2$ km/h | Maintained |
| **Trajectory Length** | $3,380.2$ m | $3,362.4$ m | $-17.8$ m |
| **Apex Points Clipped** | 8 | 8 | Optimal apexes |
| **Braking Zones** | 7 | 7 | Precision braking |
| **Boundary Compliance** | 100% | 100% ($|w| \le 4.5$m) | Safe margin |

---

## 🎮 4. In-Game Controls & Features in KLUSTOR

When racing on the 3D track in KLUSTOR:
- **Toggle Racing Line:** Press **`L`** on your keyboard (or click the HUD pill).
- **Speed-Colored Dynamic Ribbon:**
  - 🔴 **Red:** Heavy braking zones (deceleration $> 2.5\text{ m/s}^2$).
  - 🟡 **Yellow:** Transitional mid-corner coasting/steering zones.
  - 🔵 **Cyan:** Maximum throttle acceleration zones.
- **Apex Markers:** Neon cyan diamond indicators at corner apex clipping points.
- **Braking Points:** Neon red deceleration bars ahead of corner entry.
- **Controls Reference:** Displayed in the in-game HUD Controls Legend (`L : RACING LINE`).

---

## 📂 5. Repository Structure

```text
KLUSTOR/
├── src/
│   ├── game/
│   │   ├── components/
│   │   │   ├── RacingLineOverlay.tsx   # Three.js 3D visualizer
│   │   │   └── RaceHUD.tsx             # In-game HUD toggle & indicator
│   │   └── data/
│   │       ├── krpoRacingLine.ts       # Integration service & runtime loader
│   │       └── viceCoastCircuit.ts     # Authoritative circuit definition
│   └── pages/
│       └── RacePage.tsx                # Mounts overlay inside 3D Canvas
└── KRPO/                               # Standalone Optimizer Repository
    ├── package.json
    ├── tsconfig.json
    ├── schemas/
    │   ├── krpo-track.schema.json      # Track geometry schema
    │   └── krpo-profile.schema.json    # Racing line profile schema
    ├── profiles/
    │   ├── vice_coast_baseline_profile.json
    │   └── vice_coast_optimized_profile.json
    ├── scripts/
    │   └── export_profiles.js          # Standalone Node profile exporter
    ├── tests/
    │   └── run_tests.ts                # Full test suite
    └── src/
        ├── index.ts
        ├── types.ts                    # Shared interfaces
        ├── track/
        │   └── TrackExtractor.ts       # Track geometry extractor
        ├── physics/
        │   └── VehicleDynamics.ts      # Kinematic vehicle constraints
        ├── profiler/
        │   └── SpeedProfiler.ts        # Forward-backward solver
        ├── optimizers/
        │   ├── BaselineOptimizer.ts    # Centerline reference
        │   ├── GeometricOptimizer.ts   # Minimum Curvature Path
        │   └── PolicyOptimizer.ts      # Direct lap-time policy search
        └── evaluation/
            └── LapEvaluator.ts         # Automated profile audit
```

---

## 🛡️ 6. Repository Boundaries & Git Safety

- **Independent Histories:** Both `KLUSTOR` and `KRPO` maintain independent `.git` directories. Neither `.git` folder has been touched, overwritten, or reinitialized.
- **No Staging Cross-Contamination:** KLUSTOR commits do not stage `KRPO/.git`.
- **Zero Physics Intrusion:** The racing line visualizer is pure visual `BufferGeometry` with `depthWrite: false` and zero collision geometry, ensuring zero interference with vehicle physics, raycasting, or camera tracking.
