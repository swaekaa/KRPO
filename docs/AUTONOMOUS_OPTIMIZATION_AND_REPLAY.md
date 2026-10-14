# KRPO — Autonomous Lap-Time Optimization & Simulation Recorder for KLUSTOR

## 1. System Overview & Architecture

KRPO includes an **Autonomous Simulation & Experimentation Engine** and a **3D Replay & Recording Studio** integrated directly into **KLUSTOR // THE FIXER**.

The system enables automated bulk experimentation:
1. Simulates the car on the actual Vice Coast Circuit ($3.38$ km) at thousands of ticks per second headlessly using the exact discrete physics formulas from `useCarPhysics.ts`.
2. Evaluates candidate driving policies against strict validity criteria (crossing all 10 checkpoints in sequence, boundary compliance, timeout budget, and wall collision caps).
3. Archives every candidate run, logs generation-level metrics, and updates the best-known policy checkpoint.
4. Generates full 10Hz replay datasets storing timestamp, 3D position, yaw orientation, speed, steering, throttle, brake, and collision flags.
5. Loads into the **Replay Studio** in KLUSTOR (`/replay`), rendering the real recorded car trajectory with a side-by-side translucent Ghost car, broadcast telemetry, and clean cinematic recording mode for desktop video capture.

---

## 2. Headless Simulation Engine (`HeadlessSimulator.ts`)

To avoid the overhead of WebGL rendering during bulk optimization runs, `HeadlessSimulator` executes the exact physics equations deterministically:

### Fixed-Step Physics ($\Delta t = 1/60$s)
- **Longitudinal Dynamics**:
  - Throttle acceleration: $v_{t+1} = \min(V_{\text{max}}, v_t + a_{\text{accel}} \cdot \text{throttle} \cdot \Delta t)$
  - Braking: $v_{t+1} = \max(0, v_t - a_{\text{brake}} \cdot \text{brake} \cdot \Delta t)$
  - Friction: $v_{t+1} = v_t - \text{sign}(v_t) \cdot a_{\text{friction}} \cdot \Delta t$
- **Yaw Steering Dynamics**:
  - Steer factor: $f_{\text{steer}} = \min(1.0, |v| / 5.0)$
  - Angular yaw delta: $\Delta \theta = \text{steer} \cdot \Omega_{\text{max}} \cdot f_{\text{steer}} \cdot \Delta t \cdot \text{sign}(v)$
- **Corridor Clamping & Barrier Penalty**:
  - Distance from centerline: $|w| = (\mathbf{p} - \mathbf{p}_{\text{center}}) \cdot \hat{\mathbf{n}}$
  - Clamping threshold: $W_{\text{corridor}} = 4.9$m ($12.0\text{m}/2 - 1.1\text{m}$)
  - When $|w| > 4.9$m: clamped to boundary and speed penalised by $15\%$ ($v \leftarrow v \cdot 0.85$).
- **Checkpoint Validation**:
  - 10 authoritative checkpoints from `viceCoastCircuit.ts`.
  - Enforces forward plane crossing: $\mathbf{v}_{\text{prev}} \cdot \hat{\mathbf{t}} < 0$ and $\mathbf{v}_{\text{curr}} \cdot \hat{\mathbf{t}} \ge 0$.
  - Requires all 10 checkpoints to be cleared in numerical order.

---

## 3. Trajectory Follower Controller (`TrajectoryFollower.ts`)

Autonomous driving is driven by an adaptive pure pursuit lookahead controller:
- **Adaptive Lookahead**:
  $$L_d(v) = L_0 + k_v \cdot v = 7.0\text{ m} + 0.35 \cdot v$$
- **Pure Pursuit Steering**:
  $$\alpha = \arctan2(\mathbf{fwd} \times \hat{\mathbf{d}}_{\text{target}}, \mathbf{fwd} \cdot \hat{\mathbf{d}}_{\text{target}})$$
  $$\delta_{\text{steer}} = \text{clamp}(k_p \cdot \alpha, -1.0, 1.0)$$
- **Longitudinal Speed Regulation**:
  - Compares current speed $v$ with the target waypoint speed $v_{\text{target}}$:
    - If $v < v_{\text{target}} - 0.2$: throttle = 1.0, brake = 0.0
    - If $v > v_{\text{target}} + 0.5$: throttle = 0.0, brake = $\min(1.0, (v - v_{\text{target}}) / 3.5)$
    - NOS boost activates on straightaways ($v > 19.5$ m/s and $|\delta| < 0.15$).

---

## 4. Optimization Results & Experiment Summary

The autonomous optimization experiment executed 36 simulation episodes across 6 generations:

### Generation & Milestone Overview

| Episode / Milestone | Policy Identifier | Valid? | Lap Time | Avg Speed | Collisions | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Run 01 (Baseline)** | `policy-centerline-v1` | **YES ✓** | **178.45 s** | $68.2$ km/h | 0 | Exact centerline, safe cornering |
| **Run 02 (Early Candidate)** | `policy-early-candidate` | **YES ✓** | **191.35 s** | $63.6$ km/h | 2 | Under-tuned lookahead, late turn-in |
| **Run 03 (Failed / Crash)** | `policy-reckless-v1` | **NO ⚠** | **205.18 s** | $59.3$ km/h | 7 | Overspeed in corners, barrier penalty |
| **Generation 2 Best** | `policy-g2` | **YES ✓** | **182.15 s** | $66.8$ km/h | 1 | Coarse out-in-out exploration |
| **Generation 3 Best** | `policy-g3` | **YES ✓** | **174.32 s** | $69.7$ km/h | 0 | Refined braking points |
| **Generation 4 Best** | `policy-g4` | **YES ✓** | **168.91 s** | $72.0$ km/h | 0 | High-speed apex clipping |
| **Run 36 (Best-Known)** | `policy-krpo-mcp-v1` | **YES ✓** | **164.82 s** | **73.8 km/h** | **0** | **Optimal lap: -13.63s faster (+7.64%)** |

---

## 5. Replay Studio & Screen Recording Guide

The **Replay Studio** is integrated directly into KLUSTOR at `/replay` (or via button in `RacePage` / `RaceHUD`).

### Key Features
1. **Side-by-Side Ghost Car**:
   - Renders the Main Car (Best Run) alongside the Translucent Cyan Ghost Car (Baseline Run).
   - Visually reveals where the optimized car brakes later, cuts apexes tighter, and accelerates earlier.
2. **Interactive Transport Controls**:
   - Play / Pause (Spacebar)
   - Timeline Scrubber: scrub to any second of the lap
   - Speed Multipliers: `0.5x`, `1x`, `2x`, `4x`
   - Run Selector: `🏆 BEST` | `⚪ BASELINE` | `🟡 EARLY` | `🔴 FAILED`
3. **Camera Views (Press `V`)**:
   - **Chase Cam**: Dynamic following camera
   - **Overhead Cam**: High bird's eye / helicopter perspective
   - **Hood Cam**: Immersive driver viewpoint
4. **Cinematic Clean Recording Mode (Press `H`)**:
   - Instantly hides all UI overlays, leaving a clean, cinematic 3D viewport.
   - Ideal for desktop screen recording with OBS, Windows Game Bar (`Win + Alt + R`), or ShareX.

---

## 6. How to Access the Replay Studio

1. **Start the game**:
   ```bash
   npm run dev
   ```
2. Navigate to [http://localhost:5173/replay](http://localhost:5173/replay) in your browser.
3. Or during any race on `/race`:
   - Click the **`🎬 KRPO REPLAY STUDIO →`** button on the HUD.
   - Or press **`P`** on your keyboard.
   - Or click **`🎬 KRPO REPLAY`** on the post-race results screen.
