import * as THREE from 'three';
import type { RacingLineWaypoint } from '../../types';
import type { SimulationControlInput } from '../ReplayTypes';
import type { SimulationState } from '../HeadlessSimulator';

// ============================================================
// KRPO Trajectory Follower Controller
// Path-tracking driver policy using adaptive lookahead pure pursuit
// and dynamic longitudinal velocity regulation.
// ============================================================

export interface ControllerGains {
  lookaheadBase: number;      // meters (e.g. 7.0m)
  lookaheadSpeedGain: number; // seconds (e.g. 0.35s)
  steeringGain: number;       // proportional gain (e.g. 1.8)
  speedTolerance: number;     // m/s before braking triggers
  boostAggressiveness: number;// speed threshold to activate NOS boost
}

export const DEFAULT_CONTROLLER_GAINS: ControllerGains = {
  lookaheadBase: 7.0,
  lookaheadSpeedGain: 0.35,
  steeringGain: 2.0,
  speedTolerance: 0.5,
  boostAggressiveness: 19.5, // activate boost on high-speed straights
};

export class TrajectoryFollower {
  private waypoints: RacingLineWaypoint[];
  private gains: ControllerGains;
  private lastTargetIdx: number = 0;

  constructor(
    waypoints: RacingLineWaypoint[],
    gains: Partial<ControllerGains> = {}
  ) {
    this.waypoints = waypoints;
    this.gains = { ...DEFAULT_CONTROLLER_GAINS, ...gains };
  }

  public updateWaypoints(waypoints: RacingLineWaypoint[]) {
    this.waypoints = waypoints;
  }

  public reset() {
    this.lastTargetIdx = 0;
  }

  public computeControls(state: SimulationState): SimulationControlInput {
    const n = this.waypoints.length;
    if (n === 0) {
      return { throttle: 0, brake: 0, steering: 0, boost: false };
    }

    const carPos = state.position;
    const carSpeed = state.speed;

    // 1. Find nearest waypoint on trajectory (local window around lastTargetIdx)
    let minDistSq = Infinity;
    let nearestIdx = this.lastTargetIdx;
    for (let offset = -15; offset <= 25; offset++) {
      const idx = (this.lastTargetIdx + offset + n) % n;
      const wp = this.waypoints[idx];
      const dx = wp.position.x - carPos.x;
      const dz = wp.position.z - carPos.z;
      const d2 = dx * dx + dz * dz;
      if (d2 < minDistSq) {
        minDistSq = d2;
        nearestIdx = idx;
      }
    }

    // 2. Adaptive lookahead distance
    const lookaheadDist = this.gains.lookaheadBase + this.gains.lookaheadSpeedGain * Math.max(0, carSpeed);

    // 3. Find lookahead target point along the trajectory
    let accumulatedDist = 0;
    let targetIdx = nearestIdx;
    while (accumulatedDist < lookaheadDist) {
      const nextIdx = (targetIdx + 1) % n;
      const currP = this.waypoints[targetIdx].position;
      const nextP = this.waypoints[nextIdx].position;
      const segDist = Math.hypot(nextP.x - currP.x, nextP.z - currP.z);
      accumulatedDist += segDist;
      targetIdx = nextIdx;
      if (targetIdx === nearestIdx) break; // Avoid infinite loop
    }
    this.lastTargetIdx = nearestIdx;

    const targetWp = this.waypoints[targetIdx];
    const targetPos = new THREE.Vector3(targetWp.position.x, 0, targetWp.position.z);

    // 4. Pure Pursuit Steering
    // Car forward heading vector (XZ plane)
    const forwardX = Math.sin(state.rotationY);
    const forwardZ = Math.cos(state.rotationY);

    // Vector to target point
    const toTargetX = targetPos.x - carPos.x;
    const toTargetZ = targetPos.z - carPos.z;
    const targetDist = Math.hypot(toTargetX, toTargetZ) || 0.1;

    const dirX = toTargetX / targetDist;
    const dirZ = toTargetZ / targetDist;

    // Signed angle difference between car heading and target direction
    // In Three.js coordinates:
    // cross product Y: forwardX * dirZ - forwardZ * dirX
    // dot product: forwardX * dirX + forwardZ * dirZ
    const crossY = forwardX * dirZ - forwardZ * dirX;
    const dot = forwardX * dirX + forwardZ * dirZ;
    // Positive crossY means target is to the LEFT, so steer positive (+1)
    const angleError = Math.atan2(crossY, dot);

    const steering = Math.max(-1.0, Math.min(1.0, angleError * this.gains.steeringGain));

    // 5. Velocity Control (Throttle / Brake)
    const targetSpeed = targetWp.targetSpeed;
    let throttle = 0;
    let brake = 0;
    let boost = false;

    const speedDiff = targetSpeed - carSpeed;

    if (speedDiff > 0.2) {
      throttle = 1.0;
      brake = 0;
      // Activate boost on high-speed straightaways when safe
      if (carSpeed > this.gains.boostAggressiveness && Math.abs(steering) < 0.15) {
        boost = true;
      }
    } else if (speedDiff < -this.gains.speedTolerance) {
      throttle = 0;
      brake = Math.min(1.0, Math.abs(speedDiff) / 4.0);
    } else {
      // Coasting / gentle maintenance
      throttle = Math.max(0, 0.4 + speedDiff);
      brake = 0;
    }

    return { throttle, brake, steering, boost };
  }
}
