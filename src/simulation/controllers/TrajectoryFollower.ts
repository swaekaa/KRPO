import * as THREE from 'three';
import type { RacingLineWaypoint } from '../../types';
import type { SimulationControlInput } from '../ReplayTypes';
import type { SimulationState } from '../HeadlessSimulator';

// ============================================================
// KRPO Trajectory Follower Controller
// Path-tracking driver policy using adaptive lookahead pure pursuit,
// dynamic longitudinal velocity regulation, smart corner-exit boost
// recoverability, and self-testing braking zone optimization.
// ============================================================

export interface ControllerGains {
  lookaheadBase: number;       // meters (e.g. 7.0m)
  lookaheadSpeedGain: number;  // seconds (e.g. 0.35s)
  steeringGain: number;        // proportional gain (e.g. 2.0)
  speedTolerance: number;      // m/s before braking triggers
  boostAggressiveness: number; // speed threshold to activate boost (m/s)
  boostSteerTolerance: number; // max steering angle allowed while boosting
  boostDisabled?: boolean;     // for strict no-boost baseline comparison
  boostMultiplier: number;     // 1.6x for 10/10 car (150 -> 240 km/h)
}

export const DEFAULT_CONTROLLER_GAINS: ControllerGains = {
  lookaheadBase: 7.0,
  lookaheadSpeedGain: 0.35,
  steeringGain: 2.0,
  speedTolerance: 0.8,
  boostAggressiveness: 12.0, // activate boost once vehicle has initial roll (~43 km/h)
  boostSteerTolerance: 0.22, // centered / mild arc during corner-exit recovery
  boostDisabled: false,
  boostMultiplier: 1.6,      // 150 km/h * 1.6 = 240 km/h
};

export interface CornerBrakingTestEntry {
  cornerId: number;
  brakingMarginMeters: number;
  testCount: number;
  cleanPasses: number;
  wallStrikes: number;
  lastMaxLateralOffset: number;
  recommendedAction: string;
}

export class TrajectoryFollower {
  private waypoints: RacingLineWaypoint[];
  private gains: ControllerGains;
  private lastTargetIdx: number = 0;
  
  // Per-corner adaptive braking self-test records
  private cornerBrakingMargins: Map<number, number> = new Map();
  private cornerTestStats: Map<number, CornerBrakingTestEntry> = new Map();

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

  /**
   * Autonomous Braking Self-Test Feedback:
   * Dynamically adjusts braking points based on corner entry & exit telemetry.
   * If the car ran wide or hit a wall, advances braking point earlier.
   * If the car negotiated the corner cleanly, tests braking deeper to maximize boost straightaway length.
   */
  public recordCornerTestResult(cornerId: number, maxLateralOffset: number, hadWallContact: boolean) {
    const currentMargin = this.cornerBrakingMargins.get(cornerId) ?? 3.5;
    const stats = this.cornerTestStats.get(cornerId) ?? {
      cornerId,
      brakingMarginMeters: currentMargin,
      testCount: 0,
      cleanPasses: 0,
      wallStrikes: 0,
      lastMaxLateralOffset: maxLateralOffset,
      recommendedAction: 'Testing nominal brake point',
    };

    stats.testCount++;
    stats.lastMaxLateralOffset = maxLateralOffset;

    if (hadWallContact || Math.abs(maxLateralOffset) > 4.5) {
      // Failed braking test: brake earlier and cut boost sooner ahead of this corner
      const newMargin = Math.min(10.0, currentMargin + 2.5);
      this.cornerBrakingMargins.set(cornerId, newMargin);
      stats.wallStrikes++;
      stats.brakingMarginMeters = newMargin;
      stats.recommendedAction = `Advance braking by +2.5m (New margin: ${newMargin.toFixed(1)}m)`;
    } else if (Math.abs(maxLateralOffset) < 3.2) {
      // Clean pass with high safety margin: self-test braking deeper (later) to maximize boost run-up
      const newMargin = Math.max(1.5, currentMargin - 0.4);
      this.cornerBrakingMargins.set(cornerId, newMargin);
      stats.cleanPasses++;
      stats.brakingMarginMeters = newMargin;
      stats.recommendedAction = `Clean pass. Trimming brake margin to ${newMargin.toFixed(1)}m for deeper entry`;
    } else {
      stats.cleanPasses++;
      stats.recommendedAction = `Optimal equilibrium reached (${currentMargin.toFixed(1)}m margin)`;
    }

    this.cornerTestStats.set(cornerId, stats);
  }

  public getBrakingTestReport(): CornerBrakingTestEntry[] {
    return Array.from(this.cornerTestStats.values());
  }

  public computeControls(state: SimulationState): SimulationControlInput {
    const n = this.waypoints.length;
    if (n === 0) {
      return { throttle: 0, brake: 0, steering: 0, boost: false };
    }

    const carPos = state.position;
    const carSpeed = state.speed;

    // 1. Find nearest waypoint on trajectory (local window search)
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
      if (targetIdx === nearestIdx) break;
    }
    this.lastTargetIdx = nearestIdx;

    const targetWp = this.waypoints[targetIdx];
    const targetPos = new THREE.Vector3(targetWp.position.x, 0, targetWp.position.z);

    // 4. Pure Pursuit Steering
    const forwardX = Math.sin(state.rotationY);
    const forwardZ = Math.cos(state.rotationY);

    const toTargetX = targetPos.x - carPos.x;
    const toTargetZ = targetPos.z - carPos.z;
    const targetDist = Math.hypot(toTargetX, toTargetZ) || 0.1;

    const dirX = toTargetX / targetDist;
    const dirZ = toTargetZ / targetDist;

    const crossY = forwardX * dirZ - forwardZ * dirX;
    const dot = forwardX * dirX + forwardZ * dirZ;
    const angleError = Math.atan2(crossY, dot);

    const steering = Math.max(-1.0, Math.min(1.0, angleError * this.gains.steeringGain));

    // 5. Kinematic Lookahead & Boost Recoverability Evaluation
    // Examine upcoming 35 waypoints (~150m) along track to detect:
    // a) Is the car currently in or recovering from a corner exit?
    // b) How far away is the next critical braking zone?
    let distAhead = 0;
    let upcomingCornerSpeed = Infinity;
    let distToUpcomingBrakingPoint = Infinity;
    let upcomingCornerId = 0;

    const currentWp = this.waypoints[nearestIdx];
    const currCurvature = Math.abs(currentWp.curvature);

    for (let offset = 1; offset <= 35; offset++) {
      const checkIdx = (nearestIdx + offset) % n;
      const wp = this.waypoints[checkIdx];
      const prevWp = this.waypoints[(nearestIdx + offset - 1 + n) % n];
      distAhead += Math.hypot(wp.position.x - prevWp.position.x, wp.position.z - prevWp.position.z);

      if (wp.targetSpeed < upcomingCornerSpeed) {
        upcomingCornerSpeed = wp.targetSpeed;
        upcomingCornerId = wp.cornerId ?? Math.floor(checkIdx / 50);
      }

      // Check if this waypoint represents a genuine decelerating corner entry (tight hairpins)
      // High-speed sweepers, start chicane cut, and exit kinks (|kappa| <= 0.040) are flat out and do NOT require braking
      const isTrueBrakingZone = wp.isBrakingPoint || (wp.targetSpeed < carSpeed * 0.82 && Math.abs(wp.curvature) > 0.040);
      if (isTrueBrakingZone) {
        if (distToUpcomingBrakingPoint === Infinity) {
          distToUpcomingBrakingPoint = distAhead;
        }
      }
    }

    // Kinematic stopping distance calculation: d = (v^2 - v_target^2) / (2 * a_brake)
    const effectiveBrakingAccel = 28.0; // m/s^2 carbon ceramic braking
    const cornerMargin = this.cornerBrakingMargins.get(upcomingCornerId) ?? 3.5;
    const targetSpeedAhead = Math.min(upcomingCornerSpeed, targetWp.targetSpeed);
    const speedDeltaSq = Math.max(0, carSpeed * carSpeed - targetSpeedAhead * targetSpeedAhead);
    const requiredStoppingDist = (speedDeltaSq / (2 * effectiveBrakingAccel)) + cornerMargin;

    // Boost Cutoff: Cut boost only when genuinely approaching an unavoidable braking zone
    const isApproachingBrakingZone = distToUpcomingBrakingPoint < (requiredStoppingDist + 6.0);

    // Boost Recoverability:
    // Car has passed the apex or is navigating flat-out sweepers/kinks (|curvature| < 0.035),
    // steering is within tolerance, and car is stable on track.
    // Trigger boost immediately and maintain it through flat-out turns!
    const isCornerExitRecovery = 
      currCurvature < 0.035 &&
      Math.abs(steering) <= this.gains.boostSteerTolerance * 1.5 &&
      Math.abs(state.lateralOffset) < 4.4;

    const canBoost = 
      !this.gains.boostDisabled &&
      state.boost > 0.02 &&
      carSpeed >= this.gains.boostAggressiveness &&
      !isApproachingBrakingZone &&
      isCornerExitRecovery;

    // 6. Longitudinal Velocity Regulation (Throttle / Brake / Boost)
    let boost = false;
    let effectiveTargetSpeed = targetWp.targetSpeed;

    if (canBoost) {
      boost = true;
      // Allow surging up to boosted top speed (e.g., 240 km/h = 66.67 m/s)
      effectiveTargetSpeed = targetWp.targetSpeed * this.gains.boostMultiplier;
    }

    const speedDiff = effectiveTargetSpeed - carSpeed;
    let throttle = 0;
    let brake = 0;

    // Hard braking trigger if inside stopping distance for upcoming corner
    const mustBrakeForCorner = distToUpcomingBrakingPoint <= requiredStoppingDist && speedDiff < -0.5;

    if (mustBrakeForCorner) {
      throttle = 0;
      brake = Math.min(1.0, Math.max(0.7, Math.abs(speedDiff) / 3.0));
      boost = false; // Never boost during active pre-corner braking!
    } else if (speedDiff > 0.1) {
      throttle = 1.0;
      brake = 0;
    } else if (speedDiff < -this.gains.speedTolerance) {
      throttle = 0;
      brake = Math.min(1.0, Math.abs(speedDiff) / 4.0);
    } else {
      // Coasting / speed maintenance
      throttle = Math.max(0, 0.4 + speedDiff * 0.5);
      brake = 0;
    }

    return { throttle, brake, steering, boost };
  }
}
