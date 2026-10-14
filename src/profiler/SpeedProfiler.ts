import type { Vec3, VehicleParameters, RacingLineWaypoint } from '../types';
import { BASE_VEHICLE_CONFIG, maxCornerSpeed } from '../physics/VehicleDynamics';

// ============================================================
// KRPO Forward-Backward Speed Profiler
// Solves the optimal speed profile along a given 3D path
// using dynamic programming with vehicle acceleration & braking.
// ============================================================

export interface PathInputPoint {
  position: Vec3;
  tangent: Vec3;
  normal: Vec3;
  lateralOffset: number;
  curvature: number;
  sector: number;
}

export interface SpeedProfileResult {
  waypoints: RacingLineWaypoint[];
  lapTime: number;          // seconds
  averageSpeedKmh: number;
  maxSpeedKmh: number;
  totalLength: number;
  maxCurvature: number;
  averageCurvature: number;
  apexCount: number;
  brakingZoneCount: number;
}

export function computeSpeedProfile(
  pathPoints: PathInputPoint[],
  vehicle: VehicleParameters = BASE_VEHICLE_CONFIG
): SpeedProfileResult {
  const n = pathPoints.length;
  if (n === 0) {
    throw new Error('Cannot compute speed profile on empty path');
  }

  // 1. Calculate segment lengths
  const ds: number[] = new Array(n);
  let totalLength = 0;
  for (let i = 0; i < n; i++) {
    const curr = pathPoints[i].position;
    const next = pathPoints[(i + 1) % n].position;
    const dx = next.x - curr.x;
    const dy = next.y - curr.y;
    const dz = next.z - curr.z;
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    ds[i] = Math.max(dist, 0.01);
    totalLength += dist;
  }

  // 2. Initial corner speed limits (kinematic steering and lateral grip limits)
  const vProfile: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    vProfile[i] = maxCornerSpeed(pathPoints[i].curvature, vehicle);
  }

  // Max effective deceleration and acceleration
  const aBrake = vehicle.braking + vehicle.friction * 0.5; // m/s^2
  const aAccel = vehicle.acceleration;                   // m/s^2

  // 3. Backward Pass (Braking limit: ensure car can slow down for upcoming corners)
  // Run 2 full cycles around closed loop to ensure periodic convergence
  for (let cycle = 0; cycle < 2; cycle++) {
    for (let i = n - 1; i >= 0; i--) {
      const nextIdx = (i + 1) % n;
      const vNext = vProfile[nextIdx];
      const maxEntrySpeed = Math.sqrt(vNext * vNext + 2 * aBrake * ds[i]);
      vProfile[i] = Math.min(vProfile[i], maxEntrySpeed);
    }
  }

  // 4. Forward Pass (Acceleration limit: engine power limits speed gain out of corners)
  for (let cycle = 0; cycle < 2; cycle++) {
    for (let i = 0; i < n; i++) {
      const nextIdx = (i + 1) % n;
      const vCurr = vProfile[i];
      const maxExitSpeed = Math.sqrt(vCurr * vCurr + 2 * aAccel * ds[i]);
      vProfile[nextIdx] = Math.min(vProfile[nextIdx], maxExitSpeed);
    }
  }

  // 5. Build waypoints and evaluate lap time
  let lapTime = 0;
  let maxSpeed = 0;
  let arcLength = 0;
  let apexCount = 0;
  let brakingZoneCount = 0;
  let totalCurvature = 0;
  let maxCurvature = 0;

  const waypoints: RacingLineWaypoint[] = new Array(n);

  for (let i = 0; i < n; i++) {
    const pt = pathPoints[i];
    const v = vProfile[i];
    const nextIdx = (i + 1) % n;
    const vNext = vProfile[nextIdx];
    const segDist = ds[i];

    // Segment traversal time: dt = 2 * ds / (v1 + v2)
    const avgSegSpeed = Math.max((v + vNext) * 0.5, 0.5);
    const dt = segDist / avgSegSpeed;
    lapTime += dt;

    if (v > maxSpeed) maxSpeed = v;

    const absKappa = Math.abs(pt.curvature);
    if (absKappa > maxCurvature) maxCurvature = absKappa;
    totalCurvature += absKappa;

    // Longitudinal acceleration
    const aLong = (vNext * vNext - v * v) / (2 * segDist);

    let throttle = 0;
    let brake = 0;
    let isBrakingPoint = false;

    if (aLong > 0.1) {
      throttle = Math.min(1.0, aLong / vehicle.acceleration);
    } else if (aLong < -0.5) {
      brake = Math.min(1.0, Math.abs(aLong) / vehicle.braking);
      if (aLong < -2.0) {
        isBrakingPoint = true;
      }
    }

    // Apex detection: local peak in curvature where cornering is tight
    const prevIdx = (i - 1 + n) % n;
    const prevKappa = Math.abs(pathPoints[prevIdx].curvature);
    const nextKappa = Math.abs(pathPoints[nextIdx].curvature);
    const isApex = absKappa > 0.02 && absKappa >= prevKappa && absKappa >= nextKappa;

    if (isApex) apexCount++;
    if (isBrakingPoint && !waypoints[(i - 1 + n) % n]?.isBrakingPoint) brakingZoneCount++;

    // Straightaway boost zone detection (low curvature with >= 35m clear straight)
    let isBoostZone = false;
    if (absKappa < 0.003 && !isBrakingPoint) {
      let lookDist = 0;
      let clearAhead = true;
      for (let look = 1; look <= 15; look++) {
        const checkIdx = (i + look) % n;
        lookDist += ds[checkIdx];
        if (Math.abs(pathPoints[checkIdx].curvature) > 0.012 || vProfile[checkIdx] < v * 0.8) {
          clearAhead = false;
          break;
        }
        if (lookDist >= 35.0) break;
      }
      isBoostZone = clearAhead && lookDist >= 30.0;
    }

    waypoints[i] = {
      index: i,
      arcLength,
      position: { ...pt.position },
      tangent: { ...pt.tangent },
      normal: { ...pt.normal },
      lateralOffset: pt.lateralOffset,
      curvature: pt.curvature,
      targetSpeed: v,
      targetSpeedKmh: Math.round(v * 3.6 * 10) / 10,
      throttle: Math.round(throttle * 100) / 100,
      brake: Math.round(brake * 100) / 100,
      isBrakingPoint,
      isApex,
      isBoostZone,
      sector: pt.sector,
    };

    arcLength += segDist;
  }

  const averageSpeed = lapTime > 0 ? totalLength / lapTime : 0;

  return {
    waypoints,
    lapTime: Math.round(lapTime * 1000) / 1000,
    averageSpeedKmh: Math.round(averageSpeed * 3.6 * 10) / 10,
    maxSpeedKmh: Math.round(maxSpeed * 3.6 * 10) / 10,
    totalLength: Math.round(totalLength * 100) / 100,
    maxCurvature: Math.round(maxCurvature * 10000) / 10000,
    averageCurvature: Math.round((totalCurvature / n) * 10000) / 10000,
    apexCount,
    brakingZoneCount,
  };
}
