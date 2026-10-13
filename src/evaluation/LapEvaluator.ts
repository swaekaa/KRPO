import type { RacingLineProfile, TrackExportData } from '../types';
import { MAX_LEGAL_OFFSET } from '../track/TrackExtractor';

// ============================================================
// KRPO Lap Evaluator & Profile Validator
// Performs comprehensive automated audits on racing lines.
// ============================================================

export interface ValidationIssue {
  type: 'error' | 'warning';
  message: string;
  waypointIndex?: number;
  details?: Record<string, any>;
}

export interface ValidationReport {
  isValid: boolean;
  issues: ValidationIssue[];
  metrics: {
    totalWaypoints: number;
    maxLateralOffset: number;
    maxAccelSeen: number;
    maxDecelSeen: number;
    maxSpeedKmh: number;
    minSpeedKmh: number;
  };
}

export function validateRacingLineProfile(
  profile: RacingLineProfile,
  expectedTrack?: TrackExportData
): ValidationReport {
  const issues: ValidationIssue[] = [];
  const waypoints = profile.waypoints;
  const n = waypoints.length;

  if (!waypoints || n < 10) {
    issues.push({ type: 'error', message: 'Profile contains insufficient or missing waypoints.' });
    return {
      isValid: false,
      issues,
      metrics: { totalWaypoints: 0, maxLateralOffset: 0, maxAccelSeen: 0, maxDecelSeen: 0, maxSpeedKmh: 0, minSpeedKmh: 0 },
    };
  }

  // Check Schema & Circuit Match
  if (profile.circuitId !== 'vice-coast-circuit') {
    issues.push({ type: 'error', message: `Mismatched circuitId: expected 'vice-coast-circuit', got '${profile.circuitId}'.` });
  }

  if (expectedTrack && profile.trackGeometryHash !== expectedTrack.geometryHash) {
    issues.push({
      type: 'warning',
      message: `Track geometry hash mismatch: expected '${expectedTrack.geometryHash}', profile has '${profile.trackGeometryHash}'. Track may have changed.`,
    });
  }

  let maxLateral = 0;
  let maxAccel = 0;
  let maxDecel = 0;
  let maxSpeed = -Infinity;
  let minSpeed = Infinity;

  const vehicle = profile.vehicleConfig;
  const maxAllowedAccel = vehicle.acceleration * 1.25; // 25% tolerance for discrete steps
  const maxAllowedBrake = (vehicle.braking + vehicle.friction) * 1.25;

  for (let i = 0; i < n; i++) {
    const wp = waypoints[i];
    const nextWp = waypoints[(i + 1) % n];

    // Check lateral boundary
    const absOffset = Math.abs(wp.lateralOffset);
    if (absOffset > maxLateral) maxLateral = absOffset;

    if (absOffset > MAX_LEGAL_OFFSET + 0.05) {
      issues.push({
        type: 'error',
        waypointIndex: i,
        message: `Boundary violation at waypoint ${i}: offset ${wp.lateralOffset.toFixed(2)}m exceeds max allowed corridor (${MAX_LEGAL_OFFSET}m).`,
      });
    }

    if (wp.targetSpeed > maxSpeed) maxSpeed = wp.targetSpeed;
    if (wp.targetSpeed < minSpeed) minSpeed = wp.targetSpeed;

    // Check distance between consecutive waypoints (continuity)
    const dx = nextWp.position.x - wp.position.x;
    const dy = nextWp.position.y - wp.position.y;
    const dz = nextWp.position.z - wp.position.z;
    const segDist = Math.sqrt(dx * dx + dy * dy + dz * dz);

    if (segDist < 0.1 || segDist > 25.0) {
      issues.push({
        type: 'error',
        waypointIndex: i,
        message: `Discontinuous path: segment ${i}->${(i + 1) % n} has abnormal length ${segDist.toFixed(2)}m.`,
      });
    }

    // Check dynamic acceleration feasibility
    const dv2 = nextWp.targetSpeed * nextWp.targetSpeed - wp.targetSpeed * wp.targetSpeed;
    const accel = dv2 / (2 * Math.max(segDist, 0.01));

    if (accel > maxAccel) maxAccel = accel;
    if (accel < -maxDecel) maxDecel = -accel;

    if (accel > maxAllowedAccel) {
      issues.push({
        type: 'error',
        waypointIndex: i,
        message: `Infeasible acceleration at waypoint ${i}: ${accel.toFixed(2)} m/s^2 exceeds vehicle acceleration limit (${vehicle.acceleration} m/s^2).`,
      });
    }

    if (-accel > maxAllowedBrake) {
      issues.push({
        type: 'error',
        waypointIndex: i,
        message: `Infeasible deceleration at waypoint ${i}: ${(-accel).toFixed(2)} m/s^2 exceeds vehicle braking limit (${vehicle.braking} m/s^2).`,
      });
    }

    // Check coordinate system (Y should be ground level ~ 0)
    if (Math.abs(wp.position.y) > 0.5) {
      issues.push({
        type: 'warning',
        waypointIndex: i,
        message: `Unexpected non-zero elevation: y = ${wp.position.y.toFixed(2)}.`,
      });
    }
  }

  const hasErrors = issues.some(issue => issue.type === 'error');

  return {
    isValid: !hasErrors,
    issues,
    metrics: {
      totalWaypoints: n,
      maxLateralOffset: Math.round(maxLateral * 100) / 100,
      maxAccelSeen: Math.round(maxAccel * 100) / 100,
      maxDecelSeen: Math.round(maxDecel * 100) / 100,
      maxSpeedKmh: Math.round(maxSpeed * 3.6 * 10) / 10,
      minSpeedKmh: Math.round(minSpeed * 3.6 * 10) / 10,
    },
  };
}
