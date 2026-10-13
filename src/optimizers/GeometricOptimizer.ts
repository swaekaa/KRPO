import * as THREE from 'three';
import type { TrackExportData, VehicleParameters, RacingLineProfile, Vec3 } from '../types';
import { extractTrackGeometry, MAX_LEGAL_OFFSET, SAFETY_MARGIN } from '../track/TrackExtractor';
import { BASE_VEHICLE_CONFIG } from '../physics/VehicleDynamics';
import { computeSpeedProfile, type PathInputPoint } from '../profiler/SpeedProfiler';
import { generateBaselineProfile } from './BaselineOptimizer';

// ============================================================
// KRPO Geometric Minimum Curvature Path (MCP) Optimizer
// Solves the optimal racing line by minimizing path curvature
// subject to track corridor boundaries (out-in-out trajectory).
// ============================================================

export interface GeometricOptimizerOptions {
  iterations?: number;
  learningRate?: number;
  curvatureWeight?: number;
  smoothnessWeight?: number;
  lengthWeight?: number;
  maxOffset?: number; // max lateral offset in meters
}

export function optimizeGeometricRacingLine(
  trackData: TrackExportData = extractTrackGeometry(),
  vehicle: VehicleParameters = BASE_VEHICLE_CONFIG,
  options: GeometricOptimizerOptions = {}
): RacingLineProfile {
  const {
    iterations = 250,
    learningRate = 0.05,
    curvatureWeight = 1.0,
    smoothnessWeight = 0.8,
    lengthWeight = 0.02,
    maxOffset = MAX_LEGAL_OFFSET - SAFETY_MARGIN, // 4.5m
  } = options;

  const baseline = generateBaselineProfile(trackData, vehicle);
  const n = trackData.samples.length;

  // Initialize lateral offsets w_i along the normal vector
  // Initial heuristic: opposite to track curvature (swing wide before turn)
  const w: number[] = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    const s = trackData.samples[i];
    // Natural racing line heuristic: if turn bends left (positive curvature), swing right (+normal)
    // If turn bends right (negative curvature), swing left (-normal)
    w[i] = Math.max(-maxOffset, Math.min(maxOffset, -s.curvature * 80.0));
  }

  // Iterative Optimization (Constrained Projected Gradient Descent)
  for (let iter = 0; iter < iterations; iter++) {
    const grad: number[] = new Array(n).fill(0);

    for (let i = 0; i < n; i++) {
      const prevIdx = (i - 1 + n) % n;
      const nextIdx = (i + 1) % n;

      // 1. Curvature smoothing term: minimizes second derivative of offset d^2w / ds^2
      const laplacian = w[prevIdx] - 2 * w[i] + w[nextIdx];
      grad[i] += smoothnessWeight * laplacian;

      // 2. Curvature reduction term:
      // A point with high curvature should move towards the center of curvature (clip apex)
      const s = trackData.samples[i];
      const targetApexShift = -s.curvature * 60.0;
      grad[i] += curvatureWeight * (targetApexShift - w[i]) * 0.1;

      // 3. Length penalty: slight pull towards 0 to avoid unnecessary detours
      grad[i] -= lengthWeight * w[i];
    }

    // Gradient step & projection onto corridor bounds [-maxOffset, maxOffset]
    for (let i = 0; i < n; i++) {
      w[i] = w[i] + learningRate * grad[i];
      if (w[i] > maxOffset) w[i] = maxOffset;
      if (w[i] < -maxOffset) w[i] = -maxOffset;
    }
  }

  // Smooth the final offsets using a 5-point Gaussian kernel
  const smoothedW: number[] = new Array(n);
  const kernel = [0.06136, 0.24477, 0.38774, 0.24477, 0.06136];
  for (let i = 0; i < n; i++) {
    let val = 0;
    for (let k = -2; k <= 2; k++) {
      const idx = (i + k + n) % n;
      val += w[idx] * kernel[k + 2];
    }
    smoothedW[i] = Math.max(-maxOffset, Math.min(maxOffset, val));
  }

  // Construct 3D positions and recalculate local curvature along the racing line
  const positions: THREE.Vector3[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const s = trackData.samples[i];
    const normal = new THREE.Vector3(s.normal.x, s.normal.y, s.normal.z);
    const pos = new THREE.Vector3(s.position.x, s.position.y, s.position.z);
    pos.addScaledVector(normal, smoothedW[i]);
    positions[i] = pos;
  }

  const pathPoints: PathInputPoint[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const curr = positions[i];
    const prev = positions[(i - 1 + n) % n];
    const next = positions[(i + 1) % n];

    const tangent = new THREE.Vector3().subVectors(next, curr).normalize();
    const normal = new THREE.Vector3(-tangent.z, 0, tangent.x).normalize();

    // Discrete curvature of the new racing line
    const tPrev = new THREE.Vector3().subVectors(curr, prev).normalize();
    const tNext = new THREE.Vector3().subVectors(next, curr).normalize();
    const crossY = tPrev.x * tNext.z - tPrev.z * tNext.x;
    const dot = Math.max(-1, Math.min(1, tPrev.dot(tNext)));
    const angleDelta = Math.atan2(crossY, dot);
    const ds = (curr.distanceTo(prev) + curr.distanceTo(next)) * 0.5;
    const kappa = ds > 1e-4 ? angleDelta / ds : 0;

    const sector = Math.min(10, Math.floor((i / n) * 10) + 1);

    pathPoints[i] = {
      position: { x: curr.x, y: curr.y, z: curr.z },
      tangent: { x: tangent.x, y: tangent.y, z: tangent.z },
      normal: { x: normal.x, y: normal.y, z: normal.z },
      lateralOffset: Math.round(smoothedW[i] * 100) / 100,
      curvature: kappa,
      sector,
    };
  }

  const optResult = computeSpeedProfile(pathPoints, vehicle);
  const timeDelta = Math.round((optResult.lapTime - baseline.metrics.baselineLapTime) * 1000) / 1000;
  const pctImprovement = Math.round(((-timeDelta) / baseline.metrics.baselineLapTime) * 10000) / 100;

  // Maximum offset reached
  let maxOffsetSeen = 0;
  for (let i = 0; i < n; i++) {
    const absW = Math.abs(smoothedW[i]);
    if (absW > maxOffsetSeen) maxOffsetSeen = absW;
  }

  return {
    schemaVersion: '1.0.0',
    optimizerVersion: 'KRPO-Geometric-MCP-v1.0',
    circuitId: trackData.circuitId,
    trackGeometryHash: trackData.geometryHash,
    generatedAt: new Date().toISOString(),
    vehicleConfig: vehicle,
    metrics: {
      simulatedLapTime: optResult.lapTime,
      baselineLapTime: baseline.metrics.baselineLapTime,
      timeDelta,
      percentageImprovement: pctImprovement,
      averageSpeedKmh: optResult.averageSpeedKmh,
      maxSpeedKmh: optResult.maxSpeedKmh,
      racingLineLength: optResult.totalLength,
      centerlineLength: trackData.totalLength,
      maxCurvature: optResult.maxCurvature,
      averageCurvature: optResult.averageCurvature,
      isBoundaryCompliant: maxOffsetSeen <= MAX_LEGAL_OFFSET,
      maxLateralOffset: Math.round(maxOffsetSeen * 100) / 100,
      apexCount: optResult.apexCount,
      brakingZoneCount: optResult.brakingZoneCount,
    },
    waypoints: optResult.waypoints,
  };
}
