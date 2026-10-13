import * as THREE from 'three';
import type { TrackExportData, VehicleParameters, RacingLineProfile } from '../types';
import { extractTrackGeometry, MAX_LEGAL_OFFSET, SAFETY_MARGIN } from '../track/TrackExtractor';
import { BASE_VEHICLE_CONFIG } from '../physics/VehicleDynamics';
import { computeSpeedProfile, type PathInputPoint } from '../profiler/SpeedProfiler';
import { generateBaselineProfile } from './BaselineOptimizer';
import { optimizeGeometricRacingLine } from './GeometricOptimizer';

// ============================================================
// KRPO Policy Trajectory Optimizer
// Direct Lap-Time Policy Search with Trajectory Exploration
// Refines the racing line to achieve minimum possible lap time.
// ============================================================

export interface PolicyOptimizerOptions {
  numEpochs?: number;
  batchSize?: number;
  explorationNoise?: number;
  maxOffset?: number;
}

export function optimizePolicyRacingLine(
  trackData: TrackExportData = extractTrackGeometry(),
  vehicle: VehicleParameters = BASE_VEHICLE_CONFIG,
  options: PolicyOptimizerOptions = {}
): RacingLineProfile {
  const {
    numEpochs = 40,
    batchSize = 8,
    explorationNoise = 0.35,
    maxOffset = MAX_LEGAL_OFFSET - SAFETY_MARGIN, // 4.5m
  } = options;

  const baseline = generateBaselineProfile(trackData, vehicle);
  // Start from the geometrically optimized trajectory as initial policy
  const initialGeo = optimizeGeometricRacingLine(trackData, vehicle, { iterations: 150 });
  const n = trackData.samples.length;

  let bestOffsets = initialGeo.waypoints.map(wp => wp.lateralOffset);
  let bestLapTime = initialGeo.metrics.simulatedLapTime;
  let bestWaypoints = initialGeo.waypoints;
  let bestResult = {
    averageSpeedKmh: initialGeo.metrics.averageSpeedKmh,
    maxSpeedKmh: initialGeo.metrics.maxSpeedKmh,
    totalLength: initialGeo.metrics.racingLineLength,
    maxCurvature: initialGeo.metrics.maxCurvature,
    averageCurvature: initialGeo.metrics.averageCurvature,
    apexCount: initialGeo.metrics.apexCount,
    brakingZoneCount: initialGeo.metrics.brakingZoneCount,
  };

  // Helper: evaluate offsets
  const evaluateOffsets = (offsets: number[]) => {
    const positions: THREE.Vector3[] = new Array(n);
    for (let i = 0; i < n; i++) {
      const s = trackData.samples[i];
      const normal = new THREE.Vector3(s.normal.x, s.normal.y, s.normal.z);
      const pos = new THREE.Vector3(s.position.x, s.position.y, s.position.z);
      pos.addScaledVector(normal, offsets[i]);
      positions[i] = pos;
    }

    const pathPoints: PathInputPoint[] = new Array(n);
    for (let i = 0; i < n; i++) {
      const curr = positions[i];
      const prev = positions[(i - 1 + n) % n];
      const next = positions[(i + 1) % n];

      const tangent = new THREE.Vector3().subVectors(next, curr).normalize();
      const normal = new THREE.Vector3(-tangent.z, 0, tangent.x).normalize();

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
        lateralOffset: offsets[i],
        curvature: kappa,
        sector,
      };
    }

    return { pathPoints, result: computeSpeedProfile(pathPoints, vehicle) };
  };

  // Policy Search Loop (Covariance-free Local Policy Exploration)
  let noise = explorationNoise;
  for (let epoch = 0; epoch < numEpochs; epoch++) {
    for (let b = 0; b < batchSize; b++) {
      // Generate perturbation candidate using smooth low-frequency harmonics
      const candidate = [...bestOffsets];
      const waveFreq = 2 + Math.floor(Math.random() * 8); // 2-10 harmonic frequencies
      const wavePhase = Math.random() * Math.PI * 2;
      const waveScale = (Math.random() - 0.5) * noise;

      for (let i = 0; i < n; i++) {
        const u = (i / n) * Math.PI * 2;
        const perturbation = Math.sin(u * waveFreq + wavePhase) * waveScale;
        const newOffset = candidate[i] + perturbation;
        candidate[i] = Math.max(-maxOffset, Math.min(maxOffset, newOffset));
      }

      // Smooth candidate
      const smoothedCandidate: number[] = new Array(n);
      for (let i = 0; i < n; i++) {
        const prev = candidate[(i - 1 + n) % n];
        const curr = candidate[i];
        const next = candidate[(i + 1) % n];
        smoothedCandidate[i] = 0.25 * prev + 0.5 * curr + 0.25 * next;
      }

      const { pathPoints, result } = evaluateOffsets(smoothedCandidate);

      // Reward function: lower lap time is better
      if (result.lapTime < bestLapTime) {
        bestLapTime = result.lapTime;
        bestOffsets = smoothedCandidate;
        bestWaypoints = result.waypoints;
        bestResult = {
          averageSpeedKmh: result.averageSpeedKmh,
          maxSpeedKmh: result.maxSpeedKmh,
          totalLength: result.totalLength,
          maxCurvature: result.maxCurvature,
          averageCurvature: result.averageCurvature,
          apexCount: result.apexCount,
          brakingZoneCount: result.brakingZoneCount,
        };
      }
    }
    // Decay exploration noise
    noise *= 0.95;
  }

  const timeDelta = Math.round((bestLapTime - baseline.metrics.baselineLapTime) * 1000) / 1000;
  const pctImprovement = Math.round(((-timeDelta) / baseline.metrics.baselineLapTime) * 10000) / 100;

  let maxOffsetSeen = 0;
  for (let i = 0; i < n; i++) {
    const absW = Math.abs(bestOffsets[i]);
    if (absW > maxOffsetSeen) maxOffsetSeen = absW;
  }

  return {
    schemaVersion: '1.0.0',
    optimizerVersion: 'KRPO-RL-Policy-v1.0',
    circuitId: trackData.circuitId,
    trackGeometryHash: trackData.geometryHash,
    generatedAt: new Date().toISOString(),
    vehicleConfig: vehicle,
    metrics: {
      simulatedLapTime: bestLapTime,
      baselineLapTime: baseline.metrics.baselineLapTime,
      timeDelta,
      percentageImprovement: pctImprovement,
      averageSpeedKmh: bestResult.averageSpeedKmh,
      maxSpeedKmh: bestResult.maxSpeedKmh,
      racingLineLength: bestResult.totalLength,
      centerlineLength: trackData.totalLength,
      maxCurvature: bestResult.maxCurvature,
      averageCurvature: bestResult.averageCurvature,
      isBoundaryCompliant: maxOffsetSeen <= MAX_LEGAL_OFFSET,
      maxLateralOffset: Math.round(maxOffsetSeen * 100) / 100,
      apexCount: bestResult.apexCount,
      brakingZoneCount: bestResult.brakingZoneCount,
    },
    waypoints: bestWaypoints,
  };
}
