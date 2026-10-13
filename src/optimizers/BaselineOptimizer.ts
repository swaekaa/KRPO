import type { TrackExportData, VehicleParameters, RacingLineProfile } from '../types';
import { extractTrackGeometry } from '../track/TrackExtractor';
import { BASE_VEHICLE_CONFIG } from '../physics/VehicleDynamics';
import { computeSpeedProfile, type PathInputPoint } from '../profiler/SpeedProfiler';

// ============================================================
// KRPO Baseline Racing Line Optimizer
// Computes the authoritative centerline reference trajectory.
// Used as the benchmark to measure all optimization gains.
// ============================================================

export function generateBaselineProfile(
  trackData: TrackExportData = extractTrackGeometry(),
  vehicle: VehicleParameters = BASE_VEHICLE_CONFIG
): RacingLineProfile {
  const n = trackData.samples.length;
  const pathPoints: PathInputPoint[] = new Array(n);

  for (let i = 0; i < n; i++) {
    const s = trackData.samples[i];
    const sector = Math.min(10, Math.floor((i / n) * 10) + 1);

    pathPoints[i] = {
      position: { ...s.position },
      tangent: { ...s.tangent },
      normal: { ...s.normal },
      lateralOffset: 0, // Exact centerline
      curvature: s.curvature,
      sector,
    };
  }

  const result = computeSpeedProfile(pathPoints, vehicle);

  return {
    schemaVersion: '1.0.0',
    optimizerVersion: 'KRPO-Baseline-Centerline-v1.0',
    circuitId: trackData.circuitId,
    trackGeometryHash: trackData.geometryHash,
    generatedAt: new Date().toISOString(),
    vehicleConfig: vehicle,
    metrics: {
      simulatedLapTime: result.lapTime,
      baselineLapTime: result.lapTime,
      timeDelta: 0,
      percentageImprovement: 0,
      averageSpeedKmh: result.averageSpeedKmh,
      maxSpeedKmh: result.maxSpeedKmh,
      racingLineLength: result.totalLength,
      centerlineLength: trackData.totalLength,
      maxCurvature: result.maxCurvature,
      averageCurvature: result.averageCurvature,
      isBoundaryCompliant: true,
      maxLateralOffset: 0,
      apexCount: result.apexCount,
      brakingZoneCount: result.brakingZoneCount,
    },
    waypoints: result.waypoints,
  };
}
