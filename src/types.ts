// ============================================================
// KRPO — Types and Schema Definitions
// ============================================================

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface TrackSample {
  index: number;
  position: Vec3;
  tangent: Vec3;
  normal: Vec3;
  leftEdge: Vec3;
  rightEdge: Vec3;
  curvature: number;
  arcLength: number;
}

export interface CheckpointSample {
  id: string;
  index: number;
  position: Vec3;
  tangent: Vec3;
  normal: Vec3;
  leftEdge: Vec3;
  rightEdge: Vec3;
}

export interface TrackExportData {
  schemaVersion: string;
  circuitId: string;
  circuitName: string;
  geometryHash: string;
  roadWidth: number;
  carHalfWidth: number;
  maxLegalOffset: number;
  safetyMargin: number;
  totalLength: number;
  totalSamples: number;
  samples: TrackSample[];
  checkpoints: CheckpointSample[];
  startTransform: {
    position: Vec3;
    rotation: number;
    tangent: Vec3;
  };
}

export interface VehicleParameters {
  topSpeed: number;        // m/s
  acceleration: number;    // m/s^2
  braking: number;         // m/s^2
  steeringStrength: number;// rad/s (yaw rate limit)
  friction: number;        // m/s^2
  carHalfWidth: number;    // m
  designScore?: number;
}

export interface RacingLineWaypoint {
  index: number;
  arcLength: number;
  position: Vec3;
  tangent: Vec3;
  normal: Vec3;
  lateralOffset: number;   // w in meters from centerline (-left, +right)
  curvature: number;       // kappa in 1/m
  targetSpeed: number;     // m/s
  targetSpeedKmh: number;  // km/h
  throttle: number;        // 0 to 1
  brake: number;           // 0 to 1
  isBrakingPoint: boolean;
  isApex: boolean;
  cornerId?: number;
  sector: number;
}

export interface OptimizationMetrics {
  simulatedLapTime: number;       // seconds
  baselineLapTime: number;        // seconds
  timeDelta: number;              // seconds (negative is faster)
  percentageImprovement: number;  // %
  averageSpeedKmh: number;
  maxSpeedKmh: number;
  racingLineLength: number;       // meters
  centerlineLength: number;       // meters
  maxCurvature: number;           // 1/m
  averageCurvature: number;       // 1/m
  isBoundaryCompliant: boolean;
  maxLateralOffset: number;       // meters
  apexCount: number;
  brakingZoneCount: number;
}

export interface RacingLineProfile {
  schemaVersion: string;
  optimizerVersion: string;
  circuitId: string;
  trackGeometryHash: string;
  generatedAt: string;
  vehicleConfig: VehicleParameters;
  metrics: OptimizationMetrics;
  waypoints: RacingLineWaypoint[];
}
