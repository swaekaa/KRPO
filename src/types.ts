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
  designId?: string;
  boostMultiplier?: number;      // e.g. 1.6 for 10/10 car (150 km/h -> 240 km/h)
  maxBoostCapacity?: number;     // 0.0 to 1.0 (100% for 10/10 car)
  unboostedTopSpeedKmh?: number; // 150 km/h for 10/10 car
  boostedTopSpeedKmh?: number;   // 240 km/h for 10/10 car
}

export type TemplateView = 'left' | 'right' | 'front' | 'rear' | 'top';

export interface CarStats {
  topSpeed: number;       // 100–200 (km/h scale)
  acceleration: number;  // 1–10
  handling: number;      // 1–10
  designScore: number;   // 0–10
  overallRating: number; // computed weighted average
  symmetry?: number;
  contrast?: number;
  variety?: number;
  complexity?: number;
}

export interface LiveryData {
  name: string;
  textures: Partial<Record<TemplateView, string>>;
  stats: CarStats;
  createdAt: string;
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
  isBoostZone?: boolean;
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
