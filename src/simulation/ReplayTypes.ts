// ============================================================
// KRPO — Replay and Autonomous Simulation Data Types
// ============================================================

export interface SimulationControlInput {
  throttle: number;   // 0.0 to 1.0
  brake: number;      // 0.0 to 1.0
  steering: number;   // -1.0 (right) to +1.0 (left)
  boost: boolean;     // shift / NOS
}

export interface ReplayFrame {
  tick: number;
  time: number;          // seconds elapsed
  position: [number, number, number]; // [x, y, z]
  rotationY: number;     // radians
  speed: number;         // m/s
  speedKmh: number;      // km/h
  steering: number;      // -1.0 to 1.0
  throttle: number;      // 0.0 to 1.0
  brake: number;         // 0.0 to 1.0
  isBoosting: boolean;
  checkpointIndex: number;
  lapProgress: number;   // 0.0 to 1.0
  lateralOffset: number; // meters from centerline (-left, +right)
  isCollision: boolean;  // whether clamped to wall on this tick
}

export interface ReplayDataset {
  schemaVersion: string;
  runId: string;
  runType: 'baseline' | 'early_candidate' | 'failed' | 'intermediate' | 'best';
  policyId: string;
  generation: number;
  timestamp: string;
  isValid: boolean;
  invalidationReason?: string;
  lapTime: number;          // seconds
  formattedLapTime: string; // MM:SS.ms
  splits: number[];         // ms elapsed per checkpoint (10 checkpoints)
  avgSpeedKmh: number;
  topSpeedKmh: number;
  collisionCount: number;
  totalDistance: number;    // meters
  frames: ReplayFrame[];
}

export interface ExperimentSummary {
  experimentId: string;
  circuitId: string;
  circuitName: string;
  totalRuns: number;
  validRuns: number;
  invalidRuns: number;
  completionRate: number; // 0.0 to 1.0
  baselineLapTime: number;
  bestValidLapTime: number;
  timeImprovement: number;
  percentageImprovement: number;
  generations: {
    generation: number;
    bestLapTime: number;
    avgLapTime: number;
    validCount: number;
    bestPolicyId: string;
  }[];
  archivedRuns: {
    runId: string;
    runType: string;
    generation: number;
    lapTime: number;
    isValid: boolean;
    avgSpeedKmh: number;
    collisionCount: number;
  }[];
}
