// ============================================================
// KRPO Automated Test Suite
// Validates geometry extraction, boundary compliance,
// dynamic feasibility, baseline & optimized profiles.
// ============================================================

import { extractTrackGeometry, ROAD_WIDTH, MAX_LEGAL_OFFSET } from '../src/track/TrackExtractor';
import { BASE_VEHICLE_CONFIG, getScaledVehicleConfig, maxCornerSpeed } from '../src/physics/VehicleDynamics';
import { generateBaselineProfile } from '../src/optimizers/BaselineOptimizer';
import { optimizeGeometricRacingLine } from '../src/optimizers/GeometricOptimizer';
import { optimizePolicyRacingLine } from '../src/optimizers/PolicyOptimizer';
import { validateRacingLineProfile } from '../src/evaluation/LapEvaluator';

interface TestResult {
  name: string;
  passed: boolean;
  message?: string;
  durationMs: number;
}

const results: TestResult[] = [];

function assert(condition: boolean, message: string) {
  if (!condition) {
    throw new Error(`Assertion failed: ${message}`);
  }
}

function test(name: string, fn: () => void) {
  const start = Date.now();
  try {
    fn();
    results.push({ name, passed: true, durationMs: Date.now() - start });
    console.log(`  ✓ ${name}`);
  } catch (err: any) {
    results.push({ name, passed: false, message: err?.message || String(err), durationMs: Date.now() - start });
    console.error(`  ✗ ${name}: ${err?.message}`);
  }
}

console.log('\n--- Running KRPO Test Suite ---\n');

// 1. Track Geometry Extraction Tests
test('Track geometry extracts exactly 500 samples and correct dimensions', () => {
  const track = extractTrackGeometry();
  assert(track.samples.length === 500, `Expected 500 samples, got ${track.samples.length}`);
  assert(track.roadWidth === 12.0, `Expected road width 12.0, got ${track.roadWidth}`);
  assert(track.maxLegalOffset === 4.9, `Expected max legal offset 4.9, got ${track.maxLegalOffset}`);
  assert(track.totalLength > 3000 && track.totalLength < 3800, `Expected track length ~3.4km, got ${track.totalLength.toFixed(1)}m`);
  assert(track.checkpoints.length === 10, `Expected 10 checkpoints, got ${track.checkpoints.length}`);
});

test('Track boundary corridor conforms to road width', () => {
  const track = extractTrackGeometry();
  for (let i = 0; i < track.samples.length; i++) {
    const s = track.samples[i];
    const dx = s.rightEdge.x - s.leftEdge.x;
    const dy = s.rightEdge.y - s.leftEdge.y;
    const dz = s.rightEdge.z - s.leftEdge.z;
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    assert(Math.abs(dist - ROAD_WIDTH) < 0.01, `Corridor width at sample ${i} is ${dist.toFixed(2)}, expected ${ROAD_WIDTH}`);
  }
});

// 2. Coordinate Transforms & Continuity Tests
test('Track loop has continuous closed trajectory without coordinate gaps', () => {
  const track = extractTrackGeometry();
  const n = track.samples.length;
  for (let i = 0; i < n; i++) {
    const s1 = track.samples[i].position;
    const s2 = track.samples[(i + 1) % n].position;
    const dist = Math.hypot(s2.x - s1.x, s2.z - s1.z);
    assert(dist < 15.0, `Excessive distance between consecutive samples ${i}->${(i+1)%n}: ${dist}m`);
    assert(Math.abs(s1.y) < 0.01, `Track elevation deviation at sample ${i}: y=${s1.y}`);
  }
});

// 3. Vehicle Dynamics Tests
test('Vehicle dynamics computes accurate cornering limits', () => {
  const vehicle = BASE_VEHICLE_CONFIG;
  // High curvature corner (kappa = 0.1, radius = 10m)
  const speedHairpin = maxCornerSpeed(0.1, vehicle);
  // Straightaway (kappa = 0)
  const speedStraight = maxCornerSpeed(0, vehicle);
  assert(speedHairpin < speedStraight, `Hairpin speed (${speedHairpin}) must be less than straight speed (${speedStraight})`);
  assert(speedStraight === vehicle.topSpeed, `Straight speed must equal vehicle top speed (${vehicle.topSpeed})`);
  assert(speedHairpin <= vehicle.steeringStrength / 0.1, `Corner speed must respect yaw rate steering strength limit`);
});

test('Scaled vehicle configuration matches design score scaling formula', () => {
  const base = getScaledVehicleConfig(100, 3, 3, 0);
  assert(Math.abs(base.topSpeed - 22.0) < 0.01, 'Base top speed mismatch');
  
  const highPerformance = getScaledVehicleConfig(140, 10, 10, 10);
  assert(highPerformance.topSpeed > base.topSpeed, 'High performance top speed must be greater than base');
  assert(highPerformance.acceleration > base.acceleration, 'High performance accel must be greater than base');
});

// 4. Baseline Optimizer Tests
test('Baseline profile is strictly on centerline with 100% boundary compliance', () => {
  const baseline = generateBaselineProfile();
  assert(baseline.waypoints.length === 500, 'Baseline must contain 500 waypoints');
  assert(baseline.metrics.isBoundaryCompliant === true, 'Baseline must be boundary compliant');
  assert(baseline.metrics.maxLateralOffset === 0, 'Baseline lateral offset must be 0');
  assert(baseline.metrics.baselineLapTime > 100 && baseline.metrics.baselineLapTime < 250, `Unreasonable baseline lap time: ${baseline.metrics.baselineLapTime}s`);
});

// 5. Geometric & Policy Optimization Tests
test('Geometric MCP optimizer reduces curvature and produces faster lap time', () => {
  const baseline = generateBaselineProfile();
  const geo = optimizeGeometricRacingLine();
  
  assert(geo.metrics.simulatedLapTime < baseline.metrics.baselineLapTime, `Geometric lap time (${geo.metrics.simulatedLapTime}s) should beat baseline (${baseline.metrics.baselineLapTime}s)`);
  assert(geo.metrics.timeDelta < 0, 'Time delta must be negative (time saved)');
  assert(geo.metrics.isBoundaryCompliant === true, 'Geometric racing line must remain within track limits');
  assert(geo.metrics.maxLateralOffset <= MAX_LEGAL_OFFSET, `Max lateral offset ${geo.metrics.maxLateralOffset}m exceeds ${MAX_LEGAL_OFFSET}m`);
});

test('Policy optimizer achieves valid trajectory and improves over baseline', () => {
  const baseline = generateBaselineProfile();
  const policy = optimizePolicyRacingLine(undefined, undefined, { numEpochs: 15, batchSize: 4 });
  
  assert(policy.metrics.simulatedLapTime < baseline.metrics.baselineLapTime, 'Policy optimizer must be faster than baseline');
  assert(policy.metrics.isBoundaryCompliant === true, 'Policy optimizer must be boundary compliant');
  assert(policy.waypoints.length === 500, 'Waypoints count must be 500');
});

// 6. Profile Audit & Lap Evaluator Tests
test('Lap Evaluator detects profile validity correctly', () => {
  const policy = optimizeGeometricRacingLine();
  const report = validateRacingLineProfile(policy);
  assert(report.isValid === true, `Profile should be valid, but had issues: ${JSON.stringify(report.issues)}`);
  assert(report.metrics.maxLateralOffset <= MAX_LEGAL_OFFSET, 'Max lateral offset within limits');
});

console.log('\n--- Test Results Summary ---');
const total = results.length;
const passed = results.filter(r => r.passed).length;
console.log(`Total: ${total} | Passed: ${passed} | Failed: ${total - passed}\n`);

export { results };
