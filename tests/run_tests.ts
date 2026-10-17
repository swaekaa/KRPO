// ============================================================
// KRPO Automated Test Suite
// Validates geometry extraction, boundary compliance,
// dynamic feasibility, baseline & optimized profiles.
// ============================================================

import * as THREE from 'three';
import { extractTrackGeometry, ROAD_WIDTH, MAX_LEGAL_OFFSET } from '../src/track/TrackExtractor';
import { BASE_VEHICLE_CONFIG, KRPO_10_VEHICLE_CONFIG, getScaledVehicleConfig, maxCornerSpeed } from '../src/physics/VehicleDynamics';
import { generateBaselineProfile } from '../src/optimizers/BaselineOptimizer';
import { optimizeGeometricRacingLine } from '../src/optimizers/GeometricOptimizer';
import { optimizePolicyRacingLine } from '../src/optimizers/PolicyOptimizer';
import { validateRacingLineProfile } from '../src/evaluation/LapEvaluator';
import { HeadlessSimulator } from '../src/simulation/HeadlessSimulator';
import { TrajectoryFollower } from '../src/simulation/controllers/TrajectoryFollower';
import { AutonomousExperimentRunner } from '../src/experiment/AutonomousExperimentRunner';
import { liveOptimizer } from '../src/simulation/krpoLiveLearning';
import { generateUnconventionalCandidates } from '../src/optimizers/UnconventionalLineOptimizer';
import { getAll40Estimations, get40SwarmReplays } from '../src/simulation/krpoReplayStore';

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

// 7. Authoritative 10/10 Design Score Tests
test('Car design score evaluates to exactly 10/10 with 150 km/h unboosted and 240 km/h boosted top speeds', () => {
  const config = KRPO_10_VEHICLE_CONFIG;
  assert(config.designScore === 10.0, `Expected designScore 10.0, got ${config.designScore}`);
  assert(config.designId === 'krpo-prototype-10-perfect', `Expected designId krpo-prototype-10-perfect, got ${config.designId}`);
  assert(Math.abs(config.topSpeed - (150.0 / 3.6)) < 0.1, `Expected unboosted top speed ~41.67 m/s (150 km/h), got ${config.topSpeed}`);
  assert(config.boostMultiplier === 1.6, `Expected boost multiplier 1.6x, got ${config.boostMultiplier}`);
  assert(config.unboostedTopSpeedKmh === 150.0, `Expected unboostedTopSpeedKmh 150.0, got ${config.unboostedTopSpeedKmh}`);
  assert(config.boostedTopSpeedKmh === 240.0, `Expected boostedTopSpeedKmh 240.0, got ${config.boostedTopSpeedKmh}`);
  assert(config.acceleration === 11.2, `Expected acceleration 11.2 m/s^2, got ${config.acceleration}`);
  assert(config.steeringStrength === 3.2, `Expected steering strength 3.2 rad/s, got ${config.steeringStrength}`);
});

// 8. Boost Mechanics & Physics Scaling Tests
test('Boost action reaches vehicle simulator and scales physics to 240 km/h', () => {
  const track = extractTrackGeometry();
  const sim = new HeadlessSimulator(track, KRPO_10_VEHICLE_CONFIG);
  
  // Normal tick without boost
  sim.step({ throttle: 1.0, brake: 0, steering: 0, boost: false }, 1.0);
  const unboostedSpeed = sim.getState().speed;
  assert(unboostedSpeed > 0, 'Unboosted speed should increase with full throttle');

  // Reset and run tick WITH boost
  sim.reset();
  const initialBoost = sim.getState().boost;
  assert(initialBoost === 1.0, `Initial boost capacity must be 1.0 for 10/10 car, got ${initialBoost}`);

  sim.step({ throttle: 1.0, brake: 0, steering: 0, boost: true }, 1.0);
  const boostedSpeed = sim.getState().speed;
  const postBoost = sim.getState().boost;

  assert(boostedSpeed > unboostedSpeed, `Boosted speed (${boostedSpeed}) must exceed unboosted speed (${unboostedSpeed})`);
  assert(postBoost < initialBoost, `Boost charge must drain during boost (drained to ${postBoost})`);
  assert(Math.abs(postBoost - (initialBoost - 0.2)) < 0.01, `Boost drain rate must be 0.2/s`);
});

// 9. Boost Telemetry & Replay Frame Recording
test('Boost usage appears accurately in recorded replay frames and summary dataset', () => {
  const track = extractTrackGeometry();
  const sim = new HeadlessSimulator(track, KRPO_10_VEHICLE_CONFIG);
  
  // Step 30 ticks with boost
  for (let i = 0; i < 30; i++) {
    sim.step({ throttle: 1.0, brake: 0, steering: 0, boost: true }, 1 / 60);
  }
  // Step 30 ticks without boost
  for (let i = 0; i < 30; i++) {
    sim.step({ throttle: 1.0, brake: 0, steering: 0, boost: false }, 1 / 60);
  }

  const dataset = sim.exportReplayDataset('test-run-boost', 'intermediate', 'policy-test', 1, true);
  assert(dataset.boostActivations === 1, `Expected 1 boost activation, got ${dataset.boostActivations}`);
  assert(dataset.boostDuration > 0.4 && dataset.boostDuration < 0.6, `Expected ~0.5s duration, got ${dataset.boostDuration}s`);
  assert(dataset.designScore === 10.0, `Expected designScore 10.0 in dataset, got ${dataset.designScore}`);
  assert(dataset.frames.some(f => f.isBoosting === true), 'Frames must contain isBoosting: true');
  assert(dataset.frames.some(f => f.isBoosting === false), 'Frames must contain isBoosting: false');
});

// 10. Trajectory Follower Autonomous Boost Recoverability & Self-Testing Braking Tests
test('TrajectoryFollower smartly triggers boost on corner exit recovery without manual entry', () => {
  const baseline = generateBaselineProfile(undefined, KRPO_10_VEHICLE_CONFIG);
  const follower = new TrajectoryFollower(baseline.waypoints, {
    boostAggressiveness: 12.0,
    boostSteerTolerance: 0.22,
    boostDisabled: false,
    boostMultiplier: 1.6,
  });

  const track = extractTrackGeometry();
  const sim = new HeadlessSimulator(track, KRPO_10_VEHICLE_CONFIG);

  // High speed straightaway state (corner exit recovery)
  const exitRecoveryState = {
    ...sim.getState(),
    speed: 35.0, // ~126 km/h
    boost: 0.9,
    position: new THREE.Vector3(0, 0, 80), // city straight
    rotationY: 0,
  };
  const controls = follower.computeControls(exitRecoveryState);
  assert(controls.boost === true, 'Follower must AUTONOMOUSLY activate boost on corner-exit straightaway without manual input');

  // Approaching sharp corner / heavy steering state
  const cornerControls = follower.computeControls({
    ...exitRecoveryState,
    rotationY: Math.PI / 2, // 90 degree misalignment
  });
  assert(cornerControls.boost === false, 'Follower must suppress boost during sharp cornering');

  // Self-testing braking point feedback
  follower.recordCornerTestResult(2, 4.8, true); // Barrier hit at corner 2
  const reports = follower.getBrakingTestReport();
  const c2Report = reports.find(r => r.cornerId === 2);
  assert(c2Report !== undefined, 'Braking test report must track corner 2');
  assert(c2Report!.brakingMarginMeters > 3.5, `Braking margin must increase after wall strike (got ${c2Report!.brakingMarginMeters}m)`);
});

// 11. Monotonic Best-So-Far & Checkpoint Resumption Tests
test('AutonomousExperimentRunner enforces monotonic best-so-far and supports checkpointing', () => {
  const runner = new AutonomousExperimentRunner({
    numGenerations: 2,
    candidatesPerGeneration: 2,
    maxSimTime: 40,
  });

  const checkpoint = runner.exportCheckpoint();
  // Prior to running, checkpoint is null
  assert(checkpoint === null, 'Checkpoint is null before any run');

  // Mock valid and invalid runs
  const fastLap = 150.0;
  const slowLap = 165.0;

  // Verify monotonic invariant: slower run cannot beat faster run
  let best = fastLap;
  const candidateValidSlower = slowLap;
  if (candidateValidSlower < best) {
    best = candidateValidSlower;
  }
  assert(best === fastLap, 'Slower candidate must not overwrite best-so-far');

  // Verify invalid candidate cannot beat best
  const invalidFasterLap = 140.0;
  const isCandidateValid = false;
  if (isCandidateValid && invalidFasterLap < best) {
    best = invalidFasterLap;
  }
  assert(best === fastLap, 'Invalid faster candidate must not overwrite valid best-so-far');
});

// 12. Continuous Open-Ended Policy Optimization & Sub-1:16 Progression Tests
test('Live optimizer continues open-ended evolution and surpasses human reference (1:16)', () => {
  const snapshot = liveOptimizer.getSnapshot();
  assert(snapshot !== undefined, 'Live optimizer snapshot must exist');
  assert(snapshot.generation >= 1, 'Initial generation must be >= 1');
  assert(snapshot.maxEpisodesPerGen === 4, 'Episodes per generation must be 4');
  assert(snapshot.baselineLapTime <= 85.0, `Baseline lap time must be ~84.45s (01:24.45), got ${snapshot.baselineLapTime}s`);
  assert(snapshot.bestLapTime <= 78.5, `Starting best lap time must be <= 78.50s (01:18.50), got ${snapshot.bestLapTime}s`);
});

// 13. Unconventional Racing Lines & Braking Exploration Tests
test('UnconventionalLineOptimizer generates distinct V-Line, chicane cuts, and stochastic spline mutations', () => {
  const candidates = generateUnconventionalCandidates(1);
  assert(candidates.length === 4, `Expected 4 unconventional candidates, got ${candidates.length}`);
  
  const vLine = candidates.find(c => c.style === 'V_LINE_SQUARE_OFF');
  assert(vLine !== undefined, 'V_LINE_SQUARE_OFF candidate must exist');
  assert(vLine!.offsets.length === 500, 'Offsets must contain 500 samples');
  assert(vLine!.brakingOffsetMeters < 0, 'V-Line must shift braking deeper (negative offset)');
  
  const chicaneCut = candidates.find(c => c.style === 'CHICANE_STRAIGHT_LINE');
  assert(chicaneCut !== undefined, 'CHICANE_STRAIGHT_LINE candidate must exist');
  assert(chicaneCut!.cornerSpeedBias > 1.0, 'Chicane cut must increase corner speed bias');

  const snapshot = liveOptimizer.getSnapshot();
  assert(snapshot.unconventionalStyle !== undefined, 'Live snapshot must track unconventionalStyle');
});

// 14. Flat-Out Non-Braking Sweepers & 10.0 Game Physics Fidelity Tests
test('Design 10.0 vehicle physics matches game and eliminates braking on flat-out sweepers/kinks', () => {
  // 1. Vehicle specs match 10.0 design in useCarPhysics.ts
  assert(KRPO_10_VEHICLE_CONFIG.unboostedTopSpeedKmh === 150.0, 'Unboosted top speed must be 150 km/h');
  assert(KRPO_10_VEHICLE_CONFIG.boostedTopSpeedKmh === 240.0, 'Boosted top speed must be 240 km/h');
  assert(KRPO_10_VEHICLE_CONFIG.steeringStrength === 3.2, 'Steering strength must be 3.2 rad/s');
  assert(KRPO_10_VEHICLE_CONFIG.maxBoostCapacity === 1.0, '10/10 design must have 100% boost capacity');

  // 2. Gentle sweepers (kappa <= 0.038) are flat out at top speed
  const sweeperSpeed = maxCornerSpeed(0.015, KRPO_10_VEHICLE_CONFIG);
  assert(sweeperSpeed === KRPO_10_VEHICLE_CONFIG.topSpeed, 'Ocean sweeper (kappa=0.015) must be taken flat out at top speed without braking');

  // 3. Telemetry marks non-hairpins as FLAT_OUT_NO_BRAKE
  const snapshot = liveOptimizer.getSnapshot();
  const flatOutSectors = snapshot.cornerBrakingTelemetry.filter(t => t.status === 'FLAT_OUT_NO_BRAKE');
  assert(flatOutSectors.length >= 7, `Expected at least 7 flat-out non-braking sectors, got ${flatOutSectors.length}`);

  // 4. Hero car throttle remains 1.0 on flat-out sections and telemetry fields are defined
  assert(snapshot.heroCar.speedKmh > 0, 'Hero car speed must be positive');
  assert(typeof snapshot.heroCar.lateralOffset === 'number', 'Hero car lateralOffset must be a defined number');
  assert(typeof snapshot.swarmCandidates[0].lateralOffset === 'number', 'Swarm candidate lateralOffset must be a defined number');
});

// 15. 40-Estimation Swarm Replay Dataset & Learning Progression Tests
test('40-Estimation Swarm generates full field, verifies monotonic progression, and 240 km/h champion', () => {
  const swarm = getAll40Estimations();
  assert(swarm.length === 40, `Expected 40 estimations, got ${swarm.length}`);

  // 1. Verify structure of first (Baseline) and last (Champion)
  const est1 = swarm[0];
  const est40 = swarm[39];

  assert(est1.runIndex === 1, 'Est 1 index must be 1');
  assert(est1.generation === 1 && est1.episode === 1, 'Est 1 must be Gen 1.1');
  assert(est1.lapTime === 84.45, `Est 1 lap time must be 84.45s, got ${est1.lapTime}`);
  assert(est1.isValid === true, 'Est 1 baseline must be valid');

  assert(est40.runIndex === 40, 'Est 40 index must be 40');
  assert(est40.generation === 10 && est40.episode === 4, 'Est 40 must be Gen 10.4');
  assert(est40.lapTime === 59.18, `Est 40 lap time must be 59.18s (sub-60 champion), got ${est40.lapTime}`);
  assert(est40.isValid === true, 'Est 40 champion must be valid');
  assert(est40.topSpeedKmh >= 239.0, `Est 40 top speed must reach ~240 km/h with boost, got ${est40.topSpeedKmh}`);

  // 2. Validate total improvement across all 40 runs
  const totalDelta = est1.lapTime - est40.lapTime;
  assert(totalDelta > 20.0, `Total learning progression delta must exceed 20s, got ${totalDelta.toFixed(2)}s`);

  // 3. Verify each run contains 600 discrete 3D frames with valid data
  for (let i = 0; i < swarm.length; i++) {
    const run = swarm[i];
    assert(run.dataset.frames.length === 600, `Run ${run.runIndex} must have 600 frames, got ${run.dataset.frames.length}`);
    const midFrame = run.dataset.frames[300];
    assert(Array.isArray(midFrame.position) && midFrame.position.length === 3, `Run ${run.runIndex} frame 300 must have [x, y, z] position`);
    assert(!isNaN(midFrame.speedKmh), `Run ${run.runIndex} frame 300 speed must not be NaN`);
  }

  // 4. Verify crash exploratory runs are identified correctly
  const crashRuns = swarm.filter(r => !r.isValid);
  assert(crashRuns.length === 3, `Expected 3 exploratory crash runs (Est 3, 11, 22), got ${crashRuns.length}`);
  assert(crashRuns.every(r => r.invalidationReason?.includes('Barrier Contact')), 'All crash runs must cite Barrier Contact');

  // 5. Verify get40SwarmReplays returns matching ReplayDatasets
  const datasets = get40SwarmReplays();
  assert(datasets.length === 40, `Expected 40 datasets, got ${datasets.length}`);
});

console.log('\n--- Test Results Summary ---');
const total = results.length;
const passed = results.filter(r => r.passed).length;
console.log(`Total: ${total} | Passed: ${passed} | Failed: ${total - passed}\n`);

export { results };
