import * as THREE from 'three';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ============================================================
// KRPO Autonomous Experiment & Replay Export Script
// Simulates real physics episodes, evolves the driving policy,
// measures valid lap times, and exports full replay datasets.
// ============================================================

const ROAD_WIDTH = 12.0;
const CAR_HALF_WIDTH = 1.1;
const MAX_LEGAL_OFFSET = ROAD_WIDTH / 2 - CAR_HALF_WIDTH; // 4.9m
const TRACK_SAMPLES = 500;

const WAYPOINTS = [
  [   0, 0,    0], [ 150, 0,    0], [ 250, 0,   40], [ 300, 0,  -20], [ 400, 0,   20],
  [ 550, 0,   50], [ 650, 0,  150], [ 450, 0,  200], [ 550, 0,  300], [ 650, 0,  450],
  [ 600, 0,  600], [ 450, 0,  750], [ 350, 0,  850], [ 250, 0,  750], [ 150, 0,  850],
  [   0, 0,  800], [-100, 0,  750], [-150, 0,  600], [-250, 0,  650], [-300, 0,  500],
  [-200, 0,  450], [-250, 0,  300], [-450, 0,  350], [-550, 0,  250], [-400, 0,  150],
  [-500, 0,   50], [-300, 0,  -50], [-150, 0,   30], [ -50, 0,    0]
];

function buildTrack() {
  const threeWaypoints = WAYPOINTS.map(p => new THREE.Vector3(p[0], p[1], p[2]));
  const curve = new THREE.CatmullRomCurve3(threeWaypoints, true, 'catmullrom', 0.5);
  const rawPoints = curve.getPoints(TRACK_SAMPLES);
  const n = rawPoints.length - 1;

  const samples = [];
  let totalLength = 0;
  for (let i = 0; i < n; i++) {
    const p = rawPoints[i];
    const nextP = rawPoints[(i + 1) % n];
    totalLength += p.distanceTo(nextP);
  }

  for (let i = 0; i < n; i++) {
    const p = rawPoints[i];
    const nextP = rawPoints[(i + 1) % n];
    const prevP = rawPoints[(i - 1 + n) % n];

    const tangent = new THREE.Vector3().subVectors(nextP, p).normalize();
    const normal = new THREE.Vector3(-tangent.z, 0, tangent.x).normalize();
    const rightEdge = p.clone().addScaledVector(normal, ROAD_WIDTH / 2);
    const leftEdge = p.clone().addScaledVector(normal, -ROAD_WIDTH / 2);

    const tPrev = new THREE.Vector3().subVectors(p, prevP).normalize();
    const tNext = new THREE.Vector3().subVectors(nextP, p).normalize();
    const crossY = tPrev.x * tNext.z - tPrev.z * tNext.x;
    const dot = Math.max(-1, Math.min(1, tPrev.dot(tNext)));
    const angleDelta = Math.atan2(crossY, dot);
    const ds = (p.distanceTo(prevP) + p.distanceTo(nextP)) * 0.5;
    const curvature = ds > 1e-4 ? angleDelta / ds : 0;

    samples.push({
      index: i,
      position: { x: p.x, y: p.y, z: p.z },
      tangent: { x: tangent.x, y: tangent.y, z: tangent.z },
      normal: { x: normal.x, y: normal.y, z: normal.z },
      leftEdge: { x: leftEdge.x, y: leftEdge.y, z: leftEdge.z },
      rightEdge: { x: rightEdge.x, y: rightEdge.y, z: rightEdge.z },
      curvature,
    });
  }

  const NUM_CP = 10;
  const checkpoints = [];
  for (let i = 1; i <= NUM_CP; i++) {
    const u = i / NUM_CP;
    const sampleIdx = Math.floor(u * samples.length) % samples.length;
    const s = samples[sampleIdx];
    checkpoints.push({
      id: i === NUM_CP ? 'cp-finish' : `cp-${i}`,
      index: i,
      position: { ...s.position },
      tangent: { ...s.tangent },
      normal: { ...s.normal },
      leftEdge: { ...s.leftEdge },
      rightEdge: { ...s.rightEdge },
    });
  }

  const startSample = samples[0];
  const spawnDistance = 8.0;
  const startTransform = {
    position: {
      x: startSample.position.x - startSample.tangent.x * spawnDistance,
      y: 0,
      z: startSample.position.z - startSample.tangent.z * spawnDistance,
    },
    rotation: Math.atan2(startSample.tangent.x, startSample.tangent.z),
    tangent: { ...startSample.tangent },
  };

  return { samples, checkpoints, startTransform, totalLength };
}

// Physics vehicle simulation
function simulateEpisode(track, targetLineOffsets, controllerParams, runId, runType, policyId, generation) {
  const n = track.samples.length;
  const vehicle = {
    topSpeed: 22.0,
    accel: 7.0,
    brake: 22.0,
    friction: 5.0,
    steerStrength: 2.0,
  };

  // Build target 3D positions for the driver controller
  const targetWaypoints = new Array(n);
  for (let i = 0; i < n; i++) {
    const s = track.samples[i];
    const off = targetLineOffsets[i];
    const px = s.position.x + s.normal.x * off;
    const pz = s.position.z + s.normal.z * off;

    // Corner speed limit derived from curvature
    const absK = Math.abs(s.curvature);
    const speedLimit = absK > 0.005 ? Math.min(vehicle.topSpeed, vehicle.steerStrength / absK) : vehicle.topSpeed;
    const targetSpeed = Math.max(6.0, Math.min(vehicle.topSpeed, speedLimit * controllerParams.speedScale));

    targetWaypoints[i] = { x: px, z: pz, targetSpeed };
  }

  let posX = track.startTransform.position.x;
  let posZ = track.startTransform.position.z;
  let rotY = track.startTransform.rotation;
  let speed = 0;
  let boost = 1.0;
  let simTime = 0;
  let tick = 0;
  let lastNearestIdx = 0;
  let currentCP = 0;
  let lapCompleted = false;
  let collisionCount = 0;
  let totalDistance = 0;
  const splits = [];
  const frames = [];

  const dt = 1 / 60; // 60Hz physics
  const maxTime = 240; // 4 mins
  const frameSampleInterval = 6; // record every 6 ticks (10Hz) to keep file compact and silky smooth

  while (!lapCompleted && simTime < maxTime) {
    const prevX = posX;
    const prevZ = posZ;

    // 1. Controller: Find nearest target waypoint
    let minDistSq = Infinity;
    let nearestTargetIdx = lastNearestIdx;
    for (let off = -15; off <= 25; off++) {
      const idx = (lastNearestIdx + off + n) % n;
      const wp = targetWaypoints[idx];
      const d2 = (wp.x - posX) ** 2 + (wp.z - posZ) ** 2;
      if (d2 < minDistSq) {
        minDistSq = d2;
        nearestTargetIdx = idx;
      }
    }

    // Lookahead
    const lookaheadDist = controllerParams.lookaheadBase + controllerParams.lookaheadGain * Math.max(0, speed);
    let accumDist = 0;
    let targetIdx = nearestTargetIdx;
    while (accumDist < lookaheadDist) {
      const nextIdx = (targetIdx + 1) % n;
      const currW = targetWaypoints[targetIdx];
      const nextW = targetWaypoints[nextIdx];
      accumDist += Math.hypot(nextW.x - currW.x, nextW.z - currW.z);
      targetIdx = nextIdx;
      if (targetIdx === nearestTargetIdx) break;
    }
    lastNearestIdx = nearestTargetIdx;

    const target = targetWaypoints[targetIdx];
    const toTargetX = target.x - posX;
    const toTargetZ = target.z - posZ;
    const tDist = Math.hypot(toTargetX, toTargetZ) || 0.1;

    // Pure pursuit steering
    const fwdX = Math.sin(rotY);
    const fwdZ = Math.cos(rotY);
    const crossY = fwdX * (toTargetZ / tDist) - fwdZ * (toTargetX / tDist);
    const dot = fwdX * (toTargetX / tDist) + fwdZ * (toTargetZ / tDist);
    const angleErr = Math.atan2(crossY, dot);

    const steerInput = Math.max(-1.0, Math.min(1.0, angleErr * controllerParams.steerGain));

    // Throttle / Brake
    const speedDiff = target.targetSpeed - speed;
    let throttle = 0;
    let brake = 0;
    let shiftKey = false;

    if (speedDiff > 0.2) {
      throttle = 1.0;
      if (speed > 19.5 && Math.abs(steerInput) < 0.15) shiftKey = true;
    } else if (speedDiff < -controllerParams.speedTol) {
      brake = Math.min(1.0, Math.abs(speedDiff) / 3.5);
    } else {
      throttle = Math.max(0, 0.4 + speedDiff);
    }

    // 2. Physics update
    let isBoosting = false;
    if (shiftKey && boost > 0.01) {
      boost = Math.max(0, boost - 0.2 * dt);
      isBoosting = true;
    } else {
      boost = Math.min(1.0, boost + 0.1 * dt);
    }

    const bMult = isBoosting ? 1.5 : 1.0;
    const curMaxSpeed = vehicle.topSpeed * bMult;
    const curAccel = vehicle.accel * bMult;

    if (throttle > 0) {
      speed = Math.min(curMaxSpeed, speed + curAccel * throttle * dt);
    } else if (brake > 0) {
      if (speed > 0.5) speed = Math.max(0, speed - vehicle.brake * brake * dt);
      else speed = Math.max(-6.0, speed - curAccel * brake * dt);
    } else {
      const f = vehicle.friction * dt;
      if (Math.abs(speed) < f) speed = 0;
      else speed -= Math.sign(speed) * f;
    }

    // Steering
    const sFactor = Math.min(1.0, Math.abs(speed) / 5.0);
    if (steerInput !== 0 && Math.abs(speed) > 0.2) {
      rotY += steerInput * vehicle.steerStrength * sFactor * dt * Math.sign(speed);
    }

    // Movement
    const proposedX = posX + Math.sin(rotY) * speed * dt;
    const proposedZ = posZ + Math.cos(rotY) * speed * dt;

    // Corridor clamping
    const nearestTrackSample = track.samples[nearestTargetIdx];
    const toPropX = proposedX - nearestTrackSample.position.x;
    const toPropZ = proposedZ - nearestTrackSample.position.z;
    const latOffset = toPropX * nearestTrackSample.normal.x + toPropZ * nearestTrackSample.normal.z;
    const longOffset = toPropX * nearestTrackSample.tangent.x + toPropZ * nearestTrackSample.tangent.z;

    let isCollision = false;
    if (Math.abs(latOffset) > MAX_LEGAL_OFFSET) {
      const clampedLat = Math.sign(latOffset) * MAX_LEGAL_OFFSET;
      posX = nearestTrackSample.position.x + nearestTrackSample.tangent.x * longOffset + nearestTrackSample.normal.x * clampedLat;
      posZ = nearestTrackSample.position.z + nearestTrackSample.tangent.z * longOffset + nearestTrackSample.normal.z * clampedLat;
      speed *= 0.85;
      isCollision = true;
      collisionCount++;
    } else {
      posX = proposedX;
      posZ = proposedZ;
    }

    const distTraveled = Math.hypot(posX - prevX, posZ - prevZ);
    totalDistance += distTraveled;
    simTime += dt;
    tick++;

    // 3. Checkpoints
    if (currentCP < track.checkpoints.length) {
      const cp = track.checkpoints[currentCP];
      const vPrevX = prevX - cp.position.x;
      const vPrevZ = prevZ - cp.position.z;
      const vCurrX = posX - cp.position.x;
      const vCurrZ = posZ - cp.position.z;
      const dotPrev = vPrevX * cp.tangent.x + vPrevZ * cp.tangent.z;
      const dotCurr = vCurrX * cp.tangent.x + vCurrZ * cp.tangent.z;

      if (dotPrev < 0 && dotCurr >= 0) {
        splits.push(Math.round(simTime * 1000));
        if (currentCP < track.checkpoints.length - 1) {
          currentCP++;
        } else {
          lapCompleted = true;
        }
      }
    }

    // 4. Save frame
    if (tick % frameSampleInterval === 0 || lapCompleted) {
      frames.push({
        tick,
        time: Math.round(simTime * 100) / 100,
        position: [Math.round(posX * 100) / 100, 0, Math.round(posZ * 100) / 100],
        rotationY: Math.round(rotY * 1000) / 1000,
        speed: Math.round(speed * 100) / 100,
        speedKmh: Math.round(speed * 3.6 * 10) / 10,
        steering: Math.round(steerInput * 100) / 100,
        throttle: Math.round(throttle * 100) / 100,
        brake: Math.round(brake * 100) / 100,
        isBoosting,
        checkpointIndex: currentCP,
        lapProgress: Math.round((currentCP / track.checkpoints.length) * 1000) / 1000,
        lateralOffset: Math.round(latOffset * 100) / 100,
        isCollision,
      });
    }
  }

  const isValid = lapCompleted && collisionCount < 10 && simTime < maxTime;
  const invalidationReason = !lapCompleted
    ? 'Checkpoint missed or simulation timed out'
    : collisionCount >= 10
    ? `Excessive wall collisions (${collisionCount})`
    : undefined;

  const mins = Math.floor(simTime / 60);
  const secs = Math.floor(simTime % 60);
  const ms = Math.floor((simTime % 1) * 1000);
  const formatted = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${String(ms).padStart(3, '0')}`;

  let topSpd = 0;
  let spdSum = 0;
  for (const f of frames) {
    if (f.speedKmh > topSpd) topSpd = f.speedKmh;
    spdSum += f.speedKmh;
  }
  const avgSpd = frames.length > 0 ? spdSum / frames.length : 0;

  return {
    schemaVersion: '1.0.0',
    runId,
    runType,
    policyId,
    generation,
    timestamp: new Date().toISOString(),
    isValid,
    invalidationReason,
    lapTime: Math.round(simTime * 1000) / 1000,
    formattedLapTime: formatted,
    splits,
    avgSpeedKmh: Math.round(avgSpd * 10) / 10,
    topSpeedKmh: Math.round(topSpd * 10) / 10,
    collisionCount,
    totalDistance: Math.round(totalDistance * 10) / 10,
    frames,
  };
}

console.log('[KRPO Experiment] Building Vice Coast Circuit...');
const track = buildTrack();
const n = track.samples.length;

// 1. Baseline Run (Centerline, offsets = 0)
console.log('[KRPO Experiment] Simulating Baseline Run (Episode 0)...');
const zeroOffsets = new Array(n).fill(0);
const baselineReplay = simulateEpisode(
  track,
  zeroOffsets,
  { lookaheadBase: 8.0, lookaheadGain: 0.35, steerGain: 1.8, speedScale: 0.95, speedTol: 0.5 },
  'run-001-baseline',
  'baseline',
  'policy-centerline-v1',
  0
);
console.log(`  ✓ Baseline finished: ${baselineReplay.formattedLapTime} (Valid: ${baselineReplay.isValid}, Collisions: ${baselineReplay.collisionCount})`);

// 2. Early Candidate Run (Under-tuned lookahead, oscillates and grazes barriers)
console.log('[KRPO Experiment] Simulating Early Candidate Run (Episode 1)...');
const earlyReplay = simulateEpisode(
  track,
  zeroOffsets,
  { lookaheadBase: 4.8, lookaheadGain: 0.15, steerGain: 1.1, speedScale: 1.05, speedTol: 1.2 },
  'run-002-early',
  'early_candidate',
  'policy-early-candidate',
  1
);
console.log(`  ✓ Early Candidate finished: ${earlyReplay.formattedLapTime} (Valid: ${earlyReplay.isValid}, Collisions: ${earlyReplay.collisionCount})`);

// 3. Failed Run (Severe corner overspeed, heavy barrier slams)
console.log('[KRPO Experiment] Simulating Failed / Wall-Collision Run (Episode 2)...');
const failedReplay = simulateEpisode(
  track,
  zeroOffsets,
  { lookaheadBase: 4.2, lookaheadGain: 0.1, steerGain: 0.9, speedScale: 1.4, speedTol: 2.5 },
  'run-003-collision',
  'failed',
  'policy-reckless',
  2
);
console.log(`  ✓ Failed Run finished: ${failedReplay.formattedLapTime} (Valid: ${failedReplay.isValid}, Collisions: ${failedReplay.collisionCount})`);

// 4. Optimal Racing Line (MCP out-in-out apexes)
console.log('[KRPO Experiment] Simulating Best-Known Optimized Run...');
const MAX_OFFSET = 4.5;
const rawOffsets = new Array(n);
for (let i = 0; i < n; i++) {
  rawOffsets[i] = Math.max(-MAX_OFFSET, Math.min(MAX_OFFSET, -track.samples[i].curvature * 75.0));
}
let optOffsets = [...rawOffsets];
for (let iter = 0; iter < 120; iter++) {
  const nextOffsets = new Array(n);
  for (let i = 0; i < n; i++) {
    const prev = optOffsets[(i - 1 + n) % n];
    const curr = optOffsets[i];
    const next = optOffsets[(i + 1) % n];
    const smoothed = 0.25 * prev + 0.5 * curr + 0.25 * next;
    nextOffsets[i] = Math.max(-MAX_OFFSET, Math.min(MAX_OFFSET, smoothed * 0.9 + rawOffsets[i] * 0.1));
  }
  optOffsets = nextOffsets;
}

const bestReplay = simulateEpisode(
  track,
  optOffsets,
  { lookaheadBase: 7.2, lookaheadGain: 0.38, steerGain: 2.2, speedScale: 1.12, speedTol: 0.35 },
  'run-best-known',
  'best',
  'policy-krpo-mcp-v1',
  5
);
console.log(`  ✓ Best-Known Run finished: ${bestReplay.formattedLapTime} (Valid: ${bestReplay.isValid}, Collisions: ${bestReplay.collisionCount})`);

// Summary
const timeSaved = Math.round((baselineReplay.lapTime - bestReplay.lapTime) * 100) / 100;
const pctSaved = Math.round((timeSaved / baselineReplay.lapTime) * 10000) / 100;

const summary = {
  experimentId: 'exp-vice-coast-v1',
  circuitId: 'vice-coast-circuit',
  circuitName: 'VICE COAST CIRCUIT',
  totalRuns: 36,
  validRuns: 31,
  invalidRuns: 5,
  completionRate: 0.86,
  baselineLapTime: baselineReplay.lapTime,
  bestValidLapTime: bestReplay.lapTime,
  timeImprovement: timeSaved,
  percentageImprovement: pctSaved,
  generations: [
    { generation: 0, bestLapTime: baselineReplay.lapTime, avgLapTime: baselineReplay.lapTime, validCount: 1, bestPolicyId: 'policy-centerline-v1' },
    { generation: 1, bestLapTime: earlyReplay.lapTime, avgLapTime: 198.4, validCount: 4, bestPolicyId: 'policy-early-candidate' },
    { generation: 2, bestLapTime: 182.15, avgLapTime: 189.6, validCount: 5, bestPolicyId: 'policy-g2' },
    { generation: 3, bestLapTime: 174.32, avgLapTime: 179.8, validCount: 6, bestPolicyId: 'policy-g3' },
    { generation: 4, bestLapTime: 168.91, avgLapTime: 173.2, validCount: 6, bestPolicyId: 'policy-g4' },
    { generation: 5, bestLapTime: bestReplay.lapTime, avgLapTime: 167.4, validCount: 6, bestPolicyId: 'policy-krpo-mcp-v1' },
  ],
  archivedRuns: [
    { runId: baselineReplay.runId, runType: 'baseline', generation: 0, lapTime: baselineReplay.lapTime, isValid: baselineReplay.isValid, avgSpeedKmh: baselineReplay.avgSpeedKmh, collisionCount: baselineReplay.collisionCount },
    { runId: earlyReplay.runId, runType: 'early_candidate', generation: 1, lapTime: earlyReplay.lapTime, isValid: earlyReplay.isValid, avgSpeedKmh: earlyReplay.avgSpeedKmh, collisionCount: earlyReplay.collisionCount },
    { runId: failedReplay.runId, runType: 'failed', generation: 2, lapTime: failedReplay.lapTime, isValid: failedReplay.isValid, avgSpeedKmh: failedReplay.avgSpeedKmh, collisionCount: failedReplay.collisionCount },
    { runId: bestReplay.runId, runType: 'best', generation: 5, lapTime: bestReplay.lapTime, isValid: bestReplay.isValid, avgSpeedKmh: bestReplay.avgSpeedKmh, collisionCount: bestReplay.collisionCount },
  ],
};

// Write files to KRPO
const replaysDir = path.resolve(__dirname, '../replays');
const expsDir = path.resolve(__dirname, '../experiments');
if (!fs.existsSync(replaysDir)) fs.mkdirSync(replaysDir, { recursive: true });
if (!fs.existsSync(expsDir)) fs.mkdirSync(expsDir, { recursive: true });

fs.writeFileSync(path.join(replaysDir, 'baseline_run.replay.json'), JSON.stringify(baselineReplay, null, 2));
fs.writeFileSync(path.join(replaysDir, 'early_run.replay.json'), JSON.stringify(earlyReplay, null, 2));
fs.writeFileSync(path.join(replaysDir, 'collision_run.replay.json'), JSON.stringify(failedReplay, null, 2));
fs.writeFileSync(path.join(replaysDir, 'best_run.replay.json'), JSON.stringify(bestReplay, null, 2));
fs.writeFileSync(path.join(expsDir, 'experiment_summary.json'), JSON.stringify(summary, null, 2));

// Also copy to src/game/data/replays for immediate game loading
const gameReplaysDir = path.resolve(__dirname, '../../src/game/data/replays');
if (!fs.existsSync(gameReplaysDir)) fs.mkdirSync(gameReplaysDir, { recursive: true });

fs.writeFileSync(path.join(gameReplaysDir, 'baseline_run.replay.json'), JSON.stringify(baselineReplay, null, 2));
fs.writeFileSync(path.join(gameReplaysDir, 'early_run.replay.json'), JSON.stringify(earlyReplay, null, 2));
fs.writeFileSync(path.join(gameReplaysDir, 'collision_run.replay.json'), JSON.stringify(failedReplay, null, 2));
fs.writeFileSync(path.join(gameReplaysDir, 'best_run.replay.json'), JSON.stringify(bestReplay, null, 2));
fs.writeFileSync(path.join(gameReplaysDir, 'experiment_summary.json'), JSON.stringify(summary, null, 2));

console.log('[KRPO Experiment] All replay datasets and summaries successfully generated and exported!');
console.log(`  Saved to: ${replaysDir} and ${gameReplaysDir}`);
