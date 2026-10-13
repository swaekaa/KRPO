import * as THREE from 'three';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ============================================================
// Standalone Profile Exporter Script (Node.js ESM)
// Exports baseline and optimized racing line profiles to JSON.
// Run: node KRPO/scripts/export_profiles.js
// ============================================================

const ROAD_WIDTH = 12.0;
const CAR_HALF_WIDTH = 1.1;
const MAX_LEGAL_OFFSET = ROAD_WIDTH / 2 - CAR_HALF_WIDTH; // 4.9m
const SAFETY_MARGIN = 0.4;
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
  const arcLengths = [0];

  for (let i = 0; i < n; i++) {
    const p = rawPoints[i];
    const nextP = rawPoints[(i + 1) % n];
    const dist = p.distanceTo(nextP);
    totalLength += dist;
    arcLengths.push(totalLength);
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
      arcLength: arcLengths[i],
    });
  }

  return { samples, totalLength };
}

function solveProfile(samples, offsets) {
  const n = samples.length;
  const positions = new Array(n);
  for (let i = 0; i < n; i++) {
    const s = samples[i];
    const norm = new THREE.Vector3(s.normal.x, s.normal.y, s.normal.z);
    const p = new THREE.Vector3(s.position.x, s.position.y, s.position.z).addScaledVector(norm, offsets[i]);
    p.y = 0;
    positions[i] = p;
  }

  const ds = new Array(n);
  let totalLength = 0;
  for (let i = 0; i < n; i++) {
    const dist = positions[i].distanceTo(positions[(i + 1) % n]);
    ds[i] = Math.max(0.01, dist);
    totalLength += dist;
  }

  const curvatures = new Array(n);
  for (let i = 0; i < n; i++) {
    const prev = positions[(i - 1 + n) % n];
    const curr = positions[i];
    const next = positions[(i + 1) % n];
    const t1 = new THREE.Vector3().subVectors(curr, prev).normalize();
    const t2 = new THREE.Vector3().subVectors(next, curr).normalize();
    const crossY = t1.x * t2.z - t1.z * t2.x;
    const dot = Math.max(-1, Math.min(1, t1.dot(t2)));
    const angle = Math.atan2(crossY, dot);
    const segLen = (curr.distanceTo(prev) + curr.distanceTo(next)) * 0.5;
    curvatures[i] = segLen > 1e-4 ? angle / segLen : 0;
  }

  const TOP_SPEED = 22.0;
  const STEERING_RATE = 2.0;
  const A_ACCEL = 7.0;
  const A_BRAKE = 24.0;

  const vProfile = new Array(n);
  for (let i = 0; i < n; i++) {
    const absK = Math.abs(curvatures[i]);
    const maxCornerV = absK > 1e-4 ? Math.min(TOP_SPEED, Math.sqrt((9.81 * 1.3) / absK), STEERING_RATE / absK) : TOP_SPEED;
    vProfile[i] = Math.max(5.0, Math.min(TOP_SPEED, maxCornerV));
  }

  for (let cycle = 0; cycle < 2; cycle++) {
    for (let i = n - 1; i >= 0; i--) {
      const nextIdx = (i + 1) % n;
      const maxEntry = Math.sqrt(vProfile[nextIdx] * vProfile[nextIdx] + 2 * A_BRAKE * ds[i]);
      vProfile[i] = Math.min(vProfile[i], maxEntry);
    }
  }

  for (let cycle = 0; cycle < 2; cycle++) {
    for (let i = 0; i < n; i++) {
      const nextIdx = (i + 1) % n;
      const maxExit = Math.sqrt(vProfile[i] * vProfile[i] + 2 * A_ACCEL * ds[i]);
      vProfile[nextIdx] = Math.min(vProfile[nextIdx], maxExit);
    }
  }

  let lapTime = 0;
  let maxV = 0;
  let arcLength = 0;
  let apexCount = 0;
  let brakingCount = 0;
  const waypoints = new Array(n);

  for (let i = 0; i < n; i++) {
    const v = vProfile[i];
    const nextV = vProfile[(i + 1) % n];
    const dt = ds[i] / Math.max(1.0, (v + nextV) * 0.5);
    lapTime += dt;
    if (v > maxV) maxV = v;

    const aLong = (nextV * nextV - v * v) / (2 * ds[i]);
    const isBrakingPoint = aLong < -2.5;

    const absK = Math.abs(curvatures[i]);
    const prevK = Math.abs(curvatures[(i - 1 + n) % n]);
    const nextK = Math.abs(curvatures[(i + 1) % n]);
    const isApex = absK > 0.015 && absK >= prevK && absK >= nextK;

    if (isApex) apexCount++;
    if (isBrakingPoint) brakingCount++;

    const curr = positions[i];
    const next = positions[(i + 1) % n];
    const tangent = new THREE.Vector3().subVectors(next, curr).normalize();
    const normal = new THREE.Vector3(-tangent.z, 0, tangent.x).normalize();

    waypoints[i] = {
      index: i,
      arcLength: Math.round(arcLength * 10) / 10,
      position: { x: Math.round(curr.x * 100) / 100, y: 0, z: Math.round(curr.z * 100) / 100 },
      tangent: { x: Math.round(tangent.x * 1000) / 1000, y: 0, z: Math.round(tangent.z * 1000) / 1000 },
      normal: { x: Math.round(normal.x * 1000) / 1000, y: 0, z: Math.round(normal.z * 1000) / 1000 },
      lateralOffset: Math.round(offsets[i] * 100) / 100,
      curvature: Math.round(curvatures[i] * 10000) / 10000,
      targetSpeed: Math.round(v * 100) / 100,
      targetSpeedKmh: Math.round(v * 3.6 * 10) / 10,
      throttle: Math.round(Math.max(0, Math.min(1, aLong / A_ACCEL)) * 100) / 100,
      brake: Math.round(Math.max(0, Math.min(1, -aLong / A_BRAKE)) * 100) / 100,
      isBrakingPoint,
      isApex,
      sector: Math.min(10, Math.floor((i / n) * 10) + 1),
    };

    arcLength += ds[i];
  }

  return {
    lapTime: Math.round(lapTime * 100) / 100,
    averageSpeedKmh: Math.round((totalLength / lapTime) * 3.6 * 10) / 10,
    maxSpeedKmh: Math.round(maxV * 3.6 * 10) / 10,
    totalLength: Math.round(totalLength * 10) / 10,
    apexCount,
    brakingCount,
    waypoints,
  };
}

console.log('[KRPO Exporter] Extracting track...');
const { samples, totalLength } = buildTrack();
const n = samples.length;

// Baseline (centerline offsets = 0)
console.log('[KRPO Exporter] Solving Baseline Profile...');
const zeroOffsets = new Array(n).fill(0);
const baseline = solveProfile(samples, zeroOffsets);

// Optimized Profile (out-in-out MCP)
console.log('[KRPO Exporter] Solving Optimized Profile...');
const MAX_OFFSET = MAX_LEGAL_OFFSET - SAFETY_MARGIN;
const rawOffsets = new Array(n);
for (let i = 0; i < n; i++) {
  rawOffsets[i] = Math.max(-MAX_OFFSET, Math.min(MAX_OFFSET, -samples[i].curvature * 75.0));
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
const optimized = solveProfile(samples, optOffsets);

const timeSaved = Math.round((baseline.lapTime - optimized.lapTime) * 100) / 100;
const pctSaved = Math.round((timeSaved / baseline.lapTime) * 10000) / 100;

const baselineProfile = {
  schemaVersion: '1.0.0',
  optimizerVersion: 'KRPO-Baseline-v1.0',
  circuitId: 'vice-coast-circuit',
  trackGeometryHash: 'hash-vc-500',
  generatedAt: new Date().toISOString(),
  vehicleConfig: {
    topSpeed: 22.0,
    acceleration: 7.0,
    braking: 22.0,
    steeringStrength: 2.0,
    friction: 5.0,
    carHalfWidth: 1.1,
  },
  metrics: {
    simulatedLapTime: baseline.lapTime,
    baselineLapTime: baseline.lapTime,
    timeDelta: 0,
    percentageImprovement: 0,
    averageSpeedKmh: baseline.averageSpeedKmh,
    maxSpeedKmh: baseline.maxSpeedKmh,
    racingLineLength: baseline.totalLength,
    centerlineLength: Math.round(totalLength * 10) / 10,
    isBoundaryCompliant: true,
    maxLateralOffset: 0,
  },
  waypoints: baseline.waypoints,
};

const optimizedProfile = {
  schemaVersion: '1.0.0',
  optimizerVersion: 'KRPO-Optimized-MCP-v1.0',
  circuitId: 'vice-coast-circuit',
  trackGeometryHash: 'hash-vc-500',
  generatedAt: new Date().toISOString(),
  vehicleConfig: {
    topSpeed: 22.0,
    acceleration: 7.0,
    braking: 22.0,
    steeringStrength: 2.0,
    friction: 5.0,
    carHalfWidth: 1.1,
  },
  metrics: {
    simulatedLapTime: optimized.lapTime,
    baselineLapTime: baseline.lapTime,
    timeDelta: -timeSaved,
    percentageImprovement: pctSaved,
    averageSpeedKmh: optimized.averageSpeedKmh,
    maxSpeedKmh: optimized.maxSpeedKmh,
    racingLineLength: optimized.totalLength,
    centerlineLength: Math.round(totalLength * 10) / 10,
    isBoundaryCompliant: true,
    maxLateralOffset: MAX_OFFSET,
  },
  waypoints: optimized.waypoints,
};

const profilesDir = path.resolve(__dirname, '../profiles');
if (!fs.existsSync(profilesDir)) {
  fs.mkdirSync(profilesDir, { recursive: true });
}

fs.writeFileSync(path.join(profilesDir, 'vice_coast_baseline_profile.json'), JSON.stringify(baselineProfile, null, 2));
fs.writeFileSync(path.join(profilesDir, 'vice_coast_optimized_profile.json'), JSON.stringify(optimizedProfile, null, 2));

console.log(`[KRPO Exporter] Successfully exported profiles!`);
console.log(`  Baseline Lap Time:  ${baseline.lapTime}s (Avg ${baseline.averageSpeedKmh} km/h)`);
console.log(`  Optimized Lap Time: ${optimized.lapTime}s (Avg ${optimized.averageSpeedKmh} km/h)`);
console.log(`  Time Saved:         ${timeSaved}s (${pctSaved}% improvement)`);
console.log(`  Saved to: ${profilesDir}`);
