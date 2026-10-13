import * as THREE from 'three';
import type { TrackExportData, TrackSample, CheckpointSample, Vec3 } from '../types';

// ============================================================
// KRPO Track Extractor
// Extracts authoritative track geometry from KLUSTOR's
// Vice Coast Circuit into a reproducible, versioned format.
// ============================================================

export const ROAD_WIDTH = 12.0;
export const CAR_HALF_WIDTH = 1.1;
export const MAX_LEGAL_OFFSET = ROAD_WIDTH / 2 - CAR_HALF_WIDTH; // 4.9m
export const SAFETY_MARGIN = 0.4; // Safe racing line stays within 4.5m
export const TRACK_SAMPLES = 500;

export const AUTHORITATIVE_WAYPOINTS: [number, number, number][] = [
  // Start / City Straight
  [   0, 0,    0],
  [ 150, 0,    0],
  // Twist 1: Chicane right after start
  [ 250, 0,   40],
  [ 300, 0,  -20],
  [ 400, 0,   20],
  // Hairpin 1 (Outer Coast)
  [ 550, 0,   50],
  [ 650, 0,  150],
  [ 450, 0,  200],
  [ 550, 0,  300],
  // Sweep
  [ 650, 0,  450],
  [ 600, 0,  600],
  // Hairpin 2 (Deep Corner)
  [ 450, 0,  750],
  [ 350, 0,  850],
  [ 250, 0,  750],
  [ 150, 0,  850],
  [   0, 0,  800],
  // Technical section (Serpentine)
  [-100, 0,  750],
  [-150, 0,  600],
  [-250, 0,  650],
  [-300, 0,  500],
  [-200, 0,  450],
  [-250, 0,  300],
  // Hairpin 3 (Return)
  [-450, 0,  350],
  [-550, 0,  250],
  [-400, 0,  150],
  [-500, 0,   50],
  // Final Kinks
  [-300, 0,  -50],
  [-150, 0,   30],
  [ -50, 0,    0],
];

/**
 * Computes deterministic hash of track waypoints and width
 */
export function computeGeometryHash(waypoints: [number, number, number][], width: number): string {
  let hash = 0x811c9dc5;
  const str = JSON.stringify({ waypoints, width });
  for (let i = 0; i < str.length; i++) {
    hash ^= str.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return `hash-${(hash >>> 0).toString(16).padStart(8, '0')}`;
}

/**
 * Extracts and samples the full track data exactly as KLUSTOR renders it.
 */
export function extractTrackGeometry(): TrackExportData {
  const threeWaypoints = AUTHORITATIVE_WAYPOINTS.map(p => new THREE.Vector3(p[0], p[1], p[2]));
  const curve = new THREE.CatmullRomCurve3(threeWaypoints, true, 'catmullrom', 0.5);

  const rawPoints = curve.getPoints(TRACK_SAMPLES);
  const n = rawPoints.length - 1; // Drop duplicate last point for closed loop

  const samples: TrackSample[] = [];
  let totalLength = 0;
  const arcLengths: number[] = [0];

  // Pass 1: compute positions, tangents, normals, and arc lengths
  for (let i = 0; i < n; i++) {
    const p = rawPoints[i];
    const nextP = rawPoints[(i + 1) % n];
    const segLen = p.distanceTo(nextP);
    totalLength += segLen;
    arcLengths.push(totalLength);
  }

  for (let i = 0; i < n; i++) {
    const p = rawPoints[i];
    const nextP = rawPoints[(i + 1) % n];
    const prevP = rawPoints[(i - 1 + n) % n];

    // Tangent centered difference
    const tangent = new THREE.Vector3().subVectors(nextP, p).normalize();
    // Normal is perpendicular to tangent on XZ plane pointing right
    const normal = new THREE.Vector3(-tangent.z, 0, tangent.x).normalize();

    const rightEdge = p.clone().addScaledVector(normal, ROAD_WIDTH / 2);
    const leftEdge = p.clone().addScaledVector(normal, -ROAD_WIDTH / 2);

    // Discrete curvature calculation kappa = d(heading) / ds
    const tPrev = new THREE.Vector3().subVectors(p, prevP).normalize();
    const tNext = new THREE.Vector3().subVectors(nextP, p).normalize();
    
    // Cross product Y gives signed angular change
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

  // Checkpoints: 10 evenly spaced checkpoints
  const NUM_CP = 10;
  const checkpoints: CheckpointSample[] = [];
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
  const spawnDistance = 8.0; // 8 meters behind start line
  const spawnPos = {
    x: startSample.position.x - startSample.tangent.x * spawnDistance,
    y: 0,
    z: startSample.position.z - startSample.tangent.z * spawnDistance,
  };

  const geometryHash = computeGeometryHash(AUTHORITATIVE_WAYPOINTS, ROAD_WIDTH);

  return {
    schemaVersion: '1.0.0',
    circuitId: 'vice-coast-circuit',
    circuitName: 'VICE COAST CIRCUIT',
    geometryHash,
    roadWidth: ROAD_WIDTH,
    carHalfWidth: CAR_HALF_WIDTH,
    maxLegalOffset: MAX_LEGAL_OFFSET,
    safetyMargin: SAFETY_MARGIN,
    totalLength,
    totalSamples: samples.length,
    samples,
    checkpoints,
    startTransform: {
      position: spawnPos,
      rotation: Math.atan2(startSample.tangent.x, startSample.tangent.z),
      tangent: { ...startSample.tangent },
    },
  };
}
