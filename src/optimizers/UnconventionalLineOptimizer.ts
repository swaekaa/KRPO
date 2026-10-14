import * as THREE from 'three';
import { extractTrackGeometry, ROAD_WIDTH, MAX_LEGAL_OFFSET, SAFETY_MARGIN } from '../track/TrackExtractor';
import type { TrackExportData } from '../types';

// ============================================================
// KRPO Unconventional Racing Line & Braking Optimizer
// Generates and explores unconventional racing trajectories:
// 1. V-Line / Diamond Corner Squaring (delays turn-in, sharp late apex for maximum boost run-up)
// 2. Chicane & S-Bend Straight-Lining (slices diagonals across chicanes)
// 3. Early Dive-Bomb & Late Trail-Braking (brakes inside turn to scrub velocity with lateral friction)
// 4. Stochastic RL Deformation Splines (Fourier harmonics & Gaussian kernel mutations)
// ============================================================

export type UnconventionalStyle = 
  | 'V_LINE_SQUARE_OFF'
  | 'CHICANE_STRAIGHT_LINE'
  | 'DEEP_TRAIL_BRAKE'
  | 'STOCHASTIC_RL_MUTATION'
  | 'SUPERHUMAN_HYBRID';

export interface UnconventionalLineCandidate {
  style: UnconventionalStyle;
  name: string;
  description: string;
  offsets: number[]; // 500 lateral offsets in meters
  brakingOffsetMeters: number; // Shift in braking point (negative = deeper / later)
  boostExitCurvature: number; // Curvature threshold to trigger boost
  cornerSpeedBias: number; // Multiplier on allowable cornering velocity
}

const MAX_OFFSET = MAX_LEGAL_OFFSET - SAFETY_MARGIN; // 4.5m legal track boundary

/**
 * 1. V-Line / Diamond Squaring Line:
 * In hairpins and 90-degree turns, stays wide on entry (+normal), delays turn-in,
 * snaps rotation late, and squares off the corner to maximize straight-line boost exit!
 */
export function generateVLineSquareOffOffsets(trackData: TrackExportData = extractTrackGeometry()): number[] {
  const n = trackData.samples.length;
  const offsets = new Array(n).fill(0);

  for (let i = 0; i < n; i++) {
    const sCurr = trackData.samples[i];
    const absK = Math.abs(sCurr.curvature || 0);

    if (absK > 0.015) {
      // Find local apex window
      let isLateApex = false;
      for (let look = 1; look <= 8; look++) {
        const nextIdx = (i + look) % n;
        if (Math.abs(trackData.samples[nextIdx].curvature || 0) < absK * 0.7) {
          isLateApex = true; // Curvature is dropping ahead -> we are near late apex
          break;
        }
      }

      const sign = Math.sign(sCurr.curvature || 1);
      if (isLateApex) {
        // Sharp inside clip at late apex (-sign * 4.4m)
        offsets[i] = -sign * Math.min(MAX_OFFSET, 4.4);
      } else {
        // Stay wide on entry to square off (+sign * 4.2m)
        offsets[i] = sign * Math.min(MAX_OFFSET, 4.2);
      }
    } else {
      offsets[i] = 0; // Centerline on straights
    }
  }

  return smoothOffsets(offsets, 3);
}

/**
 * 2. Chicane Straight-Lining:
 * In quick alternating S-bends (e.g. chicane after start and serpentine),
 * cuts a straight diagonal directly across the curb apexes.
 */
export function generateChicaneStraightLineOffsets(trackData: TrackExportData = extractTrackGeometry()): number[] {
  const n = trackData.samples.length;
  const offsets = new Array(n).fill(0);

  for (let i = 0; i < n; i++) {
    const prevIdx = (i - 12 + n) % n;
    const nextIdx = (i + 12) % n;
    const kPrev = trackData.samples[prevIdx].curvature || 0;
    const kCurr = trackData.samples[i].curvature || 0;
    const kNext = trackData.samples[nextIdx].curvature || 0;

    // Detect alternating chicane: curvature flips sign within 24 samples
    if (kPrev * kNext < -1e-4) {
      // S-curve chicane detected! Straight-line by opposing the curvature oscillation
      const chicanePhase = (kCurr - kPrev) / (Math.abs(kCurr) + Math.abs(kPrev) + 1e-4);
      offsets[i] = Math.max(-MAX_OFFSET, Math.min(MAX_OFFSET, -chicanePhase * 4.4));
    } else if (Math.abs(kCurr) > 0.02) {
      offsets[i] = Math.max(-MAX_OFFSET, Math.min(MAX_OFFSET, -kCurr * 60.0));
    }
  }

  return smoothOffsets(offsets, 4);
}

/**
 * 3. Deep Trail-Brake & Inverted Dive Line:
 * Dives sharply to the inside on entry, carries speed deep while trail-braking,
 * and lets centrifugal force wash the car out to the extreme outside on exit.
 */
export function generateDeepTrailBrakeOffsets(trackData: TrackExportData = extractTrackGeometry()): number[] {
  const n = trackData.samples.length;
  const offsets = new Array(n).fill(0);

  for (let i = 0; i < n; i++) {
    const sCurr = trackData.samples[i];
    const k = sCurr.curvature || 0;
    const absK = Math.abs(k);

    if (absK > 0.012) {
      const sign = Math.sign(k || 1);
      // Aggressive inside dive on entry
      offsets[i] = -sign * Math.min(MAX_OFFSET, 4.3);
    } else {
      offsets[i] = 0;
    }
  }

  return smoothOffsets(offsets, 3);
}

/**
 * 4. Stochastic RL Trajectory Mutation:
 * Applies localized Gaussian bumps and multi-frequency Fourier harmonics
 * to explore unconventional racing corridors humans wouldn't think of.
 */
export function mutateTrajectoryStochastically(
  baseOffsets: number[],
  noiseLevel: number = 0.5,
  seed: number = Date.now()
): number[] {
  const n = baseOffsets.length;
  const mutated = [...baseOffsets];

  // 1. Multi-frequency Fourier waves (global harmonic deformations)
  const numWaves = 3;
  for (let w = 0; w < numWaves; w++) {
    const freq = 2 + (w * 3) + (Math.abs(seed) % 4);
    const phase = ((seed * (w + 1)) % 1000) / 1000 * Math.PI * 2;
    const amplitude = (0.3 + (w * 0.15)) * noiseLevel;

    for (let i = 0; i < n; i++) {
      const u = (i / n) * Math.PI * 2;
      mutated[i] += Math.sin(u * freq + phase) * amplitude;
    }
  }

  // 2. Localized Gaussian exploration kernels at random corners
  const numKernels = 4;
  for (let k = 0; k < numKernels; k++) {
    const centerIdx = Math.floor(((seed * (k + 7)) % 1000) / 1000 * n);
    const kernelWidth = 15 + ((k * 5) % 15);
    const kernelHeight = (((k % 2 === 0 ? 1 : -1) * 1.2) * noiseLevel);

    for (let offset = -kernelWidth; offset <= kernelWidth; offset++) {
      const idx = (centerIdx + offset + n) % n;
      const dist = Math.abs(offset) / kernelWidth;
      const weight = Math.exp(-dist * dist * 3.0);
      mutated[idx] += kernelHeight * weight;
    }
  }

  // Clamp within legal track boundaries (-4.5m to +4.5m)
  for (let i = 0; i < n; i++) {
    mutated[i] = Math.max(-MAX_OFFSET, Math.min(MAX_OFFSET, mutated[i]));
  }

  return smoothOffsets(mutated, 2);
}

/**
 * Helper: Gaussian/Laplacian smoothing pass
 */
export function smoothOffsets(offsets: number[], passes: number = 2): number[] {
  const n = offsets.length;
  let current = [...offsets];

  for (let p = 0; p < passes; p++) {
    const next = new Array(n);
    for (let i = 0; i < n; i++) {
      const prev = current[(i - 1 + n) % n];
      const curr = current[i];
      const nxt = current[(i + 1) % n];
      next[i] = 0.25 * prev + 0.50 * curr + 0.25 * nxt;
    }
    current = next;
  }

  return current;
}

/**
 * Generates an unconventional candidate bundle for live evolutionary search.
 */
export function generateUnconventionalCandidates(
  generation: number,
  championOffsets?: number[]
): UnconventionalLineCandidate[] {
  const track = extractTrackGeometry();
  const vLine = generateVLineSquareOffOffsets(track);
  const chicaneCut = generateChicaneStraightLineOffsets(track);
  const trailBrake = generateDeepTrailBrakeOffsets(track);

  const base = championOffsets ?? vLine;

  return [
    {
      style: 'V_LINE_SQUARE_OFF',
      name: 'V-Line Square-Off & Late Apex Boost Launch',
      description: 'Squares off hairpins late to aim straight early, flooring 240 km/h boost down straights',
      offsets: vLine,
      brakingOffsetMeters: -4.5, // brakes 4.5m deeper
      boostExitCurvature: 0.024,  // launches boost earlier at late apex
      cornerSpeedBias: 1.08,
    },
    {
      style: 'CHICANE_STRAIGHT_LINE',
      name: 'Chicane & S-Bend Diagonal Straight-Line Cut',
      description: 'Ignores weaving geometry and slices straight through chicanes at 210+ km/h',
      offsets: chicaneCut,
      brakingOffsetMeters: -6.0, // eliminates chicane braking
      boostExitCurvature: 0.022,
      cornerSpeedBias: 1.14,
    },
    {
      style: 'DEEP_TRAIL_BRAKE',
      name: 'Deep Trail-Brake & Inverted Inside Dive',
      description: 'Brakes deep inside the corner using lateral tire scrub, pivoting heading directly to exit',
      offsets: trailBrake,
      brakingOffsetMeters: -7.5, // ultra-deep late braking
      boostExitCurvature: 0.020,
      cornerSpeedBias: 1.10,
    },
    {
      style: 'STOCHASTIC_RL_MUTATION',
      name: `RL Harmonic Spline Deformation (Gen ${generation})`,
      description: 'Unconstrained Fourier wave & Gaussian kernel trajectory exploration',
      offsets: mutateTrajectoryStochastically(base, 0.45 + (generation % 4) * 0.15, generation * 100 + 42),
      brakingOffsetMeters: -5.0 - (generation % 3),
      boostExitCurvature: 0.023,
      cornerSpeedBias: 1.12,
    },
  ];
}
