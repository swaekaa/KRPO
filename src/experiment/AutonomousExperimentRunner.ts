import * as THREE from 'three';
import { extractTrackGeometry } from '../track/TrackExtractor';
import { BASE_VEHICLE_CONFIG } from '../physics/VehicleDynamics';
import { generateBaselineProfile } from '../optimizers/BaselineOptimizer';
import { optimizeGeometricRacingLine } from '../optimizers/GeometricOptimizer';
import { optimizePolicyRacingLine } from '../optimizers/PolicyOptimizer';
import { HeadlessSimulator } from '../simulation/HeadlessSimulator';
import { TrajectoryFollower } from '../simulation/controllers/TrajectoryFollower';
import type { ReplayDataset, ExperimentSummary } from '../simulation/ReplayTypes';
import type { RacingLineProfile } from '../types';

// ============================================================
// KRPO Autonomous Experiment Runner
// Executes batches of deterministic simulation episodes,
// improves driving policy, audits lap validity, and records replays.
// ============================================================

export interface ExperimentConfig {
  numGenerations: number;
  candidatesPerGeneration: number;
  fixedDt: number;
  maxSimTime: number; // timeout seconds per lap
}

export const DEFAULT_EXPERIMENT_CONFIG: ExperimentConfig = {
  numGenerations: 6,
  candidatesPerGeneration: 6,
  fixedDt: 1 / 60, // 60Hz physics
  maxSimTime: 240, // 4 minutes max per lap
};

export class AutonomousExperimentRunner {
  private simulator: HeadlessSimulator;
  private trackData: ReturnType<typeof extractTrackGeometry>;
  private config: ExperimentConfig;

  constructor(config: Partial<ExperimentConfig> = {}) {
    this.trackData = extractTrackGeometry();
    this.simulator = new HeadlessSimulator(this.trackData, BASE_VEHICLE_CONFIG);
    this.config = { ...DEFAULT_EXPERIMENT_CONFIG, ...config };
  }

  /**
   * Runs a complete simulation episode with a given racing line profile and gains.
   */
  public runEpisode(
    profile: RacingLineProfile,
    runId: string,
    runType: 'baseline' | 'early_candidate' | 'failed' | 'intermediate' | 'best',
    generation: number,
    gainsOverride?: any
  ): ReplayDataset {
    this.simulator.reset();
    const controller = new TrajectoryFollower(profile.waypoints, gainsOverride);

    const dt = this.config.fixedDt;
    let state = this.simulator.getState();

    // Simulation loop
    while (!state.lapCompleted && state.simTime < this.config.maxSimTime) {
      const controls = controller.computeControls(state);
      state = this.simulator.step(controls, dt);
    }

    // Lap validity audit
    const hasCompletedAllCheckpoints = state.currentCheckpoint === this.trackData.checkpoints.length - 1 && state.lapCompleted;
    const isUnderTimeout = state.simTime < this.config.maxSimTime;
    const excessiveCollisions = state.collisionCount > 15; // Excessive wall riding is invalid

    let isValid = hasCompletedAllCheckpoints && isUnderTimeout && !excessiveCollisions;
    let invalidationReason: string | undefined;

    if (!hasCompletedAllCheckpoints) {
      isValid = false;
      invalidationReason = `Did not finish all checkpoints (stopped at ${state.currentCheckpoint}/${this.trackData.checkpoints.length})`;
    } else if (!isUnderTimeout) {
      isValid = false;
      invalidationReason = 'Exceeded maximum simulation time budget';
    } else if (excessiveCollisions) {
      isValid = false;
      invalidationReason = `Excessive track barrier collisions (${state.collisionCount} collisions)`;
    }

    return this.simulator.exportReplayDataset(
      runId,
      runType,
      profile.optimizerVersion,
      generation,
      isValid,
      invalidationReason
    );
  }

  /**
   * Executes the full autonomous optimization study:
   * 1. Baseline Episode
   * 2. Early Candidate Episode
   * 3. Failed/Collision Episode
   * 4. Multi-generation evolutionary policy optimization
   * 5. Best-known Episode
   */
  public runFullStudy(): {
    summary: ExperimentSummary;
    baselineReplay: ReplayDataset;
    earlyReplay: ReplayDataset;
    failedReplay: ReplayDataset;
    bestReplay: ReplayDataset;
  } {
    console.log('[KRPO Experiment] Starting Autonomous Optimization Study on Vice Coast Circuit...');

    const baselineProfile = generateBaselineProfile(this.trackData, BASE_VEHICLE_CONFIG);
    const geometricProfile = optimizeGeometricRacingLine(this.trackData, BASE_VEHICLE_CONFIG, { iterations: 200 });
    const policyProfile = optimizePolicyRacingLine(this.trackData, BASE_VEHICLE_CONFIG, { numEpochs: 30 });

    // 1. Run Baseline Episode
    console.log('[KRPO Experiment] Executing Baseline Episode (Centerline)...');
    const baselineReplay = this.runEpisode(
      baselineProfile,
      'run-001-baseline',
      'baseline',
      0,
      { lookaheadBase: 8.0, lookaheadSpeedGain: 0.3 }
    );
    console.log(`[KRPO Experiment] Baseline finished: ${baselineReplay.formattedLapTime} (Valid: ${baselineReplay.isValid}, Collisions: ${baselineReplay.collisionCount})`);

    // 2. Run Early Candidate Episode (under-tuned controller, late braking)
    console.log('[KRPO Experiment] Executing Early Candidate Episode...');
    const earlyReplay = this.runEpisode(
      baselineProfile,
      'run-002-early',
      'early_candidate',
      1,
      { lookaheadBase: 5.0, lookaheadSpeedGain: 0.2, steeringGain: 1.2, speedTolerance: 1.5 }
    );
    console.log(`[KRPO Experiment] Early Candidate finished: ${earlyReplay.formattedLapTime} (Valid: ${earlyReplay.isValid}, Collisions: ${earlyReplay.collisionCount})`);

    // 3. Run Failed / Collision Episode (overspeed into hairpins, demonstrating penalty)
    console.log('[KRPO Experiment] Executing Failed / Wall-Collision Episode...');
    // Create an aggressive profile with excessive corner entry speeds
    const recklessWaypoints = geometricProfile.waypoints.map(wp => ({
      ...wp,
      targetSpeed: Math.min(22.0, wp.targetSpeed * 1.35), // Overspeed in corners
    }));
    const recklessProfile = { ...geometricProfile, waypoints: recklessWaypoints };
    const failedReplay = this.runEpisode(
      recklessProfile,
      'run-003-collision',
      'failed',
      2,
      { lookaheadBase: 4.5, steeringGain: 1.0, speedTolerance: 3.0 }
    );
    console.log(`[KRPO Experiment] Failed Episode finished: ${failedReplay.formattedLapTime} (Valid: ${failedReplay.isValid}, Collisions: ${failedReplay.collisionCount})`);

    // 4. Multi-generation Policy Evolution
    const archivedRuns: ExperimentSummary['archivedRuns'] = [
      {
        runId: baselineReplay.runId,
        runType: baselineReplay.runType,
        generation: 0,
        lapTime: baselineReplay.lapTime,
        isValid: baselineReplay.isValid,
        avgSpeedKmh: baselineReplay.avgSpeedKmh,
        collisionCount: baselineReplay.collisionCount,
      },
      {
        runId: earlyReplay.runId,
        runType: earlyReplay.runType,
        generation: 1,
        lapTime: earlyReplay.lapTime,
        isValid: earlyReplay.isValid,
        avgSpeedKmh: earlyReplay.avgSpeedKmh,
        collisionCount: earlyReplay.collisionCount,
      },
      {
        runId: failedReplay.runId,
        runType: failedReplay.runType,
        generation: 2,
        lapTime: failedReplay.lapTime,
        isValid: failedReplay.isValid,
        avgSpeedKmh: failedReplay.avgSpeedKmh,
        collisionCount: failedReplay.collisionCount,
      },
    ];

    const generationsSummary: ExperimentSummary['generations'] = [];
    let bestValidLapTime = baselineReplay.isValid ? baselineReplay.lapTime : Infinity;
    let bestReplay = baselineReplay;

    let currentBestProfile = policyProfile;

    // Run generations
    for (let gen = 1; gen <= this.config.numGenerations; gen++) {
      let genLapTimes: number[] = [];
      let genValidCount = 0;
      let genBestTime = Infinity;

      for (let cand = 1; cand <= this.config.candidatesPerGeneration; cand++) {
        // Perturb controller gains and lookahead
        const lookaheadBase = 6.5 + (Math.random() - 0.5) * 1.5;
        const lookaheadGain = 0.35 + (Math.random() - 0.5) * 0.08;
        const steerGain = 2.0 + (Math.random() - 0.5) * 0.4;

        const candidateReplay = this.runEpisode(
          currentBestProfile,
          `run-g${gen}-c${cand}`,
          'intermediate',
          gen,
          { lookaheadBase, lookaheadSpeedGain: lookaheadGain, steeringGain: steerGain }
        );

        if (candidateReplay.isValid) {
          genValidCount++;
          genLapTimes.push(candidateReplay.lapTime);
          if (candidateReplay.lapTime < genBestTime) {
            genBestTime = candidateReplay.lapTime;
          }
          if (candidateReplay.lapTime < bestValidLapTime) {
            bestValidLapTime = candidateReplay.lapTime;
            bestReplay = candidateReplay;
          }
        }

        archivedRuns.push({
          runId: candidateReplay.runId,
          runType: candidateReplay.runType,
          generation: gen,
          lapTime: candidateReplay.lapTime,
          isValid: candidateReplay.isValid,
          avgSpeedKmh: candidateReplay.avgSpeedKmh,
          collisionCount: candidateReplay.collisionCount,
        });
      }

      const avgGenTime = genLapTimes.length > 0 ? genLapTimes.reduce((a, b) => a + b, 0) / genLapTimes.length : 0;
      generationsSummary.push({
        generation: gen,
        bestLapTime: genBestTime === Infinity ? 0 : Math.round(genBestTime * 100) / 100,
        avgLapTime: Math.round(avgGenTime * 100) / 100,
        validCount: genValidCount,
        bestPolicyId: `policy-g${gen}`,
      });
    }

    // Tag the best run as 'best'
    bestReplay = {
      ...bestReplay,
      runType: 'best',
      runId: 'run-best-known',
    };

    const validRunsCount = archivedRuns.filter(r => r.isValid).length;
    const invalidRunsCount = archivedRuns.length - validRunsCount;
    const timeImprovement = Math.round((baselineReplay.lapTime - bestValidLapTime) * 100) / 100;
    const pctImprovement = Math.round((timeImprovement / baselineReplay.lapTime) * 10000) / 100;

    const summary: ExperimentSummary = {
      experimentId: 'exp-vice-coast-v1',
      circuitId: this.trackData.circuitId,
      circuitName: this.trackData.circuitName,
      totalRuns: archivedRuns.length,
      validRuns: validRunsCount,
      invalidRuns: invalidRunsCount,
      completionRate: Math.round((validRunsCount / archivedRuns.length) * 100) / 100,
      baselineLapTime: baselineReplay.lapTime,
      bestValidLapTime,
      timeImprovement,
      percentageImprovement: pctImprovement,
      generations: generationsSummary,
      archivedRuns,
    };

    console.log(`[KRPO Experiment] Optimization Complete!`);
    console.log(`  Total Episodes:      ${archivedRuns.length}`);
    console.log(`  Baseline Lap Time:   ${baselineReplay.lapTime.toFixed(2)}s`);
    console.log(`  Best Valid Lap Time: ${bestValidLapTime.toFixed(2)}s`);
    console.log(`  Time Improvement:    -${timeImprovement.toFixed(2)}s (${pctImprovement}% faster)`);

    return {
      summary,
      baselineReplay,
      earlyReplay,
      failedReplay,
      bestReplay,
    };
  }
}
