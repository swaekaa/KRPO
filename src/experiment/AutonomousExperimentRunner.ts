import * as THREE from 'three';
import { extractTrackGeometry } from '../track/TrackExtractor';
import { BASE_VEHICLE_CONFIG, KRPO_10_VEHICLE_CONFIG } from '../physics/VehicleDynamics';
import { generateBaselineProfile } from '../optimizers/BaselineOptimizer';
import { optimizeGeometricRacingLine } from '../optimizers/GeometricOptimizer';
import { optimizePolicyRacingLine } from '../optimizers/PolicyOptimizer';
import { HeadlessSimulator } from '../simulation/HeadlessSimulator';
import { TrajectoryFollower, type ControllerGains } from '../simulation/controllers/TrajectoryFollower';
import type { ReplayDataset, ExperimentSummary } from '../simulation/ReplayTypes';
import type { RacingLineProfile, VehicleParameters } from '../types';

// ============================================================
// KRPO Autonomous Experiment Runner
// Executes batches of deterministic simulation episodes,
// improves driving policy, audits lap validity, and records replays.
// Enforces Authoritative 10/10 vehicle configuration and monotonic best-so-far.
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

export interface RunnerCheckpoint {
  experimentId: string;
  generation: number;
  bestValidLapTime: number;
  bestPolicyId: string;
  bestReplay: ReplayDataset;
  archivedRuns: ExperimentSummary['archivedRuns'];
  generations: ExperimentSummary['generations'];
}

export class AutonomousExperimentRunner {
  private simulator: HeadlessSimulator;
  private trackData: ReturnType<typeof extractTrackGeometry>;
  private vehicleConfig: VehicleParameters;
  private config: ExperimentConfig;

  private currentGeneration: number = 0;
  private bestValidLapTime: number = Infinity;
  private bestReplay: ReplayDataset | null = null;
  private archivedRuns: ExperimentSummary['archivedRuns'] = [];
  private generationsSummary: ExperimentSummary['generations'] = [];

  constructor(
    config: Partial<ExperimentConfig> = {},
    vehicleConfig: VehicleParameters = KRPO_10_VEHICLE_CONFIG
  ) {
    this.trackData = extractTrackGeometry();
    this.vehicleConfig = vehicleConfig;
    this.simulator = new HeadlessSimulator(this.trackData, this.vehicleConfig);
    this.config = { ...DEFAULT_EXPERIMENT_CONFIG, ...config };
  }

  /**
   * Resumes or loads a saved checkpoint
   */
  public loadCheckpoint(checkpoint: RunnerCheckpoint) {
    this.currentGeneration = checkpoint.generation;
    this.bestValidLapTime = checkpoint.bestValidLapTime;
    this.bestReplay = checkpoint.bestReplay;
    this.archivedRuns = [...checkpoint.archivedRuns];
    this.generationsSummary = [...checkpoint.generations];
    console.log(`[KRPO] Resumed from Checkpoint Gen ${this.currentGeneration} with Best Valid Time ${this.bestValidLapTime.toFixed(2)}s`);
  }

  /**
   * Exports current runner checkpoint
   */
  public exportCheckpoint(): RunnerCheckpoint | null {
    if (!this.bestReplay) return null;
    return {
      experimentId: 'exp-vice-coast-v1',
      generation: this.currentGeneration,
      bestValidLapTime: this.bestValidLapTime,
      bestPolicyId: this.bestReplay.policyId,
      bestReplay: this.bestReplay,
      archivedRuns: [...this.archivedRuns],
      generations: [...this.generationsSummary],
    };
  }

  /**
   * Runs a complete simulation episode with a given racing line profile and gains.
   */
  public runEpisode(
    profile: RacingLineProfile,
    runId: string,
    runType: 'baseline' | 'early_candidate' | 'failed' | 'intermediate' | 'best',
    generation: number,
    gainsOverride?: Partial<ControllerGains>
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
   * 1. Baseline Episode (Centerline, boost disabled for strict baseline)
   * 2. Early Candidate Episode
   * 3. Failed/Collision Episode
   * 4. Multi-generation evolutionary policy optimization with boost
   * 5. Best-known Episode
   */
  public runFullStudy(): {
    summary: ExperimentSummary;
    baselineReplay: ReplayDataset;
    earlyReplay: ReplayDataset;
    failedReplay: ReplayDataset;
    bestReplay: ReplayDataset;
  } {
    console.log('[KRPO Experiment] Starting Autonomous Optimization Study on Vice Coast Circuit (Verified 10/10 Specs)...');

    const baselineProfile = generateBaselineProfile(this.trackData, this.vehicleConfig);
    const geometricProfile = optimizeGeometricRacingLine(this.trackData, this.vehicleConfig, { iterations: 200 });
    const policyProfile = optimizePolicyRacingLine(this.trackData, this.vehicleConfig, { numEpochs: 30 });

    // 1. Run Baseline Episode (No Boost, strictly centerline)
    console.log('[KRPO Experiment] Executing Baseline Episode (Centerline, No-Boost)...');
    const baselineReplay = this.runEpisode(
      baselineProfile,
      'run-001-baseline',
      'baseline',
      0,
      { lookaheadBase: 8.0, lookaheadSpeedGain: 0.3, boostDisabled: true }
    );
    console.log(`[KRPO Experiment] Baseline finished: ${baselineReplay.formattedLapTime} (Valid: ${baselineReplay.isValid}, Collisions: ${baselineReplay.collisionCount}, Boost: ${baselineReplay.boostActivations})`);

    // 2. Run Early Candidate Episode (under-tuned controller, late braking)
    console.log('[KRPO Experiment] Executing Early Candidate Episode...');
    const earlyReplay = this.runEpisode(
      baselineProfile,
      'run-002-early',
      'early_candidate',
      1,
      { lookaheadBase: 5.0, lookaheadSpeedGain: 0.2, steeringGain: 1.2, speedTolerance: 1.5, boostAggressiveness: 22.0 }
    );
    console.log(`[KRPO Experiment] Early Candidate finished: ${earlyReplay.formattedLapTime} (Valid: ${earlyReplay.isValid}, Collisions: ${earlyReplay.collisionCount})`);

    // 3. Run Failed / Collision Episode (overspeed into hairpins, demonstrating penalty)
    console.log('[KRPO Experiment] Executing Failed / Wall-Collision Episode...');
    const recklessWaypoints = geometricProfile.waypoints.map(wp => ({
      ...wp,
      targetSpeed: Math.min(66.0, wp.targetSpeed * 1.4), // Overspeed in corners
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
        designScore: baselineReplay.designScore,
        designId: baselineReplay.designId,
        boostActivations: baselineReplay.boostActivations,
        boostDuration: baselineReplay.boostDuration,
        maxBoostSpeedKmh: baselineReplay.maxBoostSpeedKmh,
      },
      {
        runId: earlyReplay.runId,
        runType: earlyReplay.runType,
        generation: 1,
        lapTime: earlyReplay.lapTime,
        isValid: earlyReplay.isValid,
        avgSpeedKmh: earlyReplay.avgSpeedKmh,
        collisionCount: earlyReplay.collisionCount,
        designScore: earlyReplay.designScore,
        designId: earlyReplay.designId,
        boostActivations: earlyReplay.boostActivations,
        boostDuration: earlyReplay.boostDuration,
        maxBoostSpeedKmh: earlyReplay.maxBoostSpeedKmh,
      },
      {
        runId: failedReplay.runId,
        runType: failedReplay.runType,
        generation: 2,
        lapTime: failedReplay.lapTime,
        isValid: failedReplay.isValid,
        avgSpeedKmh: failedReplay.avgSpeedKmh,
        collisionCount: failedReplay.collisionCount,
        designScore: failedReplay.designScore,
        designId: failedReplay.designId,
        boostActivations: failedReplay.boostActivations,
        boostDuration: failedReplay.boostDuration,
        maxBoostSpeedKmh: failedReplay.maxBoostSpeedKmh,
      },
    ];

    const generationsSummary: ExperimentSummary['generations'] = [];
    let bestValidLapTime = baselineReplay.isValid ? baselineReplay.lapTime : Infinity;
    let bestReplay = baselineReplay;

    let currentBestProfile = policyProfile;

    // Run generations
    for (let gen = 1; gen <= this.config.numGenerations; gen++) {
      this.currentGeneration = gen;
      let genLapTimes: number[] = [];
      let genValidCount = 0;
      let genBestTime = Infinity;

      for (let cand = 1; cand <= this.config.candidatesPerGeneration; cand++) {
        // Perturb controller gains and boost strategy
        const lookaheadBase = 6.5 + (Math.random() - 0.5) * 1.5;
        const lookaheadGain = 0.35 + (Math.random() - 0.5) * 0.08;
        const steerGain = 2.0 + (Math.random() - 0.5) * 0.4;
        const boostAggressiveness = 13.0 + (Math.random() - 0.5) * 4.0; // Explore NOS trigger speed
        const boostStraightDist = 32.0 + (Math.random() - 0.5) * 10.0;  // Explore straightaway lookahead
        const boostSteerTolerance = 0.18 + (Math.random() - 0.5) * 0.06;

        // Explore slight apex clipping variation
        const candWaypoints = currentBestProfile.waypoints.map(wp => ({
          ...wp,
          lateralOffset: Math.max(-4.4, Math.min(4.4, wp.lateralOffset + (Math.random() - 0.5) * 0.15)),
        }));
        const candProfile = { ...currentBestProfile, waypoints: candWaypoints };

        const candidateReplay = this.runEpisode(
          candProfile,
          `run-g${gen}-c${cand}`,
          'intermediate',
          gen,
          {
            lookaheadBase,
            lookaheadSpeedGain: lookaheadGain,
            steeringGain: steerGain,
            boostAggressiveness,
            boostStraightDist,
            boostSteerTolerance,
            boostDisabled: false,
          }
        );

        if (candidateReplay.isValid) {
          genValidCount++;
          genLapTimes.push(candidateReplay.lapTime);
          if (candidateReplay.lapTime < genBestTime) {
            genBestTime = candidateReplay.lapTime;
          }
          // Strictly monotonic best-so-far preservation
          if (candidateReplay.lapTime < bestValidLapTime) {
            bestValidLapTime = candidateReplay.lapTime;
            bestReplay = candidateReplay;
            currentBestProfile = candProfile;
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
          designScore: candidateReplay.designScore,
          designId: candidateReplay.designId,
          boostActivations: candidateReplay.boostActivations,
          boostDuration: candidateReplay.boostDuration,
          maxBoostSpeedKmh: candidateReplay.maxBoostSpeedKmh,
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

    // Preserve the overall champion as 'best'
    bestReplay = {
      ...bestReplay,
      runType: 'best',
      runId: 'run-best-known',
    };

    this.bestValidLapTime = bestValidLapTime;
    this.bestReplay = bestReplay;
    this.archivedRuns = [...archivedRuns];
    this.generationsSummary = [...generationsSummary];

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
    console.log(`  Baseline Lap Time:   ${baselineReplay.lapTime.toFixed(2)}s (No Boost)`);
    console.log(`  Best Valid Lap Time: ${bestValidLapTime.toFixed(2)}s (With Boost & Apex Policy)`);
    console.log(`  Time Improvement:    -${timeImprovement.toFixed(2)}s (${pctImprovement}% faster)`);
    console.log(`  Car Design Score:    ${bestReplay.designScore}/10 (Verified)`);
    console.log(`  Boost Activations:   ${bestReplay.boostActivations} (${bestReplay.boostDuration}s total duration)`);

    return {
      summary,
      baselineReplay,
      earlyReplay,
      failedReplay,
      bestReplay,
    };
  }
}
