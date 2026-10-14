import * as THREE from 'three';
import { extractTrackGeometry, ROAD_WIDTH, CAR_HALF_WIDTH, MAX_LEGAL_OFFSET } from '../track/TrackExtractor';
import { BASE_VEHICLE_CONFIG } from '../physics/VehicleDynamics';
import type { TrackExportData, VehicleParameters } from '../types';
import type { SimulationControlInput, ReplayFrame, ReplayDataset } from './ReplayTypes';

// ============================================================
// KRPO Headless Simulation Engine
// Discrete, deterministic simulation replicating the exact physics
// of KLUSTOR (useCarPhysics.ts) and race logic (useRaceState.ts).
// Supports bulk execution at thousands of steps per second.
// ============================================================

export interface SimulationState {
  position: THREE.Vector3;
  rotationY: number;
  speed: number;
  boost: number;
  currentCheckpoint: number;
  lapCompleted: boolean;
  simTime: number;
  tick: number;
  totalDistance: number;
  collisionCount: number;
  isCollision: boolean;
  lateralOffset: number;
  lastNearestIdx: number;
  splits: number[];
}

export class HeadlessSimulator {
  private trackData: TrackExportData;
  private vehicle: VehicleParameters;
  private state: SimulationState;
  private prevPosition: THREE.Vector3;
  private recordedFrames: ReplayFrame[] = [];

  constructor(
    trackData: TrackExportData = extractTrackGeometry(),
    vehicle: VehicleParameters = BASE_VEHICLE_CONFIG
  ) {
    this.trackData = trackData;
    this.vehicle = vehicle;
    this.state = this.createInitialState();
    this.prevPosition = this.state.position.clone();
  }

  private createInitialState(): SimulationState {
    const startPos = new THREE.Vector3(
      this.trackData.startTransform.position.x,
      0,
      this.trackData.startTransform.position.z
    );
    const maxBoost = Math.max(0, Math.min(1.0, (this.vehicle.designScore ?? 10.0) / 10.0));
    return {
      position: startPos,
      rotationY: this.trackData.startTransform.rotation,
      speed: 0,
      boost: maxBoost,
      currentCheckpoint: 0,
      lapCompleted: false,
      simTime: 0,
      tick: 0,
      totalDistance: 0,
      collisionCount: 0,
      isCollision: false,
      lateralOffset: 0,
      lastNearestIdx: 0,
      splits: [],
    };
  }

  public reset(): SimulationState {
    this.state = this.createInitialState();
    this.prevPosition = this.state.position.clone();
    this.recordedFrames = [];
    return this.getState();
  }

  public getState(): SimulationState {
    return {
      ...this.state,
      position: this.state.position.clone(),
      splits: [...this.state.splits],
    };
  }

  public getTrackData(): TrackExportData {
    return this.trackData;
  }

  public getRecordedFrames(): ReplayFrame[] {
    return [...this.recordedFrames];
  }

  /**
   * Finds nearest track sample using the local search window from useCarPhysics.ts
   */
  private findNearest(pos: THREE.Vector3, lastIdx: number) {
    const samples = this.trackData.samples;
    const numSamples = samples.length;
    let minDistSq = Infinity;
    let nearestIdx = 0;

    if (lastIdx >= 0) {
      for (let offset = -20; offset <= 20; offset++) {
        const i = (lastIdx + offset + numSamples) % numSamples;
        const dx = pos.x - samples[i].position.x;
        const dz = pos.z - samples[i].position.z;
        const d2 = dx * dx + dz * dz;
        if (d2 < minDistSq) {
          minDistSq = d2;
          nearestIdx = i;
        }
      }
    } else {
      for (let i = 0; i < numSamples; i++) {
        const dx = pos.x - samples[i].position.x;
        const dz = pos.z - samples[i].position.z;
        const d2 = dx * dx + dz * dz;
        if (d2 < minDistSq) {
          minDistSq = d2;
          nearestIdx = i;
        }
      }
    }

    return { nearest: samples[nearestIdx], nearestIdx };
  }

  /**
   * Executes one discrete physics tick (default dt = 1/60s).
   */
  public step(input: SimulationControlInput, dt: number = 1 / 60): SimulationState {
    if (this.state.lapCompleted) return this.getState();

    this.prevPosition = this.state.position.clone();

    // 1. Boost Logic — Capacity directly governed by designScore (10/10 => 100%)
    let isBoosting = false;
    const maxBoost = Math.max(0, Math.min(1.0, (this.vehicle.designScore ?? 10.0) / 10.0));
    if (input.boost && this.state.boost > 0.01 && maxBoost > 0) {
      this.state.boost = Math.max(0, this.state.boost - 0.2 * dt);
      isBoosting = true;
    } else if (!input.boost) {
      this.state.boost = Math.min(maxBoost, this.state.boost + 0.1 * dt);
    }

    const boostMult = isBoosting ? (this.vehicle.boostMultiplier ?? 1.6) : 1.0;
    const currentMaxSpeed = this.vehicle.topSpeed * boostMult;
    const currentAccel = this.vehicle.acceleration * boostMult;

    // 2. Speed update
    if (input.throttle > 0) {
      this.state.speed = Math.min(currentMaxSpeed, this.state.speed + currentAccel * input.throttle * dt);
    } else if (input.brake > 0) {
      if (this.state.speed > 0.5) {
        this.state.speed = Math.max(0, this.state.speed - this.vehicle.braking * input.brake * dt);
      } else {
        this.state.speed = Math.max(-6.0, this.state.speed - currentAccel * input.brake * dt);
      }
    } else {
      const friction = this.vehicle.friction * dt;
      if (Math.abs(this.state.speed) < friction) {
        this.state.speed = 0;
      } else {
        this.state.speed -= Math.sign(this.state.speed) * friction;
      }
    }

    // 3. Steering update
    const steerFactor = Math.min(1.0, Math.abs(this.state.speed) / 5.0);
    const clampedSteer = Math.max(-1.0, Math.min(1.0, input.steering));

    if (clampedSteer !== 0 && Math.abs(this.state.speed) > 0.2) {
      const steerDelta = clampedSteer * this.vehicle.steeringStrength * steerFactor * dt * Math.sign(this.state.speed);
      this.state.rotationY += steerDelta;
    }

    // 4. Movement forward: forward = (sin(rotY), 0, cos(rotY))
    const forwardX = Math.sin(this.state.rotationY);
    const forwardZ = Math.cos(this.state.rotationY);
    const proposed = new THREE.Vector3(
      this.state.position.x + forwardX * this.state.speed * dt,
      0,
      this.state.position.z + forwardZ * this.state.speed * dt
    );

    // 5. Boundary Clamping & Collision
    const { nearest, nearestIdx } = this.findNearest(proposed, this.state.lastNearestIdx);
    this.state.lastNearestIdx = nearestIdx;

    const toProposed = proposed.clone().sub(new THREE.Vector3(nearest.position.x, 0, nearest.position.z));
    const normalVec = new THREE.Vector3(nearest.normal.x, 0, nearest.normal.z);
    const tangentVec = new THREE.Vector3(nearest.tangent.x, 0, nearest.tangent.z);

    const lateralOffset = toProposed.dot(normalVec);
    const longitudinalOffset = toProposed.dot(tangentVec);
    this.state.lateralOffset = lateralOffset;

    const maxOffset = (ROAD_WIDTH / 2) - CAR_HALF_WIDTH; // 4.9m

    if (Math.abs(lateralOffset) > maxOffset) {
      const clampedLateral = Math.sign(lateralOffset) * maxOffset;
      const clampedPos = new THREE.Vector3(nearest.position.x, 0, nearest.position.z)
        .addScaledVector(tangentVec, longitudinalOffset)
        .addScaledVector(normalVec, clampedLateral);
      clampedPos.y = 0;
      this.state.position.copy(clampedPos);
      this.state.speed *= 0.85; // Wall hit penalty
      this.state.isCollision = true;
      this.state.collisionCount++;
    } else {
      this.state.position.copy(proposed);
      this.state.isCollision = false;
    }

    this.state.position.y = 0;

    // Accumulate total distance
    const segDist = this.prevPosition.distanceTo(this.state.position);
    this.state.totalDistance += segDist;
    this.state.simTime += dt;
    this.state.tick++;

    // 6. Checkpoint Crossing Detection (exact logic from useRaceState.ts)
    const checkpoints = this.trackData.checkpoints;
    const totalCP = checkpoints.length;
    const nextCPIdx = this.state.currentCheckpoint;

    if (nextCPIdx < totalCP) {
      const cp = checkpoints[nextCPIdx];
      const cpPos = new THREE.Vector3(cp.position.x, 0, cp.position.z);
      const cpTan = new THREE.Vector3(cp.tangent.x, 0, cp.tangent.z);

      const vPrev = this.prevPosition.clone().sub(cpPos);
      const vCurr = this.state.position.clone().sub(cpPos);

      const dotPrev = vPrev.dot(cpTan);
      const dotCurr = vCurr.dot(cpTan);

      if (dotPrev < 0 && dotCurr >= 0) {
        // Horizontal distance check along normal
        const cpNormal = new THREE.Vector3(-cpTan.z, 0, cpTan.x).normalize();
        const distFromCenter = Math.abs(vCurr.dot(cpNormal));
        const cpLeft = new THREE.Vector3(cp.leftEdge.x, 0, cp.leftEdge.z);
        const cpRight = new THREE.Vector3(cp.rightEdge.x, 0, cp.rightEdge.z);
        const cpWidth = cpLeft.distanceTo(cpRight);

        if (distFromCenter <= (cpWidth / 2) + 2.0) {
          this.state.splits.push(Math.round(this.state.simTime * 1000));
          if (nextCPIdx < totalCP - 1) {
            this.state.currentCheckpoint++;
          } else {
            this.state.lapCompleted = true;
          }
        }
      }
    }

    // 7. Record Replay Frame (store every tick or decimate if needed)
    const lapProgress = Math.min(1.0, this.state.currentCheckpoint / totalCP);
    this.recordedFrames.push({
      tick: this.state.tick,
      time: Math.round(this.state.simTime * 1000) / 1000,
      position: [
        Math.round(this.state.position.x * 100) / 100,
        0,
        Math.round(this.state.position.z * 100) / 100,
      ],
      rotationY: Math.round(this.state.rotationY * 1000) / 1000,
      speed: Math.round(this.state.speed * 100) / 100,
      speedKmh: Math.round(this.state.speed * 3.6 * 10) / 10,
      steering: Math.round(clampedSteer * 100) / 100,
      throttle: Math.round(input.throttle * 100) / 100,
      brake: Math.round(input.brake * 100) / 100,
      isBoosting,
      boostReserve: Math.round(this.state.boost * 100) / 100,
      checkpointIndex: this.state.currentCheckpoint,
      lapProgress: Math.round(lapProgress * 1000) / 1000,
      lateralOffset: Math.round(this.state.lateralOffset * 100) / 100,
      isCollision: this.state.isCollision,
    });

    return this.getState();
  }

  public exportReplayDataset(
    runId: string,
    runType: 'baseline' | 'early_candidate' | 'failed' | 'intermediate' | 'best',
    policyId: string,
    generation: number,
    isValid: boolean,
    invalidationReason?: string
  ): ReplayDataset {
    const totalSeconds = this.state.simTime;
    const mins = Math.floor(totalSeconds / 60);
    const secs = Math.floor(totalSeconds % 60);
    const ms = Math.floor((totalSeconds % 1) * 1000);
    const formatted = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${String(ms).padStart(3, '0')}`;

    let topSpeed = 0;
    let speedSum = 0;
    let boostActivations = 0;
    let boostDuration = 0;
    let maxBoostSpeedKmh = 0;
    let wasBoosting = false;

    for (const f of this.recordedFrames) {
      if (f.speedKmh > topSpeed) topSpeed = f.speedKmh;
      speedSum += f.speedKmh;
      if (f.isBoosting) {
        boostDuration += (1 / 60);
        if (f.speedKmh > maxBoostSpeedKmh) maxBoostSpeedKmh = f.speedKmh;
        if (!wasBoosting) {
          boostActivations++;
        }
      }
      wasBoosting = f.isBoosting;
    }
    const avgSpeed = this.recordedFrames.length > 0 ? speedSum / this.recordedFrames.length : 0;

    return {
      schemaVersion: '1.0.0',
      runId,
      runType,
      policyId,
      generation,
      timestamp: new Date().toISOString(),
      isValid,
      invalidationReason,
      lapTime: Math.round(totalSeconds * 1000) / 1000,
      formattedLapTime: formatted,
      splits: [...this.state.splits],
      avgSpeedKmh: Math.round(avgSpeed * 10) / 10,
      topSpeedKmh: Math.round(topSpeed * 10) / 10,
      collisionCount: this.state.collisionCount,
      totalDistance: Math.round(this.state.totalDistance * 10) / 10,
      designScore: this.vehicle.designScore ?? 10.0,
      designId: this.vehicle.designId ?? 'krpo-prototype-10-perfect',
      boostActivations,
      boostDuration: Math.round(boostDuration * 100) / 100,
      maxBoostSpeedKmh: Math.round(maxBoostSpeedKmh * 10) / 10,
      frames: [...this.recordedFrames],
    };
  }
}
