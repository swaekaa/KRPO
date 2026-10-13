import type { VehicleParameters } from '../types';

// ============================================================
// KRPO Vehicle Dynamics Model
// Models the exact kinematic and physics limits of KLUSTOR's
// arcade car physics engine (from useCarPhysics.ts).
// ============================================================

export const BASE_VEHICLE_CONFIG: VehicleParameters = {
  topSpeed: 22.0,          // m/s (~79.2 km/h)
  acceleration: 7.0,      // m/s^2
  braking: 22.0,          // m/s^2
  steeringStrength: 2.0,  // rad/s (yaw rate limit)
  friction: 5.0,          // m/s^2
  carHalfWidth: 1.1,      // m
  designScore: 0.0,
};

/**
 * Creates scaled vehicle parameters matching KLUSTOR's designScore multipliers.
 */
export function getScaledVehicleConfig(
  topSpeedStat: number = 100,
  accelStat: number = 3,
  handlingStat: number = 3,
  designScore: number = 0
): VehicleParameters {
  const safeTop = Number.isNaN(topSpeedStat) || !topSpeedStat ? 100 : topSpeedStat;
  const safeAccel = Number.isNaN(accelStat) || !accelStat ? 3 : accelStat;
  const safeHandling = Number.isNaN(handlingStat) || !handlingStat ? 3 : handlingStat;

  const speedMult = ((safeTop - 100) / 40) * 0.8 + 1.0;
  const accelMult = ((safeAccel - 3) / 7) * 0.9 + 0.7;
  const steerMult = ((safeHandling - 3) / 7) * 0.9 + 0.7;

  return {
    topSpeed: BASE_VEHICLE_CONFIG.topSpeed * speedMult,
    acceleration: BASE_VEHICLE_CONFIG.acceleration * accelMult,
    braking: BASE_VEHICLE_CONFIG.braking,
    steeringStrength: BASE_VEHICLE_CONFIG.steeringStrength * steerMult,
    friction: BASE_VEHICLE_CONFIG.friction,
    carHalfWidth: BASE_VEHICLE_CONFIG.carHalfWidth,
    designScore,
  };
}

/**
 * Maximum velocity allowable around a curve of given curvature kappa (1/m)
 * constrained by the vehicle's yaw rate (steeringStrength).
 * 
 * In KLUSTOR, omega = d(theta)/dt <= steeringStrength.
 * Since v * kappa = omega, |v| <= steeringStrength / |kappa|.
 * Also clamped by the vehicle's top speed.
 */
export function maxCornerSpeed(
  kappa: number,
  vehicle: VehicleParameters = BASE_VEHICLE_CONFIG,
  lateralGForceLimit: number = 9.81 * 1.3 // ~1.3g max lateral tire adhesion equivalent
): number {
  const absKappa = Math.abs(kappa);
  if (absKappa < 1e-4) {
    return vehicle.topSpeed;
  }

  // 1. Steering yaw-rate limit: omega <= steeringStrength => v <= steeringStrength / kappa
  const steerSpeedLimit = vehicle.steeringStrength / absKappa;

  // 2. Lateral acceleration friction limit: v^2 * kappa <= a_lat_max => v <= sqrt(a_lat / kappa)
  const gripSpeedLimit = Math.sqrt(lateralGForceLimit / absKappa);

  const cornerLimit = Math.min(steerSpeedLimit, gripSpeedLimit);
  return Math.min(vehicle.topSpeed, Math.max(3.0, cornerLimit));
}
