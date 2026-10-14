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

export const KRPO_10_UNBOOSTED_SPEED_MS = 150.0 / 3.6; // ~41.67 m/s (150 km/h)
export const KRPO_10_BOOSTED_SPEED_MS   = 240.0 / 3.6; // ~66.67 m/s (240 km/h)
export const KRPO_10_BOOST_MULTIPLIER   = 1.6;         // 150 km/h * 1.6 = 240 km/h

/**
 * Creates scaled vehicle parameters matching KLUSTOR's designScore multipliers.
 * For 10.0 design score: top speed is 150 km/h unboosted and 240 km/h under boost.
 */
export function getScaledVehicleConfig(
  topSpeedStat: number = 100,
  accelStat: number = 3,
  handlingStat: number = 3,
  designScore: number = 0,
  designId?: string
): VehicleParameters {
  const safeTop = Number.isNaN(topSpeedStat) || !topSpeedStat ? 100 : topSpeedStat;
  const safeAccel = Number.isNaN(accelStat) || !accelStat ? 3 : accelStat;
  const safeHandling = Number.isNaN(handlingStat) || !handlingStat ? 3 : handlingStat;
  const safeDesign = Number.isNaN(designScore) || designScore === undefined ? 0 : designScore;

  // Design score 0.0 -> 22.0 m/s (79.2 km/h); Design score 10.0 -> 41.67 m/s (150.0 km/h)
  const designRatio = Math.max(0, Math.min(1.0, safeDesign / 10.0));
  const baseMs = BASE_VEHICLE_CONFIG.topSpeed; // 22.0 m/s
  const topSpeedMs = designRatio > 0.01 
    ? baseMs + (KRPO_10_UNBOOSTED_SPEED_MS - baseMs) * designRatio
    : baseMs * (((safeTop - 100) / 40) * 0.8 + 1.0);

  const accelMult = ((safeAccel - 3) / 7) * 0.9 + 0.7;
  const steerMult = ((safeHandling - 3) / 7) * 0.9 + 0.7;
  const boostMult = 1.0 + (KRPO_10_BOOST_MULTIPLIER - 1.0) * designRatio; // 1.0x to 1.6x

  return {
    topSpeed: Math.round(topSpeedMs * 100) / 100,
    acceleration: Math.round(BASE_VEHICLE_CONFIG.acceleration * accelMult * 100) / 100,
    braking: 28.0, // High-performance carbon ceramic brakes
    steeringStrength: Math.round(BASE_VEHICLE_CONFIG.steeringStrength * steerMult * 100) / 100,
    friction: BASE_VEHICLE_CONFIG.friction,
    carHalfWidth: BASE_VEHICLE_CONFIG.carHalfWidth,
    designScore: safeDesign,
    designId,
    boostMultiplier: boostMult,
    maxBoostCapacity: designRatio,
    unboostedTopSpeedKmh: Math.round(topSpeedMs * 3.6 * 10) / 10,
    boostedTopSpeedKmh: Math.round(topSpeedMs * boostMult * 3.6 * 10) / 10,
  };
}

/**
 * Authoritative 10/10 Vehicle Configuration used across all KRPO experiments.
 * Mathematically calibrated to:
 * - Unboosted top speed: 150.0 km/h (41.67 m/s)
 * - Boosted top speed:   240.0 km/h (66.67 m/s with 1.6x multiplier)
 * - Acceleration:        11.2 m/s^2 (0-150 km/h in ~3.8s)
 * - Braking:             28.0 m/s^2 (rapid deceleration into hairpins)
 * - Steering:            3.2 rad/s (sharp racing turn-in)
 * - Boost capacity:      100% (1.0)
 */
export const KRPO_10_VEHICLE_CONFIG: VehicleParameters = {
  topSpeed: Math.round(KRPO_10_UNBOOSTED_SPEED_MS * 100) / 100, // 41.67 m/s (~150 km/h)
  acceleration: 11.2,                                          // m/s^2
  braking: 28.0,                                               // m/s^2 carbon ceramic
  steeringStrength: 3.2,                                       // rad/s
  friction: 5.0,                                               // m/s^2
  carHalfWidth: 1.1,                                           // m
  designScore: 10.0,
  designId: 'krpo-prototype-10-perfect',
  boostMultiplier: 1.6,                                        // 150 km/h * 1.6 = 240 km/h
  maxBoostCapacity: 1.0,
  unboostedTopSpeedKmh: 150.0,
  boostedTopSpeedKmh: 240.0,
};

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
