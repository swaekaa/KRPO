import { extractTrackGeometry } from './track/TrackExtractor';
import { BASE_VEHICLE_CONFIG } from './physics/VehicleDynamics';
import { generateBaselineProfile } from './optimizers/BaselineOptimizer';
import { optimizeGeometricRacingLine } from './optimizers/GeometricOptimizer';
import { optimizePolicyRacingLine } from './optimizers/PolicyOptimizer';
import { validateRacingLineProfile } from './evaluation/LapEvaluator';

export function runFullOptimizationPipeline() {
  console.log('[KRPO] 1. Extracting track geometry from KLUSTOR...');
  const track = extractTrackGeometry();
  console.log(`[KRPO] Extracted track: ${track.circuitName}, total length: ${track.totalLength.toFixed(1)}m, samples: ${track.totalSamples}`);

  console.log('[KRPO] 2. Generating Baseline Profile (Centerline)...');
  const baseline = generateBaselineProfile(track, BASE_VEHICLE_CONFIG);
  console.log(`[KRPO] Baseline Lap Time: ${baseline.metrics.baselineLapTime.toFixed(2)}s, Avg Speed: ${baseline.metrics.averageSpeedKmh} km/h`);

  console.log('[KRPO] 3. Optimizing Geometric Minimum Curvature Path (MCP)...');
  const geometric = optimizeGeometricRacingLine(track, BASE_VEHICLE_CONFIG, { iterations: 300 });
  console.log(`[KRPO] Geometric MCP Lap Time: ${geometric.metrics.simulatedLapTime.toFixed(2)}s (Delta: ${geometric.metrics.timeDelta.toFixed(2)}s, ${geometric.metrics.percentageImprovement}% faster)`);

  console.log('[KRPO] 4. Running Policy Trajectory Optimization...');
  const policy = optimizePolicyRacingLine(track, BASE_VEHICLE_CONFIG, { numEpochs: 50, batchSize: 10 });
  console.log(`[KRPO] Policy Lap Time: ${policy.metrics.simulatedLapTime.toFixed(2)}s (Delta: ${policy.metrics.timeDelta.toFixed(2)}s, ${policy.metrics.percentageImprovement}% faster)`);

  console.log('[KRPO] 5. Validating optimized profile...');
  const validation = validateRacingLineProfile(policy, track);
  console.log(`[KRPO] Profile Valid: ${validation.isValid}, Issues: ${validation.issues.length}`);

  return { track, baseline, geometric, policy, validation };
}
