// ============================================================
// KRPO Leaderboard & Training Progression Store
// Self-contained inside KRPO repository.
// Reactive, persistent store synchronizing autonomous KRPO
// simulation episodes, vehicle 10/10 design, boost usage telemetry,
// and monotonic best-so-far leaderboard records.
// ============================================================

export interface KRPOLeaderboardEntry {
  runId: string;
  checkpointId: string;
  policyName: string;
  generation: number;
  episode: number;
  lapTime: number;          // seconds
  formattedLapTime: string; // MM:SS.ms
  avgSpeedKmh: number;
  topSpeedKmh: number;
  isValid: boolean;
  invalidationReason?: string;
  isNewBest: boolean;
  designScore: number;      // 10.0 verified
  designId: string;
  boostActivations: number;
  boostDuration: number;    // seconds
  maxBoostSpeedKmh: number;
  trackVersion: string;
  timestamp: string;
}

export interface KRPOLeaderboardState {
  trackName: string;
  trackVersion: string;
  baselineLapTime: number;
  formattedBaselineTime: string;
  bestRecord: KRPOLeaderboardEntry | null;
  latestCandidate: KRPOLeaderboardEntry | null;
  previousBestTime: number | null;
  totalTimeSaved: number;
  percentageImproved: number;
  currentGeneration: number;
  currentEpisode: number;
  totalRuns: number;
  validRuns: number;
  invalidRuns: number;
  isTrainingActive: boolean;
  lastUpdated: string;
  history: KRPOLeaderboardEntry[];
}

const STORAGE_KEY = 'klustor_krpo_leaderboard_v2';

function formatLapTime(sec: number): string {
  if (!sec || sec === Infinity) return '--:--.---';
  const mins = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.floor((sec % 1) * 1000);
  return `${String(mins).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms).padStart(3, '0')}`;
}

const INITIAL_BASELINE: KRPOLeaderboardEntry = {
  runId: 'krpo-run-001-baseline',
  checkpointId: 'ckpt-gen0-base',
  policyName: 'Centerline Zero-Knowledge Baseline (No Boost)',
  generation: 0,
  episode: 1,
  lapTime: 178.45,
  formattedLapTime: formatLapTime(178.45),
  avgSpeedKmh: 75.4,
  topSpeedKmh: 82.1,
  isValid: true,
  isNewBest: true,
  designScore: 10.0,
  designId: 'krpo-prototype-10-perfect',
  boostActivations: 0,
  boostDuration: 0.0,
  maxBoostSpeedKmh: 0.0,
  trackVersion: 'vice-coast-v2',
  timestamp: '2026-10-14T10:00:00.000Z',
};

class KRPOLeaderboardService {
  private state: KRPOLeaderboardState;
  private listeners: Set<(state: KRPOLeaderboardState) => void> = new Set();

  constructor() {
    this.state = this.loadFromStorage();
  }

  private loadFromStorage(): KRPOLeaderboardState {
    const defaultState: KRPOLeaderboardState = {
      trackName: 'Vice Coast Circuit',
      trackVersion: 'v2.0-catmull-rom',
      baselineLapTime: INITIAL_BASELINE.lapTime,
      formattedBaselineTime: INITIAL_BASELINE.formattedLapTime,
      bestRecord: { ...INITIAL_BASELINE },
      latestCandidate: { ...INITIAL_BASELINE },
      previousBestTime: INITIAL_BASELINE.lapTime,
      totalTimeSaved: 0.0,
      percentageImproved: 0.0,
      currentGeneration: 1,
      currentEpisode: 1,
      totalRuns: 1,
      validRuns: 1,
      invalidRuns: 0,
      isTrainingActive: true,
      lastUpdated: new Date().toISOString(),
      history: [{ ...INITIAL_BASELINE }],
    };

    if (typeof window === 'undefined' || !window.localStorage) {
      return defaultState;
    }

    try {
      const saved = window.localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (parsed.bestRecord && parsed.history) {
          return {
            ...defaultState,
            ...parsed,
          };
        }
      }
    } catch (e) {
      console.warn('[KRPO Leaderboard] Error loading saved state:', e);
    }

    return defaultState;
  }

  private persist() {
    if (typeof window !== 'undefined' && window.localStorage) {
      try {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(this.state));
      } catch (e) {
        console.warn('[KRPO Leaderboard] Storage quota or persistence error:', e);
      }
    }
    this.notify();
  }

  private notify() {
    const clone = this.getState();
    this.listeners.forEach(fn => fn(clone));
  }

  public subscribe(fn: (state: KRPOLeaderboardState) => void): () => void {
    this.listeners.add(fn);
    fn(this.getState());
    return () => {
      this.listeners.delete(fn);
    };
  }

  public getState(): KRPOLeaderboardState {
    return {
      ...this.state,
      bestRecord: this.state.bestRecord ? { ...this.state.bestRecord } : null,
      latestCandidate: this.state.latestCandidate ? { ...this.state.latestCandidate } : null,
      history: [...this.state.history],
    };
  }

  public setTrainingStatus(active: boolean) {
    this.state.isTrainingActive = active;
    this.state.lastUpdated = new Date().toISOString();
    this.persist();
  }

  /**
   * Records a completed simulation episode.
   * Guarantees MONOTONIC best-so-far preservation.
   */
  public recordRun(params: {
    runId: string;
    checkpointId?: string;
    policyName?: string;
    generation: number;
    episode: number;
    lapTime: number;
    avgSpeedKmh: number;
    topSpeedKmh: number;
    isValid: boolean;
    invalidationReason?: string;
    designScore?: number;
    designId?: string;
    boostActivations?: number;
    boostDuration?: number;
    maxBoostSpeedKmh?: number;
    trackVersion?: string;
  }): { isNewBest: boolean; entry: KRPOLeaderboardEntry } {
    const currentBest = this.state.bestRecord?.lapTime ?? Infinity;
    const isNewBest = params.isValid && params.lapTime < currentBest;

    const entry: KRPOLeaderboardEntry = {
      runId: params.runId,
      checkpointId: params.checkpointId || `ckpt-g${params.generation}-e${params.episode}`,
      policyName: params.policyName || `Gen ${params.generation} Candidate ${params.episode}`,
      generation: params.generation,
      episode: params.episode,
      lapTime: Math.round(params.lapTime * 1000) / 1000,
      formattedLapTime: formatLapTime(params.lapTime),
      avgSpeedKmh: Math.round(params.avgSpeedKmh * 10) / 10,
      topSpeedKmh: Math.round(params.topSpeedKmh * 10) / 10,
      isValid: params.isValid,
      invalidationReason: params.invalidationReason,
      isNewBest,
      designScore: params.designScore ?? 10.0,
      designId: params.designId ?? 'krpo-prototype-10-perfect',
      boostActivations: params.boostActivations ?? 0,
      boostDuration: params.boostDuration ? Math.round(params.boostDuration * 10) / 10 : 0,
      maxBoostSpeedKmh: params.maxBoostSpeedKmh ? Math.round(params.maxBoostSpeedKmh * 10) / 10 : params.topSpeedKmh,
      trackVersion: params.trackVersion || this.state.trackVersion,
      timestamp: new Date().toISOString(),
    };

    const existingIdx = this.state.history.findIndex(h => h.runId === entry.runId);
    if (existingIdx >= 0) {
      this.state.history[existingIdx] = entry;
    } else {
      this.state.history.unshift(entry);
    }

    if (this.state.history.length > 150) {
      this.state.history = this.state.history.slice(0, 150);
    }

    this.state.latestCandidate = entry;
    this.state.totalRuns++;
    if (params.isValid) {
      this.state.validRuns++;
    } else {
      this.state.invalidRuns++;
    }
    this.state.currentGeneration = params.generation;
    this.state.currentEpisode = params.episode;
    this.state.lastUpdated = entry.timestamp;

    if (isNewBest) {
      this.state.previousBestTime = currentBest === Infinity ? entry.lapTime : currentBest;
      this.state.bestRecord = { ...entry };
      const timeSaved = Math.max(0, this.state.baselineLapTime - entry.lapTime);
      this.state.totalTimeSaved = Math.round(timeSaved * 100) / 100;
      this.state.percentageImproved = Math.round((timeSaved / this.state.baselineLapTime) * 10000) / 100;
      console.log(`[KRPO Leaderboard] 🏆 MONOTONIC BEST UPDATED: ${entry.formattedLapTime} (-${this.state.totalTimeSaved}s)`);
    }

    this.persist();
    return { isNewBest, entry };
  }

  public resetLeaderboard() {
    this.state.baselineLapTime = INITIAL_BASELINE.lapTime;
    this.state.formattedBaselineTime = INITIAL_BASELINE.formattedLapTime;
    this.state.bestRecord = { ...INITIAL_BASELINE };
    this.state.latestCandidate = { ...INITIAL_BASELINE };
    this.state.previousBestTime = INITIAL_BASELINE.lapTime;
    this.state.totalTimeSaved = 0.0;
    this.state.percentageImproved = 0.0;
    this.state.currentGeneration = 1;
    this.state.currentEpisode = 1;
    this.state.totalRuns = 1;
    this.state.validRuns = 1;
    this.state.invalidRuns = 0;
    this.state.history = [{ ...INITIAL_BASELINE }];
    this.state.lastUpdated = new Date().toISOString();
    this.persist();
  }
}

export const krpoLeaderboardStore = new KRPOLeaderboardService();
