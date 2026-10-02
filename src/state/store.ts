import { create } from 'zustand';
import type {
  DatasetSpec,
  Hyperparams,
  NetworkSpec,
  ProbeRef,
  SessionData,
  Snapshot,
  Speed,
} from '../worker';

/** Everything needed to reproduce a run. Plain JSON. */
export interface PlaygroundConfig {
  dataset: DatasetSpec;
  network: NetworkSpec;
  training: Hyperparams;
  /** Seeds weight initialisation, shuffling and dropout. */
  seed: number;
}

export const MAX_HIDDEN_LAYERS = 6;
export const MAX_UNITS = 8;

export const DEFAULT_CONFIG: PlaygroundConfig = {
  dataset: { kind: 'spirals', n: 400, noise: 0, seed: 1 },
  network: {
    hidden: [
      { units: 8, activation: 'tanh' },
      { units: 8, activation: 'tanh' },
    ],
  },
  // Tuned for the spirals: every seed tried reaches ≥ 95 % train accuracy (see session.test.ts).
  training: { lr: 0.03, batchSize: 10, optimiser: 'sgd', l2: 0, dropout: 0 },
  seed: 1,
};

/** Step-through mode: one data point traced forward then backward, one stage at a time. */
export interface StepThrough {
  /** The traced point; null until one is picked. */
  probe: ProbeRef | null;
  /**
   * 0 = input; 1 … C−1 = forward through column k; C = loss; then C+1 … 2C−1 =
   * backward through columns C−1 … 1 (C = number of columns).
   */
  stage: number;
}

export interface AppState {
  config: PlaygroundConfig;
  running: boolean;
  speed: Speed;
  showTestData: boolean;
  /** Bumped to rebuild the session from the same config (Reset). */
  resetCount: number;
  /** The current session's point sets; null until the worker reports them. */
  sessionData: SessionData | null;
  /** Latest snapshot from the training worker; null before the first one. */
  snapshot: Snapshot | null;
  /** Non-null while step-through mode is open. */
  stepThrough: StepThrough | null;

  setDataset(patch: Partial<DatasetSpec>): void;
  setNetwork(network: NetworkSpec): void;
  setTraining(patch: Partial<Hyperparams>): void;
  setSeed(seed: number): void;
  /** Starting training closes step-through. */
  setRunning(running: boolean): void;
  setSpeed(speed: Speed): void;
  setShowTestData(show: boolean): void;
  reset(): void;
  setSessionData(data: SessionData): void;
  setSnapshot(snapshot: Snapshot): void;
  /** Opens step-through (pausing training) or closes it. */
  setStepThroughOpen(open: boolean): void;
  setProbe(probe: ProbeRef): void;
  setStage(stage: number): void;
}

export const useAppStore = create<AppState>()((set) => ({
  config: DEFAULT_CONFIG,
  running: false,
  speed: 300,
  showTestData: false,
  resetCount: 0,
  sessionData: null,
  snapshot: null,
  stepThrough: null,

  setDataset: (patch) =>
    set(({ config }) => ({
      config: { ...config, dataset: { ...config.dataset, ...patch } },
      // Point indices refer to the old data.
      stepThrough: null,
    })),
  setNetwork: (network) =>
    set(({ config, stepThrough }) => ({
      config: { ...config, network },
      // Stage numbers depend on the column count.
      stepThrough: stepThrough && { ...stepThrough, stage: 0 },
    })),
  setTraining: (patch) =>
    set(({ config }) => ({ config: { ...config, training: { ...config.training, ...patch } } })),
  setSeed: (seed) => set(({ config }) => ({ config: { ...config, seed } })),
  setRunning: (running) =>
    set(({ stepThrough }) => ({ running, stepThrough: running ? null : stepThrough })),
  setSpeed: (speed) => set({ speed }),
  setShowTestData: (showTestData) =>
    set(({ stepThrough }) => ({
      showTestData,
      // A hidden test point can't stay selected.
      stepThrough:
        !showTestData && stepThrough?.probe?.set === 'test'
          ? { probe: null, stage: 0 }
          : stepThrough,
    })),
  reset: () => set(({ resetCount }) => ({ resetCount: resetCount + 1 })),
  setSessionData: (sessionData) => set({ sessionData }),
  setSnapshot: (snapshot) => set({ snapshot }),
  setStepThroughOpen: (open) =>
    set(({ running }) => ({
      stepThrough: open ? { probe: null, stage: 0 } : null,
      running: open ? false : running,
    })),
  setProbe: (probe) => set({ stepThrough: { probe, stage: 0 } }),
  setStage: (stage) =>
    set(({ stepThrough }) => ({ stepThrough: stepThrough && { ...stepThrough, stage } })),
}));

/** A fresh random seed for "Regenerate" / "New weights" (UI-only randomness). */
export function randomSeed(): number {
  return Math.floor(Math.random() * 1e9);
}
