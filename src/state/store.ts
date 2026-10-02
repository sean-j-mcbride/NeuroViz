import { create } from 'zustand';
import type { DatasetSpec, Hyperparams, NetworkSpec, Snapshot } from '../worker';

/** Everything needed to reproduce a run. Plain JSON. */
export interface PlaygroundConfig {
  dataset: DatasetSpec;
  network: NetworkSpec;
  training: Hyperparams;
  /** Seeds weight initialisation and shuffling. */
  seed: number;
}

/** Epochs per animation frame, or 'max' to train for a fixed time budget each frame. */
export type Speed = 1 | 5 | 20 | 'max';

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
  training: { lr: 0.03, batchSize: 10 },
  seed: 1,
};

export interface AppState {
  config: PlaygroundConfig;
  running: boolean;
  speed: Speed;
  showTestData: boolean;
  /** Bumped to rebuild the session from the same config (Reset). */
  resetCount: number;
  /** Latest snapshot from the training session; null before the first one. */
  snapshot: Snapshot | null;

  setDataset(patch: Partial<DatasetSpec>): void;
  setNetwork(network: NetworkSpec): void;
  setTraining(patch: Partial<Hyperparams>): void;
  setSeed(seed: number): void;
  setRunning(running: boolean): void;
  setSpeed(speed: Speed): void;
  setShowTestData(show: boolean): void;
  reset(): void;
  setSnapshot(snapshot: Snapshot): void;
}

export const useAppStore = create<AppState>()((set) => ({
  config: DEFAULT_CONFIG,
  running: false,
  speed: 5,
  showTestData: false,
  resetCount: 0,
  snapshot: null,

  setDataset: (patch) =>
    set(({ config }) => ({ config: { ...config, dataset: { ...config.dataset, ...patch } } })),
  setNetwork: (network) => set(({ config }) => ({ config: { ...config, network } })),
  setTraining: (patch) =>
    set(({ config }) => ({ config: { ...config, training: { ...config.training, ...patch } } })),
  setSeed: (seed) => set(({ config }) => ({ config: { ...config, seed } })),
  setRunning: (running) => set({ running }),
  setSpeed: (speed) => set({ speed }),
  setShowTestData: (showTestData) => set({ showTestData }),
  reset: () => set(({ resetCount }) => ({ resetCount: resetCount + 1 })),
  setSnapshot: (snapshot) => set({ snapshot }),
}));

/** A fresh random seed for "Regenerate" / "New weights" (UI-only randomness). */
export function randomSeed(): number {
  return Math.floor(Math.random() * 1e9);
}
