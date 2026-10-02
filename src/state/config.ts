import type { Toy2DKind } from '../data';
import type {
  Activation,
  Checkpoint,
  DatasetSpec,
  Hyperparams,
  NetworkSpec,
  SessionConfig,
} from '../worker';

/** Everything needed to reproduce a run. Plain JSON. */
export interface PlaygroundConfig {
  dataset: DatasetSpec;
  network: NetworkSpec;
  training: Hyperparams;
  /** Seeds weight initialisation, shuffling and dropout. */
  seed: number;
}

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

// The ranges and choices the controls offer. Loaded configs (files, links) are held to them.
export const MAX_HIDDEN_LAYERS = 6;
export const MAX_UNITS = 8;
export const POINTS = { min: 100, max: 1000, step: 50 } as const;
export const NOISE = { min: 0, max: 0.5, step: 0.05 } as const;
export const LEARNING_RATES = [0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3] as const;
export const BATCH_SIZES: readonly Hyperparams['batchSize'][] = [1, 5, 10, 25, 50, 'full'];
export const L2_STRENGTHS = [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1] as const;
export const DROPOUT_RATES = [0, 0.1, 0.2, 0.3, 0.5] as const;
/** Seeds are 32-bit unsigned integers. */
export const MAX_SEED = 2 ** 32 - 1;

export const DATASET_NAMES: Record<Toy2DKind, string> = {
  circle: 'Circle',
  xor: 'XOR',
  spirals: 'Spirals',
  gaussians: 'Gaussians',
};

export const ACTIVATION_NAMES: Record<Activation, string> = {
  tanh: 'Tanh',
  relu: 'ReLU',
  sigmoid: 'Sigmoid',
  linear: 'Linear',
};

export const OPTIMISER_NAMES: Record<Hyperparams['optimiser'], string> = {
  sgd: 'SGD',
  momentum: 'Momentum',
  adam: 'Adam',
};

/** "8 tanh, 8 tanh"-style summary of the hidden layers. */
export function describeNetwork({ hidden }: NetworkSpec): string {
  if (hidden.length === 0) return 'No hidden layers';
  return hidden.map((l) => `${l.units} ${ACTIVATION_NAMES[l.activation]}`).join(' → ');
}

export function describeBatchSize(b: Hyperparams['batchSize']): string {
  return b === 'full' ? 'Full batch' : String(b);
}

/** The worker session for a config, optionally resuming from a checkpoint. */
export function toSessionConfig(
  { dataset, network, training, seed }: PlaygroundConfig,
  resume?: Checkpoint | null,
): SessionConfig {
  return { dataset, network, seed, ...training, ...(resume && { resume }) };
}

export function sameConfig(a: PlaygroundConfig, b: PlaygroundConfig): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}
