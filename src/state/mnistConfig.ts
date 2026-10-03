import type { MnistSubset } from '../data';
import {
  ACTIVATIONS,
  type HiddenLayerSpec,
  type Hyperparams,
  type MnistCheckpoint,
  type MnistSessionConfig,
  type NetworkSpec,
} from '../worker';
import {
  ACTIVATION_NAMES,
  DROPOUT_RATES,
  L2_STRENGTHS,
  LEARNING_RATES,
  MAX_SEED,
  OPTIMISER_NAMES,
} from './config';

/** Everything needed to reproduce an MNIST run (the data is the bundled subset). Plain JSON. */
export interface MnistConfig {
  network: NetworkSpec;
  training: Hyperparams;
  /** Seeds weight initialisation, shuffling and dropout. */
  seed: number;
}

/** 784 → 128 ReLU → 10 with Adam: ≥ 95 % test accuracy within 6 epochs (see mnistSession.test.ts). */
export const MNIST_DEFAULT_CONFIG: MnistConfig = {
  network: { hidden: [{ units: 128, activation: 'relu' }] },
  training: { lr: 0.001, batchSize: 64, optimiser: 'adam', l2: 0, dropout: 0 },
  seed: 1,
};

// The choices the MNIST controls offer. Loaded files are held to them.
export const MNIST_MAX_HIDDEN_LAYERS = 3;
export const MNIST_UNITS = [16, 32, 64, 128, 256] as const;
export const MNIST_BATCH_SIZES = [16, 32, 64, 128, 256] as const;
/** Units for a newly added layer. */
export const MNIST_NEW_LAYER: HiddenLayerSpec = { units: 64, activation: 'relu' };

/** "784 → 128 ReLU → 10"-style summary. */
export function describeMnistNetwork({ hidden }: NetworkSpec): string {
  return ['784', ...hidden.map((l) => `${l.units} ${ACTIVATION_NAMES[l.activation]}`), '10'].join(
    ' → ',
  );
}

/** Weights and biases in the network. */
export function mnistParamCount({ hidden }: NetworkSpec): number {
  let n = 0;
  let inputs = 784;
  for (const { units } of [...hidden, { units: 10 }]) {
    n += inputs * units + units;
    inputs = units;
  }
  return n;
}

export function toMnistSessionConfig(
  { network, training, seed }: MnistConfig,
  data: MnistSubset,
  resume?: MnistCheckpoint | null,
): MnistSessionConfig {
  return { task: 'mnist', network, seed, data, ...training, ...(resume && { resume }) };
}

export function sameMnistConfig(a: MnistConfig, b: MnistConfig): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

const OPTIMISERS = Object.keys(OPTIMISER_NAMES) as Hyperparams['optimiser'][];

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

const oneOf = (options: readonly unknown[], v: unknown) => options.includes(v);

/**
 * Checks MNIST settings from a file against the choices the controls offer.
 * Returns the settings, or the first problem in words.
 */
export function parseMnistConfig(raw: unknown): { config: MnistConfig } | { error: string } {
  if (!isRecord(raw)) return { error: 'the settings are missing' };
  const { network, training, seed } = raw;
  if (!isRecord(network) || !Array.isArray(network.hidden)) {
    return { error: 'the network is missing' };
  }
  const hidden = network.hidden as unknown[];
  if (hidden.length > MNIST_MAX_HIDDEN_LAYERS) {
    return { error: `${hidden.length} hidden layers (at most ${MNIST_MAX_HIDDEN_LAYERS})` };
  }
  for (const l of hidden) {
    if (!isRecord(l) || !oneOf(MNIST_UNITS, l.units) || !oneOf(ACTIVATIONS, l.activation)) {
      return {
        error:
          `hidden layer ${JSON.stringify(l)} is not one the controls offer ` +
          `(${MNIST_UNITS.join(', ')} units; ${ACTIVATIONS.join(', ')})`,
      };
    }
  }
  if (!isRecord(training)) return { error: 'the training settings are missing' };
  const checks: [string, unknown, readonly unknown[]][] = [
    ['learning rate', training.lr, LEARNING_RATES],
    ['batch size', training.batchSize, MNIST_BATCH_SIZES],
    ['optimiser', training.optimiser, OPTIMISERS],
    ['L2 regularisation', training.l2, L2_STRENGTHS],
    ['dropout', training.dropout, DROPOUT_RATES],
  ];
  for (const [label, v, options] of checks) {
    if (!oneOf(options, v)) {
      return { error: `${label} ${JSON.stringify(v)} (choices: ${options.join(', ')})` };
    }
  }
  if (typeof seed !== 'number' || !Number.isInteger(seed) || seed < 0 || seed > MAX_SEED) {
    return { error: `weight seed ${JSON.stringify(seed)} (a whole number from 0 to ${MAX_SEED})` };
  }
  return {
    config: {
      network: { hidden: hidden as HiddenLayerSpec[] },
      training: {
        lr: training.lr as number,
        batchSize: training.batchSize as number,
        optimiser: training.optimiser as Hyperparams['optimiser'],
        l2: training.l2 as number,
        dropout: training.dropout as number,
      },
      seed,
    },
  };
}
