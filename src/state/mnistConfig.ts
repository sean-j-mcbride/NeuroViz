import type { MnistSubset } from '../data';
import { type LayerConfig, type Padding, inferShapes, layerCost } from '../engine';
import {
  ACTIVATIONS,
  type ConvLayerSpec,
  type HiddenLayerSpec,
  type Hyperparams,
  MNIST_IMAGE_SHAPE,
  type MnistCheckpoint,
  type MnistNetworkSpec,
  type MnistSessionConfig,
  checkConvStack,
  mnistNetworkToLayerConfig,
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
  network: MnistNetworkSpec;
  training: Hyperparams;
  /** Seeds weight initialisation, shuffling and dropout. */
  seed: number;
}

/** 784 → 128 ReLU → 10 with Adam: ≥ 95 % test accuracy within 6 epochs (see mnistSession.test.ts). */
export const MNIST_DEFAULT_CONFIG: MnistConfig = {
  network: { conv: [], hidden: [{ units: 128, activation: 'relu' }] },
  training: { lr: 0.001, batchSize: 64, optimiser: 'adam', l2: 0, dropout: 0 },
  seed: 1,
};

// The choices the MNIST controls offer. Loaded files are held to them.
export const MNIST_MAX_HIDDEN_LAYERS = 3;
export const MNIST_UNITS = [16, 32, 64, 128, 256] as const;
export const MNIST_BATCH_SIZES = [16, 32, 64, 128, 256] as const;
/** Units for a newly added layer. */
export const MNIST_NEW_LAYER: HiddenLayerSpec = { units: 64, activation: 'relu' };

// CNN rows. Pools are size × size with stride = size.
export const MNIST_MAX_CONV_LAYERS = 6;
export const CONV_FILTERS = [4, 8, 16, 32] as const;
export const CONV_KERNELS = [3, 5] as const;
export const CONV_STRIDES = [1, 2] as const;
export const CONV_PADDINGS: readonly Padding[] = ['valid', 'same'];
export const POOL_SIZES = [2] as const;
/** Rows added by "+ Conv" and "+ Pool". */
export const MNIST_NEW_CONV: ConvLayerSpec = {
  kind: 'conv',
  filters: 8,
  kernel: 3,
  stride: 1,
  padding: 'same',
  activation: 'relu',
};
export const MNIST_NEW_POOL: ConvLayerSpec = { kind: 'pool', size: 2 };

/** "Start from" networks in the builder. */
export const MNIST_TEMPLATES: Record<'mlp' | 'cnn', MnistNetworkSpec> = {
  mlp: MNIST_DEFAULT_CONFIG.network,
  cnn: {
    conv: [
      { kind: 'conv', filters: 8, kernel: 5, stride: 1, padding: 'valid', activation: 'relu' },
      { kind: 'pool', size: 2 },
      { kind: 'conv', filters: 16, kernel: 5, stride: 1, padding: 'valid', activation: 'relu' },
      { kind: 'pool', size: 2 },
    ],
    hidden: [],
  },
};

/** One conv row in words, e.g. "conv 8 × 5×5 ReLU" or "pool 2×2". */
export function describeConvRow(row: ConvLayerSpec): string {
  if (row.kind === 'pool') return `pool ${row.size}×${row.size}`;
  const extras = [
    row.stride !== 1 && `stride ${row.stride}`,
    row.padding === 'same' && 'same',
  ].filter(Boolean);
  return [
    `conv ${row.filters} × ${row.kernel}×${row.kernel}`,
    ...extras,
    ACTIVATION_NAMES[row.activation],
  ].join(' ');
}

/** "784 → 128 ReLU → 10"-style summary; a CNN starts "28×28 → conv 8 × 5×5 ReLU → pool 2×2 → …". */
export function describeMnistNetwork({ conv, hidden }: MnistNetworkSpec): string {
  return [
    conv.length > 0 ? '28×28' : '784',
    ...conv.map(describeConvRow),
    ...hidden.map((l) => `${l.units} ${ACTIVATION_NAMES[l.activation]}`),
    '10',
  ].join(' → ');
}

/** Why the network can't be built ("Conv layer 5: a 5 × 5 kernel doesn’t fit a 4 × 4 input"), or null. */
export function mnistNetworkError({ conv }: MnistNetworkSpec): string | null {
  const check = checkConvStack(conv);
  return check.ok ? null : `Conv layer ${check.row + 1}: ${check.message}`;
}

/**
 * Weights and biases in the network, and forward multiply-adds per image (a
 * rough guide to speed). null if the conv rows don't fit the image.
 */
export function mnistNetworkCost(
  network: MnistNetworkSpec,
): { params: number; multiplyAdds: number } | null {
  if (!checkConvStack(network.conv).ok) return null;
  const model = mnistNetworkToLayerConfig(network);
  const layers: LayerConfig[] = model.kind === 'sequential' ? model.layers : [model];
  const r = inferShapes(layers, network.conv.length > 0 ? MNIST_IMAGE_SHAPE : [784]);
  if ('error' in r) return null;
  let params = 0;
  let multiplyAdds = 0;
  layers.forEach((layer, i) => {
    const c = layerCost(layer, r.shapes[i]!);
    params += c.params;
    multiplyAdds += c.multiplyAdds;
  });
  return { params, multiplyAdds };
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

/** A conv/pool row from a file, if it is one the controls offer. */
function parseConvRow(v: unknown): ConvLayerSpec | null {
  if (!isRecord(v)) return null;
  if (v.kind === 'pool') {
    return oneOf(POOL_SIZES, v.size) ? { kind: 'pool', size: v.size as number } : null;
  }
  if (v.kind !== 'conv') return null;
  const { filters, kernel, stride, padding, activation } = v;
  if (
    !oneOf(CONV_FILTERS, filters) ||
    !oneOf(CONV_KERNELS, kernel) ||
    !oneOf(CONV_STRIDES, stride) ||
    !oneOf(CONV_PADDINGS, padding) ||
    !oneOf(ACTIVATIONS, activation)
  ) {
    return null;
  }
  return {
    kind: 'conv',
    filters: filters as number,
    kernel: kernel as number,
    stride: stride as number,
    padding: padding as Padding,
    activation: activation as HiddenLayerSpec['activation'],
  };
}

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
  const rawConv = network.conv ?? [];
  if (!Array.isArray(rawConv)) return { error: 'the conv layers must be a list' };
  if (rawConv.length > MNIST_MAX_CONV_LAYERS) {
    return { error: `${rawConv.length} conv and pool layers (at most ${MNIST_MAX_CONV_LAYERS})` };
  }
  const conv: ConvLayerSpec[] = [];
  for (const row of rawConv as unknown[]) {
    const parsed = parseConvRow(row);
    if (!parsed) {
      return {
        error:
          `conv layer ${JSON.stringify(row)} is not one the controls offer ` +
          `(${CONV_FILTERS.join(', ')} filters; ${CONV_KERNELS.map((k) => `${k}×${k}`).join(' or ')} ` +
          `kernels; stride ${CONV_STRIDES.join(' or ')}; ${CONV_PADDINGS.join(' or ')} padding; ` +
          `or a ${POOL_SIZES.map((p) => `${p}×${p}`).join(' or ')} pool)`,
      };
    }
    conv.push(parsed);
  }
  const stack = checkConvStack(conv);
  if (!stack.ok) return { error: `conv layer ${stack.row + 1}: ${stack.message}` };
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
      network: { conv, hidden: hidden as HiddenLayerSpec[] },
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
