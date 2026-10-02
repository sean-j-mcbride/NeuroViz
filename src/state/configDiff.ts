import type { HyperparamChange, Hyperparams } from '../worker';
import {
  DATASET_NAMES,
  DEFAULT_CONFIG,
  OPTIMISER_NAMES,
  type PlaygroundConfig,
  describeBatchSize,
  describeNetwork,
} from './config';

export interface ConfigChange {
  label: string;
  from: string;
  to: string;
}

const pct = (v: number) => `${Math.round(v * 100)} %`;
const orNone = (v: number, show: (v: number) => string = String) => (v === 0 ? 'None' : show(v));

/** Every setting, as a label and a readable value, in the order the controls show them. */
const FIELDS: [string, (c: PlaygroundConfig) => string][] = [
  ['Dataset', (c) => DATASET_NAMES[c.dataset.kind]],
  ['Points', (c) => String(c.dataset.n)],
  ['Noise', (c) => pct(c.dataset.noise)],
  ['Data seed', (c) => String(c.dataset.seed)],
  ['Hidden layers', (c) => describeNetwork(c.network)],
  ['Learning rate', (c) => String(c.training.lr)],
  ['Batch size', (c) => describeBatchSize(c.training.batchSize)],
  ['Optimiser', (c) => OPTIMISER_NAMES[c.training.optimiser]],
  ['L2 regularisation', (c) => orNone(c.training.l2)],
  ['Dropout', (c) => orNone(c.training.dropout, pct)],
  ['Weight seed', (c) => String(c.seed)],
];

/** The settings that differ between two runs, e.g. { label: 'Learning rate', from: '0.03', to: '3' }. */
export function diffConfigs(a: PlaygroundConfig, b: PlaygroundConfig): ConfigChange[] {
  return FIELDS.flatMap(([label, show]) => {
    const [from, to] = [show(a), show(b)];
    return from === to ? [] : [{ label, from, to }];
  });
}

/** The hyperparameters that differ between two settings. */
export function diffHyperparams(a: Hyperparams, b: Hyperparams): ConfigChange[] {
  return diffConfigs({ ...DEFAULT_CONFIG, training: a }, { ...DEFAULT_CONFIG, training: b });
}

/** A mid-run change of settings: the epoch it applied from and what changed. */
export interface RunChange {
  epoch: number;
  changes: ConfigChange[];
  /** e.g. "Learning rate 3 → 0.03; Optimiser SGD → Adam". */
  text: string;
}

/** Each change in a settings log after the starting settings. */
export function changesDuringRun(log: readonly HyperparamChange[]): RunChange[] {
  return log.slice(1).map((entry, i) => {
    const changes = diffHyperparams(log[i]!.hyperparams, entry.hyperparams);
    const text = changes.map((c) => `${c.label} ${c.from} → ${c.to}`).join('; ');
    return { epoch: entry.epoch, changes, text };
  });
}
