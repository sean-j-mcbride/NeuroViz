import {
  DATASET_NAMES,
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
