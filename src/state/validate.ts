import { TOY2D_KINDS } from '../data';
import { ACTIVATIONS, type HiddenLayerSpec, type Hyperparams } from '../worker';
import {
  BATCH_SIZES,
  DEFAULT_CONFIG,
  DROPOUT_RATES,
  L2_STRENGTHS,
  LEARNING_RATES,
  MAX_HIDDEN_LAYERS,
  MAX_SEED,
  MAX_UNITS,
  NOISE,
  POINTS,
  type PlaygroundConfig,
  describeNetwork,
} from './config';

export interface ParsedConfig {
  /** Always complete and within the controls' ranges. */
  config: PlaygroundConfig;
  /** One line per value that was missing or invalid and replaced by its default. */
  warnings: string[];
}

export interface ParseOptions {
  /** Missing values silently take their defaults (for links); otherwise each is a warning. */
  partial?: boolean;
}

const OPTIMISERS: readonly Hyperparams['optimiser'][] = ['sgd', 'momentum', 'adam'];

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function show(v: unknown): string {
  const s = typeof v === 'string' ? `“${v}”` : JSON.stringify(v);
  return s.length > 40 ? `${s.slice(0, 37)}…` : s;
}

const isInt = (v: unknown, lo: number, hi: number): v is number =>
  typeof v === 'number' && Number.isInteger(v) && v >= lo && v <= hi;
const inRange = (v: unknown, lo: number, hi: number): v is number =>
  typeof v === 'number' && Number.isFinite(v) && v >= lo && v <= hi;
const oneOf =
  <T>(options: readonly T[]) =>
  (v: unknown): v is T =>
    options.includes(v as T);

/** Layer specs as they were given, e.g. “99 tanh, 2 gelu”. */
function showLayers(v: unknown): string {
  if (!Array.isArray(v)) return show(v);
  const text = v
    .map((l) => (isRecord(l) ? `${String(l.units)} ${String(l.activation)}` : JSON.stringify(l)))
    .join(', ');
  return show(text);
}

function isHiddenLayers(v: unknown): v is HiddenLayerSpec[] {
  return (
    Array.isArray(v) &&
    v.length <= MAX_HIDDEN_LAYERS &&
    v.every(
      (l) =>
        isRecord(l) &&
        isInt(l.units, 1, MAX_UNITS) &&
        ACTIVATIONS.includes(l.activation as HiddenLayerSpec['activation']),
    )
  );
}

/**
 * Checks a config from outside the app (a saved file or a link) against the
 * ranges and choices the controls offer. Anything missing or invalid falls
 * back to the default for that value, with a warning.
 */
export function parseConfig(raw: unknown, { partial = false }: ParseOptions = {}): ParsedConfig {
  const warnings: string[] = [];
  const root = isRecord(raw) ? raw : {};
  if (!isRecord(raw)) warnings.push('Settings are missing; using the defaults');
  const section = (key: string) => (isRecord(root[key]) ? root[key] : {});

  /**
   * The value at `obj[key]` if `valid`, else `fallback` with a warning that
   * names what is `allowed`.
   */
  function pick<T>(
    obj: Record<string, unknown>,
    key: string,
    label: string,
    valid: (v: unknown) => v is T,
    fallback: T,
    allowed: string,
    describe: (v: T) => string = show,
    showBad: (v: unknown) => string = show,
  ): T {
    const v = obj[key];
    if (valid(v)) return v;
    if (v === undefined) {
      if (!partial) warnings.push(`${label} is missing; using ${describe(fallback)}`);
    } else {
      warnings.push(
        `Ignored ${label.toLowerCase()} ${showBad(v)} (${allowed}); using ${describe(fallback)}`,
      );
    }
    return fallback;
  }
  const choices = (options: readonly unknown[]) => `choices: ${options.join(', ')}`;
  const between = (lo: number, hi: number) => `allowed: ${lo}–${hi}`;

  const d = section('dataset');
  const n = section('network');
  const t = section('training');
  const D = DEFAULT_CONFIG;
  const [P, N] = [POINTS, NOISE];
  const config: PlaygroundConfig = {
    dataset: {
      kind: pick(d, 'kind', 'Dataset', oneOf(TOY2D_KINDS), D.dataset.kind, choices(TOY2D_KINDS)),
      n: pick(
        d,
        'n',
        'Number of points',
        (v) => isInt(v, P.min, P.max),
        D.dataset.n,
        between(P.min, P.max),
      ),
      noise: pick(
        d,
        'noise',
        'Noise',
        (v) => inRange(v, N.min, N.max),
        D.dataset.noise,
        between(N.min, N.max),
      ),
      seed: pick(
        d,
        'seed',
        'Data seed',
        (v) => isInt(v, 0, MAX_SEED),
        D.dataset.seed,
        `a whole number from 0 to ${MAX_SEED}`,
      ),
    },
    network: {
      hidden: pick(
        n,
        'hidden',
        'Hidden layers',
        isHiddenLayers,
        D.network.hidden,
        `up to ${MAX_HIDDEN_LAYERS} layers of 1–${MAX_UNITS} neurons, each ${ACTIVATIONS.join(', ')}`,
        (hidden) => describeNetwork({ hidden }),
        showLayers,
      ),
    },
    training: {
      lr: pick(
        t,
        'lr',
        'Learning rate',
        oneOf<number>(LEARNING_RATES),
        D.training.lr,
        choices(LEARNING_RATES),
      ),
      batchSize: pick(
        t,
        'batchSize',
        'Batch size',
        oneOf(BATCH_SIZES),
        D.training.batchSize,
        choices(BATCH_SIZES),
      ),
      optimiser: pick(
        t,
        'optimiser',
        'Optimiser',
        oneOf(OPTIMISERS),
        D.training.optimiser,
        choices(OPTIMISERS),
      ),
      l2: pick(
        t,
        'l2',
        'L2 regularisation',
        oneOf<number>(L2_STRENGTHS),
        D.training.l2,
        choices(L2_STRENGTHS),
      ),
      dropout: pick(
        t,
        'dropout',
        'Dropout',
        oneOf<number>(DROPOUT_RATES),
        D.training.dropout,
        choices(DROPOUT_RATES),
      ),
    },
    seed: pick(
      root,
      'seed',
      'Weight seed',
      (v) => isInt(v, 0, MAX_SEED),
      D.seed,
      `a whole number from 0 to ${MAX_SEED}`,
    ),
  };
  return { config, warnings };
}
