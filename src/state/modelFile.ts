import type { OptimiserKind, OptimiserState, RngState } from '../engine';
import {
  type Checkpoint,
  type HyperparamChange,
  type LossCurveState,
  LossHistory,
  type LossHistoryState,
  type ParamHistoryState,
  type TimelineState,
  TrainingSession,
} from '../worker';
import {
  type UintWidth,
  base64ToFloat32,
  base64ToUints,
  float32ToBase64,
  uintWidthFor,
  uintsToBase64,
} from './base64';
import { DEFAULT_CONFIG, type PlaygroundConfig, toSessionConfig } from './config';
import { parseConfig } from './validate';

/*
 * A saved model: the settings plus a checkpoint, as human-readable JSON.
 * Every number is written as the shortest decimal that reads back to the same
 * float32, so a save → load round-trip is exact and the file stays legible.
 * Non-finite values (a diverged loss) are written as the strings "NaN",
 * "Infinity" and "-Infinity".
 *
 * The exception is `checkpoint.history`: the charts' history (histogram
 * timeline and hover sparklines), which is display-only and bulky, so it is
 * stored compactly as base64 (histograms as whole counts).
 */

export const MODEL_FILE_FORMAT = 'neuroviz-model';
/** Saved MNIST runs (see mnistModelFile.ts); named here so each loader can redirect the other's files. */
export const MNIST_MODEL_FILE_FORMAT = 'neuroviz-mnist-model';
/**
 * 2 added the settings log (`checkpoint.hyperparamLog`), the charts' history
 * (`checkpoint.history`) and bounded loss curves (`checkpoint.losses`, which
 * replaces the per-epoch `trainLoss` / `testLoss`). Version 1 files still load:
 * as if the settings never changed during the run, with the charts restarting.
 */
export const MODEL_FILE_VERSION = 2;

export interface ModelFile {
  config: PlaygroundConfig;
  checkpoint: Checkpoint;
  /** ISO timestamp, if the file has one. */
  savedAt: string | null;
}

/** A file that can't be loaded; the message says why, for the user. */
export class ModelFileError extends Error {
  override name = 'ModelFileError';
}

/** The shortest decimal that reads back as exactly `v` in float32 (as JSON). */
export function formatFloat32(v: number): string {
  if (Number.isNaN(v)) return '"NaN"';
  if (v === Infinity) return '"Infinity"';
  if (v === -Infinity) return '"-Infinity"';
  if (Object.is(v, -0)) return '-0';
  for (let p = 1; p < 9; p++) {
    const d = Number(v.toPrecision(p));
    if (Math.fround(d) === v) return String(d);
  }
  return String(v);
}

export const formatArray = (a: Float32Array | Uint32Array): string =>
  `[${Array.from(a, a instanceof Float32Array ? formatFloat32 : String).join(', ')}]`;

export const mapValues = <A, B>(
  o: Record<string, A>,
  f: (a: A, key: string) => B,
): Record<string, B> => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, f(v, k)]));

const HISTORY_NOTE =
  'Display only: the histogram timeline and hover sparklines, so the charts carry on after ' +
  'loading. Histograms are whole counts per bin (count / size is the share); all binary ' +
  'values are little-endian base64.';

/** Each histogram bin as a whole count, written at the narrowest width that holds `size`. */
function encodeCounts(hist: Float32Array, size: number, width: UintWidth): string {
  return uintsToBase64(
    Array.from(hist, (f) => Math.round(f * size)),
    width,
  );
}

function historyBlock(
  { timeline, params }: NonNullable<Checkpoint['observations']>,
  raw: (a: Float32Array) => string,
) {
  return {
    note: HISTORY_NOTE,
    timeline: {
      interval: timeline.interval,
      epochs: raw(timeline.epochs),
      layers: timeline.layers.map((l) => {
        const bytes = uintWidthFor(l.size);
        return {
          size: l.size,
          bytes,
          weightCounts: encodeCounts(l.weightHist, l.size, bytes),
          gradCounts: encodeCounts(l.gradHist, l.size, bytes),
          weightRms: float32ToBase64(l.weightRms),
          gradRms: float32ToBase64(l.gradRms),
        };
      }),
    },
    params: {
      head: params.head,
      length: params.length,
      epochs: raw(params.epochs),
      values: float32ToBase64(params.ring),
    },
  };
}

/** A float64 that JSON can hold exactly; non-finite values as strings. */
const exactNumber = (v: number): number | string => (Number.isFinite(v) ? v : String(v));

function lossCurveBlock(c: LossCurveState, merged: boolean, raw: (a: Float32Array) => string) {
  return {
    // Until buckets merge, min = max = mean, so only the means are written.
    mean: raw(c.mean),
    ...(merged && { min: raw(c.min), max: raw(c.max) }),
    partial: mapValues(c.partial, exactNumber),
    latest: exactNumber(c.latest),
    best: exactNumber(c.best),
    bestEpoch: c.bestEpoch,
  };
}

export function lossesBlock(l: LossHistoryState, raw: (a: Float32Array) => string) {
  const merged = l.width > 1;
  return {
    width: l.width,
    count: l.count,
    train: lossCurveBlock(l.train, merged, raw),
    test: lossCurveBlock(l.test, merged, raw),
  };
}

export function serialiseModelFile(
  config: PlaygroundConfig,
  c: Checkpoint,
  savedAt: string = new Date().toISOString(),
): string {
  // Typed arrays and log entries go in as placeholders, then are spliced in on one line each.
  const arrays: string[] = [];
  const raw = (a: Float32Array | Uint32Array) => `@@${arrays.push(formatArray(a)) - 1}@@`;
  const oneLine = (v: unknown) => `@@${arrays.push(JSON.stringify(v).replace(/,"/g, ', "')) - 1}@@`;
  const doc = {
    format: MODEL_FILE_FORMAT,
    version: MODEL_FILE_VERSION,
    savedAt,
    config,
    checkpoint: {
      epoch: c.epoch,
      step: c.step,
      losses: lossesBlock(c.losses, raw),
      params: mapValues(c.params, raw),
      optimiser: {
        kind: c.optimiser.kind,
        t: c.optimiser.t,
        slots: mapValues(c.optimiser.slots, (list) => list.map(raw)),
      },
      rng: c.rng,
      order: raw(c.order),
      hyperparamLog: c.hyperparamLog.map(oneLine),
      history: c.observations && historyBlock(c.observations, raw),
    },
  };
  return `${JSON.stringify(doc, null, 2).replace(/"@@(\d+)@@"/g, (_, i: string) => arrays[Number(i)]!)}\n`;
}

// ── Reading ─────────────────────────────────────────────────────────────────

export const fail = (message: string): never => {
  throw new ModelFileError(message);
};

export function record(v: unknown, path: string): Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : fail(`${path} is missing or not an object`);
}

export function int(v: unknown, path: string, max = Number.MAX_SAFE_INTEGER): number {
  return Number.isInteger(v) && (v as number) >= 0 && (v as number) <= max
    ? (v as number)
    : fail(`${path} must be a whole number from 0 to ${max}`);
}

const NON_FINITE: Record<string, number> = {
  NaN: Number.NaN,
  Infinity: Infinity,
  '-Infinity': -Infinity,
};

function floats(v: unknown, path: string): Float32Array {
  if (!Array.isArray(v)) return fail(`${path} must be a list of numbers`);
  const out = new Float32Array(v.length);
  v.forEach((x: unknown, i) => {
    if (typeof x === 'number') out[i] = x;
    else if (typeof x === 'string' && x in NON_FINITE) out[i] = NON_FINITE[x]!;
    else fail(`${path}[${i}] is not a number`);
  });
  return out;
}

function uints(v: unknown, path: string): Uint32Array {
  if (!Array.isArray(v)) return fail(`${path} must be a list of whole numbers`);
  return Uint32Array.from(v, (x: unknown, i) => int(x, `${path}[${i}]`, 2 ** 32 - 1));
}

export function rngState(v: unknown, path: string): RngState {
  const r = record(v, path);
  const spare = r.spare;
  if (spare !== null && !(typeof spare === 'number' && Number.isFinite(spare))) {
    fail(`${path}.spare must be a number or null`);
  }
  return { state: int(r.state, `${path}.state`, 2 ** 32 - 1), spare: spare as number | null };
}

export const OPTIMISERS: readonly OptimiserKind[] = ['sgd', 'momentum', 'adam'];

function optimiserState(v: unknown, path: string): OptimiserState {
  const o = record(v, path);
  const kind = OPTIMISERS.includes(o.kind as OptimiserKind)
    ? (o.kind as OptimiserKind)
    : fail(`${path}.kind must be one of ${OPTIMISERS.join(', ')}`);
  return {
    kind,
    t: int(o.t, `${path}.t`),
    slots: mapValues(record(o.slots, `${path}.slots`), (list, name) =>
      Array.isArray(list)
        ? list.map((a, k) => floats(a, `${path}.slots["${name}"][${k}]`))
        : fail(`${path}.slots["${name}"] must be a list`),
    ),
  };
}

/** One logged setting change; the settings must be ones the controls offer. */
function hyperparamChange(v: unknown, path: string): HyperparamChange {
  const e = record(v, path);
  const { config, warnings } = parseConfig({
    ...DEFAULT_CONFIG,
    training: record(e.hyperparams, `${path}.hyperparams`),
  });
  if (warnings.length > 0) fail(`${path} is invalid: ${warnings[0]}`);
  return { epoch: int(e.epoch, `${path}.epoch`), hyperparams: config.training };
}

export function decoded<T>(decode: () => T, path: string): T {
  try {
    return decode();
  } catch (e) {
    return fail(`${path} is not valid: ${e instanceof Error ? e.message : e}`);
  }
}

export function str(v: unknown, path: string): string {
  return typeof v === 'string' ? v : fail(`${path} must be text`);
}

/** Shares from whole counts, as `histogramInto` computes them. */
function sharesFrom(counts: Uint32Array, size: number, path: string): Float32Array {
  const out = new Float32Array(counts.length);
  counts.forEach((c, i) => {
    if (c > size) fail(`${path} has a count above the layer size`);
    out[i] = c / size;
  });
  return out;
}

function history(v: unknown): Checkpoint['observations'] {
  if (v === null) return null;
  const h = record(v, 'checkpoint.history');
  const t = record(h.timeline, 'checkpoint.history.timeline');
  if (!Array.isArray(t.layers)) fail('checkpoint.history.timeline.layers must be a list');
  const timeline: TimelineState = {
    interval: int(t.interval, 'checkpoint.history.timeline.interval'),
    epochs: floats(t.epochs, 'checkpoint.history.timeline.epochs'),
    layers: (t.layers as unknown[]).map((raw, k) => {
      const path = `checkpoint.history.timeline.layers[${k}]`;
      const l = record(raw, path);
      const size = int(l.size, `${path}.size`);
      const bytes =
        l.bytes === 1 || l.bytes === 2 || l.bytes === 4
          ? l.bytes
          : fail(`${path}.bytes must be 1, 2 or 4`);
      const counts = (key: string) =>
        sharesFrom(
          decoded(() => base64ToUints(str(l[key], `${path}.${key}`), bytes), `${path}.${key}`),
          size,
          `${path}.${key}`,
        );
      const f32 = (key: string) =>
        decoded(() => base64ToFloat32(str(l[key], `${path}.${key}`)), `${path}.${key}`);
      return {
        size,
        weightHist: counts('weightCounts'),
        gradHist: counts('gradCounts'),
        weightRms: f32('weightRms'),
        gradRms: f32('gradRms'),
      };
    }),
  };
  const p = record(h.params, 'checkpoint.history.params');
  const params: ParamHistoryState = {
    head: int(p.head, 'checkpoint.history.params.head'),
    length: int(p.length, 'checkpoint.history.params.length'),
    epochs: floats(p.epochs, 'checkpoint.history.params.epochs'),
    ring: decoded(
      () => base64ToFloat32(str(p.values, 'checkpoint.history.params.values')),
      'checkpoint.history.params.values',
    ),
  };
  return { timeline, params };
}

function exact(v: unknown, path: string): number {
  if (typeof v === 'number') return v;
  if (typeof v === 'string' && v in NON_FINITE) return NON_FINITE[v]!;
  return fail(`${path} is not a number`);
}

function lossCurve(v: unknown, width: number, path: string): LossCurveState {
  const c = record(v, path);
  const mean = floats(c.mean, `${path}.mean`);
  const partial = record(c.partial, `${path}.partial`);
  const bestEpoch = c.bestEpoch === -1 ? -1 : int(c.bestEpoch, `${path}.bestEpoch`);
  return {
    mean,
    min: width > 1 ? floats(c.min, `${path}.min`) : mean.slice(),
    max: width > 1 ? floats(c.max, `${path}.max`) : mean.slice(),
    partial: {
      sum: exact(partial.sum, `${path}.partial.sum`),
      n: int(partial.n, `${path}.partial.n`),
      min: exact(partial.min, `${path}.partial.min`),
      max: exact(partial.max, `${path}.partial.max`),
    },
    latest: exact(c.latest, `${path}.latest`),
    best: exact(c.best, `${path}.best`),
    bestEpoch,
  };
}

function losses(c: Record<string, unknown>, version: number): LossHistoryState {
  if (version < 2) {
    // Version 1 kept every epoch: replay them.
    const train = floats(c.trainLoss, 'checkpoint.trainLoss');
    const test = floats(c.testLoss, 'checkpoint.testLoss');
    if (train.length !== test.length) fail('checkpoint.trainLoss and testLoss differ in length');
    return LossHistory.fromSeries(train, test).exportState();
  }
  return lossHistory(c.losses, 'checkpoint.losses');
}

/** A loss history as `lossesBlock` writes it. */
export function lossHistory(v: unknown, path: string): LossHistoryState {
  const l = record(v, path);
  const width = int(l.width, `${path}.width`);
  return {
    width,
    count: int(l.count, `${path}.count`),
    train: lossCurve(l.train, width, `${path}.train`),
    test: lossCurve(l.test, width, `${path}.test`),
  };
}

function checkpoint(v: unknown, version: number, config: PlaygroundConfig): Checkpoint {
  const c = record(v, 'checkpoint');
  const log = c.hyperparamLog;
  if (version >= 2 && !Array.isArray(log)) fail('checkpoint.hyperparamLog must be a list');
  const params = record(c.params, 'checkpoint.params');
  const rng = record(c.rng, 'checkpoint.rng');
  return {
    epoch: int(c.epoch, 'checkpoint.epoch'),
    step: int(c.step, 'checkpoint.step'),
    losses: losses(c, version),
    params: mapValues(params, (p, name) => floats(p, `checkpoint.params["${name}"]`)),
    optimiser: optimiserState(c.optimiser, 'checkpoint.optimiser'),
    rng: {
      shuffle: rngState(rng.shuffle, 'checkpoint.rng.shuffle'),
      dropout: rngState(rng.dropout, 'checkpoint.rng.dropout'),
    },
    order: uints(c.order, 'checkpoint.order'),
    hyperparamLog:
      version >= 2
        ? (log as unknown[]).map((e, i) => hyperparamChange(e, `checkpoint.hyperparamLog[${i}]`))
        : [{ epoch: 0, hyperparams: config.training }],
    observations: version >= 2 ? history(c.history) : null,
  };
}

/**
 * Reads a saved model, checking the format, the settings (which must be ones
 * the controls offer) and that the checkpoint really resumes with those
 * settings (by building the session it describes). Throws `ModelFileError`.
 */
export function parseModelFile(text: string): ModelFile {
  let doc: unknown;
  try {
    doc = JSON.parse(text);
  } catch {
    fail('This file is not valid JSON, so it can’t be a saved NeuroViz model.');
  }
  const d = record(doc, 'The file');
  if (d.format === MNIST_MODEL_FILE_FORMAT) {
    fail('This is a saved MNIST model: load it from the MNIST tab.');
  }
  if (d.format !== MODEL_FILE_FORMAT) fail('This is not a saved NeuroViz model.');
  if (typeof d.version === 'number' && d.version > MODEL_FILE_VERSION) {
    fail(
      `This model was saved by a newer version of NeuroViz (format ${d.version}); ` +
        `this version reads format ${MODEL_FILE_VERSION}.`,
    );
  }
  if (d.version !== 1 && d.version !== 2) fail('The file’s format version is missing or invalid.');

  const { config, warnings } = parseConfig(d.config);
  if (warnings.length > 0) fail(`The saved settings are invalid: ${warnings[0]}.`);

  const c = checkpoint(d.checkpoint, d.version as number, config);
  try {
    new TrainingSession({ ...toSessionConfig(config, c), gridSize: 1 });
  } catch (e) {
    fail(`The saved model doesn’t match its settings: ${e instanceof Error ? e.message : e}.`);
  }
  return { config, checkpoint: c, savedAt: typeof d.savedAt === 'string' ? d.savedAt : null };
}
