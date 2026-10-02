import type { OptimiserKind, OptimiserState, RngState } from '../engine';
import { type Checkpoint, type HyperparamChange, TrainingSession } from '../worker';
import { DEFAULT_CONFIG, type PlaygroundConfig, toSessionConfig } from './config';
import { parseConfig } from './validate';

/*
 * A saved model: the settings plus a checkpoint, as human-readable JSON.
 * Every number is written as the shortest decimal that reads back to the same
 * float32, so a save → load round-trip is exact and the file stays legible.
 * Non-finite values (a diverged loss) are written as the strings "NaN",
 * "Infinity" and "-Infinity".
 */

export const MODEL_FILE_FORMAT = 'neuroviz-model';
/**
 * 2 added the settings log (`checkpoint.hyperparamLog`). Version 1 files still
 * load, as if the settings never changed during the run.
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

const formatArray = (a: Float32Array | Uint32Array): string =>
  `[${Array.from(a, a instanceof Float32Array ? formatFloat32 : String).join(', ')}]`;

const mapValues = <A, B>(o: Record<string, A>, f: (a: A, key: string) => B): Record<string, B> =>
  Object.fromEntries(Object.entries(o).map(([k, v]) => [k, f(v, k)]));

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
      trainLoss: raw(c.trainLoss),
      testLoss: raw(c.testLoss),
      params: mapValues(c.params, raw),
      optimiser: {
        kind: c.optimiser.kind,
        t: c.optimiser.t,
        slots: mapValues(c.optimiser.slots, (list) => list.map(raw)),
      },
      rng: c.rng,
      order: raw(c.order),
      hyperparamLog: c.hyperparamLog.map(oneLine),
    },
  };
  return `${JSON.stringify(doc, null, 2).replace(/"@@(\d+)@@"/g, (_, i: string) => arrays[Number(i)]!)}\n`;
}

// ── Reading ─────────────────────────────────────────────────────────────────

const fail = (message: string): never => {
  throw new ModelFileError(message);
};

function record(v: unknown, path: string): Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : fail(`${path} is missing or not an object`);
}

function int(v: unknown, path: string, max = Number.MAX_SAFE_INTEGER): number {
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

function rngState(v: unknown, path: string): RngState {
  const r = record(v, path);
  const spare = r.spare;
  if (spare !== null && !(typeof spare === 'number' && Number.isFinite(spare))) {
    fail(`${path}.spare must be a number or null`);
  }
  return { state: int(r.state, `${path}.state`, 2 ** 32 - 1), spare: spare as number | null };
}

const OPTIMISERS: readonly OptimiserKind[] = ['sgd', 'momentum', 'adam'];

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

function checkpoint(v: unknown, version: number, config: PlaygroundConfig): Checkpoint {
  const c = record(v, 'checkpoint');
  const log = c.hyperparamLog;
  if (version >= 2 && !Array.isArray(log)) fail('checkpoint.hyperparamLog must be a list');
  const params = record(c.params, 'checkpoint.params');
  const rng = record(c.rng, 'checkpoint.rng');
  return {
    epoch: int(c.epoch, 'checkpoint.epoch'),
    step: int(c.step, 'checkpoint.step'),
    trainLoss: floats(c.trainLoss, 'checkpoint.trainLoss'),
    testLoss: floats(c.testLoss, 'checkpoint.testLoss'),
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
