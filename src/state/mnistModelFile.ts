import { MNIST_SUBSET_ID, type MnistSubset } from '../data';
import type { OptimiserState } from '../engine';
import { type MnistCheckpoint, MnistSession } from '../worker';
import {
  base64ToFloat32,
  base64ToUints,
  float32ToBase64,
  uintWidthFor,
  uintsToBase64,
} from './base64';
import { type MnistConfig, parseMnistConfig, toMnistSessionConfig } from './mnistConfig';
import {
  MNIST_MODEL_FILE_FORMAT,
  MODEL_FILE_FORMAT,
  OPTIMISERS,
  decoded,
  fail,
  formatArray,
  int,
  lossHistory,
  lossesBlock,
  mapValues,
  record,
  rngState,
  str,
} from './modelFile';

/*
 * A saved MNIST run: readable settings plus a checkpoint. Unlike the
 * playground's files, the bulk (about 100,000 weights, the optimiser's state
 * and the training order) is little-endian base64; as decimals it would be
 * several megabytes. The loss and error curves stay readable.
 */

export { MNIST_MODEL_FILE_FORMAT };
/**
 * 2 (Phase 6): the network gained `conv` rows. Version 1 files (MLPs, no
 * `conv`) still load; settings without `conv` are read as an MLP.
 */
export const MNIST_MODEL_FILE_VERSION = 2;

export interface MnistModelFile {
  config: MnistConfig;
  checkpoint: MnistCheckpoint;
  savedAt: string | null;
}

const NOTE =
  'Weights, optimiser state, training order and test predictions are little-endian base64: ' +
  'float32, except order (uint, `orderBytes` wide) and predicted (uint8).';

export function serialiseMnistModelFile(
  config: MnistConfig,
  data: MnistSubset,
  c: MnistCheckpoint,
  savedAt: string = new Date().toISOString(),
): string {
  // Curves go in as placeholders, then are spliced in on one line each (as in the playground's files).
  const arrays: string[] = [];
  const raw = (a: Float32Array) => `@@${arrays.push(formatArray(a)) - 1}@@`;
  const orderBytes = uintWidthFor(c.order.length);
  const doc = {
    format: MNIST_MODEL_FILE_FORMAT,
    version: MNIST_MODEL_FILE_VERSION,
    savedAt,
    dataset: { id: MNIST_SUBSET_ID, checksum: data.checksum },
    config,
    checkpoint: {
      note: NOTE,
      examples: c.examples,
      step: c.step,
      cursor: c.cursor,
      losses: lossesBlock(c.losses, raw),
      errors: lossesBlock(c.errors, raw),
      rng: c.rng,
      params: mapValues(c.params, float32ToBase64),
      optimiser: {
        kind: c.optimiser.kind,
        t: c.optimiser.t,
        slots: mapValues(c.optimiser.slots, (list) => list.map(float32ToBase64)),
      },
      orderBytes,
      order: uintsToBase64(c.order, orderBytes),
      evaluation: {
        predicted: uintsToBase64(c.evaluation.predicted, 1),
        confidence: float32ToBase64(c.evaluation.confidence),
      },
    },
  };
  return `${JSON.stringify(doc, null, 2).replace(/"@@(\d+)@@"/g, (_, i: string) => arrays[Number(i)]!)}\n`;
}

const f32 = (v: unknown, path: string) => decoded(() => base64ToFloat32(str(v, path)), path);

function optimiserState(v: unknown, path: string): OptimiserState {
  const o = record(v, path);
  const kind = OPTIMISERS.includes(o.kind as OptimiserState['kind'])
    ? (o.kind as OptimiserState['kind'])
    : fail(`${path}.kind must be one of ${OPTIMISERS.join(', ')}`);
  return {
    kind,
    t: int(o.t, `${path}.t`),
    slots: mapValues(record(o.slots, `${path}.slots`), (list, name) =>
      Array.isArray(list)
        ? list.map((a, k) => f32(a, `${path}.slots["${name}"][${k}]`))
        : fail(`${path}.slots["${name}"] must be a list`),
    ),
  };
}

function checkpoint(v: unknown): MnistCheckpoint {
  const c = record(v, 'checkpoint');
  const rng = record(c.rng, 'checkpoint.rng');
  const e = record(c.evaluation, 'checkpoint.evaluation');
  const bytes = c.orderBytes;
  const orderBytes =
    bytes === 1 || bytes === 2 || bytes === 4
      ? bytes
      : fail('checkpoint.orderBytes must be 1, 2 or 4');
  return {
    task: 'mnist',
    examples: int(c.examples, 'checkpoint.examples'),
    step: int(c.step, 'checkpoint.step'),
    cursor: int(c.cursor, 'checkpoint.cursor'),
    order: decoded(
      () => base64ToUints(str(c.order, 'checkpoint.order'), orderBytes),
      'checkpoint.order',
    ),
    losses: lossHistory(c.losses, 'checkpoint.losses'),
    errors: lossHistory(c.errors, 'checkpoint.errors'),
    params: mapValues(record(c.params, 'checkpoint.params'), (p, name) =>
      f32(p, `checkpoint.params["${name}"]`),
    ),
    optimiser: optimiserState(c.optimiser, 'checkpoint.optimiser'),
    rng: {
      shuffle: rngState(rng.shuffle, 'checkpoint.rng.shuffle'),
      dropout: rngState(rng.dropout, 'checkpoint.rng.dropout'),
    },
    evaluation: {
      predicted: Uint8Array.from(
        decoded(
          () => base64ToUints(str(e.predicted, 'checkpoint.evaluation.predicted'), 1),
          'checkpoint.evaluation.predicted',
        ),
      ),
      confidence: f32(e.confidence, 'checkpoint.evaluation.confidence'),
    },
  };
}

/**
 * Reads a saved MNIST run, checking the format, that it was trained on this
 * data, that the settings are ones the controls offer, and that the
 * checkpoint really resumes with them (by building the session it describes,
 * which is quick: resuming skips the evaluation). Throws `ModelFileError`.
 */
export function parseMnistModelFile(text: string, data: MnistSubset): MnistModelFile {
  let doc: unknown;
  try {
    doc = JSON.parse(text);
  } catch {
    fail('This file is not valid JSON, so it can’t be a saved NeuroViz model.');
  }
  const d = record(doc, 'The file');
  if (d.format === MODEL_FILE_FORMAT) {
    fail('This is a saved playground model: load it from the Playground tab.');
  }
  if (d.format !== MNIST_MODEL_FILE_FORMAT) fail('This is not a saved NeuroViz MNIST model.');
  if (typeof d.version === 'number' && d.version > MNIST_MODEL_FILE_VERSION) {
    fail(
      `This model was saved by a newer version of NeuroViz (format ${d.version}); ` +
        `this version reads format ${MNIST_MODEL_FILE_VERSION}.`,
    );
  }
  if (d.version !== 1 && d.version !== MNIST_MODEL_FILE_VERSION)
    fail('The file’s format version is missing or invalid.');

  const ds = record(d.dataset, 'dataset');
  if (ds.id !== MNIST_SUBSET_ID || ds.checksum !== data.checksum) {
    fail(
      `This model was trained on different data (${String(ds.id)}), ` +
        `not the MNIST subset this version bundles (${MNIST_SUBSET_ID}).`,
    );
  }
  const parsed = parseMnistConfig(d.config);
  if ('error' in parsed) return fail(`The saved settings are invalid: ${parsed.error}.`);
  const { config } = parsed;

  const c = checkpoint(d.checkpoint);
  try {
    new MnistSession(toMnistSessionConfig(config, data, c));
  } catch (e) {
    fail(`The saved model doesn’t match its settings: ${e instanceof Error ? e.message : e}.`);
  }
  return { config, checkpoint: c, savedAt: typeof d.savedAt === 'string' ? d.savedAt : null };
}
