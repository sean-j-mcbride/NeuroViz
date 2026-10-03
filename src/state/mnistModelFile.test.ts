import { describe, expect, it } from 'vitest';
import { loadMnistSubset } from '../test/mnist';
import { MnistSession } from '../worker';
import { MNIST_DEFAULT_CONFIG, type MnistConfig, toMnistSessionConfig } from './mnistConfig';
import {
  MNIST_MODEL_FILE_FORMAT,
  parseMnistModelFile,
  serialiseMnistModelFile,
} from './mnistModelFile';
import { ModelFileError, parseModelFile } from './modelFile';

const data = loadMnistSubset();

const SMALL: MnistConfig = {
  ...MNIST_DEFAULT_CONFIG,
  network: { conv: [], hidden: [{ units: 16, activation: 'relu' }] },
  training: { ...MNIST_DEFAULT_CONFIG.training, dropout: 0.2, l2: 0.001 },
};

function advance(s: MnistSession, batches: number): MnistSession {
  for (let i = 0; i < batches; i++) s.advance();
  return s;
}

function saved(config = SMALL, batches = 170): string {
  const s = advance(new MnistSession(toMnistSessionConfig(config, data)), batches);
  return serialiseMnistModelFile(config, data, s.checkpoint(), '2026-10-03T12:00:00.000Z');
}

/** The parts of a saved file the tests tamper with. */
interface FileDoc {
  version: unknown;
  dataset: Record<string, unknown>;
  config: {
    training: Record<string, unknown>;
    network: Record<string, unknown> & { conv?: unknown };
  };
  checkpoint: { params: Record<string, unknown>; examples: unknown };
}

function edited(text: string, change: (doc: FileDoc) => void): string {
  const doc = JSON.parse(text) as FileDoc;
  change(doc);
  return JSON.stringify(doc);
}

describe('MNIST model files', () => {
  for (const optimiser of ['sgd', 'momentum', 'adam'] as const) {
    it(`save mid-epoch → load → train on is bitwise identical to never stopping (${optimiser})`, () => {
      const config: MnistConfig = {
        ...SMALL,
        training: { ...SMALL.training, optimiser, lr: optimiser === 'adam' ? 0.001 : 0.03 },
      };
      const straight = advance(new MnistSession(toMnistSessionConfig(config, data)), 330);
      const file = parseMnistModelFile(saved(config, 170), data);
      expect(file.config).toEqual(config);
      expect(file.savedAt).toBe('2026-10-03T12:00:00.000Z');
      const resumed = advance(
        new MnistSession(toMnistSessionConfig(file.config, data, file.checkpoint)),
        160,
      );
      expect(resumed.checkpoint()).toEqual(straight.checkpoint());
      expect(resumed.snapshot()).toEqual(straight.snapshot());
    });
  }

  it('keeps the settings and curves readable, and the default network’s file compact', () => {
    const text = saved();
    const doc = JSON.parse(text);
    expect(doc.format).toBe(MNIST_MODEL_FILE_FORMAT);
    expect(doc.config.network.hidden).toEqual([{ units: 16, activation: 'relu' }]);
    expect(Array.isArray(doc.checkpoint.losses.train.mean)).toBe(true);
    expect(typeof doc.checkpoint.params['0.W']).toBe('string');

    const big = saved(MNIST_DEFAULT_CONFIG, 20);
    // 101,770 params × 3 (weights + Adam's two moments) as base64 float32 ≈ 1.6 MB.
    expect(big.length).toBeGreaterThan(1.4e6);
    expect(big.length).toBeLessThan(1.8e6);
  });

  it('rejects files that cannot resume here, before anything is built, in words', () => {
    const text = saved();
    const expectError = (t: string, message: RegExp) => {
      expect(() => parseMnistModelFile(t, data)).toThrow(ModelFileError);
      expect(() => parseMnistModelFile(t, data)).toThrow(message);
    };
    expectError('not json', /not valid JSON/);
    expectError(JSON.stringify({ format: 'neuroviz-model', version: 2 }), /Playground tab/);
    expectError(JSON.stringify({ format: 'other' }), /not a saved NeuroViz MNIST model/);
    expectError(
      edited(text, (d) => (d.version = 9)),
      /newer version/,
    );
    expectError(
      edited(text, (d) => (d.dataset.checksum = 1)),
      /different data/,
    );
    expectError(
      edited(text, (d) => (d.config.training.batchSize = 7)),
      /batch size 7/,
    );
    expectError(
      edited(text, (d) => (d.config.network.hidden = [{ units: 32, activation: 'relu' }])),
      /doesn’t match its settings/,
    );
    expectError(
      edited(text, (d) => (d.checkpoint.params['0.W'] = '!!')),
      /0\.W.*not valid/,
    );
    expectError(
      edited(text, (d) => (d.checkpoint.examples = 1e6)),
      /doesn’t match/,
    );
  });

  it('a CNN saves mid-epoch and resumes bitwise; its conv rows stay readable', () => {
    const config: MnistConfig = {
      ...SMALL,
      network: {
        conv: [
          { kind: 'conv', filters: 4, kernel: 5, stride: 1, padding: 'valid', activation: 'relu' },
          { kind: 'pool', size: 2 },
        ],
        hidden: [{ units: 16, activation: 'relu' }],
      },
    };
    const straight = advance(new MnistSession(toMnistSessionConfig(config, data)), 200);
    const text = saved(config, 170);
    const doc = JSON.parse(text);
    expect(doc.version).toBe(3);
    expect(doc.config.network.conv).toEqual(config.network.conv);
    // The gradient charts: one layer per conv/dense layer (conv, dense 16, output).
    expect(doc.checkpoint.history.timeline.layers).toHaveLength(3);
    const file = parseMnistModelFile(text, data);
    expect(file.config).toEqual(config);
    const resumed = advance(
      new MnistSession(toMnistSessionConfig(file.config, data, file.checkpoint)),
      30,
    );
    expect(resumed.checkpoint()).toEqual(straight.checkpoint());
  }, 30_000);

  it('version 2 files (no charts history) load; the charts restart there', () => {
    const v2 = edited(saved(SMALL, 20), (d) => {
      d.version = 2;
      delete (d.checkpoint as { history?: unknown }).history;
    });
    const file = parseMnistModelFile(v2, data);
    expect(file.checkpoint.timeline).toBeUndefined();
    const s = new MnistSession(toMnistSessionConfig(file.config, data, file.checkpoint));
    expect(s.snapshot().timeline.epochs).toHaveLength(0);
  });

  it('version 1 files (Phase 5, no conv rows) still load, as MLPs', () => {
    const v1 = edited(saved(SMALL, 20), (d) => {
      d.version = 1;
      delete (d.checkpoint as { history?: unknown }).history;
      delete d.config.network.conv;
    });
    expect(parseMnistModelFile(v1, data).config.network).toEqual(SMALL.network);
  });

  it('rejects conv rows the controls don’t offer, or that don’t fit the image', () => {
    const conv = (filters: number) => ({
      kind: 'conv',
      filters,
      kernel: 5,
      stride: 1,
      padding: 'valid',
      activation: 'relu',
    });
    const pool = { kind: 'pool', size: 2 };
    const text = saved();
    expect(() =>
      parseMnistModelFile(
        edited(text, (d) => (d.config.network.conv = [conv(7)])),
        data,
      ),
    ).toThrow(/conv layer .*"filters":7.* is not one the controls offer/);
    expect(() =>
      parseMnistModelFile(
        edited(text, (d) => (d.config.network.conv = [conv(4), pool, conv(4), pool, conv(4)])),
        data,
      ),
    ).toThrow('conv layer 5: a 5 × 5 kernel doesn’t fit a 4 × 4 input');
  });

  it('the playground loader sends MNIST files to the MNIST tab', () => {
    expect(() => parseModelFile(saved(SMALL, 5))).toThrow(/MNIST tab/);
  });
});
