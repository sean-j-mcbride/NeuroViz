import { describe, expect, it } from 'vitest';
import { TrainingSession } from '../worker';
import { DEFAULT_CONFIG, type PlaygroundConfig, toSessionConfig } from './config';
import { ModelFileError, formatFloat32, parseModelFile, serialiseModelFile } from './modelFile';

const CONFIG: PlaygroundConfig = {
  ...DEFAULT_CONFIG,
  dataset: { kind: 'spirals', n: 200, noise: 0.1, seed: 3 },
  training: { lr: 0.01, batchSize: 10, optimiser: 'adam', l2: 0.0001, dropout: 0.2 },
};

function trained(config: PlaygroundConfig, epochs: number): TrainingSession {
  const s = new TrainingSession({ ...toSessionConfig(config), gridSize: 2 });
  for (let e = 0; e < epochs; e++) s.trainEpoch();
  return s;
}

const SAVED_AT = '2026-10-02T12:00:00.000Z';

describe('formatFloat32', () => {
  it('writes the shortest decimal that reads back as the same float32', () => {
    expect(formatFloat32(Math.fround(0.1))).toBe('0.1');
    expect(formatFloat32(Math.fround(-1.2345678))).toBe('-1.2345678');
    expect(formatFloat32(10)).toBe('10');
    expect(formatFloat32(Math.fround(1e-7))).toBe('1e-7');
    expect(formatFloat32(-0)).toBe('-0');
    expect(formatFloat32(Number.NaN)).toBe('"NaN"');
    expect(formatFloat32(-Infinity)).toBe('"-Infinity"');
  });

  it('round-trips random float32 values exactly', () => {
    const a = new Float32Array(10_000);
    const bits = new Uint32Array(a.buffer);
    for (let i = 0; i < bits.length; i++) bits[i] = (Math.random() * 2 ** 32) >>> 0;
    for (const v of a) {
      if (!Number.isFinite(v)) continue;
      expect(Math.fround(JSON.parse(formatFloat32(v)) as number)).toBe(v);
    }
  });
});

describe('model files', () => {
  it('save → load is lossless', () => {
    const c = trained(CONFIG, 5).checkpoint();
    const text = serialiseModelFile(CONFIG, c, SAVED_AT);
    expect(parseModelFile(text)).toEqual({ config: CONFIG, checkpoint: c, savedAt: SAVED_AT });
  });

  it('train N, save to a file, load, train M ≡ train N + M, bitwise', () => {
    const straight = trained(CONFIG, 20);
    const text = serialiseModelFile(CONFIG, trained(CONFIG, 10).checkpoint(), SAVED_AT);
    const { config, checkpoint } = parseModelFile(text);
    const resumed = new TrainingSession({ ...toSessionConfig(config, checkpoint), gridSize: 2 });
    for (let e = 0; e < 10; e++) resumed.trainEpoch();
    expect(resumed.checkpoint()).toEqual(straight.checkpoint());
  });

  it('is readable: one array per line, short numbers', () => {
    const text = serialiseModelFile(CONFIG, trained(CONFIG, 1).checkpoint(), SAVED_AT);
    expect(text).toMatch(/^\{\n {2}"format": "neuroviz-model",\n {2}"version": 2,/);
    expect(text).toMatch(/\n {6}"0\.W": \[-?\d[^\n]*\],?\n/);
    expect(text.split('\n').length).toBeLessThan(150);
    // The charts' history is one labelled, compact block.
    expect(text).toMatch(/\n {4}"history": \{\n {6}"note": "Display only/);
  });

  it('keeps non-finite losses from a diverged run', () => {
    const c = trained(CONFIG, 1).checkpoint();
    c.trainLoss[1] = Infinity;
    c.testLoss[1] = Number.NaN;
    const back = parseModelFile(serialiseModelFile(CONFIG, c)).checkpoint;
    expect(back.trainLoss[1]).toBe(Infinity);
    expect(back.testLoss[1]).toBeNaN();
  });

  describe('rejects with a readable reason', () => {
    const good = () =>
      JSON.parse(serialiseModelFile(CONFIG, trained(CONFIG, 2).checkpoint())) as Record<
        string,
        Record<string, unknown>
      >;
    const attempt = (doc: unknown) => () =>
      parseModelFile(typeof doc === 'string' ? doc : JSON.stringify(doc));

    it.each([
      ['not JSON', 'hello', /not valid JSON/],
      ['another JSON file', { name: 'package' }, /not a saved NeuroViz model/],
      ['a newer format', { ...good(), version: 3 }, /newer version of NeuroViz \(format 3\)/],
      [
        'a settings log with settings the controls don’t offer',
        (() => {
          const d = good();
          d.checkpoint!.hyperparamLog = [{ epoch: 0, hyperparams: { ...CONFIG.training, lr: 2 } }];
          return d;
        })(),
        /hyperparamLog\[0\] is invalid: Ignored learning rate 2/,
      ],
      [
        'settings the controls don’t offer',
        { ...good(), config: { ...CONFIG, seed: -5 } },
        /saved settings are invalid: Ignored weight seed -5/,
      ],
      [
        'a corrupt number',
        (() => {
          const d = good();
          (d.checkpoint!.params as Record<string, unknown[]>)['0.W']![3] = 'x';
          return d;
        })(),
        /checkpoint.params\["0.W"\]\[3\] is not a number/,
      ],
      [
        'corrupt chart history',
        (() => {
          const d = good();
          const layers = (
            d.checkpoint!.history as { timeline: { layers: Record<string, unknown>[] } }
          ).timeline.layers;
          layers[0]!.weightRms = '%%%';
          return d;
        })(),
        /history\.timeline\.layers\[0\]\.weightRms is not valid/,
      ],
      [
        'a histogram count above the layer size',
        (() => {
          const d = good();
          const layers = (
            d.checkpoint!.history as { timeline: { layers: Record<string, unknown>[] } }
          ).timeline.layers;
          layers[0]!.size = 1;
          return d;
        })(),
        /weightCounts has a count above the layer size/,
      ],
      [
        'weights for another network',
        {
          ...good(),
          config: {
            ...CONFIG,
            network: { hidden: [{ units: 4, activation: 'tanh' }, CONFIG.network.hidden[1]] },
          },
        },
        /doesn’t match its settings: Parameter "0.W" has 16 values, expected 8/,
      ],
      [
        'a different dataset size',
        { ...good(), config: { ...CONFIG, dataset: { ...CONFIG.dataset, n: 300 } } },
        /doesn’t match its settings: Checkpoint is for 140 training points, not 210/,
      ],
    ])('%s', (_, doc, message) => {
      expect(attempt(doc)).toThrow(ModelFileError);
      expect(attempt(doc)).toThrow(message);
    });
  });

  it('keeps the settings log, including mid-run changes', () => {
    const s = trained(CONFIG, 3);
    s.setHyperparams({ ...CONFIG.training, lr: 0.1 });
    s.trainEpoch();
    const back = parseModelFile(serialiseModelFile(CONFIG, s.checkpoint())).checkpoint;
    expect(back.hyperparamLog).toEqual([
      { epoch: 0, hyperparams: CONFIG.training },
      { epoch: 3, hyperparams: { ...CONFIG.training, lr: 0.1 } },
    ]);
  });

  it('still reads version 1 files, as if the settings never changed', () => {
    const doc = JSON.parse(serialiseModelFile(CONFIG, trained(CONFIG, 2).checkpoint())) as {
      version: number;
      checkpoint: Record<string, unknown>;
    };
    doc.version = 1;
    delete doc.checkpoint.hyperparamLog;
    const { checkpoint } = parseModelFile(JSON.stringify(doc));
    expect(checkpoint.epoch).toBe(2);
    expect(checkpoint.hyperparamLog).toEqual([{ epoch: 0, hyperparams: CONFIG.training }]);
    expect(checkpoint.observations).toBeNull();
  });
});
