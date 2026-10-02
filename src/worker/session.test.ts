import { describe, expect, it } from 'vitest';
import type { Toy2DKind } from '../data';
import type { NetworkSpec } from './network';
import { type SessionConfig, TrainingSession } from './session';

const TANH_8_8: NetworkSpec = {
  hidden: [
    { units: 8, activation: 'tanh' },
    { units: 8, activation: 'tanh' },
  ],
};

function config(kind: Toy2DKind, overrides: Partial<SessionConfig> = {}): SessionConfig {
  return {
    dataset: { kind, n: 400, noise: 0, seed: 1 },
    network: TANH_8_8,
    lr: 0.03,
    batchSize: 10,
    seed: 1,
    ...overrides,
  };
}

function train(session: TrainingSession, epochs: number): TrainingSession {
  for (let e = 0; e < epochs; e++) session.trainEpoch();
  return session;
}

describe('TrainingSession', () => {
  it('splits 400 points 280 / 120 and counts epochs and steps', () => {
    const s = new TrainingSession(config('circle'));
    expect(s.train.x.rows).toBe(280);
    expect(s.test.x.rows).toBe(120);
    train(s, 3);
    const snap = s.snapshot();
    expect(snap.epoch).toBe(3);
    expect(snap.step).toBe(3 * 28);
    // Index 0 is the loss before training.
    expect(snap.trainLoss).toHaveLength(4);
    expect(snap.testLoss).toHaveLength(4);
  });

  it('is deterministic: same config → identical losses and weights', () => {
    const a = train(new TrainingSession(config('spirals')), 20).snapshot();
    const b = train(new TrainingSession(config('spirals')), 20).snapshot();
    expect(b.trainLoss).toEqual(a.trainLoss);
    expect(b.weights).toEqual(a.weights);
    const c = train(new TrainingSession(config('spirals', { seed: 2 })), 20).snapshot();
    expect(c.trainLoss).not.toEqual(a.trainLoss);
  });

  it('produces a structured-cloneable snapshot whose shapes match the network', () => {
    const network: NetworkSpec = {
      hidden: [
        { units: 3, activation: 'relu' },
        { units: 5, activation: 'linear' },
        { units: 4, activation: 'sigmoid' },
      ],
    };
    const s = new TrainingSession(config('xor', { network, gridSize: 10 }));
    const snap = structuredClone(s.snapshot());
    expect(snap.columns.map((c) => [c.kind, c.units, c.values.length])).toEqual([
      ['input', 2, 200],
      ['relu', 3, 300],
      ['linear', 5, 500],
      ['sigmoid', 4, 400],
      ['output', 1, 100],
    ]);
    expect(snap.weights.map((w) => [w.inFeatures, w.outFeatures, w.W.length, w.b.length])).toEqual([
      [2, 3, 6, 3],
      [3, 5, 15, 5],
      [5, 4, 20, 4],
      [4, 1, 4, 1],
    ]);
    // Output is a probability; ReLU outputs are non-negative.
    expect(Array.from(snap.columns[4]!.values).every((p) => p > 0 && p < 1)).toBe(true);
    expect(Array.from(snap.columns[1]!.values).every((v) => v >= 0)).toBe(true);
    expect(snap.train.x).toHaveLength(280 * 2);
    expect(snap.test.y).toHaveLength(120);
  });

  it('lays the grid out with row 0 at the top', () => {
    const snap = new TrainingSession(config('circle', { gridSize: 4 })).snapshot();
    const [x1, x2] = [0, 1].map((u) => snap.columns[0]!.values.subarray(u * 16, (u + 1) * 16));
    // Top-left cell: x most negative, y most positive.
    expect(x1![0]).toBeCloseTo(-4.5);
    expect(x2![0]).toBeCloseTo(4.5);
    expect(x1![3]).toBeCloseTo(4.5);
    expect(x2![12]).toBeCloseTo(-4.5);
  });

  it('applies new hyperparameters without resetting', () => {
    const s = train(new TrainingSession(config('gaussians')), 2);
    s.setHyperparams({ lr: 0.1, batchSize: 'full' });
    train(s, 1);
    const snap = s.snapshot();
    expect(snap.epoch).toBe(3);
    expect(snap.step).toBe(2 * 28 + 1);
  });

  it('snapshots are copies that later training does not touch', () => {
    const s = new TrainingSession(config('xor'));
    const before = s.snapshot();
    const w = before.weights[0]!.W.slice();
    train(s, 5);
    expect(before.weights[0]!.W).toEqual(w);
    expect(before.epoch).toBe(0);
  });
});

describe('learning (seeded, lr 0.03, batch size 10)', () => {
  it('a 2-8-8-1 tanh net untangles the two spirals', () => {
    const snap = train(new TrainingSession(config('spirals')), 2000).snapshot();
    expect(snap.trainAccuracy).toBeGreaterThanOrEqual(0.95);
    // Generalisation lags on 280 points of a 1.75-turn spiral; chance is 0.5.
    expect(snap.testAccuracy).toBeGreaterThanOrEqual(0.8);
  });

  it.each(['circle', 'xor', 'gaussians'] as const)(
    '%s reaches ≥ 95%% within 200 epochs',
    (kind) => {
      const snap = train(
        new TrainingSession(config(kind, { dataset: { kind, n: 400, noise: 0.1, seed: 1 } })),
        200,
      ).snapshot();
      expect(snap.trainAccuracy).toBeGreaterThanOrEqual(0.95);
      expect(snap.testAccuracy).toBeGreaterThanOrEqual(0.95);
    },
  );

  it('a linear-only net cannot solve XOR', () => {
    const network: NetworkSpec = { hidden: [{ units: 8, activation: 'linear' }] };
    const snap = train(new TrainingSession(config('xor', { network })), 200).snapshot();
    expect(snap.trainAccuracy).toBeLessThan(0.8);
  });
});
