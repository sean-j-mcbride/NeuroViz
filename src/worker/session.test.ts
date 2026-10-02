import { describe, expect, it } from 'vitest';
import type { Toy2DKind } from '../data';
import { range } from '../data';
import type { Checkpoint } from './checkpoint';
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
    optimiser: 'sgd',
    l2: 0,
    dropout: 0,
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
    expect(snap.losses.count).toBe(4);
    expect(snap.losses.train.mean).toHaveLength(4);
  });

  it('is deterministic: same config → identical losses and weights', () => {
    const a = train(new TrainingSession(config('spirals')), 20).snapshot();
    const b = train(new TrainingSession(config('spirals')), 20).snapshot();
    expect(b.losses).toEqual(a.losses);
    expect(b.weights).toEqual(a.weights);
    const c = train(new TrainingSession(config('spirals', { seed: 2 })), 20).snapshot();
    expect(c.losses).not.toEqual(a.losses);
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
    const data = s.data();
    expect(data.train.x).toHaveLength(280 * 2);
    expect(data.test.y).toHaveLength(120);
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
    s.setHyperparams({ lr: 0.1, batchSize: 'full', optimiser: 'sgd', l2: 0, dropout: 0 });
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

describe('observing training', () => {
  const busy = (): SessionConfig =>
    config('spirals', { optimiser: 'adam', lr: 0.01, dropout: 0.2, l2: 1e-3 });

  it('snapshots and probes never change the trajectory', () => {
    const a = new TrainingSession(busy());
    const b = new TrainingSession(busy());
    for (let e = 0; e < 15; e++) {
      a.trainEpoch();
      a.snapshot({ probe: { set: 'train', index: e } });
      b.trainEpoch();
    }
    const [sa, sb] = [a.snapshot(), b.snapshot()];
    expect(sa.weights).toEqual(sb.weights);
    expect(sa.losses).toEqual(sb.losses);
  });

  it('records a bounded, evenly spaced timeline of histograms', () => {
    const s = train(new TrainingSession(config('circle', { gridSize: 4 })), 300);
    const { timeline } = s.snapshot();
    const epochs = Array.from(timeline.epochs);
    expect(epochs.length).toBeLessThanOrEqual(128);
    expect(epochs[0]).toBe(0);
    const step = epochs[1]! - epochs[0]!;
    expect(step).toBe(4); // 300 epochs > 2·128 → interval 4
    epochs.forEach((e, t) => expect(e).toBe(t * step));
    expect(timeline.layers).toHaveLength(3);
    const l = timeline.layers[0]!;
    for (let t = 0; t < epochs.length; t++) {
      const col = l.gradHist.subarray(t * 32, (t + 1) * 32);
      expect(col.reduce((x, y) => x + y, 0)).toBeCloseTo(1, 5);
    }
  });

  it('keeps the last 100 epochs of every parameter, ending at the current weights', () => {
    const s = train(new TrainingSession(config('xor', { gridSize: 4 })), 120);
    const { paramHistory, weights } = s.snapshot();
    const n = paramHistory.epochs.length;
    expect(n).toBe(100);
    expect(paramHistory.epochs[0]).toBe(21);
    expect(paramHistory.epochs[n - 1]).toBe(120);
    const w = weights[1]!;
    const h = paramHistory.layers[1]!;
    for (let j = 0; j < w.W.length; j++) expect(h.W[j * n + n - 1]).toBe(w.W[j]);
    for (let j = 0; j < w.b.length; j++) expect(h.b[j * n + n - 1]).toBe(w.b[j]);
  });
});

describe('hyperparameters apply live', () => {
  it('optimiser, L2 and dropout all change the run without resetting it', () => {
    const base = () => train(new TrainingSession(config('circle', { gridSize: 4 })), 5);
    const ref = train(base(), 5).snapshot();
    const variants = [
      { optimiser: 'adam' as const },
      { optimiser: 'momentum' as const },
      { l2: 0.1 },
      { dropout: 0.3 },
    ];
    for (const v of variants) {
      const s = base();
      s.setHyperparams({ lr: 0.03, batchSize: 10, optimiser: 'sgd', l2: 0, dropout: 0, ...v });
      const snap = train(s, 5).snapshot();
      expect(snap.epoch).toBe(10);
      expect(snap.weights).not.toEqual(ref.weights);
    }
  });

  it('L2 shrinks the weights', () => {
    const norm = (l2: number) => {
      const snap = train(new TrainingSession(config('circle', { l2, gridSize: 4 })), 50).snapshot();
      return snap.weights.reduce((t, w) => t + w.W.reduce((u, x) => u + x * x, 0), 0);
    };
    expect(norm(0.03)).toBeLessThan(0.7 * norm(0));
  });
});

describe('probe', () => {
  const s = train(new TrainingSession(config('circle', { gridSize: 4 })), 10);

  it('traces one point forward and backward consistently', () => {
    const t = s.snapshot({ probe: { set: 'test', index: 3 } }).probe!;
    expect(t.ref).toEqual({ set: 'test', index: 3 });
    expect(Array.from(t.x)).toEqual(Array.from(s.test.x.data.subarray(6, 8)));
    expect(t.columns.map((c) => c.a.length)).toEqual([2, 8, 8, 1]);
    const out = t.columns[3]!;
    expect(out.a[0]).toBeCloseTo(t.p, 6);
    expect(out.dZ[0]).toBeCloseTo(t.p - t.label, 6);
    expect(t.loss).toBeCloseTo(-Math.log(t.label === 1 ? t.p : 1 - t.p), 4);
    // tanh: a = tanh(z) and dZ = dA·(1 − a²).
    const h = t.columns[1]!;
    for (let u = 0; u < 8; u++) {
      expect(h.a[u]).toBeCloseTo(Math.tanh(h.z[u]!), 6);
      expect(h.dZ[u]).toBeCloseTo(h.dA[u]! * (1 - h.a[u]! ** 2), 6);
    }
    // dW = a_prevᵀ · dZ.
    t.dW.forEach((dW, k) => {
      const [prev, next] = [t.columns[k]!.a, t.columns[k + 1]!.dZ];
      for (let i = 0; i < prev.length; i++)
        for (let j = 0; j < next.length; j++)
          expect(dW[i * next.length + j]).toBeCloseTo(prev[i]! * next[j]!, 6);
    });
  });

  it('per-example gradients average to the full-batch gradient', () => {
    const snap = s.snapshot();
    const n = s.train.x.rows;
    const mean = snap.weights.map((w) => new Float64Array(w.W.length));
    for (let i = 0; i < n; i++) {
      s.probe({ set: 'train', index: i })!.dW.forEach((dW, k) => {
        for (let j = 0; j < dW.length; j++) mean[k]![j]! += dW[j]! / n;
      });
    }
    snap.weights.forEach((w, k) => {
      for (let j = 0; j < w.gradW.length; j++) expect(mean[k]![j]).toBeCloseTo(w.gradW[j]!, 5);
    });
  });

  it('rejects out-of-range indices', () => {
    expect(s.probe({ set: 'train', index: 280 })).toBeUndefined();
    expect(s.snapshot({ probe: { set: 'test', index: -1 } }).probe).toBeUndefined();
  });
});

describe('deep sigmoid vs deep ReLU (the Phase 3 demonstration)', () => {
  const deep = (activation: 'sigmoid' | 'relu') =>
    new TrainingSession(
      config('circle', {
        dataset: { kind: 'circle', n: 400, noise: 0.1, seed: 1 },
        network: { hidden: Array.from({ length: 6 }, () => ({ units: 8, activation })) },
        gridSize: 4,
      }),
    );
  /** First-layer gradient RMS over output-layer gradient RMS at epoch 0. */
  const ratio = (s: TrainingSession) => {
    const layers = s.snapshot().timeline.layers;
    return layers[0]!.gradRms[0]! / layers.at(-1)!.gradRms[0]!;
  };

  it('sigmoid starves the early layers of gradient; ReLU does not', () => {
    // Measured (seeds 1–3): sigmoid 6e-5 – 3e-4, ReLU 0.07 – 0.1.
    expect(ratio(deep('sigmoid'))).toBeLessThan(1e-3);
    expect(ratio(deep('relu'))).toBeGreaterThan(0.03);
  });

  it('so ReLU learns the circle quickly while sigmoid barely moves', () => {
    // Measured: ReLU reaches 90 % in 9 epochs; sigmoid not within 600.
    expect(train(deep('relu'), 40).snapshot().trainAccuracy).toBeGreaterThanOrEqual(0.9);
    expect(train(deep('sigmoid'), 200).snapshot().trainAccuracy).toBeLessThan(0.75);
  });
});

describe('checkpoints: exact resume', () => {
  const N = 12;
  const M = 12;
  const cases = [
    ['adam', { optimiser: 'adam', lr: 0.01, dropout: 0.2 }],
    ['momentum', { optimiser: 'momentum', lr: 0.01, l2: 0.001 }],
    ['sgd', { optimiser: 'sgd' }],
  ] as const;

  /** Trains N epochs, checkpoints, clones it across a (simulated) worker boundary, resumes and trains M more. */
  function resumed(cfg: SessionConfig, tamper?: (c: Checkpoint) => void): TrainingSession {
    const c = structuredClone(train(new TrainingSession(cfg), N).checkpoint());
    tamper?.(c);
    return train(new TrainingSession({ ...cfg, resume: c }), M);
  }

  it.each(cases)('%s: train N, save, resume, train M ≡ train N + M, bitwise', (_, overrides) => {
    const cfg = config('spirals', { ...overrides, gridSize: 4 });
    const straight = train(new TrainingSession(cfg), N + M);
    const b = resumed(cfg);
    expect(b.checkpoint()).toEqual(straight.checkpoint());
    const [sa, sb] = [straight.snapshot(), b.snapshot()];
    expect(sb.weights).toEqual(sa.weights);
    expect(sb.timeline).toEqual(sa.timeline);
    expect(sb.paramHistory).toEqual(sa.paramHistory);
    expect(sb.losses).toEqual(sa.losses);
    expect([sb.epoch, sb.step, sb.trainAccuracy]).toEqual([sa.epoch, sa.step, sa.trainAccuracy]);
  });

  it('a resumed session reports the checkpoint epoch and history before training on', () => {
    const cfg = config('circle', { gridSize: 4 });
    const a = train(new TrainingSession(cfg), N);
    const b = new TrainingSession({ ...cfg, resume: a.checkpoint() });
    const [sa, sb] = [a.snapshot(), b.snapshot()];
    expect(sb.epoch).toBe(N);
    expect(sb.losses).toEqual(sa.losses);
    expect(sb.trainAccuracy).toBe(sa.trainAccuracy);
    // The charts' history carries on too.
    expect(sb.timeline).toEqual(sa.timeline);
    expect(sb.paramHistory).toEqual(sa.paramHistory);
  });

  it('chart history resumes exactly across a timeline compaction', () => {
    const cfg = config('circle', {
      dataset: { kind: 'circle', n: 100, noise: 0, seed: 1 },
      network: { hidden: [{ units: 3, activation: 'tanh' }] },
      gridSize: 2,
    });
    const straight = train(new TrainingSession(cfg), 300).snapshot();
    const half = structuredClone(train(new TrainingSession(cfg), 150).checkpoint());
    const resumed = train(new TrainingSession({ ...cfg, resume: half }), 150).snapshot();
    expect(straight.timeline.epochs.length).toBeLessThan(128); // compacted at least once
    expect(resumed.timeline).toEqual(straight.timeline);
    expect(resumed.paramHistory).toEqual(straight.paramHistory);
  });

  it('without saved chart history (older files), the charts restart at the checkpoint', () => {
    const cfg = config('circle', { gridSize: 4 });
    const c = train(new TrainingSession(cfg), N).checkpoint();
    c.observations = null;
    const sb = new TrainingSession({ ...cfg, resume: c }).snapshot();
    expect(Array.from(sb.timeline.epochs)).toEqual([N]);
    expect(Array.from(sb.paramHistory.epochs)).toEqual([N]);
  });

  it('rejects chart history that does not fit the network', () => {
    const cfg = config('circle', { gridSize: 4 });
    const c = train(new TrainingSession(cfg), 2).checkpoint();
    c.observations!.timeline.layers.pop();
    expect(() => new TrainingSession({ ...cfg, resume: c })).toThrow(/timeline does not fit/);
  });

  // Mutation checks: each piece of saved state matters.
  it.each([
    ['the shuffle order', (c: Checkpoint) => c.order.set(range(c.order.length))],
    ['the shuffle RNG', (c: Checkpoint) => (c.rng.shuffle = { state: 1, spare: null })],
    ['the dropout RNG', (c: Checkpoint) => (c.rng.dropout = { state: 1, spare: null })],
    ['the optimiser state', (c: Checkpoint) => (c.optimiser.t = 1)],
  ])('dropping %s changes the continuation', (_, tamper) => {
    const cfg = config('spirals', { optimiser: 'adam', lr: 0.01, dropout: 0.2, gridSize: 4 });
    const straight = train(new TrainingSession(cfg), N + M).checkpoint();
    expect(resumed(cfg, tamper).checkpoint().params).not.toEqual(straight.params);
  });

  it('starts the configured optimiser fresh if the checkpoint is for another', () => {
    const adam = config('circle', { optimiser: 'adam', lr: 0.01, gridSize: 4 });
    const c = train(new TrainingSession(adam), N).checkpoint();
    const s = new TrainingSession({ ...adam, optimiser: 'sgd', resume: c });
    expect(s.checkpoint().optimiser).toEqual({ kind: 'sgd', t: 0, slots: expect.any(Object) });
  });

  it('rejects a checkpoint from a different dataset size or network', () => {
    const cfg = config('circle', { gridSize: 4 });
    const c = train(new TrainingSession(cfg), 2).checkpoint();
    expect(
      () => new TrainingSession({ ...cfg, dataset: { ...cfg.dataset, n: 200 }, resume: c }),
    ).toThrow(/280 training points, not 140/);
    const wider: NetworkSpec = { hidden: [{ units: 4, activation: 'tanh' }, TANH_8_8.hidden[1]!] };
    expect(() => new TrainingSession({ ...cfg, network: wider, resume: c })).toThrow(
      /"0.W" has 16 values, expected 8/,
    );
  });
});

describe('dead ReLUs', () => {
  const relu = (layers: number): NetworkSpec => ({
    hidden: Array.from({ length: layers }, () => ({ units: 8, activation: 'relu' as const })),
  });
  const deadCount = (s: TrainingSession) =>
    s
      .snapshot()
      .columns.flatMap((c) => (c.dead ? Array.from(c.dead) : []))
      .reduce((n, d) => n + d, 0);

  it('flags ReLU neurons that output 0 for every training point; other kinds carry no flags', () => {
    const network: NetworkSpec = { hidden: [...relu(1).hidden, { units: 3, activation: 'tanh' }] };
    const s = new TrainingSession(config('circle', { network, gridSize: 4 }));
    const [, r, t] = s.snapshot().columns;
    expect(r!.dead).toHaveLength(8);
    expect(t!.dead).toBeUndefined();
  });

  it('a neuron with a large negative bias is dead', () => {
    const cfg = config('circle', { network: relu(1), gridSize: 4 });
    const c = new TrainingSession(cfg).checkpoint();
    c.params['0.b']![2] = -100; // far below any w·x on the ±6 plane
    const snap = new TrainingSession({ ...cfg, resume: c }).snapshot();
    expect(Array.from(snap.columns[1]!.dead!)).toEqual([0, 0, 1, 0, 0, 0, 0, 0]);
  });

  it('Adam at lr 0.1 kills many of a deep ReLU net’s neurons; lr 0.01 does not', () => {
    const run = (lr: number) =>
      train(
        new TrainingSession(
          config('spirals', { network: relu(4), optimiser: 'adam', lr, gridSize: 4 }),
        ),
        200,
      );
    expect(deadCount(run(0.1))).toBeGreaterThanOrEqual(13); // ≥ 40 % of 32
    expect(deadCount(run(0.01))).toBeLessThanOrEqual(6);
  });

  it('flagging never changes the run', () => {
    const cfg = config('spirals', { network: relu(2), optimiser: 'adam', lr: 0.1, gridSize: 4 });
    const watched = new TrainingSession(cfg);
    const plain = new TrainingSession(cfg);
    for (let e = 0; e < 20; e++) {
      watched.trainEpoch();
      watched.snapshot();
      plain.trainEpoch();
    }
    expect(watched.checkpoint()).toEqual(plain.checkpoint());
  });
});

describe('settings log', () => {
  const base = config('circle', { gridSize: 4 });
  const hp = (o: Partial<SessionConfig> = {}) => {
    const { lr, batchSize, optimiser, l2, dropout } = { ...base, ...o };
    return { lr, batchSize, optimiser, l2, dropout };
  };
  const log = (s: TrainingSession) => s.snapshot().hyperparamLog;

  it('starts with the initial settings and logs only real changes', () => {
    const s = new TrainingSession(base);
    s.setHyperparams(hp()); // re-sent unchanged
    expect(log(s)).toEqual([{ epoch: 0, hyperparams: hp() }]);
    train(s, 3);
    s.setHyperparams(hp({ lr: 0.1 }));
    train(s, 2);
    s.setHyperparams(hp({ lr: 0.1 }));
    s.setHyperparams(hp({ lr: 0.1, optimiser: 'adam' }));
    expect(log(s)).toEqual([
      { epoch: 0, hyperparams: hp() },
      { epoch: 3, hyperparams: hp({ lr: 0.1 }) },
      { epoch: 5, hyperparams: hp({ lr: 0.1, optimiser: 'adam' }) },
    ]);
  });

  it('within one epoch keeps only the last change, and a change back removes it', () => {
    const s = new TrainingSession(base);
    s.setHyperparams(hp({ lr: 1 })); // before training: the starting settings change
    expect(log(s)).toEqual([{ epoch: 0, hyperparams: hp({ lr: 1 }) }]);
    train(s, 2);
    s.setHyperparams(hp({ lr: 3 }));
    s.setHyperparams(hp({ lr: 0.3 }));
    expect(log(s).at(-1)).toEqual({ epoch: 2, hyperparams: hp({ lr: 0.3 }) });
    s.setHyperparams(hp({ lr: 1 }));
    expect(log(s)).toEqual([{ epoch: 0, hyperparams: hp({ lr: 1 }) }]);
  });

  it('survives a checkpoint; resuming with other settings logs the change there', () => {
    const s = train(new TrainingSession(base), 2);
    s.setHyperparams(hp({ l2: 0.01 }));
    train(s, 2);
    const c = structuredClone(s.checkpoint());
    expect(log(new TrainingSession({ ...base, l2: 0.01, resume: c }))).toEqual(log(s));
    const changed = new TrainingSession({ ...base, l2: 0.01, dropout: 0.2, resume: c });
    expect(log(changed).at(-1)).toEqual({ epoch: 4, hyperparams: hp({ l2: 0.01, dropout: 0.2 }) });
  });

  it('rejects a log that is out of order', () => {
    const c = train(new TrainingSession(base), 2).checkpoint();
    c.hyperparamLog = [{ epoch: 5, hyperparams: hp() }];
    expect(() => new TrainingSession({ ...base, resume: c })).toThrow(
      /settings log is out of order/,
    );
  });
});
