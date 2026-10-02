import { describe, expect, it } from 'vitest';
import { Tanh } from './layers/activations';
import { Dense } from './layers/dense';
import type { Param } from './layers/types';
import { BCEWithLogitsLoss } from './losses';
import { Adam, Momentum, type Optimiser, type OptimiserKind, SGD, makeOptimiser } from './optim';
import { Rng } from './random';
import { Sequential } from './sequential';
import { Tensor } from './tensor';
import { Trainer } from './train';

function param(values: number[]): Param {
  return { name: 'W', value: Tensor.from(values), grad: Tensor.zeros([values.length]) };
}

/** Runs `steps` updates with a gradient computed from the current value. */
function run(opt: Optimiser, p: Param, steps: number, grad: (w: Float32Array) => number[]): void {
  for (let s = 0; s < steps; s++) {
    p.grad.data.set(grad(p.value.data));
    opt.step([p]);
  }
}

describe('SGD', () => {
  it('steps against the gradient', () => {
    const p = param([1, -2]);
    run(new SGD(0.1), p, 1, () => [1, -4]);
    expect(Array.from(p.value.data)).toEqual([Math.fround(0.9), Math.fround(-1.6)]);
  });
});

describe('Momentum', () => {
  it('accumulates velocity: v = βv + g, w −= lr·v', () => {
    const p = param([1]);
    const opt = new Momentum(0.1, 0.9);
    run(opt, p, 1, () => [1]);
    expect(p.value.data[0]).toBeCloseTo(0.9, 6); // v = 1
    run(opt, p, 1, () => [1]);
    expect(p.value.data[0]).toBeCloseTo(0.71, 6); // v = 1.9
    run(opt, p, 1, () => [1]);
    expect(p.value.data[0]).toBeCloseTo(0.439, 6); // v = 2.71
  });

  it('keeps separate velocity per parameter', () => {
    const [a, b] = [param([0]), param([0])];
    const opt = new Momentum(1, 0.5);
    a.grad.data[0] = 1;
    b.grad.data[0] = -1;
    opt.step([a, b]);
    opt.step([a, b]);
    expect(a.value.data[0]).toBeCloseTo(-2.5);
    expect(b.value.data[0]).toBeCloseTo(2.5);
  });
});

describe('Adam', () => {
  it('moves every parameter by ≈ lr on the first steps, whatever the gradient scale', () => {
    const p = param([0, 0, 0]);
    const opt = new Adam(0.01);
    run(opt, p, 1, () => [1e-3, -1, 1e3]);
    expect(p.value.data[0]).toBeCloseTo(-0.01, 6);
    expect(p.value.data[1]).toBeCloseTo(0.01, 6);
    expect(p.value.data[2]).toBeCloseTo(-0.01, 6);
    // A constant gradient keeps the bias-corrected step at lr.
    run(opt, p, 1, () => [1e-3, -1, 1e3]);
    expect(p.value.data[0]).toBeCloseTo(-0.02, 6);
  });

  it('matches a hand-computed second step with a changing gradient', () => {
    const p = param([0]);
    const opt = new Adam(0.1, 0.9, 0.999, 0);
    run(opt, p, 1, () => [1]);
    run(opt, p, 1, () => [3]);
    // m = 0.9·0.1 + 0.1·3 = 0.39 → m̂ = 0.39/0.19; v = 0.999·0.001 + 0.001·9 → v̂ = v/(1 − 0.999²).
    const mHat = 0.39 / 0.19;
    const vHat = (0.999 * 0.001 + 0.001 * 9) / (1 - 0.999 ** 2);
    expect(p.value.data[0]).toBeCloseTo(-0.1 - (0.1 * mHat) / Math.sqrt(vHat), 6);
  });
});

describe('every optimiser', () => {
  const kinds: [OptimiserKind, number][] = [
    ['sgd', 0.05],
    ['momentum', 0.02],
    ['adam', 0.05],
  ];

  it.each(kinds)('%s minimises an ill-conditioned quadratic', (kind, lr) => {
    const p = param([3, -2]);
    const opt = makeOptimiser(kind, lr);
    expect(opt.kind).toBe(kind);
    // f(w) = ½(w₀² + 10·w₁²)
    run(opt, p, 1000, (w) => [w[0]!, 10 * w[1]!]);
    expect(Math.abs(p.value.data[0]!)).toBeLessThan(1e-2);
    expect(Math.abs(p.value.data[1]!)).toBeLessThan(1e-2);
  });

  it.each([
    ['momentum', 0.1],
    ['adam', 0.05],
  ] as const)('%s trains a 2-4-1 net on XOR', (kind, lr) => {
    const rng = new Rng(3);
    const model = new Sequential([
      new Dense(2, 4, { init: 'xavier', rng }),
      new Tanh(),
      new Dense(4, 1, { init: 'xavier', rng }),
    ]);
    const x = Tensor.from([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]);
    const y = Tensor.from([[0], [1], [1], [0]]);
    const trainer = new Trainer({
      model,
      loss: new BCEWithLogitsLoss(),
      optimiser: makeOptimiser(kind, lr),
    });
    let loss = Infinity;
    for (let i = 0; i < 1000; i++) loss = trainer.trainStep(x, y);
    expect(loss).toBeLessThan(0.05);
  });
});

describe('optimiser state', () => {
  const grad = (w: Float32Array) => Array.from(w, (v, i) => (i + 1) * v - 0.3);
  const named = (name: string, values: number[]): Param => ({ ...param(values), name });

  it.each(['sgd', 'momentum', 'adam'] as const)(
    '%s: save mid-run → load into a fresh optimiser → continues identically',
    (kind) => {
      const a = [named('0.W', [1, -2, 0.5]), named('0.b', [0.1])];
      const original = makeOptimiser(kind, 0.05);
      const step = (opt: Optimiser, ps: Param[]) => {
        for (const p of ps) p.grad.data.set(grad(p.value.data));
        opt.step(ps);
      };
      for (let s = 0; s < 7; s++) step(original, a);

      const saved = original.saveState(a);
      const b = a.map((p) => named(p.name, Array.from(p.value.data)));
      const resumed = makeOptimiser(kind, 0.05);
      resumed.loadState(b, saved);

      for (let s = 0; s < 7; s++) {
        step(original, a);
        step(resumed, b);
      }
      a.forEach((p, i) => expect(b[i]!.value.data).toEqual(p.value.data));
    },
  );

  it('saved state is a copy', () => {
    const p = named('w', [1]);
    const opt = new Momentum(0.1);
    p.grad.data[0] = 1;
    opt.step([p]);
    const saved = opt.saveState([p]);
    opt.step([p]);
    expect(saved.slots['w']![0]![0]).toBe(1);
  });

  it('rejects state for another kind, a missing or unknown parameter, or a wrong size', () => {
    const p = named('w', [1, 2]);
    const adam = new Adam(0.1);
    adam.step([p]);
    const saved = adam.saveState([p]);
    expect(() => new Momentum(0.1).loadState([p], saved)).toThrow(/adam, not momentum/);
    expect(() => new Adam(0.1).loadState([named('v', [1, 2])], saved)).toThrow(
      /unknown parameter "w"/,
    );
    expect(() => new Adam(0.1).loadState([p, named('v', [1])], saved)).toThrow(
      /missing parameter "v"/,
    );
    expect(() => new Adam(0.1).loadState([named('w', [1, 2, 3])], saved)).toThrow(/expected 3/);
  });
});
