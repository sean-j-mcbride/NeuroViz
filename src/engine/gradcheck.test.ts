import { describe, expect, it } from 'vitest';
import { assertGradsOk, checkLayer, checkLoss, checkModel } from './gradcheck';
import { ReLU, Sigmoid, Tanh } from './layers/activations';
import { Dense } from './layers/dense';
import type { Layer } from './layers/types';
import { BCEWithLogitsLoss, MSELoss, SoftmaxCrossEntropyLoss } from './losses';
import { Rng } from './random';
import { Sequential } from './sequential';
import { Tensor } from './tensor';

const BATCH = 5;

function oneHot(labels: number[], classes: number): Tensor {
  const t = Tensor.zeros([labels.length, classes]);
  labels.forEach((c, r) => (t.data[r * classes + c] = 1));
  return t;
}

describe('layer gradient checks', () => {
  it.each([
    ['dense (xavier)', (rng: Rng) => new Dense(3, 4, { init: 'xavier', rng })],
    ['dense (he)', (rng: Rng) => new Dense(4, 2, { init: 'he', rng })],
    ['tanh', () => new Tanh()],
    ['sigmoid', () => new Sigmoid()],
  ] as const)('%s', (_name, make) => {
    const rng = new Rng(1);
    const layer: Layer = make(rng);
    const inFeatures = layer instanceof Dense ? layer.inFeatures : 4;
    // Non-zero bias so the bias path is exercised too.
    for (const p of layer.params())
      if (p.name === 'b') p.value.data.set(Tensor.randn(p.value.shape, rng).data);
    const x = Tensor.randn([BATCH, inFeatures], rng);
    assertGradsOk(checkLayer(layer, x, rng));
  });

  it('relu (inputs kept away from the kink at 0)', () => {
    const rng = new Rng(2);
    const x = Tensor.randn([BATCH, 4], rng);
    // Push every input to |x| ≥ 0.1 so ±eps never crosses 0.
    for (let i = 0; i < x.size; i++)
      x.data[i] = Math.sign(x.data[i]! || 1) * (0.1 + Math.abs(x.data[i]!));
    assertGradsOk(checkLayer(new ReLU(), x, rng));
  });
});

describe('loss gradient checks', () => {
  it('mse', () => {
    const rng = new Rng(3);
    assertGradsOk(
      checkLoss(new MSELoss(), Tensor.randn([BATCH, 3], rng), Tensor.randn([BATCH, 3], rng)),
    );
  });

  it('softmax + cross-entropy', () => {
    const rng = new Rng(4);
    const logits = Tensor.randn([BATCH, 3], rng, 2);
    assertGradsOk(checkLoss(new SoftmaxCrossEntropyLoss(), logits, oneHot([0, 2, 1, 1, 0], 3)));
  });

  it('softmax + cross-entropy stays finite for huge logits', () => {
    const loss = new SoftmaxCrossEntropyLoss();
    const value = loss.forward(Tensor.from([[1000, -1000, 0]]), oneHot([1], 3));
    expect(value).toBeCloseTo(2000, 3);
    expect(Array.from(loss.backward().data).every(Number.isFinite)).toBe(true);
  });

  it('sigmoid + binary cross-entropy', () => {
    const rng = new Rng(7);
    const logits = Tensor.randn([BATCH, 1], rng, 2);
    const labels = Tensor.from([[0], [1], [1], [0], [1]]);
    assertGradsOk(checkLoss(new BCEWithLogitsLoss(), logits, labels));
  });

  it('sigmoid + binary cross-entropy stays finite for huge logits', () => {
    const loss = new BCEWithLogitsLoss();
    // Confidently wrong on both rows: loss ≈ (100 + 100) / 2.
    const value = loss.forward(Tensor.from([[100], [-100]]), Tensor.from([[0], [1]]));
    expect(value).toBeCloseTo(100, 3);
    expect(Array.from(loss.backward().data)).toEqual([0.5, -0.5]);
    // Confidently right: loss ≈ 0, gradient ≈ 0.
    expect(loss.forward(Tensor.from([[100], [-100]]), Tensor.from([[1], [0]]))).toBeLessThan(1e-12);
  });
});

/**
 * End-to-end checks compound float32 rounding through several layers; for
 * small-gradient tensors (e.g. first-layer W behind sigmoid + MSE) the float32
 * noise floor is ~1e-4 at the best eps. Per-layer/loss checks above hold the
 * 1e-4 rule; these use 1e-3, which still catches the 1% bug in the meta-test.
 */
const MODEL_TOL = { tol: 1e-3 };

describe('model gradient checks (end to end)', () => {
  it('dense → tanh → dense → sigmoid with MSE', () => {
    const rng = new Rng(5);
    const model = new Sequential([
      new Dense(2, 4, { init: 'xavier', rng }),
      new Tanh(),
      new Dense(4, 1, { init: 'xavier', rng }),
      new Sigmoid(),
    ]);
    const results = checkModel(
      model,
      new MSELoss(),
      Tensor.randn([BATCH, 2], rng),
      Tensor.randn([BATCH, 1], rng),
      MODEL_TOL,
    );
    expect(results.map((r) => r.name)).toEqual(['input', '0.W', '0.b', '2.W', '2.b']);
    assertGradsOk(results);
  });

  it('dense → tanh → dense with softmax + cross-entropy', () => {
    const rng = new Rng(6);
    const model = new Sequential([
      new Dense(3, 5, { init: 'xavier', rng }),
      new Tanh(),
      new Dense(5, 3, { init: 'xavier', rng }),
    ]);
    const y = oneHot([2, 0, 1, 2, 1], 3);
    assertGradsOk(
      checkModel(model, new SoftmaxCrossEntropyLoss(), Tensor.randn([BATCH, 3], rng), y, MODEL_TOL),
    );
  });
});

describe('the checker itself', () => {
  /** Dense with a 1% error in dW — the checker must catch it. */
  class BrokenDense extends Dense {
    override backward(gradOut: Tensor): Tensor {
      const dx = super.backward(gradOut);
      for (let i = 0; i < this.W.grad.size; i++) this.W.grad.data[i]! *= 1.01;
      return dx;
    }
  }

  it('fails on a subtly wrong gradient, and only for the broken tensor', () => {
    const rng = new Rng(7);
    const results = checkLayer(
      new BrokenDense(3, 4, { init: 'xavier', rng }),
      Tensor.randn([BATCH, 3], rng),
      rng,
    );
    expect(results.filter((r) => !r.pass).map((r) => r.name)).toEqual(['W']);
    expect(() => assertGradsOk(results)).toThrow(/W: relError=/);
  });

  it('restores perturbed values exactly', () => {
    const rng = new Rng(8);
    const layer = new Dense(3, 2, { init: 'xavier', rng });
    const x = Tensor.randn([BATCH, 3], rng);
    const [x0, w0] = [x.data.slice(), layer.W.value.data.slice()];
    checkLayer(layer, x, rng);
    expect(x.data).toEqual(x0);
    expect(layer.W.value.data).toEqual(w0);
  });
});
