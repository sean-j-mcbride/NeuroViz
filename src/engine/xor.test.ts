import { describe, expect, it } from 'vitest';
import { Sigmoid, Tanh } from './layers/activations';
import { Dense } from './layers/dense';
import { MSELoss } from './losses';
import { SGD } from './optim';
import { Rng } from './random';
import { Sequential } from './sequential';
import { Tensor } from './tensor';
import { Trainer } from './train';

const SEED = 42;
const LR = 1;
const STEPS = 3000;

const X = Tensor.from([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
]);
const Y = Tensor.from([[0], [1], [1], [0]]);

/** Builds a fresh 2-4-1 net from `seed` and trains it full-batch on XOR. */
function trainXor(seed: number) {
  const rng = new Rng(seed);
  const model = new Sequential([
    new Dense(2, 4, { init: 'xavier', rng }),
    new Tanh(),
    new Dense(4, 1, { init: 'xavier', rng }),
    new Sigmoid(),
  ]);
  const trainer = new Trainer({ model, loss: new MSELoss(), optimiser: new SGD(LR) });
  const losses = new Float64Array(STEPS);
  for (let i = 0; i < STEPS; i++) losses[i] = trainer.trainStep(X, Y);
  const predictions = Array.from(model.forward(X, false).data);
  const weights = model.params().map((p) => p.value.data.slice());
  return { losses, predictions, weights, finalLoss: losses[STEPS - 1]! };
}

describe('XOR with a 2-4-1 tanh/sigmoid net', () => {
  const run = trainXor(SEED);

  it('trains to near-zero loss', () => {
    console.log(
      `seed=${SEED} lr=${LR} steps=${STEPS} | initial loss=${run.losses[0]!.toFixed(6)} ` +
        `final loss=${run.finalLoss.toExponential(4)} | predictions=[${run.predictions.map((p) => p.toFixed(4)).join(', ')}]`,
    );
    expect(run.finalLoss).toBeLessThan(1e-3);
  });

  it('classifies all four points correctly with a clear margin', () => {
    const targets = Array.from(Y.data);
    run.predictions.forEach((p, i) => expect(Math.abs(p - targets[i]!)).toBeLessThan(0.1));
  });

  it('is deterministic: same seed → bitwise-identical losses and weights', () => {
    const again = trainXor(SEED);
    expect(again.losses).toEqual(run.losses);
    expect(again.weights).toEqual(run.weights);
    expect(again.finalLoss).toBe(run.finalLoss);
  });

  it('a different seed gives a different run (the seed is actually used)', () => {
    expect(trainXor(SEED + 1).losses).not.toEqual(run.losses);
  });
});
