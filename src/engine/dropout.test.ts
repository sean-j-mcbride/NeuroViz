import { describe, expect, it } from 'vitest';
import { layerFromConfig } from './build';
import { Tanh } from './layers/activations';
import { Dense } from './layers/dense';
import { Dropout } from './layers/dropout';
import type { Layer } from './layers/types';
import { BCEWithLogitsLoss } from './losses';
import { SGD } from './optim';
import { Rng } from './random';
import { Sequential } from './sequential';
import { Tensor } from './tensor';
import { Trainer } from './train';

describe('Dropout', () => {
  it('is the identity in evaluation mode and at rate 0, drawing no random numbers', () => {
    const x = Tensor.randn([4, 3], new Rng(1));
    const rng = new Rng(2);
    const before = new Rng(2).next();
    expect(new Dropout(0.5, rng).forward(x, false)).toBe(x);
    expect(new Dropout(0, rng).forward(x, true)).toBe(x);
    const g = Tensor.randn([4, 3], new Rng(3));
    const d = new Dropout(0.5, rng);
    d.forward(x, false);
    expect(d.backward(g)).toBe(g);
    expect(rng.next()).toBe(before); // untouched
  });

  it('drops ≈ rate of the elements and scales survivors so E[y] = x', () => {
    const n = 20000;
    const x = new Tensor(new Float32Array(n).fill(1), [n, 1]);
    const y = new Dropout(0.3, new Rng(4)).forward(x, true);
    let zeros = 0;
    let sum = 0;
    for (const v of y.data) {
      if (v === 0) zeros++;
      else expect(v).toBeCloseTo(1 / 0.7, 6);
      sum += v;
    }
    expect(zeros / n).toBeCloseTo(0.3, 1.5);
    expect(sum / n).toBeCloseTo(1, 1.5);
  });

  it('passes gradients through the same mask', () => {
    const d = new Dropout(0.5, new Rng(5));
    const x = new Tensor(new Float32Array(100).fill(1), [10, 10]);
    const y = d.forward(x, true).data.slice();
    const g = d.backward(new Tensor(new Float32Array(100).fill(1), [10, 10]));
    expect(Array.from(g.data)).toEqual(Array.from(y));
  });

  it('is reproducible from its seed', () => {
    const x = Tensor.randn([5, 5], new Rng(6));
    const a = new Dropout(0.4, new Rng(7)).forward(x, true).data.slice();
    const b = new Dropout(0.4, new Rng(7)).forward(x, true).data.slice();
    expect(b).toEqual(a);
  });

  it('round-trips through layerFromConfig without consuming the init rng', () => {
    const rng = new Rng(8);
    const layer = layerFromConfig({ kind: 'dropout', rate: 0.2 }, rng);
    expect(layer.toJSON()).toEqual({ kind: 'dropout', rate: 0.2 });
    expect(rng.next()).toBe(new Rng(8).next());
  });

  it('at rate 0, a model with dropout layers trains bitwise identically to one without', () => {
    const run = (withDropout: boolean) => {
      const rng = new Rng(12);
      const layers: Layer[] = [new Dense(2, 4, { init: 'xavier', rng }), new Tanh()];
      if (withDropout) layers.push(new Dropout(0, rng));
      layers.push(new Dense(4, 1, { init: 'xavier', rng }));
      const model = new Sequential(layers);
      const trainer = new Trainer({
        model,
        loss: new BCEWithLogitsLoss(),
        optimiser: new SGD(0.5),
      });
      const x = Tensor.randn([8, 2], new Rng(13));
      const y = new Tensor(Float32Array.from([0, 1, 1, 0, 1, 0, 0, 1]), [8, 1]);
      for (let i = 0; i < 50; i++) trainer.trainStep(x, y);
      return model.params().map((p) => p.value.data.slice());
    };
    expect(run(true)).toEqual(run(false));
  });
});
