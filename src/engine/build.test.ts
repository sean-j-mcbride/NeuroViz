import { describe, expect, it } from 'vitest';
import { layerFromConfig } from './build';
import type { LayerConfig } from './layers/types';
import { Rng } from './random';
import { Sequential } from './sequential';
import { Tensor } from './tensor';

const CONFIG: LayerConfig = {
  kind: 'sequential',
  layers: [
    { kind: 'dense', inFeatures: 2, outFeatures: 3, init: 'he' },
    { kind: 'relu' },
    { kind: 'dense', inFeatures: 3, outFeatures: 3, init: 'xavier' },
    { kind: 'tanh' },
    { kind: 'dense', inFeatures: 3, outFeatures: 1, init: 'xavier' },
    { kind: 'sigmoid' },
  ],
};

describe('layerFromConfig', () => {
  it('round-trips through toJSON', () => {
    const model = layerFromConfig(CONFIG, new Rng(1));
    expect(model).toBeInstanceOf(Sequential);
    expect(model.toJSON()).toEqual(CONFIG);
  });

  it('builds a working model, initialised deterministically from the rng', () => {
    const a = layerFromConfig(CONFIG, new Rng(9));
    const b = layerFromConfig(CONFIG, new Rng(9));
    const x = Tensor.from([
      [0.5, -1],
      [2, 0.25],
    ]);
    const ya = a.forward(x, false);
    expect(ya.shape).toEqual([2, 1]);
    expect(Array.from(b.forward(x, false).data)).toEqual(Array.from(ya.data));
  });
});
