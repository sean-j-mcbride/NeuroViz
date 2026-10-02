import { ReLU, Sigmoid, Tanh } from './layers/activations';
import { Dense } from './layers/dense';
import { Dropout } from './layers/dropout';
import type { Layer, LayerConfig } from './layers/types';
import type { Rng } from './random';
import { Sequential } from './sequential';

/**
 * Builds a freshly initialised layer (or model) from its serialisable config.
 * Dropout layers keep `rng` for their masks but draw nothing at construction,
 * so adding them does not change the initial weights.
 */
export function layerFromConfig(config: LayerConfig, rng: Rng): Layer {
  switch (config.kind) {
    case 'dense':
      return new Dense(config.inFeatures, config.outFeatures, { init: config.init, rng });
    case 'relu':
      return new ReLU();
    case 'tanh':
      return new Tanh();
    case 'sigmoid':
      return new Sigmoid();
    case 'dropout':
      return new Dropout(config.rate, rng);
    case 'sequential':
      return new Sequential(config.layers.map((c) => layerFromConfig(c, rng)));
  }
}
