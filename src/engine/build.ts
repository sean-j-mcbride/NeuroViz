import { ReLU, Sigmoid, Tanh } from './layers/activations';
import { Conv2D } from './layers/conv2d';
import { Dense } from './layers/dense';
import { Dropout } from './layers/dropout';
import { Flatten } from './layers/flatten';
import { MaxPool2D } from './layers/maxpool2d';
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
    case 'conv2d':
      return new Conv2D(config.inChannels, config.outChannels, config.kernel, {
        init: config.init,
        rng,
        stride: config.stride,
        padding: config.padding,
      });
    case 'maxpool2d':
      return new MaxPool2D(config.size, config.stride);
    case 'flatten':
      return new Flatten();
    case 'sequential':
      return new Sequential(config.layers.map((c) => layerFromConfig(c, rng)));
  }
}
