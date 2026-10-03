import type { LayerConfig, Padding } from './layers/types';
import type { Shape } from './tensor';

/** Zeros added on each side of the input for a kernel of size `kernel`. */
export function paddingAmount(kernel: number, padding: Padding): number {
  return padding === 'same' ? Math.floor((kernel - 1) / 2) : 0;
}

/**
 * Output length along one axis of a sliding window (conv or pool): the
 * number of positions where a `window`-wide window fits, every `stride`
 * steps, in `size + 2·pad`. Zero or less means it does not fit at all.
 */
export function slidingOutput(size: number, window: number, stride: number, pad = 0): number {
  return Math.floor((size + 2 * pad - window) / stride) + 1;
}

export type ShapeResult = { shape: Shape } | { error: string };

const times = (s: Shape) => s.join(' × ');

/**
 * The output shape of one layer for a per-example input shape (without the
 * batch dimension): `[features]` for flat data, `[channels, height, width]`
 * for images. Returns a readable reason when the layer cannot take the input.
 */
export function outputShape(config: LayerConfig, input: Shape): ShapeResult {
  switch (config.kind) {
    case 'relu':
    case 'tanh':
    case 'sigmoid':
    case 'dropout':
      return { shape: input };
    case 'flatten':
      return { shape: [input.reduce((a, b) => a * b, 1)] };
    case 'dense':
      if (input.length !== 1) {
        return { error: `a dense layer needs a flat input, not ${times(input)}` };
      }
      if (input[0] !== config.inFeatures) {
        return { error: `expects ${config.inFeatures} inputs but gets ${input[0]}` };
      }
      return { shape: [config.outFeatures] };
    case 'conv2d': {
      if (input.length !== 3) {
        return {
          error: `a convolution needs a channels × height × width input, not ${times(input)}`,
        };
      }
      const [c, h, w] = input as [number, number, number];
      if (c !== config.inChannels) {
        return { error: `expects ${config.inChannels} input channels but gets ${c}` };
      }
      const { kernel: k, stride } = config;
      const pad = paddingAmount(k, config.padding);
      const oh = slidingOutput(h, k, stride, pad);
      const ow = slidingOutput(w, k, stride, pad);
      // 'same' padding with an odd kernel always fits, so only 'valid' can fail here.
      if (oh < 1 || ow < 1)
        return { error: `a ${k} × ${k} kernel doesn’t fit a ${h} × ${w} input` };
      return { shape: [config.outChannels, oh, ow] };
    }
    case 'maxpool2d': {
      if (input.length !== 3) {
        return { error: `pooling needs a channels × height × width input, not ${times(input)}` };
      }
      const [c, h, w] = input as [number, number, number];
      const { size, stride } = config;
      const oh = slidingOutput(h, size, stride);
      const ow = slidingOutput(w, size, stride);
      if (oh < 1 || ow < 1) {
        return { error: `a ${size} × ${size} pool doesn’t fit a ${h} × ${w} input` };
      }
      return { shape: [c, oh, ow] };
    }
    case 'sequential': {
      const r = inferShapes(config.layers, input);
      if ('error' in r) return { error: `layer ${r.error.index + 1}: ${r.error.message}` };
      return { shape: r.shapes.at(-1) ?? input };
    }
  }
}

export type InferResult =
  { shapes: Shape[] } | { error: { index: number; message: string }; shapes: Shape[] };

/**
 * Every layer's per-example output shape for a per-example `input` shape, or
 * the first layer that cannot take its input (with the shapes up to it).
 */
export function inferShapes(layers: readonly LayerConfig[], input: Shape): InferResult {
  const shapes: Shape[] = [];
  let shape = input;
  for (let index = 0; index < layers.length; index++) {
    const r = outputShape(layers[index]!, shape);
    if ('error' in r) return { error: { index, message: r.error }, shapes };
    shape = r.shape;
    shapes.push(shape);
  }
  return { shapes };
}

/** Weights and biases of one layer, and the multiply-adds of its forward pass per example. */
export function layerCost(
  config: LayerConfig,
  output: Shape,
): { params: number; multiplyAdds: number } {
  switch (config.kind) {
    case 'dense': {
      const { inFeatures: i, outFeatures: o } = config;
      return { params: i * o + o, multiplyAdds: i * o };
    }
    case 'conv2d': {
      const taps = config.inChannels * config.kernel * config.kernel;
      const f = config.outChannels;
      const positions = (output[1] ?? 0) * (output[2] ?? 0);
      return { params: taps * f + f, multiplyAdds: positions * taps * f };
    }
    default:
      return { params: 0, multiplyAdds: 0 };
  }
}
