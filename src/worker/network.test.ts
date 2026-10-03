import { describe, expect, it } from 'vitest';
import {
  type ConvLayerSpec,
  type MnistNetworkSpec,
  checkConvStack,
  convRowEnds,
  mnistNetworkToLayerConfig,
  networkToLayerConfig,
} from './network';

const conv = (filters: number, kernel: number, extra: Partial<ConvLayerSpec> = {}): ConvLayerSpec =>
  ({
    kind: 'conv',
    filters,
    kernel,
    stride: 1,
    padding: 'valid',
    activation: 'relu',
    ...extra,
  }) as ConvLayerSpec;
const pool: ConvLayerSpec = { kind: 'pool', size: 2 };

describe('MNIST network specs', () => {
  it('with no conv rows, the model is exactly the Phase 5 MLP', () => {
    const spec: MnistNetworkSpec = { conv: [], hidden: [{ units: 32, activation: 'tanh' }] };
    expect(mnistNetworkToLayerConfig(spec, 0.2)).toEqual(networkToLayerConfig(spec, 0.2, 784, 10));
  });

  it('conv rows become conv + activation or max-pool, then flatten and the dense layers', () => {
    const spec: MnistNetworkSpec = {
      conv: [conv(8, 5), pool, conv(4, 3, { activation: 'linear', padding: 'same', stride: 2 })],
      hidden: [{ units: 16, activation: 'tanh' }],
    };
    const config = mnistNetworkToLayerConfig(spec, 0.1);
    expect(config).toEqual({
      kind: 'sequential',
      layers: [
        {
          kind: 'conv2d',
          inChannels: 1,
          outChannels: 8,
          kernel: 5,
          stride: 1,
          padding: 'valid',
          init: 'he',
        },
        { kind: 'relu' },
        { kind: 'maxpool2d', size: 2, stride: 2 },
        // 8 × 12 × 12 → stride 2, same: 4 × 6 × 6 = 144.
        {
          kind: 'conv2d',
          inChannels: 8,
          outChannels: 4,
          kernel: 3,
          stride: 2,
          padding: 'same',
          init: 'xavier',
        },
        { kind: 'flatten' },
        { kind: 'dense', inFeatures: 144, outFeatures: 16, init: 'xavier' },
        { kind: 'tanh' },
        { kind: 'dropout', rate: 0.1 },
        { kind: 'dense', inFeatures: 16, outFeatures: 10, init: 'xavier' },
      ],
    });
    // The rows' feature maps come after: relu (1), pool (2), the linear conv itself (3).
    expect(convRowEnds(spec.conv)).toEqual([1, 2, 3]);
  });

  it('checkConvStack gives each row’s shape and the flattened size', () => {
    expect(checkConvStack([conv(8, 5), pool, conv(16, 5), pool])).toEqual({
      ok: true,
      shapes: [
        [8, 24, 24],
        [8, 12, 12],
        [16, 8, 8],
        [16, 4, 4],
      ],
      flat: 256,
    });
    expect(checkConvStack([])).toEqual({ ok: true, shapes: [], flat: 784 });
  });

  it('checkConvStack names the first row that doesn’t fit, with the shapes before it', () => {
    expect(checkConvStack([conv(8, 5), pool, conv(8, 5), pool, pool, pool, conv(4, 3)])).toEqual({
      ok: false,
      shapes: [
        [8, 24, 24],
        [8, 12, 12],
        [8, 8, 8],
        [8, 4, 4],
        [8, 2, 2],
        [8, 1, 1],
      ],
      row: 6,
      message: 'a 3 × 3 kernel doesn’t fit a 1 × 1 input',
    });
    expect(checkConvStack([pool, pool, pool, pool, pool])).toMatchObject({
      ok: false,
      row: 4,
      message: 'a 2 × 2 pool doesn’t fit a 1 × 1 input',
    });
    expect(() =>
      mnistNetworkToLayerConfig({ conv: [pool, pool, pool, pool, pool], hidden: [] }),
    ).toThrow('Conv layer 5: a 2 × 2 pool doesn’t fit a 1 × 1 input');
  });
});
