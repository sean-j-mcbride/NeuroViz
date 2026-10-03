import { describe, expect, it } from 'vitest';
import type { ConvFilters, ConvLayerSpec } from '../worker';
import { channelStats, filterStats, kernelOf, mapNormalisation, rowTitle } from './layerViews';

const rows: ConvLayerSpec[] = [
  { kind: 'conv', filters: 2, kernel: 3, stride: 1, padding: 'valid', activation: 'tanh' },
  { kind: 'pool', size: 2 },
  { kind: 'conv', filters: 2, kernel: 3, stride: 1, padding: 'valid', activation: 'relu' },
  { kind: 'pool', size: 2 },
];

describe('layer views', () => {
  it('numbers rows within their kind', () => {
    expect(rows.map((_, i) => rowTitle(rows, i))).toEqual(['Conv 1', 'Pool 1', 'Conv 2', 'Pool 2']);
  });

  it('colours a pool map by the activation of the conv before it', () => {
    const v = Float32Array.from([0, 2, 4]);
    expect(mapNormalisation(rows, 1, v)).toEqual({ scale: 1, offset: 0 }); // tanh: fixed
    expect(mapNormalisation(rows, 3, v)).toEqual({ scale: 0.25, offset: 0 }); // relu: by max |v|
  });

  it('reads kernels and per-filter stats from the [inC·k², outC] layout', () => {
    // 2 input channels, 1×1 kernels, 2 filters: W rows are (c0, c1), columns (f0, f1).
    const f: ConvFilters = {
      row: 0,
      inChannels: 2,
      outChannels: 2,
      kernel: 1,
      stride: 1,
      padding: 'valid',
      W: Float32Array.from([1, -2, 3, 4]),
      b: Float32Array.from([0.5, -0.5]),
    };
    expect(Array.from(kernelOf(f, 1, 0))).toEqual([-2]);
    expect(Array.from(kernelOf(f, 0, 1))).toEqual([3]);
    expect(filterStats(f)).toEqual([
      { min: 1, max: 3, rms: Math.sqrt(5), bias: 0.5 },
      { min: -2, max: 4, rms: Math.sqrt(10), bias: -0.5 },
    ]);
  });

  it('summarises each channel of a feature map', () => {
    const data = Float32Array.from([0, 0, 2, 2, 1, 1, 1, 5]);
    expect(channelStats(data, [2, 2, 2])).toEqual([
      { mean: 1, max: 2, zeros: 0.5 },
      { mean: 2, max: 5, zeros: 0 },
    ]);
  });
});
