import { describe, expect, it } from 'vitest';
import { type Layer, Rng, Sequential, Tensor, layerFromConfig } from '../engine';
import {
  type ConvLayerSpec,
  MNIST_IMAGE_SHAPE,
  checkConvStack,
  convRowEnds,
  mnistNetworkToLayerConfig,
} from '../worker';
import { describeRect, receptiveField } from './receptiveField';

const conv = (kernel: number, stride = 1, padding: 'valid' | 'same' = 'valid'): ConvLayerSpec => ({
  kind: 'conv',
  filters: 2,
  kernel,
  stride,
  padding,
  activation: 'linear',
});
const pool: ConvLayerSpec = { kind: 'pool', size: 2 };

function shapesOf(rows: ConvLayerSpec[]) {
  const check = checkConvStack(rows);
  if (!check.ok) throw new Error(check.message);
  return check.shapes;
}

describe('receptive field', () => {
  it('worked example: 5×5 valid → pool → 5×5 valid → pool (LeNet-style)', () => {
    const rows = [conv(5), pool, conv(5), pool];
    const shapes = shapesOf(rows);
    // Pool 2 (row 3) pixel (1, 2) ← conv 2 rows 2–3, cols 4–5 ← pool 1 rows 2–7, cols 4–9
    // ← conv 1 rows 4–15, cols 8–19 ← input rows 4–19, cols 8–23.
    expect(receptiveField(rows, shapes, MNIST_IMAGE_SHAPE, 3, 1, 2)).toEqual({
      input: { y0: 4, y1: 19, x0: 8, x1: 23 },
      maps: [
        { y0: 4, y1: 15, x0: 8, x1: 19 },
        { y0: 2, y1: 7, x0: 4, x1: 9 },
        { y0: 2, y1: 3, x0: 4, x1: 5 },
      ],
    });
  });

  it('same padding is clipped at the edges', () => {
    const rows = [conv(3, 1, 'same'), conv(3, 2, 'same')];
    const shapes = shapesOf(rows);
    expect(receptiveField(rows, shapes, MNIST_IMAGE_SHAPE, 1, 0, 13).input).toEqual({
      y0: 0,
      y1: 2,
      x0: 24,
      x1: 27,
    });
  });

  it('describes a rectangle in words', () => {
    expect(describeRect({ y0: 8, y1: 19, x0: 12, x1: 23 })).toBe(
      'rows 8–19, columns 12–23 (12 × 12 pixels)',
    );
  });

  /**
   * Brute force through the engine: nudge each input pixel and see which
   * input pixels change the chosen feature-map pixel. With linear
   * activations and random weights, a conv stack depends on exactly the
   * receptive field; through max-pooling, only on part of it.
   */
  function dependsOn(rows: ConvLayerSpec[], row: number, y: number, x: number): Set<number> {
    const config = mnistNetworkToLayerConfig({ conv: rows, hidden: [] });
    const model = layerFromConfig(config, new Rng(3)) as Sequential;
    for (const p of model.params()) p.value.data.set(Tensor.randn(p.value.shape, new Rng(4)).data);
    const end = convRowEnds(rows)[row]!;
    const layers: Layer[] = model.layers.slice(0, end + 1);
    const [, h, w] = shapesOf(rows)[row] as [number, number, number];
    const valueAt = (img: Float32Array) => {
      let t = new Tensor(img.slice(), [1, ...MNIST_IMAGE_SHAPE]);
      for (const l of layers) t = l.forward(t, false);
      const out: number[] = [];
      for (let c = 0; c < t.shape[1]!; c++) out.push(t.data[(c * h + y) * w + x]!);
      return out;
    };
    const base = Tensor.randn([784], new Rng(5)).data;
    const ref = valueAt(base);
    const deps = new Set<number>();
    for (let p = 0; p < 784; p++) {
      const img = base.slice();
      img[p]! += 10;
      if (valueAt(img).some((v, c) => v !== ref[c])) deps.add(p);
    }
    return deps;
  }

  const pixelsOf = ({ y0, y1, x0, x1 }: { y0: number; y1: number; x0: number; x1: number }) => {
    const s = new Set<number>();
    for (let yy = y0; yy <= y1; yy++) for (let xx = x0; xx <= x1; xx++) s.add(yy * 28 + xx);
    return s;
  };

  it.each([
    ['5×5 valid → 3×3 stride 2', [conv(5), conv(3, 2)], 1, 4, 7],
    ['3×3 same → 3×3 same, at the corner', [conv(3, 1, 'same'), conv(3, 1, 'same')], 1, 0, 0],
    ['5×5 stride 2 same → 3×3 valid, at the far edge', [conv(5, 2, 'same'), conv(3)], 1, 11, 11],
  ] as const)('conv stacks depend on exactly the field: %s', (_, rows, row, y, x) => {
    const field = receptiveField(rows, shapesOf([...rows]), MNIST_IMAGE_SHAPE, row, y, x);
    expect(dependsOn([...rows], row, y, x)).toEqual(pixelsOf(field.input));
  });

  it('through max-pooling, everything the pixel depends on is inside the field', () => {
    const rows = [conv(5), pool, conv(3), pool];
    const field = receptiveField(rows, shapesOf(rows), MNIST_IMAGE_SHAPE, 3, 2, 1);
    const inside = pixelsOf(field.input);
    const deps = dependsOn(rows, 3, 2, 1);
    expect(deps.size).toBeGreaterThan(0);
    for (const p of deps) expect(inside.has(p)).toBe(true);
  });
});
