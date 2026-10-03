import { describe, expect, it } from 'vitest';
import { assertGradsOk, checkLayer, checkModel } from './gradcheck';
import { Tanh } from './layers/activations';
import { Conv2D } from './layers/conv2d';
import { Dense } from './layers/dense';
import { Flatten } from './layers/flatten';
import { MaxPool2D } from './layers/maxpool2d';
import type { LayerConfig, Padding } from './layers/types';
import { layerFromConfig } from './build';
import { SoftmaxCrossEntropyLoss } from './losses';
import { Rng } from './random';
import { Sequential } from './sequential';
import { inferShapes, layerCost, outputShape } from './shapes';
import { Tensor, reshape } from './tensor';

/** Fills every parameter (including biases) with fresh noise so every path is exercised. */
function randomise(params: { value: Tensor }[], rng: Rng): void {
  for (const p of params) p.value.data.set(Tensor.randn(p.value.shape, rng).data);
}

/** The definition of a convolution, one output at a time, as a reference. */
function directConv(x: Tensor, conv: Conv2D): Float32Array {
  const [n, c, h, w] = x.shape as [number, number, number, number];
  const { outChannels: f, kernel: k, stride: s } = conv;
  const pad = conv.padding === 'same' ? (k - 1) / 2 : 0;
  const oh = Math.floor((h + 2 * pad - k) / s) + 1;
  const ow = Math.floor((w + 2 * pad - k) / s) + 1;
  const W = conv.W.value.data;
  const out = new Float32Array(n * f * oh * ow);
  let o = 0;
  for (let b = 0; b < n; b++)
    for (let fi = 0; fi < f; fi++)
      for (let oy = 0; oy < oh; oy++)
        for (let ox = 0; ox < ow; ox++) {
          let sum = 0;
          for (let ci = 0; ci < c; ci++)
            for (let ky = 0; ky < k; ky++)
              for (let kx = 0; kx < k; kx++) {
                const iy = oy * s - pad + ky;
                const ix = ox * s - pad + kx;
                if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
                sum +=
                  x.data[((b * c + ci) * h + iy) * w + ix]! *
                  W[(ci * k * k + ky * k + kx) * f + fi]!;
              }
          out[o++] = Math.fround(Math.fround(sum) + conv.b.value.data[fi]!);
        }
  return out;
}

const CONVS: [string, number, number, number, number, Padding, [number, number]][] = [
  // name, inC, outC, kernel, stride, padding, [H, W]
  ['1→3, 3×3, valid', 1, 3, 3, 1, 'valid', [5, 5]],
  ['2→4, 5×5, same', 2, 4, 5, 1, 'same', [6, 6]],
  ['2→3, 3×3, stride 2, same', 2, 3, 3, 2, 'same', [7, 7]],
  ['1→2, 3×3, stride 2, valid, non-square', 1, 2, 3, 2, 'valid', [7, 6]],
];

describe('Conv2D', () => {
  it.each(CONVS)(
    '%s: forward matches the direct definition bitwise',
    (_, c, f, k, s, pad, [h, w]) => {
      const rng = new Rng(1);
      const conv = new Conv2D(c, f, k, { init: 'he', rng, stride: s, padding: pad });
      randomise(conv.params(), rng);
      const x = Tensor.randn([3, c, h, w], rng);
      // Zeros exercise the matmul kernels' zero-skipping.
      for (let i = 0; i < x.size; i += 3) x.data[i] = 0;
      expect(conv.forward(x, true).data).toEqual(directConv(x, conv));
    },
  );

  it.each(CONVS)('%s: gradient check', (_, c, f, k, s, pad, [h, w]) => {
    const rng = new Rng(2);
    const conv = new Conv2D(c, f, k, { init: 'xavier', rng, stride: s, padding: pad });
    randomise(conv.params(), rng);
    assertGradsOk(checkLayer(conv, Tensor.randn([2, c, h, w], rng), rng));
  });

  it('output shapes agree with outputShape', () => {
    for (const [, c, f, k, s, padding, [h, w]] of CONVS) {
      const conv = new Conv2D(c, f, k, { init: 'he', rng: new Rng(0), stride: s, padding });
      const out = conv.forward(Tensor.zeros([2, c, h, w]), false);
      const inferred = outputShape(conv.toJSON(), [c, h, w]);
      expect('shape' in inferred && [2, ...inferred.shape]).toEqual(out.shape);
    }
  });

  it('He init: std √(2 / fanIn) with fanIn = inC·k², weights [inC·k², outC]', () => {
    const conv = new Conv2D(4, 50, 5, { init: 'he', rng: new Rng(3) });
    expect(conv.W.value.shape).toEqual([100, 50]);
    const d = conv.W.value.data;
    const std = Math.sqrt(d.reduce((a, v) => a + v * v, 0) / d.length);
    expect(std).toBeCloseTo(Math.sqrt(2 / 100), 2);
    expect(conv.b.value.data.every((v) => v === 0)).toBe(true);
  });

  it('with inputGrad off, parameter gradients are unchanged and the input gradient is zeros', () => {
    const make = () => new Conv2D(2, 3, 3, { init: 'he', rng: new Rng(4), padding: 'same' });
    const rng = new Rng(5);
    const x = Tensor.randn([2, 2, 5, 5], rng);
    const g = Tensor.randn([2, 3, 5, 5], rng);
    const on = make();
    on.forward(x, true);
    on.backward(g);
    const off = make();
    off.inputGrad = false;
    off.forward(x, true);
    expect(off.backward(g).data.every((v) => v === 0)).toBe(true);
    expect(off.W.grad.data).toEqual(on.W.grad.data);
    expect(off.b.grad.data).toEqual(on.b.grad.data);
  });

  it('chunking a big batch (to bound im2col memory) keeps outputs and dx bitwise, dW to rounding', () => {
    const make = (colBudget?: number) =>
      new Conv2D(2, 3, 3, {
        init: 'he',
        rng: new Rng(12),
        padding: 'same',
        ...(colBudget && { colBudget }),
      });
    const rng = new Rng(13);
    const x = Tensor.randn([5, 2, 4, 4], rng);
    const g = Tensor.randn([5, 3, 4, 4], rng);
    const whole = make();
    const yWhole = whole.forward(x, true).data.slice();
    const dxWhole = whole.backward(g).data.slice();
    // 2 · 16 · 18 = 576 elements per image: a budget of 1,200 means chunks of 2, 2 and 1 images.
    const chunked = make(1200);
    expect(chunked.forward(x, true).data).toEqual(yWhole);
    expect(chunked.backward(g).data).toEqual(dxWhole);
    for (const [a, b] of [
      [chunked.W.grad, whole.W.grad],
      [chunked.b.grad, whole.b.grad],
    ] as const) {
      a.data.forEach((v, i) => expect(v).toBeCloseTo(b.data[i]!, 5));
    }
    const rng2 = new Rng(14);
    assertGradsOk(checkLayer(make(1200), Tensor.randn([3, 2, 4, 4], rng2), rng2));
  });

  it('round-trips through its config', () => {
    const conv = new Conv2D(2, 3, 5, {
      init: 'xavier',
      rng: new Rng(6),
      stride: 2,
      padding: 'same',
    });
    const config = conv.toJSON();
    expect(config).toEqual({
      kind: 'conv2d',
      inChannels: 2,
      outChannels: 3,
      kernel: 5,
      stride: 2,
      padding: 'same',
      init: 'xavier',
    });
    const rebuilt = layerFromConfig(config, new Rng(6)) as Conv2D;
    expect(rebuilt.W.value.data).toEqual(conv.W.value.data);
  });

  it('rejects input with the wrong channels', () => {
    const conv = new Conv2D(2, 3, 3, { init: 'he', rng: new Rng(0) });
    expect(() => conv.forward(Tensor.zeros([1, 1, 5, 5]), false)).toThrow(/\[N, 2, H, W\]/);
  });
});

/** Distinct values far apart (0.05 ≫ 2·eps), shuffled, so no pooling window is near a tie. */
function spreadOut(shape: number[], rng: Rng): Tensor {
  const t = Tensor.zeros(shape);
  const order = Array.from({ length: t.size }, (_, i) => i);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(rng.next() * (i + 1));
    [order[i], order[j]] = [order[j]!, order[i]!];
  }
  order.forEach((v, i) => (t.data[i] = (v - t.size / 2) * 0.05));
  return t;
}

describe('MaxPool2D', () => {
  it('takes the largest of each window, dropping a ragged edge', () => {
    // prettier-ignore
    const x = new Tensor(Float32Array.from([
      1, 2, 0, 9, 5,
      3, 4, 1, 0, 5,
      0, 0, 7, 8, 5,
      6, 0, 0, 0, 5,
      5, 5, 5, 5, 5,
    ]), [1, 1, 5, 5]);
    const out = new MaxPool2D(2).forward(x, false);
    expect(out.shape).toEqual([1, 1, 2, 2]);
    expect(Array.from(out.data)).toEqual([4, 9, 6, 8]);
  });

  it('routes each gradient to its window’s maximum (the first, on a tie)', () => {
    const pool = new MaxPool2D(2);
    const x = new Tensor(Float32Array.from([1, 3, 3, 0]), [1, 1, 2, 2]);
    pool.forward(x, true);
    const dx = pool.backward(new Tensor(Float32Array.from([5]), [1, 1, 1, 1]));
    expect(Array.from(dx.data)).toEqual([0, 5, 0, 0]);
  });

  it('overlapping windows (stride < size) add their gradients', () => {
    const pool = new MaxPool2D(2, 1);
    const x = new Tensor(Float32Array.from([0, 0, 0, 0, 9, 0, 0, 0, 0]), [1, 1, 3, 3]);
    expect(Array.from(pool.forward(x, true).data)).toEqual([9, 9, 9, 9]);
    const dx = pool.backward(new Tensor(Float32Array.from([1, 2, 3, 4]), [1, 1, 2, 2]));
    expect(dx.data[4]).toBe(10);
  });

  it.each([
    ['2×2, stride 2', 2, 2, [2, 3, 6, 6]],
    ['3×3, stride 2, odd edge', 3, 2, [2, 2, 7, 6]],
  ] as const)('gradient check: %s', (_, size, stride, shape) => {
    const rng = new Rng(7);
    assertGradsOk(checkLayer(new MaxPool2D(size, stride), spreadOut([...shape], rng), rng));
  });
});

describe('Flatten', () => {
  it('is a view both ways, channel by channel then row by row', () => {
    const flat = new Flatten();
    const x = Tensor.randn([2, 3, 2, 2], new Rng(8));
    const y = flat.forward(x, true);
    expect(y.shape).toEqual([2, 12]);
    expect(y.data).toBe(x.data);
    const g = Tensor.randn([2, 12], new Rng(9));
    const dx = flat.backward(g);
    expect(dx.shape).toEqual([2, 3, 2, 2]);
    expect(dx.data).toBe(g.data);
  });

  it('gradient check', () => {
    const rng = new Rng(10);
    assertGradsOk(checkLayer(new Flatten(), Tensor.randn([2, 2, 3, 3], rng), rng));
  });

  it('reshape refuses a different size', () => {
    expect(() => reshape(Tensor.zeros([2, 3]), [7])).toThrow(/cannot view \[2, 3\] as \[7\]/);
  });
});

describe('a small CNN, end to end', () => {
  it('conv → tanh → pool → conv → tanh → flatten → dense with softmax + cross-entropy', () => {
    const rng = new Rng(11);
    const model = new Sequential([
      new Conv2D(1, 3, 3, { init: 'xavier', rng, padding: 'same' }),
      new Tanh(),
      new MaxPool2D(2),
      new Conv2D(3, 4, 3, { init: 'xavier', rng }),
      new Tanh(),
      new Flatten(),
      new Dense(4, 3, { init: 'xavier', rng }),
    ]);
    randomise(model.params(), rng);
    const x = Tensor.randn([3, 1, 6, 6], rng);
    const y = Tensor.zeros([3, 3]);
    [2, 0, 1].forEach((c, r) => (y.data[r * 3 + c] = 1));
    const results = checkModel(model, new SoftmaxCrossEntropyLoss(), x, y, { tol: 1e-3 });
    expect(results.map((r) => r.name)).toEqual(['input', '0.W', '0.b', '3.W', '3.b', '6.W', '6.b']);
    assertGradsOk(results);
  });
});

describe('shape inference', () => {
  const conv = (
    inChannels: number,
    outChannels: number,
    kernel: number,
    padding: Padding = 'valid',
    stride = 1,
  ): LayerConfig => ({
    kind: 'conv2d',
    inChannels,
    outChannels,
    kernel,
    stride,
    padding,
    init: 'he',
  });
  const pool: LayerConfig = { kind: 'maxpool2d', size: 2, stride: 2 };

  it('follows a LeNet-style stack', () => {
    const r = inferShapes(
      [
        conv(1, 8, 5),
        { kind: 'relu' },
        pool,
        conv(8, 16, 5),
        { kind: 'relu' },
        pool,
        { kind: 'flatten' },
        { kind: 'dense', inFeatures: 256, outFeatures: 10, init: 'xavier' },
      ],
      [1, 28, 28],
    );
    expect(r).toEqual({
      shapes: [
        [8, 24, 24],
        [8, 24, 24],
        [8, 12, 12],
        [16, 8, 8],
        [16, 8, 8],
        [16, 4, 4],
        [256],
        [10],
      ],
    });
  });

  it('same padding keeps the size; stride 2 halves it, rounding up', () => {
    expect(outputShape(conv(1, 4, 3, 'same'), [1, 28, 28])).toEqual({ shape: [4, 28, 28] });
    expect(outputShape(conv(1, 4, 5, 'same', 2), [1, 7, 7])).toEqual({ shape: [4, 4, 4] });
    expect(outputShape(pool, [4, 7, 7])).toEqual({ shape: [4, 3, 3] });
  });

  it('names the first layer that cannot take its input', () => {
    const r = inferShapes([conv(1, 8, 5), pool, conv(8, 8, 5), pool, conv(8, 8, 5)], [1, 28, 28]);
    expect(r).toMatchObject({
      error: { index: 4, message: 'a 5 × 5 kernel doesn’t fit a 4 × 4 input' },
      shapes: [
        [8, 24, 24],
        [8, 12, 12],
        [8, 8, 8],
        [8, 4, 4],
      ],
    });
  });

  it.each([
    [conv(1, 4, 3, 'valid', 2), [1, 2, 5], 'a 3 × 3 kernel doesn’t fit a 2 × 5 input'],
    [conv(2, 4, 3), [1, 5, 5], 'expects 2 input channels but gets 1'],
    [conv(1, 4, 3), [784], 'a convolution needs a channels × height × width input, not 784'],
    [pool, [3, 1, 4], 'a 2 × 2 pool doesn’t fit a 1 × 4 input'],
    [pool, [10], 'pooling needs a channels × height × width input, not 10'],
    [
      { kind: 'dense', inFeatures: 3, outFeatures: 2, init: 'he' },
      [2, 2, 2],
      'a dense layer needs a flat input, not 2 × 2 × 2',
    ],
    [
      { kind: 'dense', inFeatures: 3, outFeatures: 2, init: 'he' },
      [4],
      'expects 3 inputs but gets 4',
    ],
  ] as [LayerConfig, number[], string][])('%o on %o: %s', (layer, input, message) => {
    expect(outputShape(layer, input)).toEqual({ error: message });
  });
});

describe('layer cost', () => {
  it('counts parameters and forward multiply-adds per example', () => {
    const conv: LayerConfig = {
      kind: 'conv2d',
      inChannels: 8,
      outChannels: 16,
      kernel: 5,
      stride: 1,
      padding: 'valid',
      init: 'he',
    };
    expect(layerCost(conv, [16, 8, 8])).toEqual({
      params: 200 * 16 + 16,
      multiplyAdds: 64 * 200 * 16,
    });
    expect(
      layerCost({ kind: 'dense', inFeatures: 784, outFeatures: 128, init: 'he' }, [128]),
    ).toEqual({
      params: 784 * 128 + 128,
      multiplyAdds: 784 * 128,
    });
    expect(layerCost({ kind: 'maxpool2d', size: 2, stride: 2 }, [8, 12, 12])).toEqual({
      params: 0,
      multiplyAdds: 0,
    });
  });
});
