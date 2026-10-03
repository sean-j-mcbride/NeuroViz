import { readFileSync } from 'node:fs';
import { afterEach, describe, expect, it } from 'vitest';
import { compileKernels } from '../../../scripts/wasm';
import { Conv2D } from '../layers/conv2d';
import type { Padding } from '../layers/types';
import { Rng } from '../random';
import { Tensor } from '../tensor';
import { KERNELS_WASM } from './kernels';
import { type WasmGlobal, WasmHeap, loadKernels, setKernelBackend } from './runtime';

afterEach(() => setKernelBackend('auto'));

const CONVS: [number, number, number, number, Padding, [number, number], number][] = [
  // inC, outC, kernel, stride, padding, [H, W], batch
  [1, 8, 5, 1, 'valid', [28, 28], 5],
  [8, 16, 5, 1, 'valid', [12, 12], 4],
  [2, 3, 3, 2, 'same', [7, 7], 3],
  [1, 2, 3, 2, 'valid', [7, 6], 3],
  [3, 5, 3, 1, 'same', [6, 5], 2], // odd filter count: the SIMD loop's scalar tail
];

/** Forward + backward on one backend; returns every output to compare. */
function run(
  backend: 'js' | 'wasm',
  [c, f, k, s, padding, [h, w], n]: (typeof CONVS)[number],
  opts: {
    colBudget?: number;
    inputGrad?: boolean;
    poison?: number;
  } = {},
) {
  setKernelBackend(backend);
  const rng = new Rng(7);
  const conv = new Conv2D(c, f, k, {
    init: 'he',
    rng,
    stride: s,
    padding,
    ...(opts.colBudget && { colBudget: opts.colBudget }),
  });
  conv.inputGrad = opts.inputGrad ?? true;
  conv.b.value.data.set(Tensor.randn([f], rng).data);
  const x = Tensor.randn([n, c, h, w], rng);
  for (let i = 0; i < x.size; i += 3) x.data[i] = 0; // zeros: the zero-skipping path
  if (opts.poison !== undefined) conv.W.value.data[1] = opts.poison;
  const y = conv.forward(x, true).data.slice();
  const g = Tensor.randn([n, f, ...conv.forward(x, true).shape.slice(2)], rng);
  for (let i = 0; i < g.size; i += 4) g.data[i] = 0;
  conv.forward(x, true);
  const dx = conv.backward(g).data.slice();
  return { y, dx, dW: conv.W.grad.data.slice(), db: conv.b.grad.data.slice() };
}

const same = (a: Float32Array, b: Float32Array) =>
  a.length === b.length && a.every((v, i) => Object.is(v, b[i]));

describe('Wasm conv kernels', () => {
  it('the committed bytes are kernels.wat compiled (run `npm run build:wasm` if not)', async () => {
    const wat = readFileSync(new URL('./kernels.wat', import.meta.url), 'utf8');
    expect(Array.from(await compileKernels(wat))).toEqual(Array.from(KERNELS_WASM));
  });

  it.each(CONVS)('%i→%i, %i×%i stride %i %s: bitwise the same as the JS kernels', (...cfg) => {
    const js = run('js', cfg);
    const wasm = run('wasm', cfg);
    for (const key of ['y', 'dx', 'dW', 'db'] as const) expect(same(wasm[key], js[key])).toBe(true);
  });

  it('chunked batches and inputGrad off agree too', () => {
    const cfg = CONVS[2]!;
    for (const opts of [
      { colBudget: 300 },
      { inputGrad: false },
      { colBudget: 300, inputGrad: false },
    ]) {
      const js = run('js', cfg, opts);
      const wasm = run('wasm', cfg, opts);
      for (const key of ['y', 'dx', 'dW', 'db'] as const)
        expect(same(wasm[key], js[key])).toBe(true);
    }
  });

  it('a non-finite weight turns skipping off, so 0 · Inf still gives NaN, as in JS', () => {
    for (const poison of [Infinity, NaN]) {
      const js = run('js', CONVS[0]!, { poison });
      const wasm = run('wasm', CONVS[0]!, { poison });
      expect(js.y.some(Number.isNaN)).toBe(true);
      for (const key of ['y', 'dx', 'dW', 'db'] as const)
        expect(same(wasm[key], js[key])).toBe(true);
    }
  });
});

describe('fallback', () => {
  it('without WebAssembly, or when it refuses the module, the JS kernels are used', () => {
    expect(loadKernels(null)).toBeNull();
    const refusing = {
      ...WebAssembly,
      Module: function () {
        throw new Error('no SIMD');
      },
    } as unknown as WasmGlobal;
    expect(loadKernels(refusing)).toBeNull();
    expect(loadKernels()).not.toBeNull();
  });
});

describe('WasmHeap', () => {
  it('reuses and merges released blocks, and grows memory when it must', () => {
    let pages = 1;
    const memory = {
      get buffer() {
        return new ArrayBuffer(pages * 65536);
      },
      grow(n: number) {
        pages += n;
        return pages - n;
      },
    };
    const heap = new WasmHeap(memory);
    const a = heap.alloc(100);
    const b = heap.alloc(1000);
    const c = heap.alloc(70_000); // past the first page
    expect(pages).toBe(2);
    heap.release(a, 100);
    heap.release(b, 1000);
    expect(heap.alloc(1100)).toBe(a); // a and b merged
    heap.release(c, 70_000);
    expect(heap.inUse).toBe(1104); // 1100 rounded up to 16
  });
});
