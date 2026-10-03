import { heNormal, xavierNormal } from '../init';
import type { Rng } from '../random';
import { paddingAmount, slidingOutput } from '../shapes';
import { Tensor, TensorBuffer, add, allFinite, matmul, sumAxis, transpose } from '../tensor';
import { WasmBlocks, type WasmKernels, wasmKernels } from '../wasm/runtime';
import type { InitKind, Layer, LayerConfig, Padding, Param } from './types';

/** Default cap on the im2col buffer, in elements (16 MB of float32). */
export const COL_BUDGET = 1 << 22;

export interface Conv2DOptions {
  init: InitKind;
  rng: Rng;
  stride?: number;
  padding?: Padding;
  /** Cap on the im2col buffer, in elements; bigger batches go through in chunks of images. */
  colBudget?: number;
}

/**
 * 2D convolution (strictly, cross-correlation, as in every ML library) over
 * NCHW input `[N, inC, H, W]`, giving `[N, outC, H′, W′]`.
 *
 * W is `[inC·k·k, outC]`, the same layout as `Dense`: one row per input tap
 * (channel c, row ky, column kx at row c·k² + ky·k + kx), one column per
 * filter. b is `[outC]`.
 *
 * Implemented as im2col + matmul on the exact `matmul` kernels. Every output
 * position's receptive patch becomes one row of `col` `[images·H′·W′,
 * inC·k²]`, so forward is col · W, the weight gradient colᵀ · dY and the
 * input gradient dY · Wᵀ. Each is arranged so the kernel skips the operand
 * that is mostly zeros: blank pixels and ReLU zeros in `col`, and the
 * gradients max-pooling and ReLU block in dY. (Measured on MNIST: a
 * quarter less time per batch than the transposed, dense arrangement.)
 *
 * `col` is k² times the size of the input, so the batch goes through in
 * chunks of whole images that keep it under `colBudget`. Outputs and input
 * gradients are the same bitwise however the batch is chunked; the parameter
 * gradients add one float32 rounding per extra chunk. When the whole batch
 * fits in one chunk (the usual case), backward reuses forward's `col`.
 */
export class Conv2D implements Layer {
  readonly kind = 'conv2d';
  readonly W: Param;
  readonly b: Param;
  readonly stride: number;
  readonly padding: Padding;
  /** As on `Dense`: off for a model's first layer, whose input gradient nothing reads. */
  inputGrad = true;
  private readonly init: InitKind;
  private readonly pad: number;
  private readonly colBudget: number;
  private x: Tensor | null = null;
  /** Forward's `col` when it covered the whole batch, else null (rebuilt per chunk). */
  private col: Tensor | null = null;
  private readonly colBuf = new TensorBuffer();
  private readonly yBuf = new TensorBuffer();
  private readonly out = new TensorBuffer();
  private readonly dyBuf = new TensorBuffer();
  private readonly dcolBuf = new TensorBuffer();
  private readonly dx = new TensorBuffer();
  private readonly WT = new TensorBuffer();
  private dWChunk: Tensor | null = null;
  private dbChunk: Tensor | null = null;
  /** Wasm memory for the Wasm kernels' buffers (created on first use). */
  private blocks: WasmBlocks | null = null;
  /** Which kernels the last forward ran on (backward must match: that's where `col` is). */
  private wasm: WasmKernels | null = null;
  /** The Wasm `col` holds the whole last batch (one chunk). */
  private wasmColValid = false;

  constructor(
    readonly inChannels: number,
    readonly outChannels: number,
    readonly kernel: number,
    { init, rng, stride = 1, padding = 'valid', colBudget = COL_BUDGET }: Conv2DOptions,
  ) {
    this.init = init;
    this.stride = stride;
    this.padding = padding;
    this.pad = paddingAmount(kernel, padding);
    this.colBudget = colBudget;
    const taps = inChannels * kernel * kernel;
    const fanOut = outChannels * kernel * kernel;
    const shape = [taps, outChannels];
    const w =
      init === 'he' ? heNormal(taps, fanOut, rng, shape) : xavierNormal(taps, fanOut, rng, shape);
    this.W = { name: 'W', value: w, grad: Tensor.zeros(w.shape) };
    this.b = { name: 'b', value: Tensor.zeros([outChannels]), grad: Tensor.zeros([outChannels]) };
  }

  private geometry(x: Tensor): [n: number, h: number, w: number, oh: number, ow: number] {
    if (x.shape.length !== 4 || x.shape[1] !== this.inChannels) {
      throw new Error(
        `Conv2D: expected [N, ${this.inChannels}, H, W] input, got [${x.shape.join(', ')}]`,
      );
    }
    const [n, , h, w] = x.shape as [number, number, number, number];
    const oh = slidingOutput(h, this.kernel, this.stride, this.pad);
    const ow = slidingOutput(w, this.kernel, this.stride, this.pad);
    if (oh < 1 || ow < 1) throw new Error(`Conv2D: a ${this.kernel}² kernel doesn’t fit ${h}×${w}`);
    return [n, h, w, oh, ow];
  }

  /** Images per chunk, so that `col` stays within the budget (at least one image). */
  private chunkSize(n: number, oh: number, ow: number): number {
    const perImage = oh * ow * this.inChannels * this.kernel * this.kernel;
    return Math.max(1, Math.min(n, Math.floor(this.colBudget / perImage)));
  }

  /**
   * im2col: copies every patch of images [b0, b0 + m) into `col`
   * `[m·H′·W′, inC·k²]`, one row per output position (zeros where the patch
   * hangs over the padded edge). With `add`, runs the other way (col2im):
   * adds each entry back where it came from in `x`.
   */
  private patches(
    x: Tensor,
    col: Tensor,
    b0: number,
    m: number,
    oh: number,
    ow: number,
    add: boolean,
  ): void {
    const { inChannels: c, kernel: k, stride: s, pad } = this;
    const [, , h, w] = x.shape as [number, number, number, number];
    const xd = x.data;
    const cd = col.data;
    let o = 0;
    for (let b = b0; b < b0 + m; b++) {
      for (let oy = 0; oy < oh; oy++) {
        for (let ox = 0; ox < ow; ox++) {
          for (let ci = 0; ci < c; ci++) {
            const plane = (b * c + ci) * h;
            for (let ky = 0; ky < k; ky++) {
              const iy = oy * s - pad + ky;
              if (iy < 0 || iy >= h) {
                if (!add) cd.fill(0, o, o + k);
                o += k;
                continue;
              }
              const ro = (plane + iy) * w;
              for (let kx = 0; kx < k; kx++, o++) {
                const ix = ox * s - pad + kx;
                const inside = ix >= 0 && ix < w;
                if (add) {
                  if (inside) xd[ro + ix]! += cd[o]!;
                } else cd[o] = inside ? xd[ro + ix]! : 0;
              }
            }
          }
        }
      }
    }
  }

  forward(x: Tensor, _train: boolean): Tensor {
    this.wasm = wasmKernels();
    return this.wasm ? this.forwardWasm(x, this.wasm) : this.forwardJs(x);
  }

  backward(gradOut: Tensor): Tensor {
    if (!this.x) throw new Error('Conv2D: backward called before forward');
    return this.wasm ? this.backwardWasm(gradOut, this.wasm) : this.backwardJs(gradOut);
  }

  private forwardJs(x: Tensor): Tensor {
    const [n, , , oh, ow] = this.geometry(x);
    const f = this.outChannels;
    const taps = this.inChannels * this.kernel * this.kernel;
    const area = oh * ow;
    const chunk = this.chunkSize(n, oh, ow);
    const out = this.out.take([n, f, oh, ow]);
    const od = out.data;
    const bias = this.b.value.data;
    let col: Tensor | null = null;
    for (let b0 = 0; b0 < n; b0 += chunk) {
      const m = Math.min(chunk, n - b0);
      col = this.colBuf.take([m * area, taps]);
      this.patches(x, col, b0, m, oh, ow, false);
      // y [m·H′·W′, outC] = col · W, skipping zero pixels; then add the bias while permuting to NCHW.
      const yd = matmul(col, this.W.value, {}, this.yBuf.take([m * area, f])).data;
      scatterOutput(yd, od, bias, b0, m, f, area);
    }
    this.x = x;
    this.col = chunk >= n ? col : null;
    return out;
  }

  private backwardJs(gradOut: Tensor): Tensor {
    const x = this.x!;
    const [n, , , oh, ow] = this.geometry(x);
    const f = this.outChannels;
    const taps = this.inChannels * this.kernel * this.kernel;
    const area = oh * ow;
    const chunk = this.chunkSize(n, oh, ow);
    const g = gradOut.data;
    const dx = this.dx.take(x.shape);
    dx.data.fill(0);
    // Wᵀ as its own array, so dcol = dY · Wᵀ runs on the kernel that skips dY's zeros.
    const WT = this.inputGrad ? transpose(this.W.value, this.WT.take([f, taps])) : null;
    for (let b0 = 0; b0 < n; b0 += chunk) {
      const m = Math.min(chunk, n - b0);
      let col = this.col;
      if (!col) {
        col = this.colBuf.take([m * area, taps]);
        this.patches(x, col, b0, m, oh, ow, false);
      }
      // dY as [m·H′·W′, outC], the layout of y in forward.
      const dy = this.dyBuf.take([m * area, f]);
      gatherGradient(g, dy.data, b0, m, f, area);
      if (b0 === 0) {
        matmul(col, dy, { transA: true }, this.W.grad); // dW = colᵀ·dY
        sumAxis(dy, 0, this.b.grad); // db = Σ over every position and example
      } else {
        this.dWChunk ??= Tensor.zeros(this.W.grad.shape);
        this.dbChunk ??= Tensor.zeros(this.b.grad.shape);
        add(this.W.grad, matmul(col, dy, { transA: true }, this.dWChunk), this.W.grad);
        add(this.b.grad, sumAxis(dy, 0, this.dbChunk), this.b.grad);
      }
      if (WT) {
        // dcol = dY·Wᵀ (after max-pool and ReLU, dY is mostly zeros), then col2im.
        const dcol = matmul(dy, WT, {}, this.dcolBuf.take(col.shape));
        this.patches(dx, dcol, b0, m, oh, ow, true);
      }
    }
    return dx;
  }

  /** forwardJs on the Wasm kernels: im2col and col · W run in Wasm memory, where `col` stays. */
  private forwardWasm(x: Tensor, { heap, k: K }: WasmKernels): Tensor {
    const [n, h, w, oh, ow] = this.geometry(x);
    const { inChannels: c, outChannels: f, kernel: k, stride: s, pad } = this;
    const taps = c * k * k;
    const area = oh * ow;
    const img = c * h * w;
    const chunk = this.chunkSize(n, oh, ow);
    const blocks = (this.blocks ??= new WasmBlocks(this, heap));
    // Allocate first (it may grow memory), then make views.
    const pX = blocks.get('x', chunk * img * 4);
    const pCol = blocks.get('col', chunk * area * taps * 4);
    const pW = blocks.get('W', taps * f * 4);
    const pY = blocks.get('y', chunk * area * f * 4);
    const pAcc = blocks.get('acc', f * 8);
    heap.f32(pW, taps * f).set(this.W.value.data);
    const skip = allFinite(this.W.value.data) ? 1 : 0;
    const out = this.out.take([n, f, oh, ow]);
    for (let b0 = 0; b0 < n; b0 += chunk) {
      const m = Math.min(chunk, n - b0);
      const rows = m * area;
      heap.f32(pX, m * img).set(x.data.subarray(b0 * img, (b0 + m) * img));
      K.patches(pX, pCol, 0, m, c, h, w, k, s, pad, oh, ow, 0);
      K.nn(pCol, pW, pY, pAcc, rows, taps, f, skip);
      scatterOutput(heap.f32(pY, rows * f), out.data, this.b.value.data, b0, m, f, area);
    }
    this.x = x;
    this.wasmColValid = chunk >= n;
    return out;
  }

  /** backwardJs on the Wasm kernels. */
  private backwardWasm(gradOut: Tensor, { heap, k: K }: WasmKernels): Tensor {
    const x = this.x!;
    const [n, h, w, oh, ow] = this.geometry(x);
    const { inChannels: c, outChannels: f, kernel: k, stride: s, pad } = this;
    const taps = c * k * k;
    const area = oh * ow;
    const img = c * h * w;
    const chunk = this.chunkSize(n, oh, ow);
    const blocks = this.blocks!;
    const pX = blocks.get('x', chunk * img * 4);
    const pCol = blocks.get('col', chunk * area * taps * 4);
    const pDy = blocks.get('dy', chunk * area * f * 4);
    const pDW = blocks.get('dW', taps * f * 4);
    const pAccW = blocks.get('accW', taps * f * 8);
    const grad = this.inputGrad;
    const pWT = grad ? blocks.get('WT', taps * f * 4) : 0;
    const pDcol = grad ? blocks.get('dcol', chunk * area * taps * 4) : 0;
    const pDx = grad ? blocks.get('dx', chunk * img * 4) : 0;
    const pAccT = grad ? blocks.get('accT', taps * 8) : 0;
    const Wd = this.W.value.data;
    const skipWT = allFinite(Wd) ? 1 : 0;
    if (grad) {
      const WT = heap.f32(pWT, taps * f);
      for (let t = 0; t < taps; t++)
        for (let fi = 0; fi < f; fi++) WT[fi * taps + t] = Wd[t * f + fi]!;
    }
    const dx = this.dx.take(x.shape);
    dx.data.fill(0);
    for (let b0 = 0; b0 < n; b0 += chunk) {
      const m = Math.min(chunk, n - b0);
      const rows = m * area;
      if (!this.wasmColValid) {
        heap.f32(pX, m * img).set(x.data.subarray(b0 * img, (b0 + m) * img));
        K.patches(pX, pCol, 0, m, c, h, w, k, s, pad, oh, ow, 0);
      }
      const dyd = heap.f32(pDy, rows * f);
      gatherGradient(gradOut.data, dyd, b0, m, f, area);
      K.tn(pCol, pDy, pDW, pAccW, taps, rows, f, allFinite(dyd) ? 1 : 0); // dW = colᵀ·dY
      const dW = new Tensor(heap.f32(pDW, taps * f), [taps, f]);
      const dy = new Tensor(heap.f32(pDy, rows * f), [rows, f]);
      if (b0 === 0) {
        this.W.grad.data.set(dW.data);
        sumAxis(dy, 0, this.b.grad);
      } else {
        this.dbChunk ??= Tensor.zeros(this.b.grad.shape);
        add(this.W.grad, dW, this.W.grad);
        add(this.b.grad, sumAxis(dy, 0, this.dbChunk), this.b.grad);
      }
      if (grad) {
        K.nn(pDy, pWT, pDcol, pAccT, rows, f, taps, skipWT); // dcol = dY·Wᵀ
        heap.f32(pDx, m * img).fill(0);
        K.patches(pDx, pDcol, 0, m, c, h, w, k, s, pad, oh, ow, 1); // col2im
        dx.data.set(heap.f32(pDx, m * img), b0 * img);
      }
    }
    return dx;
  }

  params(): Param[] {
    return [this.W, this.b];
  }

  toJSON(): LayerConfig {
    return {
      kind: 'conv2d',
      inChannels: this.inChannels,
      outChannels: this.outChannels,
      kernel: this.kernel,
      stride: this.stride,
      padding: this.padding,
      init: this.init,
    };
  }
}

/** y `[m·H′·W′, outC]` (one chunk) plus bias, into images [b0, b0 + m) of NCHW `out`. */
function scatterOutput(
  yd: Float32Array,
  od: Float32Array,
  bias: Float32Array,
  b0: number,
  m: number,
  f: number,
  area: number,
): void {
  for (let i = 0; i < m; i++) {
    for (let fi = 0; fi < f; fi++) {
      const bv = bias[fi]!;
      const dst = ((b0 + i) * f + fi) * area;
      let src = i * area * f + fi;
      for (let p = 0; p < area; p++, src += f) od[dst + p] = yd[src]! + bv;
    }
  }
}

/** Images [b0, b0 + m) of NCHW gradient `g`, as dY `[m·H′·W′, outC]` (the layout of y). */
function gatherGradient(
  g: Float32Array,
  dyd: Float32Array,
  b0: number,
  m: number,
  f: number,
  area: number,
): void {
  for (let i = 0; i < m; i++) {
    for (let fi = 0; fi < f; fi++) {
      const src = ((b0 + i) * f + fi) * area;
      let dst = i * area * f + fi;
      for (let p = 0; p < area; p++, dst += f) dyd[dst] = g[src + p]!;
    }
  }
}
