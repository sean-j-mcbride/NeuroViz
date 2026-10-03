import { heNormal, xavierNormal } from '../init';
import type { Rng } from '../random';
import { paddingAmount, slidingOutput } from '../shapes';
import { Tensor, TensorBuffer, add, matmul, sumAxis } from '../tensor';
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
 * Implemented as im2col + matmul: every output position's receptive patch
 * becomes one row of `col` `[images·H′·W′, inC·k²]`, so forward is col · W
 * and the weight gradient is colᵀ · dY. That reuses the exact, zero-skipping
 * `matmul` kernels (input pixels and ReLU outputs are mostly zero).
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
  private dWChunk: Tensor | null = null;
  private dbChunk: Tensor | null = null;

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
   * Copies every patch of images [b0, b0 + m) into `col`, one row per output
   * position (zeros where the patch hangs over the padded edge). With `add`,
   * runs the other way (col2im): adds each row of `col` back into `x`.
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
              const inRow = iy >= 0 && iy < h;
              const ro = (plane + iy) * w;
              for (let kx = 0; kx < k; kx++, o++) {
                const ix = ox * s - pad + kx;
                const inside = inRow && ix >= 0 && ix < w;
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
      // y [m·H′·W′, outC] = col · W, then add the bias while permuting to NCHW.
      const yd = matmul(col, this.W.value, {}, this.yBuf.take([m * area, f])).data;
      for (let i = 0; i < m; i++) {
        for (let fi = 0; fi < f; fi++) {
          const bv = bias[fi]!;
          const dst = ((b0 + i) * f + fi) * area;
          let src = i * area * f + fi;
          for (let p = 0; p < area; p++, src += f) od[dst + p] = yd[src]! + bv;
        }
      }
    }
    this.x = x;
    this.col = chunk >= n ? col : null;
    return out;
  }

  backward(gradOut: Tensor): Tensor {
    const x = this.x;
    if (!x) throw new Error('Conv2D: backward called before forward');
    const [n, , , oh, ow] = this.geometry(x);
    const f = this.outChannels;
    const taps = this.inChannels * this.kernel * this.kernel;
    const area = oh * ow;
    const chunk = this.chunkSize(n, oh, ow);
    const g = gradOut.data;
    const dx = this.dx.take(x.shape);
    dx.data.fill(0);
    for (let b0 = 0; b0 < n; b0 += chunk) {
      const m = Math.min(chunk, n - b0);
      let col = this.col;
      if (!col) {
        col = this.colBuf.take([m * area, taps]);
        this.patches(x, col, b0, m, oh, ow, false);
      }
      // dY as [m·H′·W′, outC], the layout of y in forward.
      const dy = this.dyBuf.take([m * area, f]);
      const dyd = dy.data;
      for (let i = 0; i < m; i++) {
        for (let fi = 0; fi < f; fi++) {
          const src = ((b0 + i) * f + fi) * area;
          let dst = i * area * f + fi;
          for (let p = 0; p < area; p++, dst += f) dyd[dst] = g[src + p]!;
        }
      }
      if (b0 === 0) {
        matmul(col, dy, { transA: true }, this.W.grad); // dW = colᵀ·dY
        sumAxis(dy, 0, this.b.grad); // db = Σ over every position and example
      } else {
        this.dWChunk ??= Tensor.zeros(this.W.grad.shape);
        this.dbChunk ??= Tensor.zeros(this.b.grad.shape);
        add(this.W.grad, matmul(col, dy, { transA: true }, this.dWChunk), this.W.grad);
        add(this.b.grad, sumAxis(dy, 0, this.dbChunk), this.b.grad);
      }
      if (this.inputGrad) {
        // dcol = dY·Wᵀ, then add each patch back where it came from (col2im).
        const dcol = matmul(dy, this.W.value, { transB: true }, this.dcolBuf.take(col.shape));
        this.patches(dx, dcol, b0, m, oh, ow, true);
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
