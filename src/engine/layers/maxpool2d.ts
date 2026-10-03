import { slidingOutput } from '../shapes';
import { type Tensor, TensorBuffer } from '../tensor';
import type { Layer, LayerConfig, Param } from './types';

/**
 * Max pooling over `size × size` windows every `stride` (default `size`) on
 * NCHW input. Windows that would run off the edge are dropped (the output
 * size rounds down). Each output remembers which input it came from, and the
 * gradient flows back to that input only. On a tie the first maximum (in
 * reading order) wins, so results are deterministic.
 */
export class MaxPool2D implements Layer {
  readonly kind = 'maxpool2d';
  readonly stride: number;
  /** Flat input index of each output's maximum, from the last forward. */
  private argmax = new Int32Array(0);
  private x: Tensor | null = null;
  private readonly out = new TensorBuffer();
  private readonly dx = new TensorBuffer();

  constructor(
    readonly size: number,
    stride?: number,
  ) {
    this.stride = stride ?? size;
  }

  forward(x: Tensor, _train: boolean): Tensor {
    if (x.shape.length !== 4) {
      throw new Error(`MaxPool2D: expected [N, C, H, W] input, got [${x.shape.join(', ')}]`);
    }
    const [n, c, h, w] = x.shape as [number, number, number, number];
    const { size: k, stride: s } = this;
    const oh = slidingOutput(h, k, s);
    const ow = slidingOutput(w, k, s);
    if (oh < 1 || ow < 1) throw new Error(`MaxPool2D: a ${k}² window doesn’t fit ${h}×${w}`);
    const out = this.out.take([n, c, oh, ow]);
    if (this.argmax.length < out.size) this.argmax = new Int32Array(out.size);
    const xd = x.data;
    const od = out.data;
    const arg = this.argmax;
    let o = 0;
    for (let plane = 0; plane < n * c; plane++) {
      const base = plane * h * w;
      for (let oy = 0; oy < oh; oy++) {
        for (let ox = 0; ox < ow; ox++, o++) {
          let best = base + oy * s * w + ox * s;
          let bv = xd[best]!;
          for (let ky = 0; ky < k; ky++) {
            const ro = base + (oy * s + ky) * w + ox * s;
            for (let kx = 0; kx < k; kx++) {
              const v = xd[ro + kx]!;
              if (v > bv) {
                bv = v;
                best = ro + kx;
              }
            }
          }
          od[o] = bv;
          arg[o] = best;
        }
      }
    }
    this.x = x;
    return out;
  }

  backward(gradOut: Tensor): Tensor {
    const x = this.x;
    if (!x) throw new Error('MaxPool2D: backward called before forward');
    const dx = this.dx.take(x.shape);
    const d = dx.data;
    d.fill(0);
    const g = gradOut.data;
    const arg = this.argmax;
    for (let o = 0; o < g.length; o++) d[arg[o]!]! += g[o]!;
    return dx;
  }

  params(): Param[] {
    return [];
  }

  toJSON(): LayerConfig {
    return { kind: 'maxpool2d', size: this.size, stride: this.stride };
  }
}
