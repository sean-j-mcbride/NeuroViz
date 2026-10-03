import type { Rng } from '../random';
import { type Tensor, TensorBuffer } from '../tensor';
import type { Layer, LayerConfig, Param } from './types';

/**
 * Inverted dropout: in training, each element is zeroed with probability
 * `rate` and survivors are scaled by 1/(1 − rate), so evaluation needs no
 * rescaling. With `train = false` or `rate = 0` it is the identity (returns
 * its input unchanged) and draws nothing from the RNG.
 */
export class Dropout implements Layer {
  readonly kind = 'dropout';
  /** Drop probability in [0, 1); may be changed between steps. */
  rate: number;
  /** Reuse the last mask instead of sampling a new one (for gradient checks). */
  freezeMask = false;
  private readonly rng: Rng;
  /** Per element 0 or 1/(1 − rate); null when the last forward was the identity. */
  private mask: Float32Array | null = null;
  private active = false;
  private readonly out = new TensorBuffer();
  private readonly dx = new TensorBuffer();

  constructor(rate: number, rng: Rng) {
    this.rate = rate;
    this.rng = rng;
  }

  forward(x: Tensor, train: boolean): Tensor {
    this.active = train && this.rate > 0;
    if (!this.active) return x;
    const out = this.out.take(x.shape);
    if (!this.freezeMask || this.mask?.length !== x.size) this.sampleMask(x.size);
    const m = this.mask!;
    const xd = x.data;
    const yd = out.data;
    for (let i = 0; i < xd.length; i++) yd[i] = xd[i]! * m[i]!;
    return out;
  }

  private sampleMask(size: number): void {
    if (this.mask?.length !== size) this.mask = new Float32Array(size);
    const keep = 1 - this.rate;
    const scale = 1 / keep;
    const m = this.mask;
    for (let i = 0; i < size; i++) m[i] = this.rng.next() < keep ? scale : 0;
  }

  backward(gradOut: Tensor): Tensor {
    if (!this.active) return gradOut;
    const m = this.mask!;
    const res = this.dx.take(gradOut.shape);
    const g = gradOut.data;
    const dx = res.data;
    for (let i = 0; i < dx.length; i++) dx[i] = g[i]! * m[i]!;
    return res;
  }

  params(): Param[] {
    return [];
  }

  toJSON(): LayerConfig {
    return { kind: 'dropout', rate: this.rate };
  }
}
