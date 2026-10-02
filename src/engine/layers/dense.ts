import { heNormal, xavierNormal } from '../init';
import type { Rng } from '../random';
import { Tensor, add, matmul, sumAxis } from '../tensor';
import type { InitKind, Layer, LayerConfig, Param } from './types';

export interface DenseOptions {
  init: InitKind;
  rng: Rng;
}

/** Fully connected layer: y = x·W + b, with W `[in, out]` and b `[out]`. */
export class Dense implements Layer {
  readonly kind = 'dense';
  readonly W: Param;
  readonly b: Param;
  private readonly init: InitKind;
  private x: Tensor | null = null;
  private out: Tensor | null = null;
  private dx: Tensor | null = null;

  constructor(
    readonly inFeatures: number,
    readonly outFeatures: number,
    { init, rng }: DenseOptions,
  ) {
    this.init = init;
    const w =
      init === 'he'
        ? heNormal(inFeatures, outFeatures, rng)
        : xavierNormal(inFeatures, outFeatures, rng);
    this.W = { name: 'W', value: w, grad: Tensor.zeros(w.shape) };
    this.b = { name: 'b', value: Tensor.zeros([outFeatures]), grad: Tensor.zeros([outFeatures]) };
  }

  forward(x: Tensor, _train: boolean): Tensor {
    if (x.cols !== this.inFeatures) {
      throw new Error(`Dense: expected ${this.inFeatures} input features, got ${x.cols}`);
    }
    this.x = x;
    if (this.out?.rows !== x.rows) this.out = Tensor.zeros([x.rows, this.outFeatures]);
    matmul(x, this.W.value, {}, this.out);
    return add(this.out, this.b.value, this.out);
  }

  backward(gradOut: Tensor): Tensor {
    const x = this.x;
    if (!x) throw new Error('Dense: backward called before forward');
    matmul(x, gradOut, { transA: true }, this.W.grad); // dW = xᵀ·g
    sumAxis(gradOut, 0, this.b.grad); // db = Σ_rows g
    if (this.dx?.rows !== x.rows) this.dx = Tensor.zeros(x.shape);
    return matmul(gradOut, this.W.value, { transB: true }, this.dx); // dx = g·Wᵀ
  }

  params(): Param[] {
    return [this.W, this.b];
  }

  toJSON(): LayerConfig {
    return {
      kind: 'dense',
      inFeatures: this.inFeatures,
      outFeatures: this.outFeatures,
      init: this.init,
    };
  }
}
