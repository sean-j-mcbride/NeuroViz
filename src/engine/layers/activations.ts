import { type Tensor, TensorBuffer } from '../tensor';
import type { Layer, LayerConfig, Param } from './types';

/** Base for parameter-free elementwise activations y = f(x). */
abstract class Activation implements Layer {
  abstract readonly kind: 'relu' | 'tanh' | 'sigmoid';
  private x: Tensor | null = null;
  private out: Tensor | null = null;
  private readonly outBuf = new TensorBuffer();
  private readonly dxBuf = new TensorBuffer();

  protected abstract f(x: number): number;
  /** f′ expressed in terms of the input x and the output y = f(x). */
  protected abstract df(x: number, y: number): number;

  forward(x: Tensor, _train: boolean): Tensor {
    this.x = x;
    this.out = this.outBuf.take(x.shape);
    const xd = x.data;
    const yd = this.out.data;
    for (let i = 0; i < xd.length; i++) yd[i] = this.f(xd[i]!);
    return this.out;
  }

  backward(gradOut: Tensor): Tensor {
    const { x, out } = this;
    if (!x || !out) throw new Error(`${this.kind}: backward called before forward`);
    const res = this.dxBuf.take(x.shape);
    const g = gradOut.data;
    const xd = x.data;
    const yd = out.data;
    const dx = res.data;
    for (let i = 0; i < dx.length; i++) dx[i] = g[i]! * this.df(xd[i]!, yd[i]!);
    return res;
  }

  params(): Param[] {
    return [];
  }

  toJSON(): LayerConfig {
    return { kind: this.kind };
  }
}

export class ReLU extends Activation {
  readonly kind = 'relu';
  protected f(x: number): number {
    return x > 0 ? x : 0;
  }
  protected df(x: number): number {
    return x > 0 ? 1 : 0;
  }
}

export class Tanh extends Activation {
  readonly kind = 'tanh';
  protected f(x: number): number {
    return Math.tanh(x);
  }
  protected df(_x: number, y: number): number {
    return 1 - y * y;
  }
}

export class Sigmoid extends Activation {
  readonly kind = 'sigmoid';
  protected f(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }
  protected df(_x: number, y: number): number {
    return y * (1 - y);
  }
}
