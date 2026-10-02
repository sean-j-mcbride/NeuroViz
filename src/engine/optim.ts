import type { Param } from './layers/types';
import type { Tensor } from './tensor';

export type OptimiserKind = 'sgd' | 'momentum' | 'adam';

export interface Optimiser {
  readonly kind: OptimiserKind;
  /** Learning rate; may be changed between steps. */
  lr: number;
  step(params: Param[]): void;
}

/**
 * Per-parameter state buffers, keyed by the parameter's value tensor (stable
 * for the life of a layer, unlike the `Param` objects `Sequential` builds).
 */
class StateBuffers {
  private readonly map = new Map<Tensor, Float32Array>();

  get(value: Tensor): Float32Array {
    let s = this.map.get(value);
    if (!s) {
      s = new Float32Array(value.size);
      this.map.set(value, s);
    }
    return s;
  }
}

/** Plain stochastic gradient descent: value −= lr · grad. */
export class SGD implements Optimiser {
  readonly kind = 'sgd';
  constructor(public lr: number) {}

  step(params: Param[]): void {
    const lr = this.lr;
    for (const { value, grad } of params) {
      const v = value.data;
      const g = grad.data;
      for (let i = 0; i < v.length; i++) v[i]! -= lr * g[i]!;
    }
  }
}

/** SGD with classical (heavy-ball) momentum: velocity = β·velocity + grad; value −= lr · velocity. */
export class Momentum implements Optimiser {
  readonly kind = 'momentum';
  private readonly velocity = new StateBuffers();

  constructor(
    public lr: number,
    readonly beta = 0.9,
  ) {}

  step(params: Param[]): void {
    const { lr, beta } = this;
    for (const { value, grad } of params) {
      const v = value.data;
      const g = grad.data;
      const m = this.velocity.get(value);
      for (let i = 0; i < v.length; i++) {
        m[i] = beta * m[i]! + g[i]!;
        v[i]! -= lr * m[i]!;
      }
    }
  }
}

/**
 * Adam (Kingma & Ba): bias-corrected running means of the gradient (m) and its
 * square (v); value −= lr · m̂ / (√v̂ + eps).
 */
export class Adam implements Optimiser {
  readonly kind = 'adam';
  private readonly m = new StateBuffers();
  private readonly v = new StateBuffers();
  private t = 0;

  constructor(
    public lr: number,
    readonly beta1 = 0.9,
    readonly beta2 = 0.999,
    readonly eps = 1e-8,
  ) {}

  step(params: Param[]): void {
    const { lr, beta1, beta2, eps } = this;
    this.t++;
    const c1 = 1 - beta1 ** this.t;
    const c2 = 1 - beta2 ** this.t;
    for (const { value, grad } of params) {
      const w = value.data;
      const g = grad.data;
      const m = this.m.get(value);
      const v = this.v.get(value);
      for (let i = 0; i < w.length; i++) {
        const gi = g[i]!;
        m[i] = beta1 * m[i]! + (1 - beta1) * gi;
        v[i] = beta2 * v[i]! + (1 - beta2) * gi * gi;
        w[i]! -= (lr * (m[i]! / c1)) / (Math.sqrt(v[i]! / c2) + eps);
      }
    }
  }
}

export function makeOptimiser(kind: OptimiserKind, lr: number): Optimiser {
  switch (kind) {
    case 'sgd':
      return new SGD(lr);
    case 'momentum':
      return new Momentum(lr);
    case 'adam':
      return new Adam(lr);
  }
}
