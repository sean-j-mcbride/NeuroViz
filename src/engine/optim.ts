import type { Param } from './layers/types';

export interface Optimiser {
  step(params: Param[]): void;
}

/** Plain stochastic gradient descent: value −= lr · grad. */
export class SGD implements Optimiser {
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
