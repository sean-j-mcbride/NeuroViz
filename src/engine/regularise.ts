import type { Param } from './layers/types';

/** Weight matrices are regularised; biases are not. Matches "W" and prefixed names like "0.W". */
export function isWeight(p: Param): boolean {
  return p.name === 'W' || p.name.endsWith('.W');
}

/**
 * L2 regularisation: adds λ·w to the gradient of every weight matrix (not
 * biases) and returns the penalty (λ/2)·Σw² that this gradient belongs to.
 * Call after `backward`, which overwrites the gradients.
 */
export function addL2(params: Param[], lambda: number): number {
  if (lambda === 0) return 0;
  let sumSq = 0;
  for (const p of params) {
    if (!isWeight(p)) continue;
    const w = p.value.data;
    const g = p.grad.data;
    for (let i = 0; i < w.length; i++) {
      const wi = w[i]!;
      sumSq += wi * wi;
      g[i]! += lambda * wi;
    }
  }
  return 0.5 * lambda * sumSq;
}
