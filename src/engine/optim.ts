import type { Param } from './layers/types';
import type { Tensor } from './tensor';

export type OptimiserKind = 'sgd' | 'momentum' | 'adam';

export interface Optimiser {
  readonly kind: OptimiserKind;
  /** Learning rate; may be changed between steps. */
  lr: number;
  step(params: Param[]): void;
  /** Copies of the internal state, keyed by param name, for saving a run. */
  saveState(params: Param[]): OptimiserState;
  /** Restores state saved (by an optimiser of the same kind) for params with the same names and sizes. */
  loadState(params: Param[], state: OptimiserState): void;
}

/** Serialisable optimiser state. */
export interface OptimiserState {
  kind: OptimiserKind;
  /** Steps taken so far (Adam's bias-correction counter); 0 where unused. */
  t: number;
  /** Per param name, its state buffers in a fixed order (Momentum: [velocity]; Adam: [m, v]). */
  slots: Record<string, Float32Array[]>;
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

function saveSlots(
  kind: OptimiserKind,
  t: number,
  params: Param[],
  buffers: StateBuffers[],
): OptimiserState {
  const slots: Record<string, Float32Array[]> = {};
  for (const p of params) slots[p.name] = buffers.map((b) => b.get(p.value).slice());
  return { kind, t, slots };
}

function loadSlots(
  kind: OptimiserKind,
  params: Param[],
  state: OptimiserState,
  buffers: StateBuffers[],
): void {
  if (state.kind !== kind) {
    throw new Error(`Optimiser state is for ${state.kind}, not ${kind}`);
  }
  const names = new Set(params.map((p) => p.name));
  for (const name of Object.keys(state.slots)) {
    if (!names.has(name)) throw new Error(`Optimiser state has unknown parameter "${name}"`);
  }
  for (const p of params) {
    const saved = state.slots[p.name];
    if (!saved || saved.length !== buffers.length) {
      throw new Error(`Optimiser state is missing parameter "${p.name}"`);
    }
    saved.forEach((s, k) => {
      if (s.length !== p.value.size) {
        throw new Error(
          `Optimiser state for "${p.name}" has ${s.length} values, expected ${p.value.size}`,
        );
      }
      buffers[k]!.get(p.value).set(s);
    });
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

  saveState(params: Param[]): OptimiserState {
    return saveSlots(this.kind, 0, params, []);
  }

  loadState(params: Param[], state: OptimiserState): void {
    loadSlots(this.kind, params, state, []);
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

  saveState(params: Param[]): OptimiserState {
    return saveSlots(this.kind, 0, params, [this.velocity]);
  }

  loadState(params: Param[], state: OptimiserState): void {
    loadSlots(this.kind, params, state, [this.velocity]);
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

  saveState(params: Param[]): OptimiserState {
    return saveSlots(this.kind, this.t, params, [this.m, this.v]);
  }

  loadState(params: Param[], state: OptimiserState): void {
    loadSlots(this.kind, params, state, [this.m, this.v]);
    this.t = state.t;
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
