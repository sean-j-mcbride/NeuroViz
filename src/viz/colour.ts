import type { ColumnKind } from '../worker';

export type Rgb = readonly [number, number, number];

/** Orange for negative / label 0, blue for positive / label 1. */
export const NEGATIVE: Rgb = [245, 147, 34];
export const POSITIVE: Rgb = [8, 119, 189];

export const LUT_SIZE = 256;

/**
 * Diverging colour table, `LUT_SIZE` RGB triples packed in a Uint8Array:
 * index 0 = `neg`, the middle = `mid`, the last = `pos`, linear in between.
 */
export function divergingLut(neg: Rgb, mid: Rgb, pos: Rgb): Uint8Array {
  const lut = new Uint8Array(LUT_SIZE * 3);
  for (let i = 0; i < LUT_SIZE; i++) {
    const t = (i / (LUT_SIZE - 1)) * 2 - 1; // −1 … 1
    const [from, to, f] = t < 0 ? [mid, neg, -t] : [mid, pos, t];
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = Math.round(from[c]! + (to[c]! - from[c]!) * f);
  }
  return lut;
}

/** LUT index for t ∈ [−1, 1] (clamped). */
export function lutIndex(t: number): number {
  const i = Math.round(((t + 1) / 2) * (LUT_SIZE - 1));
  return i < 0 ? 0 : i > LUT_SIZE - 1 ? LUT_SIZE - 1 : i;
}

/** Linear map v ↦ v·scale + offset, chosen so the column's natural range lands on [−1, 1]. */
export interface Normalisation {
  scale: number;
  offset: number;
}

function maxAbs(values: Float32Array): number {
  let m = 0;
  for (let i = 0; i < values.length; i++) m = Math.max(m, Math.abs(values[i]!));
  return m;
}

/**
 * How each column kind's values map onto the colour scale, registered by kind.
 * Bounded activations use their fixed range; unbounded ones are scaled per neuron.
 */
export const NORMALISERS: Record<
  ColumnKind,
  (values: Float32Array, domain: number) => Normalisation
> = {
  input: (_v, domain) => ({ scale: 1 / domain, offset: 0 }),
  tanh: () => ({ scale: 1, offset: 0 }),
  sigmoid: () => ({ scale: 2, offset: -1 }),
  output: () => ({ scale: 2, offset: -1 }),
  relu: (v) => ({ scale: 1 / (maxAbs(v) || 1), offset: 0 }),
  linear: (v) => ({ scale: 1 / (maxAbs(v) || 1), offset: 0 }),
};

export function rgbCss([r, g, b]: Rgb, alpha = 1): string {
  return alpha === 1 ? `rgb(${r} ${g} ${b})` : `rgb(${r} ${g} ${b} / ${alpha})`;
}
