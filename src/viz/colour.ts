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

function hexRgb(hex: string): Rgb {
  return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)) as unknown as Rgb;
}

/**
 * Ordinal blue ramps for "layer depth" (first layer faintest, output strongest),
 * interpolated evenly in OKLab between the reference ramp's steps 250 → 700
 * (light) and 600 → 100 (dark). Validated with the dataviz palette checker:
 * monotone lightness, adjacent ΔL ≥ 0.06, near end ≥ 2:1 against the surface.
 */
const DEPTH_LIGHT = ['#86b6ef', '#719fd8', '#5c89c1', '#4874ab', '#345f95', '#214a80', '#0d366b'];
const DEPTH_DARK = ['#184f95', '#3867a6', '#557fb8', '#7297c9', '#90b0d9', '#aec9ea', '#cde2fb'];

/** Colour of layer k of n on the depth ramp (n ≤ 7), spread evenly so steps stay distinct. */
export function depthColour(k: number, n: number, dark: boolean): string {
  const ramp = dark ? DEPTH_DARK : DEPTH_LIGHT;
  const i = n <= 1 ? ramp.length - 1 : Math.round((k * (ramp.length - 1)) / (n - 1));
  return ramp[Math.min(ramp.length - 1, i)]!;
}

/**
 * Sequential colour table for densities: the surface colour at 0, then the
 * reference blue ramp towards high contrast (darker on light, lighter on dark).
 */
export function sequentialLut(dark: boolean): Uint8Array {
  const stops = (
    dark
      ? ['#1d2025', '#184f95', '#2a78d6', '#6da7ec', '#b7d3f6', '#cde2fb']
      : ['#ffffff', '#cde2fb', '#6da7ec', '#2a78d6', '#184f95', '#0d366b']
  ).map(hexRgb);
  const lut = new Uint8Array(LUT_SIZE * 3);
  for (let i = 0; i < LUT_SIZE; i++) {
    const t = (i / (LUT_SIZE - 1)) * (stops.length - 1);
    const s = Math.min(stops.length - 2, Math.floor(t));
    const f = t - s;
    for (let c = 0; c < 3; c++) {
      lut[i * 3 + c] = Math.round(stops[s]![c]! + (stops[s + 1]![c]! - stops[s]![c]!) * f);
    }
  }
  return lut;
}

/** Compact number for small labels: 2 decimals for ordinary values, 1-digit exponent for tiny ones. */
export function formatCompact(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 0.01 && a < 100) return v.toFixed(2);
  return v.toExponential(0).replace('e+', 'e');
}

/** Number for tooltips: 4 significant figures, exponent for very small or large values. */
export function formatPrecise(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  if (v === 0) return '0';
  const a = Math.abs(v);
  return a >= 1e-3 && a < 1e4 ? v.toPrecision(4) : v.toExponential(2);
}
