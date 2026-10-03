import type { Activation, ConvFilters, ConvLayerSpec } from '../worker';
import { NORMALISERS, type Normalisation } from './colour';

/**
 * How each image-layer kind appears in the MNIST views, registered by kind:
 * its name, whether it has filters to show, and the activation its feature
 * map has been through (which sets its colour scale). A new image-layer kind
 * adds an entry here.
 */
export const LAYER_VIEWS: Record<
  ConvLayerSpec['kind'],
  {
    name: string;
    hasFilters: boolean;
    activation(rows: readonly ConvLayerSpec[], i: number): Activation;
  }
> = {
  conv: {
    name: 'Conv',
    hasFilters: true,
    activation: (rows, i) => (rows[i] as Extract<ConvLayerSpec, { kind: 'conv' }>).activation,
  },
  pool: {
    name: 'Pool',
    hasFilters: false,
    // Max-pooling keeps values as they were: the map is in the last conv's range.
    activation: (rows, i) => {
      for (let j = i - 1; j >= 0; j--) {
        const r = rows[j]!;
        if (r.kind === 'conv') return r.activation;
      }
      return 'linear';
    },
  },
};

/** "Conv 2", "Pool 1": rows numbered within their kind. */
export function rowTitle(rows: readonly ConvLayerSpec[], i: number): string {
  const kind = rows[i]!.kind;
  const n = rows.slice(0, i + 1).filter((r) => r.kind === kind).length;
  return `${LAYER_VIEWS[kind].name} ${n}`;
}

/** Colour scale for values from row i's feature map (fixed for tanh/sigmoid, else by the largest |value|). */
export function mapNormalisation(
  rows: readonly ConvLayerSpec[],
  i: number,
  values: Float32Array,
): Normalisation {
  const activation = LAYER_VIEWS[rows[i]!.kind].activation(rows, i);
  return NORMALISERS[activation](values, 1);
}

export interface FilterStats {
  min: number;
  max: number;
  rms: number;
  bias: number;
}

/** Each filter's weight range and RMS over all its taps, and its bias. */
export function filterStats({ W, b, outChannels: f }: ConvFilters): FilterStats[] {
  const taps = W.length / f;
  return Array.from({ length: f }, (_, j) => {
    let min = Infinity;
    let max = -Infinity;
    let sq = 0;
    for (let t = 0; t < taps; t++) {
      const v = W[t * f + j]!;
      min = Math.min(min, v);
      max = Math.max(max, v);
      sq += v * v;
    }
    return { min, max, rms: Math.sqrt(sq / taps), bias: b[j]! };
  });
}

/** Filter f's k×k kernel for input channel c, row-major. */
export function kernelOf(
  { W, outChannels: f, kernel: k }: ConvFilters,
  filter: number,
  c: number,
): Float32Array {
  const out = new Float32Array(k * k);
  for (let i = 0; i < k * k; i++) out[i] = W[(c * k * k + i) * f + filter]!;
  return out;
}

export interface ChannelStats {
  mean: number;
  max: number;
  /** Share of the map that is exactly 0 (after ReLU: not firing). */
  zeros: number;
}

/** Per-channel mean, max and share of zeros of a `[C, H, W]` map. */
export function channelStats(data: Float32Array, [c, h, w]: readonly number[]): ChannelStats[] {
  const area = h! * w!;
  return Array.from({ length: c! }, (_, ch) => {
    let sum = 0;
    let max = -Infinity;
    let zeros = 0;
    for (let p = ch * area; p < (ch + 1) * area; p++) {
      const v = data[p]!;
      sum += v;
      max = Math.max(max, v);
      if (v === 0) zeros++;
    }
    return { mean: sum / area, max, zeros: zeros / area };
  });
}
