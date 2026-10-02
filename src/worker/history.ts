/**
 * How values map onto the bin axis before equal-width binning.
 * - `asinh`: t = asinh(v / unit): linear for |v| ≪ unit, logarithmic beyond,
 *   and it keeps the sign. Suits weights, which span ±0.01 to ±50.
 * - `log10-abs`: t = log10|v|. Suits gradient magnitudes, which span decades.
 */
export type BinScale = { kind: 'asinh'; unit: number } | { kind: 'log10-abs' };

/** Histogram bins: `count` equal-width bins covering [lo, hi] on the `scale` axis. */
export interface BinSpec {
  lo: number;
  hi: number;
  count: number;
  scale: BinScale;
}

/** Value → bin axis. */
export function toBinAxis(v: number, scale: BinScale): number {
  if (scale.kind === 'asinh') return Math.asinh(v / scale.unit);
  return v === 0 ? -Infinity : Math.log10(Math.abs(v));
}

/** Bin axis → value (for `log10-abs`, the magnitude). */
export function fromBinAxis(t: number, scale: BinScale): number {
  return scale.kind === 'asinh' ? scale.unit * Math.sinh(t) : 10 ** t;
}

const WEIGHT_SCALE: BinScale = { kind: 'asinh', unit: 0.05 };
const WEIGHT_LIMIT = toBinAxis(50, WEIGHT_SCALE);

/**
 * Weights on a signed-log axis covering ±50 (each bin ≈ ×1.5 wide for |w| ≳ 0.1,
 * linear below 0.05). Values beyond ±50 land in the edge bins.
 */
export const WEIGHT_BINS: BinSpec = {
  lo: -WEIGHT_LIMIT,
  hi: WEIGHT_LIMIT,
  count: 40,
  scale: WEIGHT_SCALE,
};
/** Gradients on a log scale: bins over log10|g|. Zeros (and anything tinier) land in the bottom bin. */
export const GRAD_BINS: BinSpec = { lo: -10, hi: 1, count: 32, scale: { kind: 'log10-abs' } };

function binIndex(t: number, { lo, hi, count }: BinSpec): number {
  const k = Math.floor(((t - lo) / (hi - lo)) * count);
  return k < 0 ? 0 : k >= count ? count - 1 : k;
}

/** Writes the fraction of `values` in each bin into `out[offset .. offset + count)`. */
export function histogramInto(
  values: Float32Array,
  spec: BinSpec,
  out: Float32Array,
  offset: number,
): void {
  out.fill(0, offset, offset + spec.count);
  if (values.length === 0) return;
  // Whole counts first (exact in float32), then count / n: a saved count restores it bit for bit.
  for (let i = 0; i < values.length; i++) {
    out[offset + binIndex(toBinAxis(values[i]!, spec.scale), spec)]! += 1;
  }
  for (let k = offset; k < offset + spec.count; k++) out[k] = out[k]! / values.length;
}

export function rms(values: Float32Array): number {
  if (values.length === 0) return 0;
  let s = 0;
  for (let i = 0; i < values.length; i++) s += values[i]! * values[i]!;
  return Math.sqrt(s / values.length);
}

/** One dense layer's weights and their gradient, as fed to the recorders. */
export interface LayerSample {
  W: Float32Array;
  gradW: Float32Array;
}

export interface LayerTimeline {
  /** `[T, WEIGHT_BINS.count]` row-major: column t is the weight histogram at `epochs[t]`. */
  weightHist: Float32Array;
  /** `[T, GRAD_BINS.count]` row-major, over log10|∂L/∂W|. */
  gradHist: Float32Array;
  weightRms: Float32Array;
  gradRms: Float32Array;
}

/** A `HistogramTimeline`'s full state, for saving a run (arrays trimmed to `length` columns). */
export interface TimelineState {
  interval: number;
  epochs: Float32Array;
  layers: (LayerTimeline & {
    /** How many weights the histograms were taken over (each bin is a count / size). */
    size: number;
  })[];
}

export interface TimelineSnapshot {
  /** Epoch of each recorded column, ascending. */
  epochs: Float32Array;
  weightBins: BinSpec;
  gradBins: BinSpec;
  /** One per dense layer, in network order. */
  layers: LayerTimeline[];
}

/**
 * Per-layer weight and gradient histograms over the whole run, at bounded cost.
 * Records every `interval` epochs into at most `capacity` columns; when full it
 * drops every other column and doubles the interval, so the columns always
 * span the run evenly and each is an exact moment (never an average).
 */
export class HistogramTimeline {
  interval = 1;
  length = 0;
  private readonly epochs: Float32Array;
  private readonly layers: LayerTimeline[];
  /** Weights per layer, as last recorded. */
  private readonly sizes: number[];

  constructor(
    layerCount: number,
    readonly capacity = 128,
  ) {
    if (capacity < 2 || capacity % 2 !== 0) throw new Error('capacity must be even and ≥ 2');
    this.epochs = new Float32Array(capacity);
    this.layers = Array.from({ length: layerCount }, () => ({
      weightHist: new Float32Array(capacity * WEIGHT_BINS.count),
      gradHist: new Float32Array(capacity * GRAD_BINS.count),
      weightRms: new Float32Array(capacity),
      gradRms: new Float32Array(capacity),
    }));
    this.sizes = new Array<number>(layerCount).fill(0);
  }

  /**
   * Whether `epoch` should be recorded: every `interval` epochs counted from
   * the first recorded one (epoch 0, or where a resumed run picked up).
   */
  due(epoch: number): boolean {
    if (this.length === 0) return true;
    return (epoch - this.epochs[0]!) % this.interval === 0 && epoch > this.epochs[this.length - 1]!;
  }

  record(epoch: number, samples: LayerSample[]): void {
    if (this.length === this.capacity) this.compact();
    const t = this.length++;
    this.epochs[t] = epoch;
    samples.forEach((s, k) => {
      const l = this.layers[k]!;
      histogramInto(s.W, WEIGHT_BINS, l.weightHist, t * WEIGHT_BINS.count);
      histogramInto(s.gradW, GRAD_BINS, l.gradHist, t * GRAD_BINS.count);
      l.weightRms[t] = rms(s.W);
      l.gradRms[t] = rms(s.gradW);
      this.sizes[k] = s.W.length;
    });
  }

  exportState(): TimelineState {
    const n = this.length;
    return {
      interval: this.interval,
      epochs: this.epochs.slice(0, n),
      layers: this.layers.map((l, k) => ({
        size: this.sizes[k]!,
        weightHist: l.weightHist.slice(0, n * WEIGHT_BINS.count),
        gradHist: l.gradHist.slice(0, n * GRAD_BINS.count),
        weightRms: l.weightRms.slice(0, n),
        gradRms: l.gradRms.slice(0, n),
      })),
    };
  }

  /** Continues from a saved state (same layer count, at most `capacity` columns). */
  importState(state: TimelineState): void {
    const n = state.epochs.length;
    const fits =
      state.layers.length === this.layers.length &&
      n <= this.capacity &&
      Number.isInteger(state.interval) &&
      state.interval >= 1 &&
      state.layers.every(
        (l) =>
          l.weightHist.length === n * WEIGHT_BINS.count &&
          l.gradHist.length === n * GRAD_BINS.count &&
          l.weightRms.length === n &&
          l.gradRms.length === n,
      );
    if (!fits) throw new Error('Saved histogram timeline does not fit this network');
    this.interval = state.interval;
    this.length = n;
    this.epochs.set(state.epochs);
    state.layers.forEach((s, k) => {
      const l = this.layers[k]!;
      l.weightHist.set(s.weightHist);
      l.gradHist.set(s.gradHist);
      l.weightRms.set(s.weightRms);
      l.gradRms.set(s.gradRms);
      this.sizes[k] = s.size;
    });
  }

  /** Keeps the even columns (epochs s, s + 2·i, s + 4·i, … from the start s) and doubles the interval. */
  private compact(): void {
    const half = this.capacity / 2;
    const keep = (a: Float32Array, width: number) => {
      for (let t = 0; t < half; t++) a.copyWithin(t * width, 2 * t * width, (2 * t + 1) * width);
    };
    keep(this.epochs, 1);
    for (const l of this.layers) {
      keep(l.weightHist, WEIGHT_BINS.count);
      keep(l.gradHist, GRAD_BINS.count);
      keep(l.weightRms, 1);
      keep(l.gradRms, 1);
    }
    this.length = half;
    this.interval *= 2;
  }

  snapshot(): TimelineSnapshot {
    const n = this.length;
    return {
      epochs: this.epochs.slice(0, n),
      weightBins: WEIGHT_BINS,
      gradBins: GRAD_BINS,
      layers: this.layers.map((l) => ({
        weightHist: l.weightHist.slice(0, n * WEIGHT_BINS.count),
        gradHist: l.gradHist.slice(0, n * GRAD_BINS.count),
        weightRms: l.weightRms.slice(0, n),
        gradRms: l.gradRms.slice(0, n),
      })),
    };
  }
}

export interface ParamHistorySnapshot {
  /** Epoch of each sample, oldest first; length n. */
  epochs: Float32Array;
  /**
   * Per dense layer, param-major: weight j's history (`W[j]` in DenseWeights
   * order) is `W.subarray(j·n, (j+1)·n)`; likewise for `b`.
   */
  layers: { W: Float32Array; b: Float32Array }[];
}

/** A `ParamHistory`'s full state, for saving a run. */
export interface ParamHistoryState {
  head: number;
  length: number;
  /** `capacity` entries, ring order. */
  epochs: Float32Array;
  /** `capacity × width`, ring order. */
  ring: Float32Array;
}

/** The last `capacity` epochs of every weight and bias, for tooltip sparklines. */
export class ParamHistory {
  private readonly epochs: Float32Array;
  /** Time-major ring: row r holds every parameter at one epoch, layers concatenated [W0, b0, W1, …]. */
  private readonly ring: Float32Array;
  private readonly width: number;
  private readonly offsets: { W: number; b: number; wSize: number; bSize: number }[];
  private head = 0;
  length = 0;

  constructor(
    sizes: { W: number; b: number }[],
    readonly capacity = 100,
  ) {
    let o = 0;
    this.offsets = sizes.map(({ W, b }) => {
      const entry = { W: o, b: o + W, wSize: W, bSize: b };
      o += W + b;
      return entry;
    });
    this.width = o;
    this.epochs = new Float32Array(capacity);
    this.ring = new Float32Array(capacity * o);
  }

  record(epoch: number, layers: { W: Float32Array; b: Float32Array }[]): void {
    const row = this.head * this.width;
    this.epochs[this.head] = epoch;
    layers.forEach((l, k) => {
      const off = this.offsets[k]!;
      this.ring.set(l.W, row + off.W);
      this.ring.set(l.b, row + off.b);
    });
    this.head = (this.head + 1) % this.capacity;
    this.length = Math.min(this.length + 1, this.capacity);
  }

  exportState(): ParamHistoryState {
    return {
      head: this.head,
      length: this.length,
      epochs: this.epochs.slice(),
      ring: this.ring.slice(),
    };
  }

  /** Continues from a saved state of a history with the same parameter sizes and capacity. */
  importState(s: ParamHistoryState): void {
    const ok =
      s.epochs.length === this.capacity &&
      s.ring.length === this.ring.length &&
      Number.isInteger(s.head) &&
      Number.isInteger(s.length) &&
      s.head >= 0 &&
      s.head < this.capacity &&
      s.length >= 0 &&
      s.length <= this.capacity;
    if (!ok) throw new Error('Saved parameter history does not fit this network');
    this.head = s.head;
    this.length = s.length;
    this.epochs.set(s.epochs);
    this.ring.set(s.ring);
  }

  snapshot(): ParamHistorySnapshot {
    const n = this.length;
    const first = (this.head - n + this.capacity) % this.capacity;
    const rowOf = (t: number) => ((first + t) % this.capacity) * this.width;
    const epochs = new Float32Array(n);
    for (let t = 0; t < n; t++) epochs[t] = this.epochs[(first + t) % this.capacity]!;
    const unroll = (start: number, size: number) => {
      const out = new Float32Array(size * n);
      for (let t = 0; t < n; t++) {
        const row = rowOf(t) + start;
        for (let j = 0; j < size; j++) out[j * n + t] = this.ring[row + j]!;
      }
      return out;
    };
    return {
      epochs,
      layers: this.offsets.map((o) => ({ W: unroll(o.W, o.wSize), b: unroll(o.b, o.bSize) })),
    };
  }
}
