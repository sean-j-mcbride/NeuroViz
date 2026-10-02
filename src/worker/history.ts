/** Histogram bins: `count` equal-width bins covering [lo, hi]. */
export interface BinSpec {
  lo: number;
  hi: number;
  count: number;
}

/** Weights, linear scale. Values outside the range land in the edge bins. */
export const WEIGHT_BINS: BinSpec = { lo: -4, hi: 4, count: 32 };
/** Gradients on a log scale: bins over log10|g|. Zeros (and anything tinier) land in the bottom bin. */
export const GRAD_BINS: BinSpec = { lo: -10, hi: 1, count: 32 };

function binIndex(v: number, { lo, hi, count }: BinSpec): number {
  const k = Math.floor(((v - lo) / (hi - lo)) * count);
  return k < 0 ? 0 : k >= count ? count - 1 : k;
}

/**
 * Writes the fraction of `values` in each bin into `out[offset .. offset + count)`.
 * With `logAbs`, bins log10|v| instead of v.
 */
export function histogramInto(
  values: Float32Array,
  spec: BinSpec,
  logAbs: boolean,
  out: Float32Array,
  offset: number,
): void {
  out.fill(0, offset, offset + spec.count);
  if (values.length === 0) return;
  const unit = 1 / values.length;
  for (let i = 0; i < values.length; i++) {
    const v = values[i]!;
    const x = logAbs ? (v === 0 ? -Infinity : Math.log10(Math.abs(v))) : v;
    out[offset + binIndex(x, spec)]! += unit;
  }
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
  }

  /** Whether `epoch` should be recorded. */
  due(epoch: number): boolean {
    return (
      epoch % this.interval === 0 && (this.length === 0 || epoch > this.epochs[this.length - 1]!)
    );
  }

  record(epoch: number, samples: LayerSample[]): void {
    if (this.length === this.capacity) this.compact();
    const t = this.length++;
    this.epochs[t] = epoch;
    samples.forEach((s, k) => {
      const l = this.layers[k]!;
      histogramInto(s.W, WEIGHT_BINS, false, l.weightHist, t * WEIGHT_BINS.count);
      histogramInto(s.gradW, GRAD_BINS, true, l.gradHist, t * GRAD_BINS.count);
      l.weightRms[t] = rms(s.W);
      l.gradRms[t] = rms(s.gradW);
    });
  }

  /** Keeps the even columns (epochs 0, 2·i, 4·i, …) and doubles the interval. */
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
