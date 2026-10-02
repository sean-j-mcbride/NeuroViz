/**
 * Train and test loss per epoch, at bounded size. Every epoch is kept exactly
 * until `capacity` buckets are full; then neighbouring buckets are merged
 * (mean of the pair, min of the mins, max of the maxes) and the bucket width
 * doubles, as the histogram timeline does. Memory, snapshots and saved files
 * therefore stop growing, while spikes survive in the min / max. The latest
 * and the lowest loss are tracked exactly.
 */

export const LOSS_CAPACITY = 4096;

/** One curve's buckets: bucket i covers epochs [i·width, (i + 1)·width), the last possibly partial. */
export interface LossBuckets {
  mean: Float32Array;
  min: Float32Array;
  max: Float32Array;
}

export interface LossCurveSummary extends LossBuckets {
  /** The loss after the latest epoch (NaN before any). */
  latest: number;
  /** The lowest finite loss so far (Infinity if none), and the epoch it was reached. */
  best: number;
  bestEpoch: number;
}

export interface LossHistorySnapshot {
  /** Epochs recorded; epoch 0 is before training, so this is the current epoch + 1. */
  count: number;
  /** Epochs per bucket (1 until the first merge). */
  width: number;
  train: LossCurveSummary;
  test: LossCurveSummary;
}

/** A curve's full state, for saving a run: complete buckets plus the one being filled. */
export interface LossCurveState extends LossBuckets {
  /** The bucket being filled: float64 running sum, count, min and max (n = 0 when empty). */
  partial: { sum: number; n: number; min: number; max: number };
  latest: number;
  best: number;
  bestEpoch: number;
}

export interface LossHistoryState {
  width: number;
  count: number;
  train: LossCurveState;
  test: LossCurveState;
}

class Curve {
  readonly mean: Float32Array;
  readonly min: Float32Array;
  readonly max: Float32Array;
  /** Complete buckets. */
  length = 0;
  sum = 0;
  n = 0;
  lo = Infinity;
  hi = -Infinity;
  latest = Number.NaN;
  best = Infinity;
  bestEpoch = -1;

  constructor(readonly capacity: number) {
    this.mean = new Float32Array(capacity);
    this.min = new Float32Array(capacity);
    this.max = new Float32Array(capacity);
  }

  add(v: number, epoch: number): void {
    this.sum += v;
    this.n++;
    // Math.min / max propagate NaN, so a diverged epoch shows in its bucket.
    this.lo = this.n === 1 ? v : Math.min(this.lo, v);
    this.hi = this.n === 1 ? v : Math.max(this.hi, v);
    this.latest = v;
    if (v < this.best) {
      this.best = v;
      this.bestEpoch = epoch;
    }
  }

  /** Closes the bucket being filled. */
  close(): void {
    const k = this.length++;
    this.mean[k] = this.sum / this.n;
    this.min[k] = this.lo;
    this.max[k] = this.hi;
    this.sum = 0;
    this.n = 0;
  }

  /** Merges neighbouring buckets, halving the count. */
  merge(): void {
    const half = this.length / 2;
    for (let k = 0; k < half; k++) {
      const [a, b] = [2 * k, 2 * k + 1];
      this.mean[k] = (this.mean[a]! + this.mean[b]!) / 2;
      this.min[k] = Math.min(this.min[a]!, this.min[b]!);
      this.max[k] = Math.max(this.max[a]!, this.max[b]!);
    }
    this.length = half;
  }

  summary(): LossCurveSummary {
    const k = this.length;
    const size = k + (this.n > 0 ? 1 : 0);
    const out = {
      mean: new Float32Array(size),
      min: new Float32Array(size),
      max: new Float32Array(size),
    };
    out.mean.set(this.mean.subarray(0, k));
    out.min.set(this.min.subarray(0, k));
    out.max.set(this.max.subarray(0, k));
    if (this.n > 0) {
      out.mean[k] = this.sum / this.n;
      out.min[k] = this.lo;
      out.max[k] = this.hi;
    }
    return { ...out, latest: this.latest, best: this.best, bestEpoch: this.bestEpoch };
  }

  exportState(): LossCurveState {
    const k = this.length;
    return {
      mean: this.mean.slice(0, k),
      min: this.min.slice(0, k),
      max: this.max.slice(0, k),
      partial: { sum: this.sum, n: this.n, min: this.lo, max: this.hi },
      latest: this.latest,
      best: this.best,
      bestEpoch: this.bestEpoch,
    };
  }

  importState(s: LossCurveState): void {
    this.length = s.mean.length;
    this.mean.set(s.mean);
    this.min.set(s.min);
    this.max.set(s.max);
    ({ sum: this.sum, n: this.n, min: this.lo, max: this.hi } = s.partial);
    this.latest = s.latest;
    this.best = s.best;
    this.bestEpoch = s.bestEpoch;
  }
}

export class LossHistory {
  width = 1;
  count = 0;
  private readonly train: Curve;
  private readonly test: Curve;

  constructor(readonly capacity = LOSS_CAPACITY) {
    if (capacity < 2 || capacity % 2 !== 0) throw new Error('capacity must be even and ≥ 2');
    this.train = new Curve(capacity);
    this.test = new Curve(capacity);
  }

  /** Records one epoch's losses (epoch = the number recorded so far). */
  push(train: number, test: number): void {
    const epoch = this.count++;
    this.train.add(train, epoch);
    this.test.add(test, epoch);
    if (this.train.n < this.width) return;
    if (this.train.length === this.capacity) {
      // Full: merge pairs and double the width; the open bucket is now half
      // of a wider one and keeps filling.
      this.train.merge();
      this.test.merge();
      this.width *= 2;
      return;
    }
    this.train.close();
    this.test.close();
  }

  snapshot(): LossHistorySnapshot {
    return {
      count: this.count,
      width: this.width,
      train: this.train.summary(),
      test: this.test.summary(),
    };
  }

  exportState(): LossHistoryState {
    return {
      width: this.width,
      count: this.count,
      train: this.train.exportState(),
      test: this.test.exportState(),
    };
  }

  /** Continues from a saved state; throws if it is inconsistent. */
  importState(s: LossHistoryState): void {
    const { width, count, train, test } = s;
    const buckets = train.mean.length;
    const curveOk = (c: LossCurveState) =>
      c.mean.length === buckets &&
      c.min.length === buckets &&
      c.max.length === buckets &&
      c.partial.n === count - buckets * width;
    const ok =
      Number.isInteger(width) &&
      width >= 1 &&
      (width & (width - 1)) === 0 &&
      buckets <= this.capacity &&
      curveOk(train) &&
      curveOk(test) &&
      train.partial.n >= 0 &&
      train.partial.n < width;
    if (!ok) throw new Error('Saved loss history is inconsistent');
    this.width = width;
    this.count = count;
    this.train.importState(train);
    this.test.importState(test);
  }

  /** Replays per-epoch losses (as older saved files hold them). */
  static fromSeries(
    train: Float32Array,
    test: Float32Array,
    capacity = LOSS_CAPACITY,
  ): LossHistory {
    if (train.length !== test.length) throw new Error('Train and test loss lengths differ');
    const h = new LossHistory(capacity);
    for (let i = 0; i < train.length; i++) h.push(train[i]!, test[i]!);
    return h;
  }
}
