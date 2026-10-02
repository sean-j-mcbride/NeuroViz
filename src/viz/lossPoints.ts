import type { LossCurveSummary, LossHistorySnapshot } from '../worker';

/** Points to draw for one loss curve: x in epochs, the mean line and the min–max band. */
export interface LossPoints {
  x: Float64Array;
  mean: Float32Array;
  lo: Float32Array;
  hi: Float32Array;
  /** Whether any point covers more than one epoch, so the band carries information. */
  banded: boolean;
}

/** First and last epoch of bucket i. */
function bucketSpan(i: number, width: number, count: number): [number, number] {
  return [i * width, Math.min((i + 1) * width, count) - 1];
}

/**
 * Groups a curve's buckets so there are at most `maxPoints` points: each point
 * is the mean of its buckets' means at their centre epoch, with the lowest min
 * and highest max, so a spike survives as the band however long the run.
 */
export function lossPoints(
  curve: LossCurveSummary,
  { width, count }: Pick<LossHistorySnapshot, 'width' | 'count'>,
  maxPoints: number,
): LossPoints {
  const buckets = curve.mean.length;
  const group = Math.max(1, Math.ceil(buckets / Math.max(1, maxPoints)));
  const n = Math.ceil(buckets / group);
  const out: LossPoints = {
    x: new Float64Array(n),
    mean: new Float32Array(n),
    lo: new Float32Array(n),
    hi: new Float32Array(n),
    banded: width > 1 || group > 1,
  };
  for (let p = 0; p < n; p++) {
    const first = p * group;
    const last = Math.min(buckets, first + group) - 1;
    let sum = 0;
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = first; i <= last; i++) {
      sum += curve.mean[i]!;
      // Math.min / max propagate NaN, so a diverged stretch stays visible as a gap.
      lo = Math.min(lo, curve.min[i]!);
      hi = Math.max(hi, curve.max[i]!);
    }
    const [start] = bucketSpan(first, width, count);
    const [, end] = bucketSpan(last, width, count);
    out.x[p] = (start + end) / 2;
    out.mean[p] = sum / (last - first + 1);
    out.lo[p] = lo;
    out.hi[p] = hi;
  }
  return out;
}

/** The value shown on hover at `epoch`: exact, or a bucket's mean and range. */
export interface LossAt {
  mean: number;
  min: number;
  max: number;
  /** The epochs the bucket covers (equal when exact). */
  from: number;
  to: number;
}

export function lossAt(
  curve: LossCurveSummary,
  { width, count }: Pick<LossHistorySnapshot, 'width' | 'count'>,
  epoch: number,
): LossAt | null {
  if (epoch < 0 || epoch >= count) return null;
  const i = Math.floor(epoch / width);
  const [from, to] = bucketSpan(i, width, count);
  return { mean: curve.mean[i]!, min: curve.min[i]!, max: curve.max[i]!, from, to };
}
