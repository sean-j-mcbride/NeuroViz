import { type BinSpec, fromBinAxis } from '../worker';

/**
 * Fraction of column t of a `[T, bins.count]` histogram whose bins satisfy
 * `test`, applied to each bin's centre value. Approximate to within one bin,
 * because bin edges need not line up with the threshold.
 */
export function shareWhere(
  hist: Float32Array,
  t: number,
  bins: BinSpec,
  test: (centre: number) => boolean,
): number {
  const step = (bins.hi - bins.lo) / bins.count;
  let share = 0;
  for (let b = 0; b < bins.count; b++) {
    if (test(fromBinAxis(bins.lo + (b + 0.5) * step, bins.scale)))
      share += hist[t * bins.count + b]!;
  }
  return share;
}
