/** The y-range for loss curves. */
export interface LossAxis {
  lo: number;
  hi: number;
  /** The highest value, when it lies above `hi` and is drawn clipped; otherwise null. */
  clippedAbove: number | null;
}

/** Values used for the percentile; enough for a stable estimate at any run length. */
const SAMPLE = 2000;
const PERCENTILE = 0.98;

/**
 * A y-range that one spike can't squash. On a linear scale the top is the
 * smallest of the true maximum and the largest of: 1.25 × the 98th percentile,
 * 1.05 × each curve's starting value (so the early descent is never cut) and
 * 1.1 × each curve's latest value (so where training is now is always shown).
 * Anything above is clipped and reported. A log scale already copes with spikes,
 * so it keeps the full range. Returns null when there is nothing to plot.
 */
export function lossAxis(series: readonly Float32Array[], logScale: boolean): LossAxis | null {
  const usable = (v: number) => Number.isFinite(v) && (!logScale || v > 0);
  let lo = Infinity;
  let hi = -Infinity;
  for (const s of series) {
    for (const v of s) {
      if (!usable(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!Number.isFinite(lo)) return null;
  if (logScale) return { lo, hi, clippedAbove: null };

  const total = series.reduce((n, s) => n + s.length, 0);
  const stride = Math.max(1, Math.floor(total / SAMPLE));
  const sample: number[] = [];
  let floor = 0;
  for (const s of series) {
    for (let i = 0; i < s.length; i += stride) if (usable(s[i]!)) sample.push(s[i]!);
    const first = s.find(usable);
    let last: number | undefined;
    for (let i = s.length - 1; i >= 0 && last === undefined; i--) if (usable(s[i]!)) last = s[i];
    if (first !== undefined) floor = Math.max(floor, 1.05 * first);
    if (last !== undefined) floor = Math.max(floor, 1.1 * last);
  }
  sample.sort((a, b) => a - b);
  const q = sample[Math.floor(PERCENTILE * (sample.length - 1))] ?? hi;
  const top = Math.min(hi, Math.max(1.25 * q, floor));
  return { lo: 0, hi: top, clippedAbove: top < hi ? hi : null };
}
