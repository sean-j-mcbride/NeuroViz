import { describe, expect, it } from 'vitest';
import { lossAxis } from './lossAxis';

/** A smooth decay from 0.7 towards 0.1 over n epochs. */
const decay = (n: number) =>
  Float32Array.from({ length: n }, (_, i) => 0.1 + 0.6 * Math.exp(-i / 50));

describe('lossAxis', () => {
  it('keeps the full range for smooth curves, from 0 on a linear scale', () => {
    const a = lossAxis([decay(500)], false)!;
    expect(a.lo).toBe(0);
    expect(a.hi).toBeCloseTo(0.7, 5);
    expect(a.clippedAbove).toBeNull();
  });

  it('clips a lone spike and reports its height', () => {
    const s = decay(1000);
    s[600] = 13;
    const a = lossAxis([s], false)!;
    expect(a.hi).toBeLessThan(1);
    expect(a.clippedAbove).toBe(13);
  });

  it('never cuts the starting value, even when the descent is steep and the run long', () => {
    const s = Float32Array.from({ length: 20_000 }, (_, i) => 0.05 + 0.65 * Math.exp(-i / 20));
    expect(lossAxis([s], false)!.hi).toBeCloseTo(0.7, 5);
  });

  it('always shows the latest values, even if they are the highest', () => {
    const s = decay(1000);
    s[999] = 3; // diverging right now
    const a = lossAxis([s], false)!;
    expect(a.hi).toBe(3);
    expect(a.clippedAbove).toBeNull();
  });

  it('considers every series (e.g. a reference run)', () => {
    const spiky = decay(1000);
    spiky[100] = 50;
    const a = lossAxis([decay(1000), spiky], false)!;
    expect(a.clippedAbove).toBe(50);
  });

  it('keeps the full range on a log scale, ignoring values ≤ 0', () => {
    const s = decay(1000);
    s[600] = 13;
    s[601] = 0;
    expect(lossAxis([s], true)).toEqual({
      lo: Math.fround(0.1 + 0.6 * Math.exp(-999 / 50)),
      hi: 13,
      clippedAbove: null,
    });
  });

  it('returns null when nothing is plottable', () => {
    expect(lossAxis([Float32Array.of(Number.NaN, Infinity)], false)).toBeNull();
    expect(lossAxis([new Float32Array(0)], false)).toBeNull();
  });
});
