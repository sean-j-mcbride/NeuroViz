import { describe, expect, it } from 'vitest';
import {
  GRAD_BINS,
  HistogramTimeline,
  ParamHistory,
  WEIGHT_BINS,
  histogramInto,
  rms,
} from './history';

const f32 = (...v: number[]) => Float32Array.from(v);

describe('histogramInto', () => {
  it('bins fractions, clamping out-of-range values into the edge bins', () => {
    const out = new Float32Array(4);
    histogramInto(f32(-10, -0.5, 0.5, 0.6, 10), { lo: -1, hi: 1, count: 4 }, false, out, 0);
    [1, 1, 0, 3].forEach((count, k) => expect(out[k]! * 5).toBeCloseTo(count, 5));
  });

  it('bins log10|v| for gradients, with zeros in the bottom bin', () => {
    const out = new Float32Array(GRAD_BINS.count + 2);
    histogramInto(f32(0, 1e-3, -1e-3, 1), GRAD_BINS, true, out, 1);
    expect(out[0]).toBe(0); // untouched before the offset
    expect(out[1]).toBeCloseTo(0.25); // the zero
    const binOf = (log: number) =>
      1 + Math.floor(((log - GRAD_BINS.lo) / (GRAD_BINS.hi - GRAD_BINS.lo)) * GRAD_BINS.count);
    expect(out[binOf(-3)]).toBeCloseTo(0.5);
    expect(out[binOf(0)]).toBeCloseTo(0.25);
  });

  it('rms', () => {
    expect(rms(f32(3, -4))).toBeCloseTo(Math.sqrt(12.5));
    expect(rms(f32())).toBe(0);
  });
});

describe('HistogramTimeline', () => {
  const sample = (v: number) => [{ W: f32(v, v), gradW: f32(v / 10, v / 10) }];

  it('records every epoch until full, then keeps every other column and doubles the interval', () => {
    const t = new HistogramTimeline(1, 4);
    for (let e = 0; e < 20; e++) if (t.due(e)) t.record(e, sample(e));
    // 0,1,2,3 → full; 4: compact to 0,2 (interval 2) → 0,2,4,6 → 8: compact to 0,4 (interval 4)
    // → 0,4,8,12 → 16: compact to 0,8 (interval 8) → 0,8,16.
    const snap = t.snapshot();
    expect(Array.from(snap.epochs)).toEqual([0, 8, 16]);
    expect(t.interval).toBe(8);
    expect(Array.from(snap.layers[0]!.weightRms)).toEqual([0, 8, 16]);
    expect(snap.layers[0]!.weightHist).toHaveLength(3 * WEIGHT_BINS.count);
  });

  it('does not record the same epoch twice', () => {
    const t = new HistogramTimeline(1, 4);
    t.record(0, sample(1));
    expect(t.due(0)).toBe(false);
    expect(t.due(1)).toBe(true);
  });
});

describe('ParamHistory', () => {
  it('returns the most recent epochs oldest first, param-major', () => {
    const h = new ParamHistory([{ W: 2, b: 1 }], 3);
    for (let e = 0; e < 5; e++) h.record(e, [{ W: f32(e, 10 * e), b: f32(-e) }]);
    const snap = h.snapshot();
    expect(Array.from(snap.epochs)).toEqual([2, 3, 4]);
    expect(Array.from(snap.layers[0]!.W)).toEqual([2, 3, 4, 20, 30, 40]);
    expect(Array.from(snap.layers[0]!.b)).toEqual([-2, -3, -4]);
  });

  it('handles a partly filled buffer', () => {
    const h = new ParamHistory([{ W: 1, b: 1 }], 5);
    h.record(0, [{ W: f32(7), b: f32(8) }]);
    expect(Array.from(h.snapshot().layers[0]!.W)).toEqual([7]);
  });
});
