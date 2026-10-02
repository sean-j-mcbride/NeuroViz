import { describe, expect, it } from 'vitest';
import { GRAD_BINS, WEIGHT_BINS, histogramInto } from '../worker';
import { shareWhere } from './timelineStats';

describe('shareWhere', () => {
  it('sums the bins whose centre passes the test, per column', () => {
    const hist = new Float32Array(2 * WEIGHT_BINS.count);
    histogramInto(Float32Array.from([-5, -0.1, 0.2, 3]), WEIGHT_BINS, hist, 0);
    histogramInto(Float32Array.from([0.1, 0.1, 0.1, 9]), WEIGHT_BINS, hist, WEIGHT_BINS.count);
    const beyondOne = (c: number) => Math.abs(c) > 1;
    expect(shareWhere(hist, 0, WEIGHT_BINS, beyondOne)).toBeCloseTo(0.5, 5);
    expect(shareWhere(hist, 1, WEIGHT_BINS, beyondOne)).toBeCloseTo(0.25, 5);
  });

  it('works on log-magnitude bins', () => {
    const hist = new Float32Array(GRAD_BINS.count);
    histogramInto(Float32Array.from([0, 1e-8, 1e-3, 0.5]), GRAD_BINS, hist, 0);
    expect(shareWhere(hist, 0, GRAD_BINS, (c) => c < 1e-6)).toBeCloseTo(0.5, 5);
  });
});
