import { describe, expect, it } from 'vitest';
import { LossHistory } from '../worker';
import { lossAt, lossPoints } from './lossPoints';

function history(values: number[], capacity: number) {
  const h = new LossHistory(capacity);
  values.forEach((v) => h.push(v, v));
  return h.snapshot();
}

describe('lossPoints', () => {
  it('one point per epoch when they fit, with no band', () => {
    const s = history([3, 2, 1], 8);
    const p = lossPoints(s.train, s, 100);
    expect(Array.from(p.x)).toEqual([0, 1, 2]);
    expect(Array.from(p.mean)).toEqual([3, 2, 1]);
    expect(p.banded).toBe(false);
  });

  it('groups to the point budget, keeping each group’s extremes', () => {
    const s = history([1, 9, 2, 2, 3, 1], 8);
    const p = lossPoints(s.train, s, 3);
    expect(Array.from(p.x)).toEqual([0.5, 2.5, 4.5]);
    expect(Array.from(p.mean)).toEqual([5, 2, 2]);
    expect(Array.from(p.hi)).toEqual([9, 2, 3]);
    expect(Array.from(p.lo)).toEqual([1, 2, 1]);
    expect(p.banded).toBe(true);
  });

  it('places merged buckets at their centre epoch, including a partial last one', () => {
    const s = history([1, 3, 2, 6, 5], 4); // width 2: [0,1] [2,3] [4]
    const p = lossPoints(s.train, s, 100);
    expect(Array.from(p.x)).toEqual([0.5, 2.5, 4]);
    expect(p.banded).toBe(true);
  });
});

describe('lossAt', () => {
  it('is exact per epoch before merging, and a bucket after', () => {
    const exact = history([3, 2, 1], 8);
    expect(lossAt(exact.train, exact, 1)).toEqual({ mean: 2, min: 2, max: 2, from: 1, to: 1 });
    const merged = history([1, 3, 2, 6, 5], 4);
    expect(lossAt(merged.train, merged, 3)).toEqual({ mean: 4, min: 2, max: 6, from: 2, to: 3 });
    expect(lossAt(merged.train, merged, 4)).toEqual({ mean: 5, min: 5, max: 5, from: 4, to: 4 });
    expect(lossAt(merged.train, merged, 5)).toBeNull();
  });
});
