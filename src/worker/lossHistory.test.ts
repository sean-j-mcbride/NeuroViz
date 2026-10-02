import { describe, expect, it } from 'vitest';
import { LOSS_CAPACITY, LossHistory } from './lossHistory';

const f32 = (...v: number[]) => Float32Array.from(v);

function filled(values: number[], capacity: number): LossHistory {
  const h = new LossHistory(capacity);
  values.forEach((v) => h.push(v, 2 * v));
  return h;
}

describe('LossHistory', () => {
  it('keeps every epoch exactly until full', () => {
    const s = filled([0.7, 0.5, 0.6], 4).snapshot();
    expect(s).toMatchObject({ count: 3, width: 1 });
    expect(s.train.mean).toEqual(f32(0.7, 0.5, 0.6));
    expect(s.train.min).toEqual(s.train.mean);
    expect(s.train.max).toEqual(s.train.mean);
    expect(s.test.mean).toEqual(f32(1.4, 1, 1.2));
    expect([s.train.latest, s.train.best, s.train.bestEpoch]).toEqual([0.6, 0.5, 1]);
  });

  it('when full, merges neighbours (mean, min, max) and doubles the width', () => {
    // capacity 4: epochs 0–3 fill it; epoch 4 triggers the merge.
    const s = filled([1, 3, 2, 6, 5], 4).snapshot();
    expect(s).toMatchObject({ count: 5, width: 2 });
    // Buckets [0,1] [2,3] complete; [4] still filling.
    expect(s.train.mean).toEqual(f32(2, 4, 5));
    expect(s.train.min).toEqual(f32(1, 2, 5));
    expect(s.train.max).toEqual(f32(3, 6, 5));
    const t = filled([1, 3, 2, 6, 5, 7], 4).snapshot();
    expect(t.train.mean).toEqual(f32(2, 4, 6));
    expect(t.train.max).toEqual(f32(3, 6, 7));
  });

  it('keeps merging as the run grows, so the size stays bounded', () => {
    const h = new LossHistory();
    for (let i = 0; i < 100_000; i++) h.push(1 / (i + 1), 2 / (i + 1));
    const s = h.snapshot();
    expect(s.count).toBe(100_000);
    expect(s.width).toBe(32);
    expect(s.train.mean.length).toBeLessThanOrEqual(LOSS_CAPACITY + 1);
    expect(s.train.max[0]).toBe(1); // the first epoch survives as a max
    expect(s.train.best).toBe(1 / 100_000);
    expect(s.train.bestEpoch).toBe(99_999);
  });

  it('a spike survives in the max after merging', () => {
    const values = Array.from({ length: 64 }, (_, i) => (i === 37 ? 9 : 0.5));
    const s = filled(values, 8).snapshot();
    expect(s.width).toBe(8);
    expect(Math.max(...s.train.max)).toBe(9);
    expect(Math.max(...s.train.mean)).toBeLessThan(2);
  });

  it('NaN and Infinity propagate into their bucket; best ignores them', () => {
    const s = filled([0.5, Number.NaN, 0.4, Infinity, 0.3], 2).snapshot();
    expect(s.train.mean[0]).toBeNaN();
    expect(s.train.max[1]).toBe(Infinity);
    expect(s.train.best).toBeCloseTo(0.3);
  });

  it('export → import continues identically', () => {
    const values = Array.from({ length: 1000 }, (_, i) => Math.sin(i) + 2);
    const straight = filled(values, 16);
    const first = filled(values.slice(0, 437), 16);
    const resumed = new LossHistory(16);
    resumed.importState(structuredClone(first.exportState()));
    values.slice(437).forEach((v) => resumed.push(v, 2 * v));
    expect(resumed.exportState()).toEqual(straight.exportState());
    expect(resumed.snapshot()).toEqual(straight.snapshot());
  });

  it('fromSeries equals pushing the same float32 values online', () => {
    const train = Float32Array.from({ length: 300 }, (_, i) => Math.cos(i) + 1);
    const test = Float32Array.from(train, (v) => v * 1.5);
    const online = new LossHistory(32);
    train.forEach((v, i) => online.push(v, test[i]!));
    expect(LossHistory.fromSeries(train, test, 32).exportState()).toEqual(online.exportState());
  });

  it('rejects an inconsistent saved state', () => {
    const state = filled([1, 2, 3, 4, 5], 4).exportState();
    const h = new LossHistory(4);
    expect(() => h.importState({ ...state, count: 9 })).toThrow(/inconsistent/);
    expect(() => h.importState({ ...state, width: 3 })).toThrow(/inconsistent/);
  });
});
