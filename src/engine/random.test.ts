import { describe, expect, it } from 'vitest';
import { Rng } from './random';

describe('Rng', () => {
  it('produces the same sequence for the same seed', () => {
    const a = new Rng(123);
    const b = new Rng(123);
    for (let i = 0; i < 1000; i++) expect(a.next()).toBe(b.next());
  });

  it('produces different sequences for different seeds', () => {
    const a = new Rng(1);
    const b = new Rng(2);
    const same = Array.from({ length: 100 }, () => a.next() === b.next()).filter(Boolean);
    expect(same.length).toBeLessThan(5);
  });

  it('next() stays in [0, 1)', () => {
    const rng = new Rng(7);
    for (let i = 0; i < 10_000; i++) {
      const v = rng.next();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it('normal() has roughly the requested mean and std', () => {
    const rng = new Rng(42);
    const n = 20_000;
    const xs = Array.from({ length: n }, () => rng.normal(2, 3));
    const mean = xs.reduce((s, x) => s + x, 0) / n;
    const variance = xs.reduce((s, x) => s + (x - mean) ** 2, 0) / n;
    expect(mean).toBeCloseTo(2, 1);
    expect(Math.sqrt(variance)).toBeCloseTo(3, 1);
  });

  it('getState/setState resume the exact sequence, including a pending normal() spare', () => {
    const a = new Rng(9);
    for (let i = 0; i < 10; i++) a.next();
    a.normal(); // leaves a spare cached
    const b = new Rng(1);
    b.setState(JSON.parse(JSON.stringify(a.getState())) as ReturnType<Rng['getState']>);
    for (let i = 0; i < 100; i++) expect(b.normal()).toBe(a.normal());
    for (let i = 0; i < 100; i++) expect(b.next()).toBe(a.next());
  });
});
