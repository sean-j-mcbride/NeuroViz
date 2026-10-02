import { describe, expect, it } from 'vitest';
import { Rng } from '../engine';
import { range, shuffleInPlace, splitTrainTest } from './split';
import { DOMAIN, TOY2D_KINDS, generateToy2D } from './toy2d';

const OPTS = { n: 200, noise: 0.3, seed: 1 };

function points(kind: (typeof TOY2D_KINDS)[number], noise = 0, n = 200) {
  const { x, y } = generateToy2D(kind, { n, noise, seed: 3 });
  return Array.from({ length: n }, (_, i) => ({
    px: x.data[2 * i]!,
    py: x.data[2 * i + 1]!,
    label: y.data[i]!,
  }));
}

describe.each(TOY2D_KINDS)('%s', (kind) => {
  it('has the right shapes and balanced 0/1 labels', () => {
    const { x, y } = generateToy2D(kind, OPTS);
    expect(x.shape).toEqual([200, 2]);
    expect(y.shape).toEqual([200, 1]);
    expect(Array.from(y.data).every((v) => v === 0 || v === 1)).toBe(true);
    expect(Array.from(y.data).filter((v) => v === 1)).toHaveLength(100);
  });

  it('stays inside the domain, even with heavy noise', () => {
    const { x } = generateToy2D(kind, { ...OPTS, noise: 1 });
    expect(Array.from(x.data).every((v) => Math.abs(v) <= DOMAIN)).toBe(true);
  });

  it('is deterministic per seed', () => {
    const a = generateToy2D(kind, OPTS);
    const b = generateToy2D(kind, OPTS);
    expect(b.x.data).toEqual(a.x.data);
    expect(b.y.data).toEqual(a.y.data);
    expect(generateToy2D(kind, { ...OPTS, seed: 2 }).x.data).not.toEqual(a.x.data);
  });
});

describe('clean (noise = 0) geometry', () => {
  it('circle: label 1 inside, label 0 outside', () => {
    for (const { px, py, label } of points('circle'))
      expect(Math.hypot(px, py) < 3 ? 1 : 0).toBe(label);
  });

  it('xor: label 1 where x and y share a sign', () => {
    for (const { px, py, label } of points('xor')) expect(px * py > 0 ? 1 : 0).toBe(label);
  });

  it('gaussians: label mostly matches the side of x + y = 0', () => {
    const pts = points('gaussians');
    const agree = pts.filter(({ px, py, label }) => (px + py > 0 ? 1 : 0) === label).length;
    expect(agree / pts.length).toBeGreaterThan(0.95);
  });

  it('spirals: the two arms are point reflections of each other', () => {
    const pts = points('spirals');
    // Rows 2i (label 0) and 2i + 1 (label 1) share radius and differ by π in angle.
    for (let i = 0; i < pts.length; i += 2) {
      expect(pts[i]!.px).toBeCloseTo(-pts[i + 1]!.px, 5);
      expect(pts[i]!.py).toBeCloseTo(-pts[i + 1]!.py, 5);
    }
  });
});

describe('splitting', () => {
  it('shuffles to a permutation', () => {
    const idx = shuffleInPlace(range(50), new Rng(1));
    expect([...idx].sort((a, b) => a - b)).toEqual([...range(50)]);
    expect([...idx]).not.toEqual([...range(50)]);
  });

  it('splits rows without losing or duplicating any', () => {
    const data = generateToy2D('spirals', OPTS);
    const { train, test } = splitTrainTest(data, 0.3, new Rng(5));
    expect(test.x.rows).toBe(60);
    expect(train.x.rows).toBe(140);
    const key = (d: typeof train, i: number) =>
      `${d.x.data[2 * i]},${d.x.data[2 * i + 1]},${d.y.data[i]}`;
    const all = [
      ...Array.from({ length: 140 }, (_, i) => key(train, i)),
      ...Array.from({ length: 60 }, (_, i) => key(test, i)),
    ].sort();
    const orig = Array.from({ length: 200 }, (_, i) => key(data, i)).sort();
    expect(all).toEqual(orig);
  });
});
