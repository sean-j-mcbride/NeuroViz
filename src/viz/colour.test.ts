import { describe, expect, it } from 'vitest';
import { LUT_SIZE, NEGATIVE, NORMALISERS, POSITIVE, divergingLut, lutIndex } from './colour';

const MID = [255, 255, 255] as const;

describe('divergingLut', () => {
  const lut = divergingLut(NEGATIVE, MID, POSITIVE);
  const at = (i: number) => Array.from(lut.subarray(i * 3, i * 3 + 3));

  it('runs negative → mid → positive', () => {
    expect(lut).toHaveLength(LUT_SIZE * 3);
    expect(at(0)).toEqual([...NEGATIVE]);
    expect(at(LUT_SIZE - 1)).toEqual([...POSITIVE]);
    // 256 entries have no exact centre; both middle entries are within a step of white.
    for (const c of at(LUT_SIZE / 2)) expect(c).toBeGreaterThan(245);
  });
});

describe('lutIndex', () => {
  it('maps −1, 0, 1 to the ends and middle, and clamps', () => {
    expect(lutIndex(-1)).toBe(0);
    expect(lutIndex(1)).toBe(LUT_SIZE - 1);
    expect(Math.abs(lutIndex(0) - (LUT_SIZE - 1) / 2)).toBeLessThanOrEqual(0.5);
    expect(lutIndex(-7)).toBe(0);
    expect(lutIndex(7)).toBe(LUT_SIZE - 1);
  });
});

describe('NORMALISERS', () => {
  const apply = (n: { scale: number; offset: number }, v: number) => v * n.scale + n.offset;

  it('maps each kind’s natural range onto [−1, 1]', () => {
    const v = Float32Array.from([0, 0.5, 4]);
    expect(apply(NORMALISERS.input(v, 6), -6)).toBe(-1);
    expect(apply(NORMALISERS.sigmoid(v, 6), 0)).toBe(-1);
    expect(apply(NORMALISERS.output(v, 6), 1)).toBe(1);
    expect(apply(NORMALISERS.tanh(v, 6), -1)).toBe(-1);
    expect(apply(NORMALISERS.relu(v, 6), 4)).toBe(1);
    expect(apply(NORMALISERS.linear(Float32Array.from([-8, 2]), 6), -8)).toBe(-1);
  });

  it('copes with an all-zero (dead) neuron', () => {
    expect(NORMALISERS.relu(new Float32Array(4), 6).scale).toBe(1);
  });
});
