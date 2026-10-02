import { describe, expect, it } from 'vitest';
import {
  base64ToFloat32,
  base64ToUints,
  float32ToBase64,
  uintWidthFor,
  uintsToBase64,
} from './base64';

describe('base64', () => {
  it('round-trips float32 exactly, including -0, NaN and infinities', () => {
    const a = Float32Array.of(0.1, -0, Number.NaN, Infinity, -Infinity, 3.4e38, 1e-45);
    const back = base64ToFloat32(float32ToBase64(a));
    expect(Array.from(back)).toEqual(Array.from(a));
    expect(Object.is(back[1], -0)).toBe(true);
  });

  it('is little-endian', () => {
    expect(float32ToBase64(Float32Array.of(1))).toBe('AACAPw=='); // 00 00 80 3f
    expect(uintsToBase64([1, 2], 2)).toBe('AQACAA==');
  });

  it.each([1, 2, 4] as const)('round-trips %i-byte unsigned integers', (width) => {
    const max = width === 4 ? 2 ** 32 - 1 : 2 ** (8 * width) - 1;
    const values = [0, 1, max];
    expect(Array.from(base64ToUints(uintsToBase64(values, width), width))).toEqual(values);
  });

  it('picks the narrowest width', () => {
    expect([255, 256, 65535, 65536].map(uintWidthFor)).toEqual([1, 2, 2, 4]);
  });

  it('round-trips long arrays (chunked encoding)', () => {
    const a = Float32Array.from({ length: 100_000 }, (_, i) => Math.sin(i));
    expect(base64ToFloat32(float32ToBase64(a))).toEqual(a);
  });

  it('rejects bad input', () => {
    expect(() => base64ToFloat32('not base64!')).toThrow();
    expect(() => base64ToFloat32(btoa('abc'))).toThrow(/whole number of float32/);
    expect(() => base64ToUints(btoa('abc'), 2)).toThrow(/whole number of 2-byte/);
  });
});
