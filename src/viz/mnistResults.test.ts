import { describe, expect, it } from 'vitest';
import { confidentMistakes, confusionCounts, perClassAccuracy, recordEpoch } from './mnistResults';

const labels = Uint8Array.of(0, 1, 1, 2, 2, 2);
const predicted = Uint8Array.of(0, 7, 1, 2, 7, 1);
const confidence = Float32Array.of(0.9, 0.6, 0.8, 0.99, 0.95, 0.6);

describe('MNIST results', () => {
  it('counts true label × predicted class', () => {
    const c = confusionCounts(labels, predicted);
    expect(c.reduce((a, b) => a + b)).toBe(6);
    expect([
      c[0],
      c[1 * 10 + 7],
      c[1 * 10 + 1],
      c[2 * 10 + 2],
      c[2 * 10 + 7],
      c[2 * 10 + 1],
    ]).toEqual([1, 1, 1, 1, 1, 1]);
    const acc = perClassAccuracy(c);
    expect(acc.slice(0, 3)).toEqual([1, 0.5, 1 / 3]);
    expect(Number.isNaN(acc[3])).toBe(true);
  });

  it('lists mistakes most confident first, ties in index order, optionally for one cell', () => {
    expect(confidentMistakes(labels, predicted, confidence)).toEqual([4, 1, 5]);
    expect(confidentMistakes(labels, predicted, confidence, { label: 2, predicted: 7 })).toEqual([
      4,
    ]);
    expect(confidentMistakes(labels, predicted, confidence, { label: 0, predicted: 0 })).toEqual(
      [],
    );
  });

  it('names records in epochs', () => {
    expect(recordEpoch(0, 5000, 10000)).toBe('0');
    expect(recordEpoch(3, 5000, 10000)).toBe('1.5');
    expect(recordEpoch(20, 5000, 10000)).toBe('10');
  });
});
