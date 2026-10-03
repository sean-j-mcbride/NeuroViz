import { describe, expect, it } from 'vitest';
import { Tensor } from '../engine';
import { loadMnistBytes } from '../test/mnist';
import {
  MNIST_PIXELS,
  type MnistSet,
  gatherImages,
  mnistCount,
  oneHotInto,
  parseMnistSubset,
} from './mnist';

/** The checksum of the committed file; `npm run data:mnist` must reproduce it. */
const CHECKSUM = 1402161221;

const bytes = loadMnistBytes();
const subset = parseMnistSubset(bytes);

function perClass(set: MnistSet): number[] {
  const counts = new Array<number>(10).fill(0);
  for (const l of set.labels) counts[l]!++;
  return counts;
}

describe('the bundled MNIST subset', () => {
  it('has 10,000 training and 2,000 test images, balanced by digit', () => {
    expect(mnistCount(subset.train)).toBe(10_000);
    expect(mnistCount(subset.test)).toBe(2_000);
    expect(subset.train.images.length).toBe(10_000 * MNIST_PIXELS);
    expect(perClass(subset.train)).toEqual(new Array(10).fill(1000));
    expect(perClass(subset.test)).toEqual(new Array(10).fill(200));
  });

  it('is the file the build script makes', () => {
    expect(subset.checksum).toBe(CHECKSUM);
  });

  it('keeps official indices in order, without repeats', () => {
    for (const { source, max } of [
      { source: subset.train.source, max: 60_000 },
      { source: subset.test.source, max: 10_000 },
    ]) {
      for (let i = 1; i < source.length; i++) expect(source[i]!).toBeGreaterThan(source[i - 1]!);
      expect(source.at(-1)!).toBeLessThan(max);
    }
  });

  it('looks like digits: mostly background, some full ink', () => {
    const img = subset.train.images.subarray(0, MNIST_PIXELS);
    const zeros = img.filter((v) => v === 0).length;
    expect(zeros / MNIST_PIXELS).toBeGreaterThan(0.6);
    expect(Math.max(...img)).toBeGreaterThan(200);
  });
});

describe('parseMnistSubset', () => {
  it('rejects other files with a readable error', () => {
    expect(() => parseMnistSubset(new Uint8Array(100))).toThrow(/Not a NeuroViz MNIST file/);
    const truncated = bytes.slice(0, 1000);
    expect(() => parseMnistSubset(truncated)).toThrow(/bytes, expected/);
  });
});

describe('gatherImages / oneHotInto', () => {
  it('scale pixels to [0, 1] and one-hot the labels in the given order', () => {
    const order = Uint32Array.of(5, 2, 9);
    const x = gatherImages(subset.test, order, 1, Tensor.zeros([2, MNIST_PIXELS]));
    const y = oneHotInto(subset.test.labels, order, 1, Tensor.zeros([2, 10]));
    for (let r = 0; r < 2; r++) {
      const src = order[1 + r]!;
      for (let p = 0; p < MNIST_PIXELS; p += 37) {
        expect(x.data[r * MNIST_PIXELS + p]).toBeCloseTo(
          subset.test.images[src * MNIST_PIXELS + p]! / 255,
          6,
        );
      }
      const row = Array.from(y.data.subarray(r * 10, r * 10 + 10));
      expect(row.indexOf(1)).toBe(subset.test.labels[src]);
      expect(row.reduce((a, b) => a + b)).toBe(1);
    }
  });
});
