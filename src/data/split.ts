import { Rng, Tensor } from '../engine';
import type { Dataset } from './toy2d';

/** Fisher–Yates shuffle of `indices` in place. */
export function shuffleInPlace(indices: Uint32Array, rng: Rng): Uint32Array {
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(rng.next() * (i + 1));
    const t = indices[i]!;
    indices[i] = indices[j]!;
    indices[j] = t;
  }
  return indices;
}

export function range(n: number): Uint32Array {
  const r = new Uint32Array(n);
  for (let i = 0; i < n; i++) r[i] = i;
  return r;
}

/**
 * Copies rows `indices[start .. start + out.rows)` of `src` into `out`.
 * Used to assemble mini-batches without allocating.
 */
export function gatherRows(src: Tensor, indices: Uint32Array, start: number, out: Tensor): Tensor {
  const cols = src.cols;
  if (out.cols !== cols) throw new Error(`gatherRows: out has ${out.cols} cols, expected ${cols}`);
  for (let r = 0; r < out.rows; r++) {
    const s = indices[start + r]! * cols;
    out.data.set(src.data.subarray(s, s + cols), r * cols);
  }
  return out;
}

function take(src: Tensor, indices: Uint32Array, start: number, count: number): Tensor {
  return gatherRows(src, indices, start, Tensor.zeros([count, src.cols]));
}

/** Seeded random split. The test set gets round(n · testFraction) rows. */
export function splitTrainTest(
  data: Dataset,
  testFraction: number,
  rng: Rng,
): { train: Dataset; test: Dataset } {
  const n = data.x.rows;
  const nTest = Math.round(n * testFraction);
  const idx = shuffleInPlace(range(n), rng);
  return {
    test: { x: take(data.x, idx, 0, nTest), y: take(data.y, idx, 0, nTest) },
    train: { x: take(data.x, idx, nTest, n - nTest), y: take(data.y, idx, nTest, n - nTest) },
  };
}
