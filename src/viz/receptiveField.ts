import { type Shape, paddingAmount } from '../engine';
import type { ConvLayerSpec } from '../worker';

/** Inclusive row and column ranges of a map. */
export interface Rect {
  y0: number;
  y1: number;
  x0: number;
  x1: number;
}

/** A row's sliding window: kernel (or pool) size, step and zero padding. */
function windowOf(row: ConvLayerSpec): { size: number; stride: number; pad: number } {
  return row.kind === 'pool'
    ? { size: row.size, stride: row.size, pad: 0 }
    : { size: row.kernel, stride: row.stride, pad: paddingAmount(row.kernel, row.padding) };
}

/**
 * Which part of each earlier map a feature-map pixel depends on: the pixel at
 * (y, x) of conv row `row`'s output, traced back through every row's window.
 *
 * `shapes[i]` is row i's output shape `[C, H, W]`, and `input` the image's.
 * Returns the rectangle in the input image and in the output of every row
 * before `row` (`maps[i]` for i < row). Windows that hang over a padded edge
 * are clipped to the map: the padding is zeros, which depend on nothing.
 * For a conv stack this is exactly the set of inputs the pixel depends on;
 * through max-pooling it is where it can depend (each pool picks one).
 */
export function receptiveField(
  rows: readonly ConvLayerSpec[],
  shapes: readonly Shape[],
  input: Shape,
  row: number,
  y: number,
  x: number,
): { input: Rect; maps: Rect[] } {
  let r: Rect = { y0: y, y1: y, x0: x, x1: x };
  const maps: Rect[] = [];
  for (let i = row; i >= 0; i--) {
    const { size, stride, pad } = windowOf(rows[i]!);
    const [, h, w] = (i > 0 ? shapes[i - 1]! : input) as [number, number, number];
    r = {
      y0: Math.max(0, r.y0 * stride - pad),
      y1: Math.min(h - 1, r.y1 * stride - pad + size - 1),
      x0: Math.max(0, r.x0 * stride - pad),
      x1: Math.min(w - 1, r.x1 * stride - pad + size - 1),
    };
    if (i > 0) maps[i - 1] = r;
  }
  return { input: r, maps };
}

/** "rows 8–19, columns 12–23 (12 × 12 pixels)". */
export function describeRect({ y0, y1, x0, x1 }: Rect, unit = 'pixels'): string {
  return `rows ${y0}–${y1}, columns ${x0}–${x1} (${y1 - y0 + 1} × ${x1 - x0 + 1} ${unit})`;
}
