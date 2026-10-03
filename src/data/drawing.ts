import { MNIST_PIXELS, MNIST_SIDE } from './mnist';

/** MNIST digits were scaled so the longer side of their bounding box is 20 pixels. */
const BOX = 20;
/** Source pixels fainter than this are ignored when finding the bounding box. */
const INK = 0.05;

/**
 * Turns a drawing into an MNIST-style input, the way MNIST itself was made:
 * crop to the ink's bounding box, scale (keeping the aspect ratio) so the
 * longer side is 20 pixels, then place it in a 28×28 image with its centre of
 * mass in the middle. Scaling is area-averaged, so thin or thick strokes keep
 * their weight. `alpha` is `[h · w]`, row-major, ink in [0, 1].
 * Returns null when nothing has been drawn.
 */
export function preprocessDrawing(alpha: Float32Array, w: number, h: number): Float32Array | null {
  if (alpha.length !== w * h) throw new Error(`preprocessDrawing: expected ${w * h} values`);
  let x0 = w;
  let y0 = h;
  let x1 = -1;
  let y1 = -1;
  let mass = 0;
  let mx = 0;
  let my = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const a = alpha[y * w + x]!;
      if (a <= INK) continue;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
      mass += a;
      mx += a * (x + 0.5);
      my += a * (y + 0.5);
    }
  }
  if (x1 < 0) return null;
  const scale = BOX / Math.max(x1 - x0 + 1, y1 - y0 + 1);
  // Source x maps to (x − x0)·scale + offX; put the centre of mass at the image centre.
  const offX = MNIST_SIDE / 2 - (mx / mass - x0) * scale;
  const offY = MNIST_SIDE / 2 - (my / mass - y0) * scale;

  // Splat each source pixel over the target pixels it overlaps, weighted by the overlap area.
  const out = new Float32Array(MNIST_PIXELS);
  for (let y = y0; y <= y1; y++) {
    const ty0 = (y - y0) * scale + offY;
    const ty1 = ty0 + scale;
    for (let x = x0; x <= x1; x++) {
      const a = alpha[y * w + x]!;
      if (a <= INK) continue;
      const tx0 = (x - x0) * scale + offX;
      const tx1 = tx0 + scale;
      for (let ty = Math.max(0, Math.floor(ty0)); ty < Math.min(MNIST_SIDE, Math.ceil(ty1)); ty++) {
        const oy = Math.min(ty + 1, ty1) - Math.max(ty, ty0);
        for (
          let tx = Math.max(0, Math.floor(tx0));
          tx < Math.min(MNIST_SIDE, Math.ceil(tx1));
          tx++
        ) {
          const ox = Math.min(tx + 1, tx1) - Math.max(tx, tx0);
          out[ty * MNIST_SIDE + tx]! += a * ox * oy;
        }
      }
    }
  }
  for (let i = 0; i < out.length; i++) out[i] = Math.min(1, out[i]!);
  return out;
}
