import { describe, expect, it } from 'vitest';
import { preprocessDrawing } from './drawing';
import { MNIST_SIDE } from './mnist';

const W = 280;

/** A ring of thickness t around (cx, cy) with outer radii (rx, ry), as a W×W ink buffer. */
function ring(cx: number, cy: number, rx: number, ry: number, t: number): Float32Array {
  const a = new Float32Array(W * W);
  for (let y = 0; y < W; y++) {
    for (let x = 0; x < W; x++) {
      const dx = (x + 0.5 - cx) / rx;
      const dy = (y + 0.5 - cy) / ry;
      const d = Math.hypot(dx, dy);
      if (d <= 1 && d >= 1 - t) a[y * W + x] = 1;
    }
  }
  return a;
}

function centreOfMass(img: Float32Array): [number, number] {
  let m = 0;
  let mx = 0;
  let my = 0;
  img.forEach((v, i) => {
    m += v;
    mx += v * ((i % MNIST_SIDE) + 0.5);
    my += v * (Math.floor(i / MNIST_SIDE) + 0.5);
  });
  return [mx / m, my / m];
}

function meanAbsDiff(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i]! - b[i]!);
  return s / a.length;
}

describe('preprocessDrawing', () => {
  it('returns null for an empty drawing', () => {
    expect(preprocessDrawing(new Float32Array(W * W), W, W)).toBeNull();
  });

  it('puts the centre of mass at the image centre and fits a 20-pixel box', () => {
    const img = preprocessDrawing(ring(60, 200, 30, 50, 0.3), W, W)!;
    const [cx, cy] = centreOfMass(img);
    expect(Math.abs(cx - 14)).toBeLessThan(0.5);
    expect(Math.abs(cy - 14)).toBeLessThan(0.5);
    let top = MNIST_SIDE;
    let bottom = -1;
    img.forEach((v, i) => {
      const r = Math.floor(i / MNIST_SIDE);
      if (v > 0.01) {
        top = Math.min(top, r);
        bottom = Math.max(bottom, r);
      }
    });
    expect(bottom - top + 1).toBeGreaterThanOrEqual(20);
    expect(bottom - top + 1).toBeLessThanOrEqual(21);
  });

  it('gives the same image wherever the digit is drawn', () => {
    const a = preprocessDrawing(ring(80, 90, 40, 60, 0.25), W, W)!;
    const b = preprocessDrawing(ring(190, 170, 40, 60, 0.25), W, W)!;
    expect(meanAbsDiff(a, b)).toBeLessThan(1e-6);
  });

  it('gives nearly the same image whatever size the digit is drawn', () => {
    const small = preprocessDrawing(ring(140, 140, 30, 45, 0.25), W, W)!;
    const large = preprocessDrawing(ring(140, 140, 80, 120, 0.25), W, W)!;
    expect(meanAbsDiff(small, large)).toBeLessThan(0.02);
  });

  it('keeps values in [0, 1]', () => {
    const img = preprocessDrawing(ring(140, 140, 100, 100, 1), W, W)!;
    expect(Math.max(...img)).toBeLessThanOrEqual(1);
    expect(Math.min(...img)).toBeGreaterThanOrEqual(0);
  });
});
