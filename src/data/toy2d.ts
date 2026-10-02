import { Rng, Tensor } from '../engine';

export type Toy2DKind = 'circle' | 'xor' | 'spirals' | 'gaussians';

export const TOY2D_KINDS: readonly Toy2DKind[] = ['circle', 'xor', 'spirals', 'gaussians'];

/** Every point lies in [−DOMAIN, DOMAIN]² (the TF Playground convention). */
export const DOMAIN = 6;

/** Clean data spans radius ≈ 5, leaving room for noise inside the domain. */
const RADIUS = 5;

/** Std of the Gaussian position noise at `noise = 1`, in domain units. */
const NOISE_STD = 2;

export interface Toy2DOptions {
  /** Total number of points; classes are balanced (exactly, when n is even). */
  n: number;
  /** 0 = clean. Points are jittered by N(0, (noise·NOISE_STD)²); labels are kept, so classes overlap. */
  noise: number;
  seed: number;
}

export interface Dataset {
  /** Inputs `[N, 2]`. */
  x: Tensor;
  /** Labels `[N, 1]`, each 0 or 1. */
  y: Tensor;
}

type Sampler = (rng: Rng, label: 0 | 1, i: number, perClass: number) => [number, number];

function uniform(rng: Rng, lo: number, hi: number): number {
  return lo + (hi - lo) * rng.next();
}

function sign(rng: Rng): number {
  return rng.next() < 0.5 ? -1 : 1;
}

function polar(r: number, theta: number): [number, number] {
  return [r * Math.cos(theta), r * Math.sin(theta)];
}

const SAMPLERS: Record<Toy2DKind, Sampler> = {
  // Label 1 inside r < 0.5R, label 0 in a ring 0.7R..R.
  circle: (rng, label) => {
    const r = label === 1 ? uniform(rng, 0, 0.5 * RADIUS) : uniform(rng, 0.7 * RADIUS, RADIUS);
    return polar(r, uniform(rng, 0, 2 * Math.PI));
  },
  // Label 1 where x and y share a sign; points kept slightly off the axes.
  xor: (rng, label) => {
    const pad = 0.3;
    const sx = sign(rng);
    const sy = label === 1 ? sx : -sx;
    return [sx * uniform(rng, pad, RADIUS), sy * uniform(rng, pad, RADIUS)];
  },
  // Two interleaved arms, 1.75 turns each, radius growing linearly to R.
  spirals: (_rng, label, i, perClass) => {
    const t = i / perClass;
    return polar(t * RADIUS, 1.75 * 2 * Math.PI * t + (label === 1 ? 0 : Math.PI));
  },
  // Unit-variance blobs at (2, 2) [label 1] and (−2, −2) [label 0].
  gaussians: (rng, label) => {
    const c = label === 1 ? 2 : -2;
    return [rng.normal(c, 1), rng.normal(c, 1)];
  },
};

function clamp(v: number): number {
  return Math.min(DOMAIN, Math.max(-DOMAIN, v));
}

/** Generates a labelled 2D toy dataset. Deterministic for a given seed. */
export function generateToy2D(kind: Toy2DKind, { n, noise, seed }: Toy2DOptions): Dataset {
  if (!Number.isInteger(n) || n < 2)
    throw new Error(`generateToy2D: n must be an integer ≥ 2, got ${n}`);
  const rng = new Rng(seed);
  const sample = SAMPLERS[kind];
  const x = Tensor.zeros([n, 2]);
  const y = Tensor.zeros([n, 1]);
  const ones = Math.floor(n / 2);
  const std = noise * NOISE_STD;
  for (let row = 0; row < n; row++) {
    // Labels alternate 0, 1, 0, 1, … so the classes are balanced (and so is any prefix).
    const label = (row % 2) as 0 | 1;
    const perClass = label === 1 ? ones : n - ones;
    const i = Math.floor(row / 2);
    const [px, py] = sample(rng, label, i, perClass);
    x.data[2 * row] = clamp(px + (std > 0 ? rng.normal(0, std) : 0));
    x.data[2 * row + 1] = clamp(py + (std > 0 ? rng.normal(0, std) : 0));
    y.data[row] = label;
  }
  return { x, y };
}
