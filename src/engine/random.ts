/**
 * Seeded pseudo-random number generator (mulberry32). Every source of
 * randomness in the engine goes through an `Rng` so runs are reproducible.
 */
export class Rng {
  private state: number;
  private spare: number | null = null;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  /** Uniform float in [0, 1). */
  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Normally distributed float (Box–Muller; the second sample is cached). */
  normal(mean = 0, std = 1): number {
    if (this.spare !== null) {
      const z = this.spare;
      this.spare = null;
      return mean + std * z;
    }
    let u = 0;
    while (u === 0) u = this.next(); // avoid log(0)
    const v = this.next();
    const r = Math.sqrt(-2 * Math.log(u));
    this.spare = r * Math.sin(2 * Math.PI * v);
    return mean + std * r * Math.cos(2 * Math.PI * v);
  }
}
