import { describe, expect, it } from 'vitest';
import { type Snapshot, TrainingSession } from '../worker';
import { type PlaygroundConfig, toSessionConfig } from './config';
import { PRESETS, type PresetId } from './presets';
import { parseConfig } from './validate';

function run(config: PlaygroundConfig, epochs: number): Snapshot {
  const s = new TrainingSession({ ...toSessionConfig(config), gridSize: 25 });
  for (let e = 0; e < epochs; e++) s.trainEpoch();
  return s.snapshot();
}

const preset = (id: PresetId) => PRESETS.find((p) => p.id === id)!;
const last = (a: Float32Array) => a[a.length - 1]!;

/** Fraction of ReLU neurons the snapshot flags as dead (0 for every training point). */
function deadFraction(s: Snapshot): number {
  const flags = s.columns.flatMap((c) => (c.dead ? Array.from(c.dead) : []));
  return flags.reduce((n, d) => n + d, 0) / flags.length;
}

/** Fraction of epochs whose training loss rose by more than 5 %. */
function jumpiness(s: Snapshot): number {
  let ups = 0;
  for (let i = 1; i < s.losses.train.mean.length; i++) {
    if (s.losses.train.mean[i]! > 1.05 * s.losses.train.mean[i - 1]!) ups++;
  }
  return ups / (s.losses.train.mean.length - 1);
}

describe('presets', () => {
  it('are valid configs the controls can show, and each fix differs from its preset', () => {
    for (const p of PRESETS) {
      expect(parseConfig(p.config).warnings).toEqual([]);
      expect(parseConfig(p.fix).warnings).toEqual([]);
      expect(p.fix).not.toEqual(p.config);
    }
  });

  // Seeds 1–5 for 300 epochs: train accuracy 65–67 %, |train − test loss| ≤ 0.03;
  // the fix reaches 99.3 %.
  it('Underfitting: train and test loss level off together, high; the fix fits', () => {
    const p = preset('underfitting');
    const s = run(p.config, 300);
    expect(s.trainAccuracy).toBeLessThan(0.75);
    expect(last(s.losses.train.mean)).toBeGreaterThan(0.5);
    expect(Math.abs(last(s.losses.train.mean) - last(s.losses.test.mean))).toBeLessThan(0.05);
    expect(run(p.fix, 300).trainAccuracy).toBeGreaterThan(0.95);
  });

  // Seeds 1–5 for 1000 epochs: test loss ends 4–11× its minimum (1.2–4.5);
  // with L2 0.03 it ends at 0.32–0.40.
  it('Overfitting: training loss falls while test loss climbs; L2 keeps test loss low', () => {
    const p = preset('overfitting');
    const s = run(p.config, 1000);
    expect(s.trainAccuracy).toBeGreaterThan(0.9);
    expect(last(s.losses.train.mean)).toBeLessThan(0.15);
    expect(last(s.losses.test.mean)).toBeGreaterThan(3 * s.losses.test.best);
    const fixed = run(p.fix, 1000);
    expect(last(fixed.losses.test.mean)).toBeLessThan(0.5);
    expect(last(fixed.losses.test.mean)).toBeLessThan(0.3 * last(s.losses.test.mean));
  });

  // Seeds 1–5 for 200 epochs: 53–72 % of neurons dead, accuracy 48–65 %;
  // with lr 0.01: 3–19 % dead, 99 %.
  it('Dead ReLUs: many neurons die and it stalls; a lower rate keeps them alive', () => {
    const p = preset('dead-relus');
    const s = run(p.config, 200);
    expect(deadFraction(s)).toBeGreaterThan(0.4);
    expect(s.trainAccuracy).toBeLessThan(0.7);
    const fixed = run(p.fix, 200);
    expect(deadFraction(fixed)).toBeLessThanOrEqual(0.2);
    expect(fixed.trainAccuracy).toBeGreaterThan(0.95);
  });

  // Seeds 1–5 for 500 epochs: the loss rises > 5 % in 42–46 % of epochs and ends
  // above where it started; at lr 0.03, in ≤ 2.4 % of epochs.
  it('Too high LR: the loss jumps about and ends above its start; lr 0.03 is smooth', () => {
    const p = preset('too-high-lr');
    const s = run(p.config, 500);
    expect(jumpiness(s)).toBeGreaterThan(0.3);
    expect(last(s.losses.train.mean)).toBeGreaterThan(s.losses.train.mean[0]!);
    expect(jumpiness(run(p.fix, 500))).toBeLessThan(0.05);
  });
});
