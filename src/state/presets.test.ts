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
const min = (a: Float32Array) => a.reduce((m, v) => Math.min(m, v), Infinity);

/** Fraction of ReLU neurons that output 0 everywhere on the input plane. */
function deadFraction(s: Snapshot): number {
  let dead = 0;
  let total = 0;
  const g2 = s.gridSize ** 2;
  for (const c of s.columns) {
    if (c.kind !== 'relu') continue;
    for (let u = 0; u < c.units; u++) {
      total++;
      if (c.values.subarray(u * g2, (u + 1) * g2).every((v) => v === 0)) dead++;
    }
  }
  return dead / total;
}

/** Fraction of epochs whose training loss rose by more than 5 %. */
function jumpiness(s: Snapshot): number {
  let ups = 0;
  for (let i = 1; i < s.trainLoss.length; i++) {
    if (s.trainLoss[i]! > 1.05 * s.trainLoss[i - 1]!) ups++;
  }
  return ups / (s.trainLoss.length - 1);
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
    expect(last(s.trainLoss)).toBeGreaterThan(0.5);
    expect(Math.abs(last(s.trainLoss) - last(s.testLoss))).toBeLessThan(0.05);
    expect(run(p.fix, 300).trainAccuracy).toBeGreaterThan(0.95);
  });

  // Seeds 1–5 for 1000 epochs: test loss ends 4–11× its minimum (1.2–4.5);
  // with L2 0.03 it ends at 0.32–0.40.
  it('Overfitting: training loss falls while test loss climbs; L2 keeps test loss low', () => {
    const p = preset('overfitting');
    const s = run(p.config, 1000);
    expect(s.trainAccuracy).toBeGreaterThan(0.9);
    expect(last(s.trainLoss)).toBeLessThan(0.15);
    expect(last(s.testLoss)).toBeGreaterThan(3 * min(s.testLoss));
    const fixed = run(p.fix, 1000);
    expect(last(fixed.testLoss)).toBeLessThan(0.5);
    expect(last(fixed.testLoss)).toBeLessThan(0.3 * last(s.testLoss));
  });

  // Seeds 1–5 for 200 epochs: 50–69 % of neurons dead, accuracy 48–65 %;
  // with lr 0.01: 3–16 % dead, 99 %.
  it('Dead ReLUs: many neurons die and it stalls; a lower rate keeps them alive', () => {
    const p = preset('dead-relus');
    const s = run(p.config, 200);
    expect(deadFraction(s)).toBeGreaterThan(0.4);
    expect(s.trainAccuracy).toBeLessThan(0.7);
    const fixed = run(p.fix, 200);
    expect(deadFraction(fixed)).toBeLessThan(0.2);
    expect(fixed.trainAccuracy).toBeGreaterThan(0.95);
  });

  // Seeds 1–5 for 500 epochs: the loss rises > 5 % in 42–46 % of epochs and ends
  // above where it started; at lr 0.03, in ≤ 2.4 % of epochs.
  it('Too high LR: the loss jumps about and ends above its start; lr 0.03 is smooth', () => {
    const p = preset('too-high-lr');
    const s = run(p.config, 500);
    expect(jumpiness(s)).toBeGreaterThan(0.3);
    expect(last(s.trainLoss)).toBeGreaterThan(s.trainLoss[0]!);
    expect(jumpiness(run(p.fix, 500))).toBeLessThan(0.05);
  });
});
