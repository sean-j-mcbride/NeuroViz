import { describe, expect, it } from 'vitest';
import { DEFAULT_CONFIG, type PlaygroundConfig } from './config';
import { parseConfig } from './validate';

const custom: PlaygroundConfig = {
  dataset: { kind: 'xor', n: 250, noise: 0.35, seed: 4_000_000_000 },
  network: { hidden: [{ units: 1, activation: 'linear' }] },
  training: { lr: 3, batchSize: 'full', optimiser: 'adam', l2: 0.0003, dropout: 0.5 },
  seed: 0,
};

describe('parseConfig', () => {
  it('accepts every valid config unchanged, with no warnings', () => {
    for (const c of [DEFAULT_CONFIG, custom, { ...custom, network: { hidden: [] } }]) {
      expect(parseConfig(JSON.parse(JSON.stringify(c)))).toEqual({ config: c, warnings: [] });
    }
  });

  it('replaces each invalid value with its default and says so', () => {
    const { config, warnings } = parseConfig({
      ...custom,
      dataset: { ...custom.dataset, kind: 'moons', n: 5000 },
      training: { ...custom.training, lr: 0.5, batchSize: 7 },
      seed: -1,
    });
    expect(config).toEqual({
      ...custom,
      dataset: { ...custom.dataset, kind: 'spirals', n: 400 },
      training: { ...custom.training, lr: 0.03, batchSize: 10 },
      seed: 1,
    });
    expect(warnings).toEqual([
      'Ignored dataset “moons” (choices: circle, xor, spirals, gaussians); using “spirals”',
      'Ignored number of points 5000 (allowed: 100–1000); using 400',
      'Ignored learning rate 0.5 (choices: 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3); using 0.03',
      'Ignored batch size 7 (choices: 1, 5, 10, 25, 50, full); using 10',
      'Ignored weight seed -1 (a whole number from 0 to 4294967295); using 1',
    ]);
  });

  it.each([
    ['too many layers', Array.from({ length: 7 }, () => ({ units: 2, activation: 'tanh' }))],
    ['too many units', [{ units: 9, activation: 'tanh' }]],
    ['zero units', [{ units: 0, activation: 'tanh' }]],
    ['unknown activation', [{ units: 2, activation: 'gelu' }]],
    ['not a list', 'deep'],
  ])('rejects hidden layers with %s', (_, hidden) => {
    const { config, warnings } = parseConfig({ ...custom, network: { hidden } });
    expect(config.network).toEqual(DEFAULT_CONFIG.network);
    expect(warnings).toHaveLength(1);
  });

  it('quotes rejected layers as given and says what is allowed', () => {
    const { warnings } = parseConfig(
      { network: { hidden: [{ units: 99, activation: 'tanh' }] } },
      { partial: true },
    );
    expect(warnings).toEqual([
      'Ignored hidden layers “99 tanh” (up to 6 layers of 1–8 neurons, each tanh, relu, ' +
        'sigmoid, linear); using 8 Tanh → 8 Tanh',
    ]);
  });

  it('warns about missing values unless partial', () => {
    expect(parseConfig({}).warnings).toHaveLength(11);
    expect(parseConfig({}, { partial: true })).toEqual({ config: DEFAULT_CONFIG, warnings: [] });
    expect(parseConfig(null).warnings[0]).toMatch(/Settings are missing/);
  });

  it('rejects non-finite and non-integer numbers', () => {
    const { warnings } = parseConfig(
      { dataset: { n: 100.5, noise: Number.NaN }, seed: 1.5 },
      { partial: true },
    );
    expect(warnings).toHaveLength(3);
  });
});
