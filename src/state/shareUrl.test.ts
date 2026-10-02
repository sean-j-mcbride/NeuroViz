import { describe, expect, it } from 'vitest';
import { DEFAULT_CONFIG, type PlaygroundConfig } from './config';
import { decodeConfig, encodeConfig } from './shareUrl';

const custom: PlaygroundConfig = {
  dataset: { kind: 'gaussians', n: 150, noise: 0.45, seed: 123456789 },
  network: {
    hidden: [
      { units: 3, activation: 'relu' },
      { units: 8, activation: 'sigmoid' },
      { units: 1, activation: 'linear' },
    ],
  },
  training: { lr: 0.0001, batchSize: 'full', optimiser: 'momentum', l2: 0.1, dropout: 0.3 },
  seed: 42,
};

describe('share links', () => {
  it('encodes the default config as a readable hash', () => {
    expect(encodeConfig(DEFAULT_CONFIG)).toBe(
      '#data=spirals&points=400&noise=0&dataSeed=1&layers=8tanh,8tanh' +
        '&lr=0.03&batch=10&optimiser=sgd&l2=0&dropout=0&seed=1',
    );
  });

  it.each([
    ['the default', DEFAULT_CONFIG],
    ['a custom config', custom],
    ['no hidden layers', { ...custom, network: { hidden: [] } }],
  ])('round-trips %s exactly', (_, config) => {
    expect(decodeConfig(encodeConfig(config))).toEqual({ config, warnings: [] });
  });

  it('survives the browser percent-encoding the commas', () => {
    const hash = encodeConfig(custom).replaceAll(',', '%2C');
    expect(decodeConfig(hash)?.config).toEqual(custom);
  });

  it('fills missing values from the defaults', () => {
    expect(decodeConfig('#lr=1&layers=2relu')).toEqual({
      config: {
        ...DEFAULT_CONFIG,
        network: { hidden: [{ units: 2, activation: 'relu' }] },
        training: { ...DEFAULT_CONFIG.training, lr: 1 },
      },
      warnings: [],
    });
  });

  it('ignores garbage values with warnings and leaves unrelated hashes alone', () => {
    const r = decodeConfig('#data=moons&points=lots&layers=8tanh,xx&lr=&batch=0&seed=1e99');
    expect(r?.config).toEqual(DEFAULT_CONFIG);
    expect(r?.warnings).toEqual([
      'Ignored dataset “moons” (choices: circle, xor, spirals, gaussians); using “spirals”',
      'Ignored number of points “lots” (allowed: 100–1000); using 400',
      'Ignored hidden layers “8tanh,xx” (up to 6 layers of 1–8 neurons, each tanh, relu, sigmoid, ' +
        'linear); using 8 Tanh → 8 Tanh',
      'Ignored learning rate “” (choices: 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3); using 0.03',
      'Ignored batch size 0 (choices: 1, 5, 10, 25, 50, full); using 10',
      'Ignored weight seed 1e+99 (a whole number from 0 to 4294967295); using 1',
    ]);
    expect(decodeConfig('')).toBeNull();
    expect(decodeConfig('#section-2')).toBeNull();
  });
});
