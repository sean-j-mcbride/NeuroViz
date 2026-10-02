import { describe, expect, it } from 'vitest';
import { DEFAULT_CONFIG } from './config';
import { diffConfigs } from './configDiff';
import { PRESETS } from './presets';

describe('diffConfigs', () => {
  it('is empty for identical configs', () => {
    expect(diffConfigs(DEFAULT_CONFIG, structuredClone(DEFAULT_CONFIG))).toEqual([]);
  });

  it('lists each changed setting readably, in control order', () => {
    const overfitting = PRESETS.find((p) => p.id === 'overfitting')!;
    expect(diffConfigs(DEFAULT_CONFIG, overfitting.config)).toEqual([
      { label: 'Dataset', from: 'Spirals', to: 'Circle' },
      { label: 'Points', from: '400', to: '150' },
      { label: 'Noise', from: '0 %', to: '40 %' },
      {
        label: 'Hidden layers',
        from: '8 Tanh → 8 Tanh',
        to: '8 ReLU → 8 ReLU → 8 ReLU → 8 ReLU → 8 ReLU → 8 ReLU',
      },
      { label: 'Learning rate', from: '0.03', to: '0.01' },
      { label: 'Optimiser', from: 'SGD', to: 'Adam' },
    ]);
    expect(diffConfigs(overfitting.config, overfitting.fix)).toEqual([
      { label: 'L2 regularisation', from: 'None', to: '0.03' },
    ]);
  });

  it('describes batch size, dropout and an empty network', () => {
    const b = {
      ...DEFAULT_CONFIG,
      network: { hidden: [] },
      training: { ...DEFAULT_CONFIG.training, batchSize: 'full' as const, dropout: 0.2 },
    };
    expect(diffConfigs(DEFAULT_CONFIG, b).map((c) => `${c.label}: ${c.from} → ${c.to}`)).toEqual([
      'Hidden layers: 8 Tanh → 8 Tanh → No hidden layers',
      'Batch size: 10 → Full batch',
      'Dropout: None → 20 %',
    ]);
  });
});
