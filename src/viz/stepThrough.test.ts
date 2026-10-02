import { describe, expect, it } from 'vitest';
import { forwardLayer, gradLayer, stageCount, stageLabel, stageView } from './stepThrough';

const NAMES = ['Input', 'Hidden 1', 'Output'];

describe('step-through stages (3 columns)', () => {
  it('has 6 stages: 3 forward, the loss, 2 backward', () => {
    expect(stageCount(3)).toBe(6);
    expect(Array.from({ length: 6 }, (_, s) => stageLabel(s, NAMES))).toEqual([
      'Forward pass · Input',
      'Forward pass · Hidden 1',
      'Forward pass · Output',
      'Loss',
      'Backward pass · Hidden 1',
      'Backward pass · Input',
    ]);
  });

  it('lights layers forward, then gradients backward', () => {
    const at = (s: number) => {
      const v = stageView(s, 3);
      return [0, 1].map((k) => (gradLayer(v, k) ? 'g' : forwardLayer(v, k) ? 'f' : '-')).join('');
    };
    expect([0, 1, 2, 3, 4, 5].map(at)).toEqual(['--', 'f-', 'ff', 'ff', '-g', 'gg']);
    expect(stageView(5, 3).gradFrom).toBe(0);
    expect(stageView(3, 3).gradFrom).toBe(3);
  });
});
