import { beforeEach, describe, expect, it } from 'vitest';
import { DEFAULT_CONFIG, useAppStore } from './store';

const s = () => useAppStore.getState();

beforeEach(() => {
  useAppStore.setState({
    config: DEFAULT_CONFIG,
    running: false,
    stepThrough: null,
    showTestData: false,
  });
});

describe('store: step-through and training', () => {
  it('opening step-through pauses training; Play closes it', () => {
    s().setRunning(true);
    s().setStepThroughOpen(true);
    expect(s().running).toBe(false);
    expect(s().stepThrough).toEqual({ probe: null, stage: 0 });
    s().setProbe({ set: 'train', index: 3 });
    s().setStage(4);
    s().setRunning(true);
    expect(s().stepThrough).toBeNull();
  });

  it('picking a new point restarts at stage 0', () => {
    s().setStepThroughOpen(true);
    s().setProbe({ set: 'train', index: 1 });
    s().setStage(3);
    s().setProbe({ set: 'train', index: 2 });
    expect(s().stepThrough).toEqual({ probe: { set: 'train', index: 2 }, stage: 0 });
  });

  it('a new dataset closes step-through (indices refer to the old data); a new network resets the stage', () => {
    s().setStepThroughOpen(true);
    s().setProbe({ set: 'train', index: 1 });
    s().setStage(3);
    s().setNetwork({ hidden: [{ units: 2, activation: 'relu' }] });
    expect(s().stepThrough).toEqual({ probe: { set: 'train', index: 1 }, stage: 0 });
    s().setDataset({ kind: 'xor' });
    expect(s().stepThrough).toBeNull();
  });

  it('hiding the test data drops a picked test point but keeps a training point', () => {
    s().setShowTestData(true);
    s().setStepThroughOpen(true);
    s().setProbe({ set: 'test', index: 5 });
    s().setShowTestData(false);
    expect(s().stepThrough).toEqual({ probe: null, stage: 0 });
    s().setShowTestData(true);
    s().setProbe({ set: 'train', index: 5 });
    s().setShowTestData(false);
    expect(s().stepThrough?.probe).toEqual({ set: 'train', index: 5 });
  });

  it('hyperparameter patches merge without touching the rest of the config', () => {
    s().setTraining({ optimiser: 'adam', l2: 0.001 });
    expect(s().config.training).toEqual({
      ...DEFAULT_CONFIG.training,
      optimiser: 'adam',
      l2: 0.001,
    });
    expect(s().config.network).toBe(DEFAULT_CONFIG.network);
  });
});
