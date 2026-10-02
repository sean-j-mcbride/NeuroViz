import { beforeEach, describe, expect, it } from 'vitest';
import { type Checkpoint, TrainingSession } from '../worker';
import { toSessionConfig } from './config';
import { PRESETS } from './presets';
import { DEFAULT_CONFIG, presetLabel, useAppStore } from './store';

const s = () => useAppStore.getState();

beforeEach(() => {
  useAppStore.setState({
    config: DEFAULT_CONFIG,
    running: false,
    stepThrough: null,
    showTestData: false,
    resume: null,
    presetId: null,
    reference: null,
    snapshot: null,
    sessionData: null,
  });
});

function session(config = DEFAULT_CONFIG, epochs = 2): TrainingSession {
  const t = new TrainingSession({ ...toSessionConfig(config), gridSize: 4 });
  for (let e = 0; e < epochs; e++) t.trainEpoch();
  return t;
}

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

describe('store: loading, presets and the reference run', () => {
  const checkpoint = (): Checkpoint => session().checkpoint();

  it('loading a model starts a fresh, paused session that resumes the checkpoint', () => {
    s().setRunning(true);
    s().setStepThroughOpen(true);
    const before = s().resetCount;
    const c = checkpoint();
    const config = { ...DEFAULT_CONFIG, seed: 7 };
    s().loadModel({ config, checkpoint: c, savedAt: null });
    expect(s()).toMatchObject({ config, resume: c, running: false, stepThrough: null });
    expect(s().resetCount).toBe(before + 1);
  });

  it('Reset and hyperparameter changes keep the checkpoint; data, network and seed drop it', () => {
    const load = () =>
      s().loadModel({ config: DEFAULT_CONFIG, checkpoint: checkpoint(), savedAt: null });
    load();
    s().reset();
    s().setTraining({ lr: 0.1 });
    expect(s().resume).not.toBeNull();
    for (const change of [
      () => s().setDataset({ noise: 0.1 }),
      () => s().setNetwork({ hidden: [] }),
      () => s().setSeed(9),
    ]) {
      load();
      change();
      expect(s().resume).toBeNull();
    }
  });

  it('a preset loads its config and is named while the config still matches it', () => {
    const p = PRESETS[1]!;
    s().setTraining({ lr: 1 });
    s().loadPreset(p.id);
    expect(s()).toMatchObject({ config: p.config, presetId: p.id, resume: null });
    expect(presetLabel(s().config, s().presetId)).toBe(p.name);
    expect(presetLabel(p.fix, p.id)).toBe(`${p.name}, fixed`);
    s().setTraining({ dropout: 0.5 });
    expect(presetLabel(s().config, s().presetId)).toBeNull();
    s().setConfig(DEFAULT_CONFIG);
    expect(s().presetId).toBeNull();
  });

  it('pinning freezes the current run; trying a preset’s fix pins it first', () => {
    const p = PRESETS.find((q) => q.id === 'too-high-lr')!;
    s().loadPreset(p.id);
    s().pinReference();
    expect(s().reference).toBeNull(); // nothing to pin before the first snapshot

    const run = session(p.config, 3);
    s().setSnapshot(run.snapshot());
    s().setSessionData(run.data());
    s().tryPresetFix();
    const ref = s().reference!;
    expect(ref).toMatchObject({ label: 'Too high LR', config: p.config, epoch: 3 });
    expect(ref.losses.count).toBe(4);
    expect(ref.boundary.values).toHaveLength(16);
    expect(ref.data?.train.y.length).toBe(280);
    expect(s().config).toEqual(p.fix);

    s().pinReference();
    expect(s().reference?.label).toBe('Too high LR, fixed');
    s().setReference(null);
    expect(s().reference).toBeNull();
  });
});
