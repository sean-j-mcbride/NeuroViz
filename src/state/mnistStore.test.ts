import { beforeEach, describe, expect, it } from 'vitest';
import type { MnistCheckpoint } from '../worker';
import { MNIST_DEFAULT_CONFIG, mnistParamCount, useMnistStore } from './mnistStore';
import { useAppStore } from './store';
import { tabForHash, useTabStore } from './tab';

const initial = useMnistStore.getState();
const initialApp = useAppStore.getState();

beforeEach(() => {
  useMnistStore.setState(initial, true);
  useAppStore.setState(initialApp, true);
  useTabStore.setState({ tab: 'playground' });
});

const checkpoint = { task: 'mnist' } as MnistCheckpoint;

describe('MNIST store', () => {
  it('starts from the default: 784 → 128 ReLU → 10, Adam, at Max speed, paused', () => {
    const s = useMnistStore.getState();
    expect(s.config).toEqual(MNIST_DEFAULT_CONFIG);
    expect([s.running, s.speed]).toEqual([false, 'max']);
    expect(mnistParamCount(s.config.network)).toBe(784 * 128 + 128 + 128 * 10 + 10);
  });

  it('loading a model starts a fresh, paused run that resumes it; changing the network drops it', () => {
    useMnistStore.getState().setRunning(true);
    const before = useMnistStore.getState().resetCount;
    useMnistStore.getState().loadModel({ config: MNIST_DEFAULT_CONFIG, checkpoint, savedAt: null });
    let s = useMnistStore.getState();
    expect([s.running, s.resume, s.resetCount]).toEqual([false, checkpoint, before + 1]);
    s.setTraining({ lr: 0.003 }); // hyperparameters apply live
    expect(useMnistStore.getState().resume).toBe(checkpoint);
    s.setNetwork({ hidden: [] });
    s = useMnistStore.getState();
    expect(s.resume).toBeNull();
    expect(s.config.training.lr).toBe(0.003);
  });

  it('a new seed drops the loaded checkpoint', () => {
    useMnistStore.getState().setConfig(MNIST_DEFAULT_CONFIG, { resume: checkpoint });
    useMnistStore.getState().setSeed(9);
    expect(useMnistStore.getState().resume).toBeNull();
  });
});

describe('tabs', () => {
  it('#mnist opens the MNIST tab; anything else is the playground', () => {
    expect(tabForHash('#mnist')).toBe('mnist');
    expect(tabForHash('#data=spirals')).toBe('playground');
    expect(tabForHash('')).toBe('playground');
  });

  it('leaving a tab pauses its run and leaves the other alone', () => {
    useAppStore.getState().setRunning(true);
    useTabStore.getState().setTab('mnist');
    expect(useAppStore.getState().running).toBe(false);
    useMnistStore.getState().setRunning(true);
    useAppStore.getState().setRunning(true); // e.g. a stale control
    useTabStore.getState().setTab('playground');
    expect(useMnistStore.getState().running).toBe(false);
    expect(useAppStore.getState().running).toBe(true);
  });
});
