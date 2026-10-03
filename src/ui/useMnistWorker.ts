import { useEffect, useMemo } from 'react';
import { mnistNetworkError, toMnistSessionConfig, useMnistStore } from '../state/mnistStore';
import { useAppStore } from '../state/store';
import type { FromWorker, MnistCheckpoint } from '../worker';
import { loadMnist } from './mnistLoader';
import { type MnistTask, TrainingClient } from './trainingClient';

function createClient(): TrainingClient<MnistTask> {
  return new TrainingClient<MnistTask>(
    {
      onData: () => {},
      onSnapshot: (snapshot) => useMnistStore.getState().setSnapshot(snapshot),
      onError: (message) => {
        console.error(`MNIST worker: ${message}`);
        useAppStore.getState().setNotice({ kind: 'error', text: `Training failed: ${message}` });
      },
    },
    {
      now: () => performance.now(),
      drawn: () => useMnistStore.getState().drawn,
    },
  );
}

/**
 * Fetches the bundled images into the MNIST store whenever it is `idle`: on
 * first use, and again when "Try again" resets a failed load to idle.
 */
function useMnistData(): void {
  const status = useMnistStore((s) => s.data.status);
  useEffect(() => {
    if (status !== 'idle') return;
    const { setData } = useMnistStore.getState();
    setData({ status: 'loading' });
    loadMnist().then(
      (subset) => setData({ status: 'ready', subset }),
      (e: unknown) =>
        setData({ status: 'error', message: e instanceof Error ? e.message : String(e) }),
    );
  }, [status]);
}

/**
 * Runs MNIST training in its own Web Worker and keeps the MNIST store's
 * snapshot fresh: the same pattern as the playground's `useTrainingWorker`.
 */
export function useMnistWorker(): {
  step: () => void;
  requestCheckpoint: () => Promise<MnistCheckpoint>;
} {
  useMnistData();
  const data = useMnistStore((s) => s.data);
  const { network, seed, training } = useMnistStore((s) => s.config);
  const resetCount = useMnistStore((s) => s.resetCount);
  const running = useMnistStore((s) => s.running);
  const speed = useMnistStore((s) => s.speed);
  const drawn = useMnistStore((s) => s.drawn);
  const subset = data.status === 'ready' ? data.subset : null;

  const client = useMemo(() => createClient(), []);

  useEffect(() => {
    const w = new Worker(new URL('../worker/training.worker.ts', import.meta.url), {
      type: 'module',
    });
    w.onmessage = ({ data }: MessageEvent<FromWorker>) => client.receive(data);
    client.attach((msg) => w.postMessage(msg));
    return () => {
      client.detach();
      w.terminate();
    };
  }, [client]);

  // New architecture or seed (or Reset, or a load) → fresh session, resuming a loaded checkpoint.
  useEffect(() => {
    if (!subset) return;
    const { config, resume } = useMnistStore.getState();
    // The builder shows why; the last valid session stays until the network is fixed.
    if (mnistNetworkError(config.network)) return;
    client.init(toMnistSessionConfig(config, subset, resume));
  }, [client, subset, network, seed, resetCount]);

  // The effects below need a session, so they wait for the data (init above runs first).
  useEffect(() => {
    if (subset) client.setHyperparams(training, useMnistStore.getState().running);
  }, [client, subset, training]);

  // A new drawing needs a snapshot carrying its prediction.
  useEffect(() => {
    if (subset) client.requestSnapshot();
  }, [client, subset, drawn]);

  useEffect(() => {
    if (!running || !subset) return;
    client.run(speed);
    let frame = 0;
    const tick = () => {
      client.tick();
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(frame);
      client.pause();
    };
  }, [client, running, speed, subset]);

  return {
    // Before the images load there is no session to step.
    step: () => subset && client.step(),
    requestCheckpoint: () => client.requestCheckpoint(),
  };
}
