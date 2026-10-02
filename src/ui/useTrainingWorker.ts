import { useEffect, useMemo } from 'react';
import { useAppStore } from '../state/store';
import type { FromWorker } from '../worker';
import { TrainingClient } from './trainingClient';

/** A client that reports into the app store. */
function createClient(): TrainingClient {
  return new TrainingClient(
    {
      onData: (data) => useAppStore.getState().setSessionData(data),
      onSnapshot: (snapshot) => useAppStore.getState().setSnapshot(snapshot),
      onError: (message) => console.error(`Training worker: ${message}`),
    },
    {
      now: () => performance.now(),
      probe: () => useAppStore.getState().stepThrough?.probe,
    },
  );
}

/**
 * Runs training in a Web Worker and keeps the store's snapshot fresh. Store
 * changes become worker messages; the request / session logic lives in the
 * unit-tested `TrainingClient`, and this hook only wires it to React.
 */
export function useTrainingWorker(): { step: () => void } {
  const { dataset, network, seed, training } = useAppStore((s) => s.config);
  const resetCount = useAppStore((s) => s.resetCount);
  const running = useAppStore((s) => s.running);
  const speed = useAppStore((s) => s.speed);
  const probe = useAppStore((s) => s.stepThrough?.probe ?? null);

  const client = useMemo(() => createClient(), []);

  // The worker lives for the life of the component (declared first so the
  // effects below see it attached).
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

  // New data, architecture or seed (or Reset) → fresh session.
  useEffect(() => {
    const { training: t } = useAppStore.getState().config;
    client.init({ dataset, network, seed, ...t });
  }, [client, dataset, network, seed, resetCount]);

  useEffect(() => {
    client.setHyperparams(training, useAppStore.getState().running);
  }, [client, training]);

  // Picking a step-through point needs a snapshot carrying its trace.
  useEffect(() => {
    if (probe) client.requestSnapshot();
  }, [client, probe]);

  useEffect(() => {
    if (!running) return;
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
  }, [client, running, speed]);

  return { step: () => client.step() };
}
