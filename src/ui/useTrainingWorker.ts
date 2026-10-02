import { useCallback, useEffect, useRef } from 'react';
import { useAppStore } from '../state/store';
import type { FromWorker, ToWorker } from '../worker';

/** At most ~15 snapshot requests per second while training. */
const SNAPSHOT_INTERVAL_MS = 66;

/**
 * Runs training in a Web Worker and keeps the store's snapshot fresh.
 *
 * Store changes become worker messages (new session, hyperparameters,
 * run / pause). Snapshots are pulled: at most one request is in flight, and
 * while training a new one goes out at most every 66 ms, so a slow UI simply
 * asks less often instead of queueing a backlog.
 */
export function useTrainingWorker(): { step: () => void } {
  const worker = useRef<Worker | null>(null);
  const sessionId = useRef(0);
  const requestId = useRef(0);
  const inFlight = useRef(false);
  /** A request was wanted while one was in flight; send it when the reply lands. */
  const queued = useRef(false);
  const lastRequest = useRef(0);

  const { dataset, network, seed, training } = useAppStore((s) => s.config);
  const resetCount = useAppStore((s) => s.resetCount);
  const running = useAppStore((s) => s.running);
  const speed = useAppStore((s) => s.speed);
  const probe = useAppStore((s) => s.stepThrough?.probe ?? null);

  const send = useCallback((msg: ToWorker) => worker.current?.postMessage(msg), []);

  const requestSnapshot = useCallback(() => {
    if (!worker.current) return;
    if (inFlight.current) {
      queued.current = true;
      return;
    }
    inFlight.current = true;
    queued.current = false;
    lastRequest.current = performance.now();
    const p = useAppStore.getState().stepThrough?.probe;
    send({ type: 'snapshot', requestId: ++requestId.current, ...(p && { probe: p }) });
  }, [send]);

  // The worker lives for the life of the component (declared first so the
  // effects below see it).
  useEffect(() => {
    const w = new Worker(new URL('../worker/training.worker.ts', import.meta.url), {
      type: 'module',
    });
    w.onmessage = ({ data: msg }: MessageEvent<FromWorker>) => {
      const store = useAppStore.getState();
      switch (msg.type) {
        case 'ready':
          if (msg.sessionId === sessionId.current) store.setSessionData(msg.data);
          return;
        case 'snapshot':
          inFlight.current = false;
          if (msg.sessionId === sessionId.current) store.setSnapshot(msg.snapshot);
          if (queued.current) requestSnapshot();
          return;
        case 'error':
          inFlight.current = false;
          console.error(`Training worker: ${msg.message}`);
          return;
      }
    };
    worker.current = w;
    return () => {
      w.terminate();
      worker.current = null;
      inFlight.current = false;
    };
  }, [requestSnapshot]);

  // New data, architecture or seed (or Reset) → fresh session.
  useEffect(() => {
    const { training: t } = useAppStore.getState().config;
    sessionId.current++;
    send({ type: 'init', sessionId: sessionId.current, config: { dataset, network, seed, ...t } });
    requestSnapshot();
  }, [dataset, network, seed, resetCount, send, requestSnapshot]);

  // Hyperparameters apply live; refresh the view when paused (gradients change with L2).
  useEffect(() => {
    send({ type: 'hyperparams', hyperparams: training });
    if (!useAppStore.getState().running) requestSnapshot();
  }, [training, send, requestSnapshot]);

  // Picking a step-through point needs a snapshot carrying its trace.
  useEffect(() => {
    if (probe) requestSnapshot();
  }, [probe, requestSnapshot]);

  useEffect(() => {
    if (!running) return;
    send({ type: 'run', speed });
    let frame = 0;
    const tick = (now: number) => {
      if (!inFlight.current && now - lastRequest.current >= SNAPSHOT_INTERVAL_MS) {
        requestSnapshot();
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(frame);
      send({ type: 'pause' });
      // Show exactly where training stopped.
      requestSnapshot();
    };
  }, [running, speed, send, requestSnapshot]);

  const step = useCallback(() => {
    send({ type: 'step' });
    requestSnapshot();
  }, [send, requestSnapshot]);

  return { step };
}
