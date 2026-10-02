import { useCallback, useEffect, useRef } from 'react';
import { useAppStore } from '../state/store';
import { TrainingSession } from '../worker';

/** Training time per animation frame at speed 'max', leaving the rest of the frame for drawing. */
const FRAME_BUDGET_MS = 8;
/** Publish at most ~15 snapshots per second. */
const SNAPSHOT_INTERVAL_MS = 66;

/**
 * Owns the TrainingSession and drives it from requestAnimationFrame. The session
 * lives in a ref, not the store: only its Snapshots are app state. Phase 3 moves
 * the session into a Web Worker behind the same publish-a-Snapshot contract.
 */
export function useTrainingLoop(): { step: () => void } {
  const session = useRef<TrainingSession | null>(null);
  const { dataset, network, seed, training } = useAppStore((s) => s.config);
  const resetCount = useAppStore((s) => s.resetCount);
  const running = useAppStore((s) => s.running);
  const speed = useAppStore((s) => s.speed);
  const setSnapshot = useAppStore((s) => s.setSnapshot);

  // New data, architecture or seed (or Reset) → fresh session.
  useEffect(() => {
    const { training: t } = useAppStore.getState().config;
    const s = new TrainingSession({ dataset, network, seed, ...t });
    session.current = s;
    setSnapshot(s.snapshot());
  }, [dataset, network, seed, resetCount, setSnapshot]);

  // Hyperparameters apply live.
  useEffect(() => {
    session.current?.setHyperparams(training);
  }, [training]);

  useEffect(() => {
    if (!running) return;
    let frame = 0;
    let lastPublish = 0;
    const tick = (now: number) => {
      const s = session.current;
      if (s) {
        if (speed === 'max') {
          const start = performance.now();
          do s.trainEpoch();
          while (performance.now() - start < FRAME_BUDGET_MS);
        } else {
          for (let i = 0; i < speed; i++) s.trainEpoch();
        }
        if (now - lastPublish >= SNAPSHOT_INTERVAL_MS) {
          lastPublish = now;
          setSnapshot(s.snapshot());
        }
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(frame);
      // Show exactly where training stopped.
      if (session.current) setSnapshot(session.current.snapshot());
    };
  }, [running, speed, setSnapshot]);

  const step = useCallback(() => {
    const s = session.current;
    if (!s) return;
    s.trainEpoch();
    setSnapshot(s.snapshot());
  }, [setSnapshot]);

  return { step };
}
