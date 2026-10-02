import type { Hyperparams, SessionConfig } from './session';
import type { ProbeRef, SessionData, Snapshot } from './snapshot';

/** Target training speed in epochs per second, or 'max' for as fast as the worker can go. */
export type Speed = 30 | 100 | 300 | 1000 | 'max';
export const SPEEDS: readonly Speed[] = [30, 100, 300, 1000, 'max'];

/** Messages from the UI to the training worker. */
export type ToWorker =
  /** Start a fresh session. The UI picks `sessionId` so it can drop stale replies. */
  | { type: 'init'; sessionId: number; config: SessionConfig }
  | { type: 'hyperparams'; hyperparams: Hyperparams }
  | { type: 'run'; speed: Speed }
  | { type: 'pause' }
  /** Train exactly one epoch (while paused). */
  | { type: 'step' }
  /** Ask for a snapshot; the UI sends one at a time, which throttles the flow. */
  | { type: 'snapshot'; requestId: number; probe?: ProbeRef };

/** Messages from the training worker to the UI. */
export type FromWorker =
  | { type: 'ready'; sessionId: number; data: SessionData }
  | { type: 'snapshot'; sessionId: number; requestId: number; snapshot: Snapshot }
  | { type: 'error'; message: string };
