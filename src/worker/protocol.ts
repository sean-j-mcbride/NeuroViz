import type { Checkpoint } from './checkpoint';
import type { MnistSessionConfig } from './mnistSession';
import type { MnistCheckpoint, MnistSessionData, MnistSnapshot } from './mnistSnapshot';
import type { Hyperparams, SessionConfig } from './session';
import type { ProbeRef, SessionData, Snapshot } from './snapshot';

/**
 * Target training speed in units per second (epochs in the playground,
 * mini-batches for MNIST), or 'max' for as fast as the worker can go.
 */
export type Speed = 30 | 100 | 300 | 1000 | 'max';
export const SPEEDS: readonly Speed[] = [30, 100, 300, 1000, 'max'];

/** Messages from the UI to the training worker. */
export type ToWorker =
  /**
   * Start a fresh session: the 2D playground or (with `task: 'mnist'`) MNIST.
   * The UI picks `sessionId` so it can drop stale replies.
   */
  | { type: 'init'; sessionId: number; config: SessionConfig | MnistSessionConfig }
  | { type: 'hyperparams'; hyperparams: Hyperparams }
  | { type: 'run'; speed: Speed }
  | { type: 'pause' }
  /** Train exactly one unit: an epoch in the playground, a mini-batch for MNIST (while paused). */
  | { type: 'step' }
  /**
   * Ask for a snapshot; the UI sends one at a time, which throttles the flow.
   * `probe` (playground) traces a data point; `drawn` (MNIST) is a 28×28 input to classify.
   */
  | { type: 'snapshot'; requestId: number; probe?: ProbeRef; drawn?: Float32Array }
  /** Ask for everything needed to save the run and resume it exactly. */
  | { type: 'checkpoint'; requestId: number };

/** Messages from the training worker to the UI. */
export type FromWorker =
  | { type: 'ready'; sessionId: number; data: SessionData | MnistSessionData }
  | { type: 'snapshot'; sessionId: number; requestId: number; snapshot: Snapshot | MnistSnapshot }
  | {
      type: 'checkpoint';
      sessionId: number;
      requestId: number;
      checkpoint: Checkpoint | MnistCheckpoint;
    }
  /** `requestId` is set when the failed message was a request. */
  | { type: 'error'; message: string; requestId?: number };
