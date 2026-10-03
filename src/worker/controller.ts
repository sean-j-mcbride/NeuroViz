import { type Checkpoint, checkpointBuffers } from './checkpoint';
import { MnistSession } from './mnistSession';
import {
  type MnistCheckpoint,
  type MnistSessionData,
  type MnistSnapshot,
  mnistCheckpointBuffers,
  mnistSnapshotBuffers,
} from './mnistSnapshot';
import type { FromWorker, Speed, ToWorker } from './protocol';
import { type Hyperparams, TrainingSession } from './session';
import { type ProbeRef, type SessionData, type Snapshot, snapshotBuffers } from './snapshot';

/** Time source and task scheduling, injected so the controller runs in tests without a Worker. */
export interface Scheduler {
  now(): number;
  /** Runs `fn` after about `ms` milliseconds; 0 means after any pending messages. */
  defer(fn: () => void, ms: number): void;
}

/** Sends a message to the UI, transferring (not copying) the listed buffers. */
export type Post = (message: FromWorker, transfer: ArrayBuffer[]) => void;

/** Longest stretch of training between yields, so incoming messages are handled promptly. */
export const SLICE_MS = 12;
/** If pacing falls further behind than this, forget the backlog instead of racing to catch up. */
const MAX_LAG_MS = 250;

/** What the controller needs from a session, whichever task it trains. */
interface WorkerSession {
  /** Trains one unit: an epoch (playground) or a mini-batch (MNIST). */
  advance(): void;
  setHyperparams(h: Hyperparams): void;
  data(): { data: SessionData | MnistSessionData; transfer: ArrayBuffer[] };
  snapshot(req: { probe?: ProbeRef; drawn?: Float32Array; gradTarget?: number }): {
    snapshot: Snapshot | MnistSnapshot;
    transfer: ArrayBuffer[];
  };
  checkpoint(): { checkpoint: Checkpoint | MnistCheckpoint; transfer: ArrayBuffer[] };
}

function playground(s: TrainingSession): WorkerSession {
  return {
    advance: () => s.trainEpoch(),
    setHyperparams: (h) => s.setHyperparams(h),
    data: () => {
      const data = s.data();
      const { train, test } = data;
      const transfer = [train.x, train.y, test.x, test.y].map((a) => a.buffer as ArrayBuffer);
      return { data, transfer };
    },
    snapshot: ({ probe }) => {
      const snapshot = s.snapshot(probe ? { probe } : undefined);
      return { snapshot, transfer: snapshotBuffers(snapshot) };
    },
    checkpoint: () => {
      const checkpoint = s.checkpoint();
      return { checkpoint, transfer: checkpointBuffers(checkpoint) };
    },
  };
}

function mnist(s: MnistSession): WorkerSession {
  return {
    advance: () => s.advance(),
    setHyperparams: (h) => s.setHyperparams(h),
    data: () => ({ data: s.data(), transfer: [] }),
    snapshot: ({ drawn, gradTarget }) => {
      const snapshot = s.snapshot(
        drawn ? { drawn, ...(gradTarget !== undefined && { gradTarget }) } : undefined,
      );
      return { snapshot, transfer: mnistSnapshotBuffers(snapshot) };
    },
    checkpoint: () => {
      const checkpoint = s.checkpoint();
      return { checkpoint, transfer: mnistCheckpointBuffers(checkpoint) };
    },
  };
}

/**
 * Owns the training session inside the worker (the playground's
 * TrainingSession or an MnistSession): handles UI messages, runs the
 * training loop in short slices paced to the requested speed, and answers
 * snapshot requests. DOM-free; the worker entry wires it to `postMessage`.
 */
export class TrainingController {
  private session: WorkerSession | null = null;
  private sessionId = -1;
  /** null while paused. */
  private speed: Speed | null = null;
  private scheduled = false;
  /** Pacing baseline: units trained since `paceStart`. */
  private paceStart = 0;
  private paceUnits = 0;

  constructor(
    private readonly post: Post,
    private readonly scheduler: Scheduler,
  ) {}

  handle(msg: ToWorker): void {
    try {
      this.dispatch(msg);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      this.post(
        { type: 'error', message, ...('requestId' in msg && { requestId: msg.requestId }) },
        [],
      );
    }
  }

  private dispatch(msg: ToWorker): void {
    switch (msg.type) {
      case 'init': {
        const { config } = msg;
        this.session =
          config.task === 'mnist'
            ? mnist(new MnistSession(config))
            : playground(new TrainingSession(config));
        this.sessionId = msg.sessionId;
        const { data, transfer } = this.session.data();
        this.post({ type: 'ready', sessionId: msg.sessionId, data }, transfer);
        this.resetPace();
        if (this.speed !== null) this.schedule(0);
        return;
      }
      case 'hyperparams':
        this.session?.setHyperparams(msg.hyperparams);
        return;
      case 'run':
        this.speed = msg.speed;
        this.resetPace();
        this.schedule(0);
        return;
      case 'pause':
        this.speed = null;
        return;
      case 'step':
        this.requireSession().advance();
        return;
      case 'snapshot': {
        const { snapshot, transfer } = this.requireSession().snapshot({
          ...(msg.probe && { probe: msg.probe }),
          ...(msg.drawn && { drawn: msg.drawn }),
          ...(msg.gradTarget !== undefined && { gradTarget: msg.gradTarget }),
        });
        this.post(
          { type: 'snapshot', sessionId: this.sessionId, requestId: msg.requestId, snapshot },
          transfer,
        );
        return;
      }
      case 'checkpoint': {
        const { checkpoint, transfer } = this.requireSession().checkpoint();
        this.post(
          { type: 'checkpoint', sessionId: this.sessionId, requestId: msg.requestId, checkpoint },
          transfer,
        );
        return;
      }
    }
  }

  private requireSession(): WorkerSession {
    if (!this.session) throw new Error('No training session: send init first');
    return this.session;
  }

  private resetPace(): void {
    this.paceStart = this.scheduler.now();
    this.paceUnits = 0;
  }

  private schedule(ms: number): void {
    if (this.scheduled) return;
    this.scheduled = true;
    this.scheduler.defer(() => {
      this.scheduled = false;
      this.tick();
    }, ms);
  }

  private tick(): void {
    const { session, speed, scheduler } = this;
    if (!session || speed === null) return;
    const start = scheduler.now();
    if (speed === 'max') {
      do session.advance();
      while (scheduler.now() - start < SLICE_MS);
      this.schedule(0);
      return;
    }

    const msPerUnit = 1000 / speed;
    const due = () => Math.floor((scheduler.now() - this.paceStart) / msPerUnit) - this.paceUnits;
    while (due() > 0 && scheduler.now() - start < SLICE_MS) {
      session.advance();
      this.paceUnits++;
    }
    // Falling behind (slow machine, big net): drop the backlog rather than spiral.
    if (due() * msPerUnit > MAX_LAG_MS) this.resetPace();
    const nextAt = this.paceStart + (this.paceUnits + 1) * msPerUnit;
    this.schedule(Math.max(0, nextAt - scheduler.now()));
  }
}
