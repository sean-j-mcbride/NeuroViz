import type { FromWorker, Speed, ToWorker } from './protocol';
import { TrainingSession } from './session';
import { snapshotBuffers } from './snapshot';

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

/**
 * Owns the TrainingSession inside the worker: handles UI messages, runs the
 * training loop in short slices paced to the requested speed, and answers
 * snapshot requests. DOM-free; the worker entry wires it to `postMessage`.
 */
export class TrainingController {
  private session: TrainingSession | null = null;
  private sessionId = -1;
  /** null while paused. */
  private speed: Speed | null = null;
  private scheduled = false;
  /** Pacing baseline: epochs trained since `paceStart`. */
  private paceStart = 0;
  private paceEpochs = 0;

  constructor(
    private readonly post: Post,
    private readonly scheduler: Scheduler,
  ) {}

  handle(msg: ToWorker): void {
    try {
      this.dispatch(msg);
    } catch (e) {
      this.post({ type: 'error', message: e instanceof Error ? e.message : String(e) }, []);
    }
  }

  private dispatch(msg: ToWorker): void {
    switch (msg.type) {
      case 'init': {
        this.session = new TrainingSession(msg.config);
        this.sessionId = msg.sessionId;
        const data = this.session.data();
        this.post({ type: 'ready', sessionId: msg.sessionId, data }, [
          data.train.x.buffer,
          data.train.y.buffer,
          data.test.x.buffer,
          data.test.y.buffer,
        ] as ArrayBuffer[]);
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
        this.requireSession().trainEpoch();
        return;
      case 'snapshot': {
        const snapshot = this.requireSession().snapshot(
          msg.probe ? { probe: msg.probe } : undefined,
        );
        this.post(
          { type: 'snapshot', sessionId: this.sessionId, requestId: msg.requestId, snapshot },
          snapshotBuffers(snapshot),
        );
        return;
      }
    }
  }

  private requireSession(): TrainingSession {
    if (!this.session) throw new Error('No training session: send init first');
    return this.session;
  }

  private resetPace(): void {
    this.paceStart = this.scheduler.now();
    this.paceEpochs = 0;
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
      do session.trainEpoch();
      while (scheduler.now() - start < SLICE_MS);
      this.schedule(0);
      return;
    }

    const msPerEpoch = 1000 / speed;
    const due = () => Math.floor((scheduler.now() - this.paceStart) / msPerEpoch) - this.paceEpochs;
    while (due() > 0 && scheduler.now() - start < SLICE_MS) {
      session.trainEpoch();
      this.paceEpochs++;
    }
    // Falling behind (slow machine, big net): drop the backlog rather than spiral.
    if (due() * msPerEpoch > MAX_LAG_MS) this.resetPace();
    const nextAt = this.paceStart + (this.paceEpochs + 1) * msPerEpoch;
    this.schedule(Math.max(0, nextAt - scheduler.now()));
  }
}
