import type {
  Checkpoint,
  FromWorker,
  Hyperparams,
  ProbeRef,
  SessionConfig,
  SessionData,
  Snapshot,
  Speed,
  ToWorker,
} from '../worker';

/** At most ~15 snapshot requests per second while training. */
export const SNAPSHOT_INTERVAL_MS = 66;

export interface ClientHandlers {
  onData(data: SessionData): void;
  onSnapshot(snapshot: Snapshot): void;
  onError(message: string): void;
}

export interface ClientOptions {
  now(): number;
  /** The step-through point to trace, read when each request is sent. */
  probe(): ProbeRef | null | undefined;
  intervalMs?: number;
}

/**
 * The UI side of the worker protocol, DOM-free so it can be unit-tested.
 *
 * Snapshots are pulled: at most one request is in flight. A request wanted
 * while one is in flight is queued (once) and sent when the reply lands.
 * While training, `tick` (called every animation frame) asks at most every
 * `intervalMs`, so a slow UI asks less often instead of building a backlog.
 * Each `init` starts a new session id; replies from older sessions are dropped.
 */
export class TrainingClient {
  private post: ((msg: ToWorker) => void) | null = null;
  private sessionId = 0;
  private requestId = 0;
  private inFlight = false;
  private queued = false;
  private lastRequest = -Infinity;
  private readonly intervalMs: number;
  /** Checkpoint requests awaiting a reply, by request id. */
  private readonly checkpoints = new Map<
    number,
    { sessionId: number; resolve(c: Checkpoint): void; reject(e: Error): void }
  >();

  constructor(
    private readonly handlers: ClientHandlers,
    private readonly options: ClientOptions,
  ) {
    this.intervalMs = options.intervalMs ?? SNAPSHOT_INTERVAL_MS;
  }

  /** Connects to a (new) worker; any request to an old worker is forgotten. */
  attach(post: (msg: ToWorker) => void): void {
    this.post = post;
    this.inFlight = false;
    this.queued = false;
  }

  detach(): void {
    this.post = null;
    this.inFlight = false;
    this.queued = false;
    for (const { reject } of this.checkpoints.values()) {
      reject(new Error('The training worker stopped'));
    }
    this.checkpoints.clear();
  }

  init(config: SessionConfig): void {
    if (!this.post) return;
    this.sessionId++;
    this.post({ type: 'init', sessionId: this.sessionId, config });
    this.requestSnapshot();
  }

  /** Applies live; when paused, refreshes the view (e.g. gradients change with L2). */
  setHyperparams(hyperparams: Hyperparams, running: boolean): void {
    this.post?.({ type: 'hyperparams', hyperparams });
    if (!running) this.requestSnapshot();
  }

  run(speed: Speed): void {
    this.post?.({ type: 'run', speed });
  }

  /** Pauses and fetches a snapshot of exactly where training stopped. */
  pause(): void {
    this.post?.({ type: 'pause' });
    this.requestSnapshot();
  }

  step(): void {
    this.post?.({ type: 'step' });
    this.requestSnapshot();
  }

  /** Call every animation frame while training. */
  tick(): void {
    if (!this.inFlight && this.options.now() - this.lastRequest >= this.intervalMs) {
      this.requestSnapshot();
    }
  }

  requestSnapshot(): void {
    if (!this.post) return;
    if (this.inFlight) {
      this.queued = true;
      return;
    }
    this.inFlight = true;
    this.queued = false;
    this.lastRequest = this.options.now();
    const probe = this.options.probe();
    this.post({ type: 'snapshot', requestId: ++this.requestId, ...(probe && { probe }) });
  }

  /**
   * Everything needed to save the current run. Any hyperparameter change sent
   * earlier is applied first (the worker handles messages in order). Rejects
   * if the session is replaced before the reply arrives.
   */
  requestCheckpoint(): Promise<Checkpoint> {
    const post = this.post;
    if (!post) return Promise.reject(new Error('The training worker is not running'));
    const requestId = ++this.requestId;
    return new Promise((resolve, reject) => {
      this.checkpoints.set(requestId, { sessionId: this.sessionId, resolve, reject });
      post({ type: 'checkpoint', requestId });
    });
  }

  receive(msg: FromWorker): void {
    const requestId = 'requestId' in msg ? msg.requestId : undefined;
    const pending = requestId === undefined ? undefined : this.checkpoints.get(requestId);
    if (pending) {
      this.checkpoints.delete(requestId!);
      if (msg.type === 'checkpoint' && msg.sessionId === this.sessionId) {
        pending.resolve(msg.checkpoint);
      } else {
        pending.reject(
          new Error(
            msg.type === 'error' ? msg.message : 'The run was restarted before it was saved',
          ),
        );
      }
      return;
    }
    switch (msg.type) {
      case 'ready':
        if (msg.sessionId === this.sessionId) this.handlers.onData(msg.data);
        return;
      case 'snapshot':
        this.inFlight = false;
        if (msg.sessionId === this.sessionId) this.handlers.onSnapshot(msg.snapshot);
        if (this.queued) this.requestSnapshot();
        return;
      case 'checkpoint':
        return; // no longer awaited (the worker was re-attached)
      case 'error':
        this.inFlight = false;
        this.handlers.onError(msg.message);
        if (this.queued) this.requestSnapshot();
        return;
    }
  }
}
