import { describe, expect, it } from 'vitest';
import { type Post, type Scheduler, TrainingController } from './controller';
import type { FromWorker, ToWorker } from './protocol';
import type { SessionConfig } from './session';

const CONFIG: SessionConfig = {
  dataset: { kind: 'circle', n: 100, noise: 0, seed: 1 },
  network: { hidden: [{ units: 3, activation: 'tanh' }] },
  lr: 0.03,
  batchSize: 10,
  optimiser: 'sgd',
  l2: 0,
  dropout: 0,
  seed: 1,
  gridSize: 4,
};

/**
 * A fake clock where every `now()` call costs `costMs` (so training "takes
 * time" and slices end), plus a task queue run by `advance`.
 */
class FakeScheduler implements Scheduler {
  t = 0;
  private tasks: { at: number; fn: () => void }[] = [];
  constructor(private readonly costMs = 0.1) {}

  now(): number {
    this.t += this.costMs;
    return this.t;
  }

  defer(fn: () => void, ms: number): void {
    this.tasks.push({ at: this.t + ms, fn });
  }

  get pending(): number {
    return this.tasks.length;
  }

  /** Runs every task due within the next `ms`, in time order. */
  advance(ms: number): void {
    const end = this.t + ms;
    for (;;) {
      this.tasks.sort((a, b) => a.at - b.at);
      const next = this.tasks[0];
      if (!next || next.at > end) break;
      this.tasks.shift();
      this.t = Math.max(this.t, next.at);
      next.fn();
    }
    this.t = Math.max(this.t, end);
  }
}

function setup(costMs?: number) {
  const scheduler = new FakeScheduler(costMs);
  const sent: { message: FromWorker; transfer: ArrayBuffer[] }[] = [];
  const post: Post = (message, transfer) => sent.push({ message, transfer });
  const controller = new TrainingController(post, scheduler);
  const send = (m: ToWorker) => controller.handle(m);
  let requestId = 0;
  /** Requests a snapshot and returns the reply. */
  const snap = (probe?: { set: 'train' | 'test'; index: number }) => {
    send({ type: 'snapshot', requestId: ++requestId, ...(probe && { probe }) });
    const reply = sent.at(-1)!.message;
    if (reply.type !== 'snapshot') throw new Error(`expected a snapshot, got ${reply.type}`);
    expect(reply.requestId).toBe(requestId);
    return reply;
  };
  return { scheduler, sent, send, snap };
}

describe('TrainingController', () => {
  it('init → ready with the point sets (transferred), then snapshots on request', () => {
    const { sent, send, snap } = setup();
    send({ type: 'init', sessionId: 7, config: CONFIG });
    const ready = sent[0]!;
    expect(ready.message.type).toBe('ready');
    if (ready.message.type !== 'ready') return;
    expect(ready.message.sessionId).toBe(7);
    expect(ready.message.data.train.x).toHaveLength(70 * 2);
    expect(ready.transfer).toHaveLength(4);

    const s = snap();
    expect(s.sessionId).toBe(7);
    expect(s.snapshot.epoch).toBe(0);
    expect(s.snapshot.probe).toBeUndefined();
    expect(sent.at(-1)!.transfer.length).toBeGreaterThan(10);
  });

  it('step trains exactly one epoch; probes are attached on request', () => {
    const { send, snap } = setup();
    send({ type: 'init', sessionId: 1, config: CONFIG });
    send({ type: 'step' });
    send({ type: 'step' });
    const s = snap({ set: 'train', index: 0 });
    expect(s.snapshot.epoch).toBe(2);
    expect(s.snapshot.probe?.ref).toEqual({ set: 'train', index: 0 });
  });

  it('paces training to the requested epochs per second, and pause stops it', () => {
    const { scheduler, send, snap } = setup();
    send({ type: 'init', sessionId: 1, config: CONFIG });
    send({ type: 'run', speed: 100 });
    scheduler.advance(1000);
    const epochs = snap().snapshot.epoch;
    expect(epochs).toBeGreaterThanOrEqual(95);
    expect(epochs).toBeLessThanOrEqual(101);

    send({ type: 'pause' });
    scheduler.advance(1000);
    expect(snap().snapshot.epoch).toBe(epochs);
    expect(scheduler.pending).toBe(0);
  });

  it("'max' trains in slices and keeps yielding", () => {
    const { scheduler, send, snap } = setup(1);
    send({ type: 'init', sessionId: 1, config: CONFIG });
    send({ type: 'run', speed: 'max' });
    scheduler.advance(0);
    // One slice: every now() costs 1 ms, so ~12 epochs per 12 ms slice.
    const first = snap().snapshot.epoch;
    expect(first).toBeGreaterThan(5);
    expect(first).toBeLessThan(20);
    expect(scheduler.pending).toBe(1);
    scheduler.advance(0);
    expect(snap().snapshot.epoch).toBeGreaterThan(first);
  });

  it('a new session while running keeps running, under the new id', () => {
    const { scheduler, send, snap } = setup();
    send({ type: 'init', sessionId: 1, config: CONFIG });
    send({ type: 'run', speed: 300 });
    scheduler.advance(100);
    send({ type: 'init', sessionId: 2, config: { ...CONFIG, seed: 2 } });
    expect(snap().snapshot.epoch).toBe(0);
    scheduler.advance(100);
    const s = snap();
    expect(s.sessionId).toBe(2);
    expect(s.snapshot.epoch).toBeGreaterThan(20);
  });

  it('run before init starts training once a session arrives', () => {
    const { scheduler, send, snap } = setup();
    send({ type: 'run', speed: 100 });
    scheduler.advance(50);
    send({ type: 'init', sessionId: 1, config: CONFIG });
    scheduler.advance(200);
    expect(snap().snapshot.epoch).toBeGreaterThan(10);
  });

  it('applies hyperparameters to the live session', () => {
    const a = setup();
    const b = setup();
    for (const { send } of [a, b]) send({ type: 'init', sessionId: 1, config: CONFIG });
    b.send({ type: 'hyperparams', hyperparams: { ...CONFIG, optimiser: 'adam', lr: 0.1 } });
    for (const { send } of [a, b]) send({ type: 'step' });
    expect(b.snap().snapshot.weights).not.toEqual(a.snap().snapshot.weights);
  });

  it('reports errors instead of throwing, with the request id when there is one', () => {
    const { sent, send } = setup();
    send({ type: 'step' });
    expect(sent[0]!.message).toEqual({
      type: 'error',
      message: 'No training session: send init first',
    });
    send({ type: 'checkpoint', requestId: 4 });
    expect(sent[1]!.message).toMatchObject({ type: 'error', requestId: 4 });
  });

  it('answers checkpoint requests; init with resume continues from it', () => {
    const { sent, send, snap } = setup();
    send({ type: 'init', sessionId: 1, config: { ...CONFIG, optimiser: 'adam' } });
    for (let i = 0; i < 3; i++) send({ type: 'step' });
    send({ type: 'checkpoint', requestId: 9 });
    const { message, transfer } = sent.at(-1)!;
    if (message.type !== 'checkpoint')
      throw new Error(`expected a checkpoint, got ${message.type}`);
    expect([message.sessionId, message.requestId, message.checkpoint.epoch]).toEqual([1, 9, 3]);
    // Losses, order, 4 params, 2 Adam slots per param; chart history: 3 arrays + 4 per layer.
    expect(transfer).toHaveLength(3 + 4 + 8 + 3 + 2 * 4);

    const checkpoint = structuredClone(message.checkpoint);
    send({
      type: 'init',
      sessionId: 2,
      config: { ...CONFIG, optimiser: 'adam', resume: checkpoint },
    });
    expect(snap().snapshot.epoch).toBe(3);
  });

  it('every message survives structured cloning', () => {
    const { sent, send, snap } = setup();
    send({ type: 'init', sessionId: 1, config: CONFIG });
    snap({ set: 'test', index: 2 });
    send({ type: 'checkpoint', requestId: 99 });
    for (const { message } of sent) expect(structuredClone(message)).toEqual(message);
  });
});
