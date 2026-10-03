import { describe, expect, it } from 'vitest';
import type { FromWorker, SessionConfig, Snapshot, ToWorker } from '../worker';
import { TrainingController } from '../worker/controller';
import { type ClientHandlers, type MnistTask, TrainingClient } from './trainingClient';

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

/** A minimal fake snapshot: only `epoch` is read by these tests. */
const fakeSnapshot = (epoch: number) => ({ epoch }) as Snapshot;

function setup(probe: () => { set: 'train' | 'test'; index: number } | null = () => null) {
  let t = 0;
  const sent: ToWorker[] = [];
  const got = { snapshots: [] as number[], data: 0, errors: [] as string[] };
  const handlers: ClientHandlers = {
    onData: () => got.data++,
    onSnapshot: (s) => got.snapshots.push(s.epoch),
    onError: (m) => got.errors.push(m),
  };
  const client = new TrainingClient(handlers, { now: () => t, probe });
  client.attach((m) => sent.push(m));
  const requests = () => sent.filter((m) => m.type === 'snapshot');
  const lastRequestId = () => {
    const r = requests().at(-1);
    if (r?.type !== 'snapshot') throw new Error('no snapshot request');
    return r.requestId;
  };
  const reply = (sessionId: number, epoch: number) =>
    client.receive({
      type: 'snapshot',
      sessionId,
      requestId: lastRequestId(),
      snapshot: fakeSnapshot(epoch),
    });
  return { client, sent, got, requests, reply, advance: (ms: number) => (t += ms) };
}

describe('TrainingClient (unit)', () => {
  it('keeps at most one request in flight and sends one queued request when the reply lands', () => {
    const { client, requests, reply, got } = setup();
    client.init(CONFIG); // init + first request
    client.requestSnapshot();
    client.requestSnapshot();
    client.step();
    expect(requests()).toHaveLength(1);
    reply(1, 0);
    expect(got.snapshots).toEqual([0]);
    expect(requests()).toHaveLength(2); // the queued one, sent once
    reply(1, 1);
    expect(requests()).toHaveLength(2); // nothing else was queued
    expect(got.snapshots).toEqual([0, 1]);
  });

  it('tick asks at most every interval, and never while a request is in flight', () => {
    const { client, requests, reply, advance } = setup();
    client.init(CONFIG);
    client.tick();
    expect(requests()).toHaveLength(1); // in flight
    reply(1, 0);
    advance(30);
    client.tick();
    expect(requests()).toHaveLength(1); // too soon
    advance(40);
    client.tick();
    expect(requests()).toHaveLength(2);
  });

  it('drops replies from an older session, but still sends the queued request', () => {
    const { client, requests, reply, got } = setup();
    client.init(CONFIG);
    client.init({ ...CONFIG, seed: 2 }); // session 2; request queued behind session 1's
    expect(requests()).toHaveLength(1);
    reply(1, 99); // stale
    expect(got.snapshots).toEqual([]);
    expect(requests()).toHaveLength(2);
    reply(2, 0);
    expect(got.snapshots).toEqual([0]);
    client.receive({
      type: 'ready',
      sessionId: 1,
      data: {
        train: { x: new Float32Array(), y: new Float32Array() },
        test: { x: new Float32Array(), y: new Float32Array() },
      },
    });
    expect(got.data).toBe(0);
  });

  it('attaches the current probe to each request', () => {
    let probe: { set: 'train' | 'test'; index: number } | null = null;
    const { client, requests, reply } = setup(() => probe);
    client.init(CONFIG);
    probe = { set: 'test', index: 4 };
    reply(1, 0);
    client.requestSnapshot();
    const [first, second] = requests();
    expect(first).not.toHaveProperty('probe');
    expect(second).toMatchObject({ probe: { set: 'test', index: 4 } });
  });

  it('attaches a copy of the drawn digit (MNIST) to each request', () => {
    let drawn: Float32Array | null = null;
    const sent: ToWorker[] = [];
    const client = new TrainingClient<MnistTask>(
      { onData: () => {}, onSnapshot: () => {}, onError: () => {} },
      { now: () => 0, drawn: () => drawn },
    );
    client.attach((m) => sent.push(m));
    client.requestSnapshot();
    drawn = new Float32Array(784).fill(0.5);
    client.receive({ type: 'error', message: 'stale' }); // clears the in-flight request
    client.requestSnapshot();
    const [first, second] = sent.filter((m) => m.type === 'snapshot');
    expect(first).not.toHaveProperty('drawn');
    if (second?.type !== 'snapshot' || !second.drawn) throw new Error('no drawn digit');
    expect(second.drawn).toEqual(drawn);
    expect(second.drawn).not.toBe(drawn); // the pad keeps its own array
  });

  it('an error clears the in-flight request', () => {
    const { client, requests, got } = setup();
    client.init(CONFIG);
    client.receive({ type: 'error', message: 'boom' });
    expect(got.errors).toEqual(['boom']);
    client.requestSnapshot();
    expect(requests()).toHaveLength(2);
  });

  it('pause and paused hyperparameter changes fetch a snapshot; running ones do not', () => {
    const { client, sent, requests, reply } = setup();
    client.init(CONFIG);
    reply(1, 0);
    client.setHyperparams(CONFIG, true);
    expect(requests()).toHaveLength(1);
    client.setHyperparams(CONFIG, false);
    expect(requests()).toHaveLength(2);
    reply(1, 0);
    client.run(300);
    client.pause();
    expect(sent.slice(-3).map((m) => m.type)).toEqual(['run', 'pause', 'snapshot']);
  });

  it('does nothing while detached, and a re-attach forgets the old in-flight request', () => {
    const { client, sent, requests } = setup();
    client.init(CONFIG);
    client.detach();
    client.init(CONFIG);
    client.step();
    expect(sent).toHaveLength(2);
    client.attach((m) => sent.push(m));
    client.requestSnapshot();
    expect(requests()).toHaveLength(2);
  });
});

/**
 * The client wired to the real TrainingController through a simulated
 * message channel: every message is structured-cloned and delivered later,
 * in order, like postMessage.
 */
function connected() {
  let t = 0;
  const tasks: { at: number; fn: () => void }[] = [];
  const toWorker: ToWorker[] = [];
  const toUi: FromWorker[] = [];
  const snapshots: Snapshot[] = [];
  let data = 0;
  const client = new TrainingClient(
    {
      onData: () => data++,
      onSnapshot: (s) => snapshots.push(s),
      onError: (m) => {
        throw new Error(m);
      },
    },
    { now: () => t, probe: () => null },
  );
  const controller = new TrainingController((m) => toUi.push(structuredClone(m)), {
    now: () => (t += 0.1),
    defer: (fn, ms) => tasks.push({ at: t + ms, fn }),
  });
  client.attach((m) => toWorker.push(structuredClone(m)));
  /** Delivers messages both ways until the channel is quiet. */
  const flush = () => {
    while (toWorker.length || toUi.length) {
      while (toWorker.length) controller.handle(toWorker.shift()!);
      while (toUi.length) client.receive(toUi.shift()!);
    }
  };
  /** Runs the worker's scheduled training for `ms`, then flushes. */
  const runFor = (ms: number) => {
    const end = t + ms;
    for (;;) {
      tasks.sort((a, b) => a.at - b.at);
      const next = tasks[0];
      if (!next || next.at > end) break;
      tasks.shift();
      t = Math.max(t, next.at);
      next.fn();
      client.tick();
      flush();
    }
    t = Math.max(t, end);
  };
  return { client, flush, runFor, snapshots, data: () => data };
}

describe('TrainingClient ↔ TrainingController', () => {
  it('init → data + first snapshot; step → epoch 1', () => {
    const c = connected();
    c.client.init(CONFIG);
    c.flush();
    expect(c.data()).toBe(1);
    expect(c.snapshots.map((s) => s.epoch)).toEqual([0]);
    c.client.step();
    c.flush();
    expect(c.snapshots.at(-1)!.epoch).toBe(1);
  });

  it('a reset while a request is in flight never shows the old session', () => {
    const c = connected();
    c.client.init(CONFIG);
    c.flush();
    c.client.step();
    c.client.step(); // queued behind the step's request
    c.client.init({ ...CONFIG, seed: 2 }); // before any of that is delivered
    c.flush();
    // Only the new session's epoch-0 snapshot is shown after the first one.
    expect(c.snapshots.map((s) => s.epoch)).toEqual([0, 0]);
  });

  it('while running, snapshots arrive no faster than the interval', () => {
    const c = connected();
    c.client.init(CONFIG);
    c.flush();
    c.client.run(300);
    c.flush();
    c.runFor(1000);
    c.client.pause();
    c.flush();
    const shown = c.snapshots.length;
    expect(shown).toBeGreaterThan(10);
    expect(shown).toBeLessThanOrEqual(1 + 1000 / 66 + 2);
    const last = c.snapshots.at(-1)!.epoch;
    expect(last).toBeGreaterThan(250);
    expect(last).toBeLessThanOrEqual(301);
  });

  it('a checkpoint reflects hyperparameters sent before it and resumes the run', async () => {
    const c = connected();
    c.client.init(CONFIG);
    c.client.step();
    c.client.setHyperparams({ ...CONFIG, optimiser: 'adam', lr: 0.1 }, false);
    c.client.step();
    const saved = c.client.requestCheckpoint();
    c.flush();
    const checkpoint = await saved;
    expect(checkpoint.epoch).toBe(2);
    expect(checkpoint.optimiser.kind).toBe('adam');

    c.client.init({ ...CONFIG, optimiser: 'adam', lr: 0.1, resume: checkpoint });
    c.flush();
    expect(c.snapshots.at(-1)!.epoch).toBe(2);
  });

  it('a checkpoint request is rejected if the run restarts before the reply', async () => {
    const c = connected();
    c.client.init(CONFIG);
    c.flush();
    const saved = c.client.requestCheckpoint();
    c.client.init({ ...CONFIG, seed: 2 });
    c.flush();
    await expect(saved).rejects.toThrow(/restarted before it was saved/);
  });
});

describe('TrainingClient checkpoints (unit)', () => {
  it('rejects when detached, on a worker error for that request, and on detach', async () => {
    const { client, sent } = setup();
    client.init(CONFIG);
    const failed = client.requestCheckpoint();
    const req = sent.at(-1)!;
    if (req.type !== 'checkpoint') throw new Error('expected a checkpoint request');
    client.receive({ type: 'error', message: 'boom', requestId: req.requestId });
    await expect(failed).rejects.toThrow('boom');

    const dropped = client.requestCheckpoint();
    client.detach();
    await expect(dropped).rejects.toThrow(/stopped/);
    await expect(client.requestCheckpoint()).rejects.toThrow(/not running/);
  });

  it('a checkpoint error does not disturb the snapshot in flight', () => {
    const { client, sent, requests, reply, got } = setup();
    client.init(CONFIG); // snapshot request in flight
    void client.requestCheckpoint().catch(() => {});
    const req = sent.at(-1)!;
    if (req.type !== 'checkpoint') throw new Error('expected a checkpoint request');
    client.receive({ type: 'error', message: 'boom', requestId: req.requestId });
    client.requestSnapshot();
    expect(requests()).toHaveLength(1); // still in flight; queued
    reply(1, 0);
    expect(got.snapshots).toEqual([0]);
    expect(got.errors).toEqual([]);
  });
});
