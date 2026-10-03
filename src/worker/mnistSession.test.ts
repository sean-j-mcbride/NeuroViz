import { describe, expect, it } from 'vitest';
import { loadMnistSubset } from '../test/mnist';
import { type Post, TrainingController } from './controller';
import {
  MnistSession,
  type MnistSessionConfig,
  RECORD_EVERY,
  TRAIN_EVAL_SIZE,
} from './mnistSession';
import type { MnistCheckpoint } from './mnistSnapshot';
import { isMnistSnapshot, mnistCheckpointBuffers, mnistSnapshotBuffers } from './mnistSnapshot';
import type { FromWorker } from './protocol';

const data = loadMnistSubset();

/** A small network so the tests stay quick. */
const SMALL: MnistSessionConfig = {
  task: 'mnist',
  network: { hidden: [{ units: 16, activation: 'relu' }] },
  seed: 1,
  data,
  lr: 0.001,
  batchSize: 64,
  optimiser: 'adam',
  l2: 0,
  dropout: 0,
};

function paramsOf(s: MnistSession): Record<string, Float32Array> {
  return s.checkpoint().params;
}

function bitwiseEqual(a: Float32Array, b: Float32Array): boolean {
  return a.length === b.length && a.every((v, i) => Object.is(v, b[i]));
}

function sameParams(a: MnistSession, b: MnistSession): boolean {
  const pa = paramsOf(a);
  const pb = paramsOf(b);
  return Object.keys(pa).every((k) => bitwiseEqual(pa[k]!, pb[k]!));
}

/** Test image i scaled to [0, 1], as the digit pad would send it. */
function testImage(i: number): Float32Array {
  const pixels = new Float32Array(784);
  for (let p = 0; p < 784; p++) pixels[p] = data.test.images[i * 784 + p]! / 255;
  return pixels;
}

function advance(s: MnistSession, batches: number): MnistSession {
  for (let i = 0; i < batches; i++) s.advance();
  return s;
}

describe('MnistSession', () => {
  it('builds 784 → hidden → 10 and records metrics before training', () => {
    const s = new MnistSession(SMALL);
    const snap = s.snapshot();
    expect(snap.task).toBe('mnist');
    expect(snap.firstLayer).toMatchObject({ inFeatures: 784, outFeatures: 16 });
    expect(snap.firstLayer.W).toHaveLength(784 * 16);
    expect(snap.losses.count).toBe(1);
    expect(snap.trainSize).toBe(10_000);
    expect(snap.testSize).toBe(2_000);
    // Untrained: about chance (10 %), loss about ln 10.
    expect(snap.testAccuracy).toBeLessThan(0.3);
    expect(snap.losses.test.latest).toBeGreaterThan(1.5);
  });

  it('with no hidden layer, the first layer is the output (10 class templates)', () => {
    const s = new MnistSession({ ...SMALL, network: { hidden: [] } });
    const snap = s.snapshot({ drawn: new Float32Array(784) });
    expect(snap.firstLayer.outFeatures).toBe(10);
    expect(snap.prediction?.hidden).toBeUndefined();
  });

  it('trains one mini-batch per advance, records every RECORD_EVERY examples, wraps epochs', () => {
    const s = new MnistSession({ ...SMALL, batchSize: 100 });
    const perRecord = RECORD_EVERY / 100;
    advance(s, perRecord - 1);
    expect(s.snapshot().losses.count).toBe(1);
    s.advance(); // RECORD_EVERY examples
    expect(s.snapshot().losses.count).toBe(2);
    advance(s, 100 - perRecord); // 10,000: one full epoch
    const snap = s.snapshot();
    expect([snap.examples, snap.step, snap.epoch]).toEqual([10_000, 100, 1]);
    expect(snap.losses.count).toBe(10_000 / RECORD_EVERY + 1);
    expect(snap.errors.count).toBe(snap.losses.count);
  });

  it('a batch larger than the record interval still keeps the record schedule (and resumes)', () => {
    const s = new MnistSession({ ...SMALL, batchSize: 'full' });
    s.advance(); // 10,000 examples: records due at 5,000 and 10,000
    expect(s.snapshot().losses.count).toBe(10_000 / RECORD_EVERY + 1);
    const resumed = new MnistSession({ ...SMALL, batchSize: 'full', resume: s.checkpoint() });
    expect(resumed.examples).toBe(10_000);
  });

  it('the last batch of an epoch is partial', () => {
    const s = new MnistSession({ ...SMALL, batchSize: 256 });
    advance(s, 40); // 39 × 256 = 9,984, then 16
    expect(s.examples).toBe(10_000);
    s.advance();
    expect(s.examples).toBe(10_256);
  });

  it('learns: a few hundred batches take test accuracy well above chance', () => {
    const s = advance(new MnistSession(SMALL), 300);
    const snap = s.snapshot();
    expect(snap.testAccuracy).toBeGreaterThan(0.85);
    expect(snap.losses.test.latest).toBeLessThan(snap.losses.test.mean[0]!);
  });

  it('is deterministic for a seed, and seeds differ', () => {
    const a = advance(new MnistSession(SMALL), 50);
    const b = advance(new MnistSession(SMALL), 50);
    const c = advance(new MnistSession({ ...SMALL, seed: 2 }), 50);
    expect(sameParams(a, b)).toBe(true);
    expect(sameParams(a, c)).toBe(false);
  });

  it('observing (snapshots, predictions) never changes the run', () => {
    const cfg = { ...SMALL, dropout: 0.3 };
    const plain = advance(new MnistSession(cfg), 60);
    const watched = new MnistSession(cfg);
    const drawn = testImage(0);
    for (let i = 0; i < 60; i++) {
      watched.advance();
      if (i % 7 === 0) watched.snapshot({ drawn });
      if (i % 11 === 0) watched.checkpoint();
    }
    expect(sameParams(plain, watched)).toBe(true);
  });

  it('keeps every test prediction with its confidence; the counts agree with the accuracy', () => {
    const snap = advance(new MnistSession(SMALL), 100).snapshot();
    const { predicted, confidence } = snap.evaluation;
    expect(predicted).toHaveLength(2_000);
    let correct = 0;
    for (let i = 0; i < predicted.length; i++) {
      if (predicted[i] === data.test.labels[i]) correct++;
      expect(confidence[i]).toBeGreaterThanOrEqual(0.1 - 1e-6); // the argmax of 10 classes
      expect(confidence[i]).toBeLessThanOrEqual(1);
    }
    expect(correct / 2_000).toBeCloseTo(snap.testAccuracy, 6);
  });

  it('measures train metrics on a fixed subset of the training set', () => {
    expect(TRAIN_EVAL_SIZE).toBeLessThanOrEqual(10_000);
    const a = new MnistSession(SMALL).snapshot();
    const b = new MnistSession(SMALL).snapshot();
    expect(a.losses.train.latest).toBe(b.losses.train.latest);
  });

  it('classifies a drawn digit, with probabilities that sum to 1 and first-hidden-layer values', () => {
    const s = advance(new MnistSession(SMALL), 300);
    const { prediction } = s.snapshot({ drawn: testImage(0) });
    expect(prediction!.probs).toHaveLength(10);
    expect(prediction!.probs.reduce((a, b) => a + b)).toBeCloseTo(1, 5);
    expect(prediction!.hidden).toHaveLength(16);
    expect(prediction!.hidden!.every((v) => v >= 0)).toBe(true); // ReLU
    const best = prediction!.probs.indexOf(Math.max(...prediction!.probs));
    expect(best).toBe(s.snapshot().evaluation.predicted[0]);
  });

  it('hyperparameters apply live without resetting', () => {
    const s = advance(new MnistSession(SMALL), 10);
    s.setHyperparams({ ...SMALL, batchSize: 32, optimiser: 'sgd', lr: 0.1 });
    s.advance();
    expect(s.examples).toBe(10 * 64 + 32);
    expect(s.checkpoint().optimiser.kind).toBe('sgd');
  });
});

describe('MNIST done-when: ~95 % test accuracy, quickly', () => {
  /*
   * The default (784-128-10 ReLU, Adam lr 0.001, batch 64), seeds 1–5:
   * first ≥ 95 % on the 2,000 official test images after 50,000–60,000
   * examples (5–6 epochs), and 95.4–96.2 % after 10 epochs. About 2.6 s to
   * 95 % in Node; the browser's figure is in the roadmap note.
   */
  it('784-128-10 ReLU, Adam: ≥ 95 % within 6 epochs, still ≥ 95 % at 10', () => {
    const s = new MnistSession({
      ...SMALL,
      network: { hidden: [{ units: 128, activation: 'relu' }] },
    });
    let first95 = Infinity;
    let records = s.snapshot().losses.count;
    while (s.examples < 100_000) {
      s.advance();
      if (s.snapshot().losses.count === records) continue;
      records++;
      if (first95 === Infinity && s.snapshot().testAccuracy >= 0.95) first95 = s.examples;
    }
    expect(first95).toBeLessThanOrEqual(60_000 + 64);
    expect(s.snapshot().testAccuracy).toBeGreaterThanOrEqual(0.95);
  }, 60_000);
});

describe('MnistSession checkpoints', () => {
  for (const optimiser of ['sgd', 'momentum', 'adam'] as const) {
    it(`resuming mid-epoch is bitwise identical to never stopping (${optimiser}, dropout, L2)`, () => {
      const cfg: MnistSessionConfig = {
        ...SMALL,
        optimiser,
        lr: optimiser === 'adam' ? 0.001 : 0.05,
        dropout: 0.2,
        l2: 0.001,
      };
      const straight = advance(new MnistSession(cfg), 330);
      // Epoch 1 is 157 batches (156 × 64, then 16); 13 more puts us mid-way through epoch 2.
      const first = advance(new MnistSession(cfg), 170);
      const saved: MnistCheckpoint = structuredClone(first.checkpoint());
      expect(saved.cursor).toBe(13 * 64);
      // 160 more crosses into epoch 3, so its shuffle depends on the restored order and RNG.
      const resumed = advance(new MnistSession({ ...cfg, resume: saved }), 160);
      expect(sameParams(straight, resumed)).toBe(true);
      const [a, b] = [straight.snapshot(), resumed.snapshot()];
      expect([b.examples, b.step]).toEqual([a.examples, a.step]);
      expect(b.losses).toEqual(a.losses);
      expect(b.errors).toEqual(a.errors);
      expect(b.evaluation).toEqual(a.evaluation);
    });
  }

  it('a resumed run shows exactly what was saved', () => {
    const s = advance(new MnistSession(SMALL), 45);
    const resumed = new MnistSession({ ...SMALL, resume: structuredClone(s.checkpoint()) });
    expect(resumed.snapshot()).toEqual(s.snapshot());
  });

  it('rejects inconsistent checkpoints with a readable error', () => {
    const c = advance(new MnistSession(SMALL), 40).checkpoint();
    const bad = (patch: Partial<MnistCheckpoint>) => () =>
      new MnistSession({ ...SMALL, resume: { ...structuredClone(c), ...patch } });
    expect(bad({ order: new Uint32Array(5) })).toThrow(/5 training images/);
    const dup = c.order.slice();
    dup[1] = dup[0]!;
    expect(bad({ order: dup })).toThrow(/permutation/);
    expect(bad({ cursor: 10_000 })).toThrow(/cursor/);
    expect(bad({ examples: c.examples + RECORD_EVERY })).toThrow(/history/);
    expect(
      bad({ evaluation: { predicted: new Uint8Array(3), confidence: new Float32Array(3) } }),
    ).toThrow(/evaluation/);
  });
});

describe('TrainingController with an MNIST session', () => {
  function setup() {
    const sent: { message: FromWorker; transfer: ArrayBuffer[] }[] = [];
    const post: Post = (message, transfer) => sent.push({ message, transfer });
    const controller = new TrainingController(post, { now: () => 0, defer: () => {} });
    return { sent, send: controller.handle.bind(controller) };
  }

  it('init → ready with the sizes; step trains one batch; snapshots carry the drawn digit', () => {
    const { sent, send } = setup();
    send({ type: 'init', sessionId: 3, config: SMALL });
    expect(sent[0]!.message).toEqual({
      type: 'ready',
      sessionId: 3,
      data: { task: 'mnist', trainSize: 10_000, testSize: 2_000 },
    });
    send({ type: 'step' });
    send({ type: 'snapshot', requestId: 1, drawn: new Float32Array(784) });
    const { message, transfer } = sent.at(-1)!;
    if (message.type !== 'snapshot' || !isMnistSnapshot(message.snapshot)) {
      throw new Error('expected an MNIST snapshot');
    }
    expect(message.snapshot.step).toBe(1);
    expect(message.snapshot.prediction?.probs).toHaveLength(10);
    expect(transfer).toEqual(mnistSnapshotBuffers(message.snapshot));
    expect(structuredClone(message)).toEqual(message);
  });

  it('answers checkpoint requests, and init with resume continues', () => {
    const { sent, send } = setup();
    send({ type: 'init', sessionId: 1, config: SMALL });
    for (let i = 0; i < 5; i++) send({ type: 'step' });
    send({ type: 'checkpoint', requestId: 2 });
    const { message, transfer } = sent.at(-1)!;
    if (message.type !== 'checkpoint') throw new Error('expected a checkpoint');
    const c = message.checkpoint as MnistCheckpoint;
    expect(transfer).toEqual(mnistCheckpointBuffers(c));
    send({ type: 'init', sessionId: 2, config: { ...SMALL, resume: structuredClone(c) } });
    send({ type: 'snapshot', requestId: 3 });
    const reply = sent.at(-1)!.message;
    if (reply.type !== 'snapshot' || !isMnistSnapshot(reply.snapshot))
      throw new Error('no snapshot');
    expect(reply.snapshot.step).toBe(5);
  });
});
