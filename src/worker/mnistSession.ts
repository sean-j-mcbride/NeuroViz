import {
  Dense,
  Dropout,
  Rng,
  Sequential,
  SoftmaxCrossEntropyLoss,
  Tensor,
  TensorBuffer,
  Trainer,
  exportParams,
  importParams,
  layerFromConfig,
  makeOptimiser,
} from '../engine';
import {
  MNIST_CLASSES,
  MNIST_PIXELS,
  type MnistSet,
  type MnistSubset,
  gatherImages,
  mnistCount,
  oneHotInto,
  range,
  shuffleInPlace,
} from '../data';
import { LossHistory } from './lossHistory';
import type {
  MnistCheckpoint,
  MnistEvaluation,
  MnistPrediction,
  MnistSessionData,
  MnistSnapshot,
} from './mnistSnapshot';
import { type NetworkSpec, networkToLayerConfig } from './network';
import type { Hyperparams } from './session';

export interface MnistSessionConfig extends Hyperparams {
  task: 'mnist';
  network: NetworkSpec;
  /** Seeds weight initialisation, per-epoch shuffling and dropout masks. */
  seed: number;
  data: MnistSubset;
  /** Continue from this checkpoint (same data, network and seed). */
  resume?: MnistCheckpoint;
}

/** Metrics are recorded every this many training examples. */
export const RECORD_EVERY = 5000;
/** Size of the fixed training subset the train loss and accuracy are measured on. */
export const TRAIN_EVAL_SIZE = 1000;
/** Evaluation runs in chunks of this many images. */
const EVAL_CHUNK = 500;
/** The train-evaluation subset is the same for every run, so curves are comparable. */
const TRAIN_EVAL_SEED = 0;

/**
 * One MNIST training run: an MLP with 784 inputs and 10 logits, trained with
 * softmax cross-entropy. DOM-free, so it runs unchanged in a Worker.
 *
 * The unit of work is one mini-batch (an epoch of 10,000 images is far
 * longer than the worker's time slice). Only `advance` changes the model or
 * consumes randomness; snapshots and predictions run in evaluation mode.
 */
export class MnistSession {
  readonly train: MnistSet;
  readonly test: MnistSet;
  private readonly model: Sequential;
  private readonly dense: Dense[];
  private readonly dropouts: Dropout[];
  private readonly trainer: Trainer;
  /** Index into model.layers of the layer whose output is the first hidden layer's value. */
  private readonly firstHiddenEnd: number | null;
  private readonly shuffleRng: Rng;
  private readonly initRng: Rng;
  private readonly order: Uint32Array;
  private readonly trainEval: Uint32Array;
  private readonly testAll: Uint32Array;
  private readonly evalLoss = new SoftmaxCrossEntropyLoss();
  private readonly x = new TensorBuffer();
  private readonly y = new TensorBuffer();
  private batchSize: number;
  private cursor = 0;
  private readonly losses = new LossHistory();
  private readonly errors = new LossHistory();
  private evaluation: MnistEvaluation;
  examples = 0;
  step = 0;

  constructor(config: MnistSessionConfig) {
    const { network, seed, data } = config;
    this.train = data.train;
    this.test = data.test;
    const rng = new Rng(seed);
    this.initRng = rng;
    const model = layerFromConfig(
      networkToLayerConfig(network, config.dropout, MNIST_PIXELS, MNIST_CLASSES),
      rng,
    );
    if (!(model instanceof Sequential)) throw new Error('MnistSession: expected a Sequential');
    this.model = model;
    this.dense = model.layers.filter((l): l is Dense => l instanceof Dense);
    this.dense[0]!.inputGrad = false; // nothing reads ∂L/∂pixels
    this.dropouts = model.layers.filter((l): l is Dropout => l instanceof Dropout);
    let end: number | null = null;
    if (network.hidden.length > 0) {
      end = 0;
      while (!(model.layers[end + 1] instanceof Dense)) end++;
    }
    this.firstHiddenEnd = end;

    this.trainer = new Trainer({
      model,
      loss: new SoftmaxCrossEntropyLoss(),
      optimiser: makeOptimiser(config.optimiser, config.lr),
      l2: config.l2,
    });
    this.shuffleRng = new Rng(seed + 1);
    const n = mnistCount(this.train);
    this.order = range(n);
    const evalCount = Math.min(TRAIN_EVAL_SIZE, n);
    this.trainEval = shuffleInPlace(range(n), new Rng(TRAIN_EVAL_SEED)).slice(0, evalCount);
    this.testAll = range(mnistCount(this.test));
    this.batchSize = this.resolveBatchSize(config.batchSize);
    this.evaluation = {
      predicted: new Uint8Array(this.testAll.length),
      confidence: new Float32Array(this.testAll.length),
    };
    if (config.resume) this.restore(config.resume, config.optimiser);
    else this.record();
  }

  private resolveBatchSize(b: Hyperparams['batchSize']): number {
    const n = mnistCount(this.train);
    return b === 'full' ? n : Math.max(1, Math.min(n, Math.floor(b)));
  }

  /** Applied from the next mini-batch on; does not reset training. */
  setHyperparams({ lr, batchSize, optimiser, l2, dropout }: Hyperparams): void {
    if (optimiser !== this.trainer.optimiser.kind) {
      this.trainer.optimiser = makeOptimiser(optimiser, lr);
    }
    this.trainer.optimiser.lr = lr;
    this.trainer.l2 = l2;
    for (const d of this.dropouts) d.rate = dropout;
    this.batchSize = this.resolveBatchSize(batchSize);
  }

  get epoch(): number {
    return this.examples / mnistCount(this.train);
  }

  /**
   * Trains one mini-batch. Each epoch starts by shuffling the previous order in
   * place; its last batch may be smaller. Records metrics when due.
   */
  advance(): void {
    const n = mnistCount(this.train);
    if (this.cursor === 0) shuffleInPlace(this.order, this.shuffleRng);
    const rows = Math.min(this.batchSize, n - this.cursor);
    const x = this.x.take([rows, MNIST_PIXELS]);
    const y = this.y.take([rows, MNIST_CLASSES]);
    gatherImages(this.train, this.order, this.cursor, x);
    oneHotInto(this.train.labels, this.order, this.cursor, y);
    this.trainer.trainStep(x, y);
    this.step++;
    this.examples += rows;
    this.cursor += rows;
    if (this.cursor >= n) this.cursor = 0;
    if (this.examples >= this.losses.count * RECORD_EVERY) this.record();
  }

  /** Loss and accuracy over `indices` of `set` in evaluation mode, optionally keeping predictions. */
  private evaluate(
    set: MnistSet,
    indices: Uint32Array,
    keep?: MnistEvaluation,
  ): { loss: number; accuracy: number } {
    let lossSum = 0;
    let correct = 0;
    for (let start = 0; start < indices.length; start += EVAL_CHUNK) {
      const rows = Math.min(EVAL_CHUNK, indices.length - start);
      const x = gatherImages(set, indices, start, this.x.take([rows, MNIST_PIXELS]));
      const y = oneHotInto(set.labels, indices, start, this.y.take([rows, MNIST_CLASSES]));
      const logits = this.model.forward(x, false);
      lossSum += this.evalLoss.forward(logits, y) * rows;
      const z = logits.data;
      for (let r = 0; r < rows; r++) {
        const o = r * MNIST_CLASSES;
        let best = 0;
        for (let c = 1; c < MNIST_CLASSES; c++) if (z[o + c]! > z[o + best]!) best = c;
        if (best === set.labels[indices[start + r]!]) correct++;
        if (keep) {
          let sum = 0;
          for (let c = 0; c < MNIST_CLASSES; c++) sum += Math.exp(z[o + c]! - z[o + best]!);
          keep.predicted[start + r] = best;
          keep.confidence[start + r] = 1 / sum;
        }
      }
    }
    return { loss: lossSum / indices.length, accuracy: correct / indices.length };
  }

  /**
   * Evaluates once and logs it for every record now due: one, unless a batch
   * was larger than the record interval, so record k stays at k · RECORD_EVERY.
   */
  private record(): void {
    const tr = this.evaluate(this.train, this.trainEval);
    const te = this.evaluate(this.test, this.testAll, this.evaluation);
    do {
      this.losses.push(Math.fround(tr.loss), Math.fround(te.loss));
      this.errors.push(Math.fround(1 - tr.accuracy), Math.fround(1 - te.accuracy));
    } while (this.examples >= this.losses.count * RECORD_EVERY);
  }

  /** The network's output for one 28×28 input in [0, 1]. Evaluation mode. */
  predict(pixels: Float32Array): MnistPrediction {
    if (pixels.length !== MNIST_PIXELS) {
      throw new Error(`predict: expected ${MNIST_PIXELS} pixels, got ${pixels.length}`);
    }
    let h = new Tensor(pixels.slice(), [1, MNIST_PIXELS]);
    let hidden: Float32Array | undefined;
    this.model.layers.forEach((layer, i) => {
      h = layer.forward(h, false);
      if (i === this.firstHiddenEnd) hidden = h.data.slice();
    });
    const z = h.data;
    let max = -Infinity;
    for (const v of z) max = Math.max(max, v);
    const probs = new Float32Array(MNIST_CLASSES);
    let sum = 0;
    for (let c = 0; c < MNIST_CLASSES; c++) sum += Math.exp(z[c]! - max);
    for (let c = 0; c < MNIST_CLASSES; c++) probs[c] = Math.exp(z[c]! - max) / sum;
    return { probs, ...(hidden && { hidden }) };
  }

  data(): MnistSessionData {
    return { task: 'mnist', trainSize: mnistCount(this.train), testSize: mnistCount(this.test) };
  }

  snapshot(opts: { drawn?: Float32Array } = {}): MnistSnapshot {
    const losses = this.losses.snapshot();
    const errors = this.errors.snapshot();
    const first = this.dense[0]!;
    const prediction = opts.drawn && this.predict(opts.drawn);
    return {
      task: 'mnist',
      examples: this.examples,
      epoch: this.epoch,
      step: this.step,
      trainSize: mnistCount(this.train),
      testSize: mnistCount(this.test),
      recordEvery: RECORD_EVERY,
      losses,
      errors,
      trainAccuracy: 1 - errors.train.latest,
      testAccuracy: 1 - errors.test.latest,
      firstLayer: {
        inFeatures: first.inFeatures,
        outFeatures: first.outFeatures,
        W: first.W.value.data.slice(),
        b: first.b.value.data.slice(),
      },
      evaluation: this.evaluationCopy(),
      ...(prediction && { prediction }),
    };
  }

  private evaluationCopy(): MnistEvaluation {
    return {
      predicted: this.evaluation.predicted.slice(),
      confidence: this.evaluation.confidence.slice(),
    };
  }

  checkpoint(): MnistCheckpoint {
    return {
      task: 'mnist',
      examples: this.examples,
      step: this.step,
      cursor: this.cursor,
      order: this.order.slice(),
      losses: this.losses.exportState(),
      errors: this.errors.exportState(),
      params: exportParams(this.model),
      optimiser: this.trainer.optimiser.saveState(this.model.params()),
      rng: { shuffle: this.shuffleRng.getState(), dropout: this.initRng.getState() },
      evaluation: this.evaluationCopy(),
    };
  }

  /**
   * Picks up from a checkpoint. Optimiser state is restored only when it is
   * for the configured optimiser (otherwise that optimiser starts fresh).
   */
  private restore(c: MnistCheckpoint, optimiser: Hyperparams['optimiser']): void {
    const n = mnistCount(this.train);
    if (c.order.length !== n) {
      throw new Error(`Checkpoint is for ${c.order.length} training images, not ${n}`);
    }
    const seen = new Uint8Array(n);
    for (const i of c.order) {
      if (i >= n || seen[i]) throw new Error('Checkpoint order is not a permutation');
      seen[i] = 1;
    }
    if (!Number.isInteger(c.cursor) || c.cursor < 0 || c.cursor >= n) {
      throw new Error(`Checkpoint cursor ${c.cursor} is outside the training set`);
    }
    const records = Math.floor(c.examples / RECORD_EVERY) + 1;
    if (c.losses.count !== records || c.errors.count !== records) {
      throw new Error(`Checkpoint history does not cover ${c.examples} examples`);
    }
    const t = this.testAll.length;
    if (c.evaluation.predicted.length !== t || c.evaluation.confidence.length !== t) {
      throw new Error(`Checkpoint evaluation is not for ${t} test images`);
    }
    importParams(this.model, c.params);
    if (c.optimiser.kind === optimiser) {
      this.trainer.optimiser.loadState(this.model.params(), c.optimiser);
    }
    this.shuffleRng.setState(c.rng.shuffle);
    this.initRng.setState(c.rng.dropout);
    this.order.set(c.order);
    this.cursor = c.cursor;
    this.examples = c.examples;
    this.step = c.step;
    this.losses.importState(c.losses);
    this.errors.importState(c.errors);
    this.evaluation.predicted.set(c.evaluation.predicted);
    this.evaluation.confidence.set(c.evaluation.confidence);
  }
}
