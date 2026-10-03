import {
  Conv2D,
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
  ReLU,
  makeOptimiser,
  reshape,
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
import { HistogramTimeline } from './history';
import { LossHistory } from './lossHistory';
import type {
  ConvFilters,
  DeadFlags,
  FirstLayerWeights,
  FeatureMap,
  MnistCheckpoint,
  MnistEvaluation,
  MnistPrediction,
  MnistSessionData,
  MnistSnapshot,
} from './mnistSnapshot';
import {
  MNIST_IMAGE_SHAPE,
  type MnistNetworkSpec,
  convRowEnds,
  copyMnistNetwork,
  mnistNetworkToLayerConfig,
} from './network';
import type { Hyperparams } from './session';

export interface MnistSessionConfig extends Hyperparams {
  task: 'mnist';
  network: MnistNetworkSpec;
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
 * Gradient histograms, gradient RMS and dead flags are measured at every
 * record over this many training images (the first of the train-evaluation
 * subset), in chunks of GRAD_CHUNK.
 */
export const GRAD_EVAL_SIZE = 500;
const GRAD_CHUNK = 250;

/** Names of the parameterised layers: "Conv 1", "Dense 1", "Output"; or "Input → H1" for an MLP. */
function layerNamesFor({ conv, hidden }: MnistNetworkSpec): string[] {
  if (conv.length === 0) {
    const n = hidden.length + 1;
    return Array.from({ length: n }, (_, k) => {
      const from = k === 0 ? 'Input' : `H${k}`;
      return `${from} → ${k === n - 1 ? 'Output' : `H${k + 1}`}`;
    });
  }
  const convs = conv.filter((r) => r.kind === 'conv').map((_, i) => `Conv ${i + 1}`);
  return [...convs, ...hidden.map((_, i) => `Dense ${i + 1}`), 'Output'];
}

/**
 * One MNIST training run: an MLP (784 inputs) or a CNN (a 1 × 28 × 28 image)
 * with 10 logits, trained with softmax cross-entropy. DOM-free, so it runs
 * unchanged in a Worker.
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
  private readonly convs: Conv2D[];
  private readonly dropouts: Dropout[];
  private readonly trainer: Trainer;
  /** Whether the model takes images (`[N, 1, 28, 28]`) rather than rows of 784 pixels. */
  private readonly cnn: boolean;
  /** MLPs: index into model.layers of the layer whose output is the first hidden layer's value. */
  private readonly firstHiddenEnd: number | null;
  /** CNNs: index into model.layers of each conv row's last layer (its feature map). */
  private readonly mapEnds: number[];
  /** CNNs: the `conv` row of each Conv2D layer. */
  private readonly convRows: number[];
  private readonly network: MnistNetworkSpec;
  /** Conv and dense layers in order: what the timeline, layer names and dead flags index. */
  private readonly paramLayers: (Dense | Conv2D)[];
  /** For each ReLU layer (by index in the model), the param layer it follows. */
  private readonly reluOwners: Map<number, number>;
  private readonly timeline: HistogramTimeline;
  private dead: DeadFlags[] = [];
  private readonly gradEval: Uint32Array;
  /** Float64 sums of each param layer's ∂L/∂W over the gradient subset's chunks. */
  private readonly gradSums: Float64Array[];
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
    const model = layerFromConfig(mnistNetworkToLayerConfig(network, config.dropout), rng);
    if (!(model instanceof Sequential)) throw new Error('MnistSession: expected a Sequential');
    this.model = model;
    this.network = copyMnistNetwork(network);
    this.cnn = network.conv.length > 0;
    this.dense = model.layers.filter((l): l is Dense => l instanceof Dense);
    this.convs = model.layers.filter((l): l is Conv2D => l instanceof Conv2D);
    const first = model.layers.find((l) => l instanceof Dense || l instanceof Conv2D);
    if (first instanceof Dense || first instanceof Conv2D) first.inputGrad = false; // nothing reads ∂L/∂pixels
    this.dropouts = model.layers.filter((l): l is Dropout => l instanceof Dropout);
    let end: number | null = null;
    if (!this.cnn && network.hidden.length > 0) {
      end = 0;
      while (!(model.layers[end + 1] instanceof Dense)) end++;
    }
    this.firstHiddenEnd = end;
    this.mapEnds = convRowEnds(network.conv);
    this.convRows = network.conv.flatMap((row, i) => (row.kind === 'conv' ? [i] : []));
    this.paramLayers = model.layers.filter(
      (l): l is Dense | Conv2D => l instanceof Dense || l instanceof Conv2D,
    );
    this.reluOwners = new Map();
    let lastParam = -1;
    model.layers.forEach((l, i) => {
      if (l instanceof Dense || l instanceof Conv2D) lastParam++;
      else if (l instanceof ReLU) this.reluOwners.set(i, lastParam);
    });
    this.timeline = new HistogramTimeline(this.paramLayers.length);
    this.gradSums = this.paramLayers.map((l) => new Float64Array(l.W.value.size));

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
    this.gradEval = this.trainEval.slice(0, Math.min(GRAD_EVAL_SIZE, evalCount));
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
    this.trainer.trainStep(this.input(x), y);
    this.step++;
    this.examples += rows;
    this.cursor += rows;
    if (this.cursor >= n) this.cursor = 0;
    if (this.examples >= this.losses.count * RECORD_EVERY) this.record();
  }

  /** Rows of 784 pixels as the model takes them: unchanged for an MLP, a view as images for a CNN. */
  private input(x: Tensor): Tensor {
    return this.cnn ? reshape(x, [x.rows, ...MNIST_IMAGE_SHAPE]) : x;
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
      const logits = this.model.forward(this.input(x), false);
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
    const index = this.losses.count - 1;
    if (this.timeline.due(index)) {
      const gradW = this.measureGradients();
      this.timeline.record(
        index,
        this.paramLayers.map((l, k) => ({ W: l.W.value.data, gradW: gradW[k]! })),
      );
    }
  }

  /**
   * The gradient of (mean data loss + L2 penalty) over the gradient subset, in
   * evaluation mode, per parameterised layer's W; also sets the dead flags.
   * Observation only: it overwrites the params' `grad`, which the next
   * training step overwrites before the optimiser reads it, and draws no
   * randomness.
   */
  private measureGradients(): Float32Array[] {
    const total = this.gradEval.length;
    for (const sum of this.gradSums) sum.fill(0);
    const alive = new Map<number, Uint8Array>();
    for (let start = 0; start < total; start += GRAD_CHUNK) {
      const rows = Math.min(GRAD_CHUNK, total - start);
      const x = gatherImages(this.train, this.gradEval, start, this.x.take([rows, MNIST_PIXELS]));
      const y = oneHotInto(
        this.train.labels,
        this.gradEval,
        start,
        this.y.take([rows, MNIST_CLASSES]),
      );
      let h = this.input(x);
      this.model.layers.forEach((layer, i) => {
        h = layer.forward(h, false);
        if (layer instanceof ReLU) markAlive(h, alive, i);
      });
      this.evalLoss.forward(h, y);
      this.model.backward(this.evalLoss.backward());
      // The subset's mean gradient is the chunks' mean gradients weighted by their size.
      const weight = rows / total;
      this.paramLayers.forEach((l, k) => {
        const g = l.W.grad.data;
        const sum = this.gradSums[k]!;
        for (let i = 0; i < g.length; i++) sum[i]! += g[i]! * weight;
      });
    }
    const l2 = this.trainer.l2;
    this.dead = [...alive].flatMap(([i, live]) => {
      const flags = live.map((v) => 1 - v);
      return flags.some((f) => f === 1) ? [{ layer: this.reluOwners.get(i)!, flags }] : [];
    });
    return this.paramLayers.map((l, k) => {
      const w = l.W.value.data;
      const sum = this.gradSums[k]!;
      return Float32Array.from(sum, (g, i) => g + l2 * w[i]!);
    });
  }

  /**
   * The network's output for one 28×28 input in [0, 1], in evaluation mode.
   * For a CNN also each row's feature map and, for the loss against `target`
   * (default: the network's answer), the gradient of each map and of the input.
   */
  predict(pixels: Float32Array, target?: number): MnistPrediction {
    if (pixels.length !== MNIST_PIXELS) {
      throw new Error(`predict: expected ${MNIST_PIXELS} pixels, got ${pixels.length}`);
    }
    let h = this.input(new Tensor(pixels.slice(), [1, MNIST_PIXELS]));
    let hidden: Float32Array | undefined;
    const maps: FeatureMap[] = [];
    this.model.layers.forEach((layer, i) => {
      h = layer.forward(h, false);
      if (i === this.firstHiddenEnd) hidden = h.data.slice();
      const row = this.mapEnds.indexOf(i);
      if (row >= 0) {
        const [, c, ht, w] = h.shape as [number, number, number, number];
        maps.push({ row, shape: [c, ht, w], data: h.data.slice() });
      }
    });
    const z = h.data;
    let max = -Infinity;
    for (const v of z) max = Math.max(max, v);
    const probs = new Float32Array(MNIST_CLASSES);
    let sum = 0;
    for (let c = 0; c < MNIST_CLASSES; c++) sum += Math.exp(z[c]! - max);
    for (let c = 0; c < MNIST_CLASSES; c++) probs[c] = Math.exp(z[c]! - max) / sum;
    if (!this.cnn) return { probs, ...(hidden && { hidden }) };
    return { probs, maps, ...this.gradientsFor(h, probs, target) };
  }

  /**
   * Backward from the cross-entropy against `target` through the forward pass
   * `predict` just made, keeping ∂L/∂ each row's output and ∂L/∂ the input.
   */
  private gradientsFor(
    logits: Tensor,
    probs: Float32Array,
    target: number | undefined,
  ): Pick<MnistPrediction, 'target' | 'gradMaps' | 'inputGrad'> {
    const cls =
      target !== undefined && Number.isInteger(target) && target >= 0 && target < MNIST_CLASSES
        ? target
        : probs.indexOf(Math.max(...probs));
    const y = Tensor.zeros([1, MNIST_CLASSES]);
    y.data[cls] = 1;
    this.evalLoss.forward(logits, y);
    const first = this.paramLayers[0]!;
    first.inputGrad = true;
    const gradMaps: FeatureMap[] = [];
    let g = this.evalLoss.backward();
    try {
      for (let i = this.model.layers.length - 1; i >= 0; i--) {
        g = this.model.layers[i]!.backward(g); // now ∂L/∂(output of layer i − 1)
        const row = this.mapEnds.indexOf(i - 1);
        if (row >= 0) {
          const [, c, h, w] = g.shape as [number, number, number, number];
          gradMaps.unshift({ row, shape: [c, h, w], data: g.data.slice() });
        }
      }
    } finally {
      first.inputGrad = false;
    }
    return { target: cls, gradMaps, inputGrad: g.data.slice() };
  }

  data(): MnistSessionData {
    return { task: 'mnist', trainSize: mnistCount(this.train), testSize: mnistCount(this.test) };
  }

  snapshot(opts: { drawn?: Float32Array; gradTarget?: number } = {}): MnistSnapshot {
    const losses = this.losses.snapshot();
    const errors = this.errors.snapshot();
    const prediction = opts.drawn && this.predict(opts.drawn, opts.gradTarget);
    return {
      task: 'mnist',
      network: copyMnistNetwork(this.network),
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
      ...(this.cnn ? { filters: this.filters() } : { firstLayer: this.firstLayer() }),
      evaluation: this.evaluationCopy(),
      timeline: this.timeline.snapshot(),
      layerNames: layerNamesFor(this.network),
      dead: this.deadCopy(),
      gradEvalSize: this.gradEval.length,
      ...(prediction && { prediction }),
    };
  }

  private deadCopy(): DeadFlags[] {
    return this.dead.map((d) => ({ layer: d.layer, flags: d.flags.slice() }));
  }

  private firstLayer(): FirstLayerWeights {
    const first = this.dense[0]!;
    return {
      inFeatures: first.inFeatures,
      outFeatures: first.outFeatures,
      W: first.W.value.data.slice(),
      b: first.b.value.data.slice(),
    };
  }

  private filters(): ConvFilters[] {
    return this.convs.map((c, i) => ({
      row: this.convRows[i]!,
      inChannels: c.inChannels,
      outChannels: c.outChannels,
      kernel: c.kernel,
      stride: c.stride,
      padding: c.padding,
      W: c.W.value.data.slice(),
      b: c.b.value.data.slice(),
    }));
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
      timeline: this.timeline.exportState(),
      dead: this.deadCopy(),
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
    // Files from before the charts' history was saved: the charts restart here.
    if (c.timeline) this.timeline.importState(c.timeline);
    if (c.dead) {
      for (const d of c.dead) {
        const l = this.paramLayers[d.layer];
        const units = l instanceof Conv2D ? l.outChannels : l?.outFeatures;
        if (units !== d.flags.length)
          throw new Error('Checkpoint dead flags do not fit this network');
      }
      this.dead = c.dead.map((d) => ({ layer: d.layer, flags: d.flags.slice() }));
    }
  }
}

/**
 * Marks which units (dense) or channels (conv) of a ReLU output are non-zero
 * for at least one row; `alive` keeps, per ReLU layer index, 1 for alive.
 */
function markAlive(h: Tensor, alive: Map<number, Uint8Array>, layer: number): void {
  const [rows, units] = h.shape as [number, number];
  const area = h.size / (rows * units);
  let live = alive.get(layer);
  if (!live) alive.set(layer, (live = new Uint8Array(units)));
  const d = h.data;
  for (let r = 0; r < rows; r++) {
    for (let u = 0; u < units; u++) {
      if (live[u]) continue;
      const o = (r * units + u) * area;
      for (let p = 0; p < area; p++) {
        if (d[o + p] !== 0) {
          live[u] = 1;
          break;
        }
      }
    }
  }
}
