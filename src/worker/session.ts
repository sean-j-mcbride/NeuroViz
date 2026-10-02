import {
  BCEWithLogitsLoss,
  Dense,
  Dropout,
  type OptimiserKind,
  ReLU,
  Rng,
  Sequential,
  Tensor,
  Trainer,
  addL2,
  exportParams,
  importParams,
  layerFromConfig,
  makeOptimiser,
} from '../engine';
import {
  DOMAIN,
  type Dataset,
  type Toy2DKind,
  gatherRows,
  generateToy2D,
  range,
  shuffleInPlace,
  splitTrainTest,
} from '../data';
import type { Checkpoint } from './checkpoint';
import { HistogramTimeline, ParamHistory } from './history';
import { LossHistory } from './lossHistory';
import { INPUTS, type NetworkSpec, networkToLayerConfig } from './network';
import type {
  NeuronColumn,
  ProbeColumn,
  ProbeRef,
  ProbeTrace,
  SessionData,
  Snapshot,
} from './snapshot';

export interface DatasetSpec {
  kind: Toy2DKind;
  n: number;
  noise: number;
  seed: number;
}

/** Everything that may change mid-run without resetting training. */
export interface Hyperparams {
  lr: number;
  /** Mini-batch size; 'full' (or anything ≥ the training set) means full-batch. */
  batchSize: number | 'full';
  /** Switching optimiser starts it with fresh state (zero momentum / moments). */
  optimiser: OptimiserKind;
  /** L2 regularisation strength λ on the weights (not biases). */
  l2: number;
  /** Dropout rate after every hidden layer, in [0, 1). */
  dropout: number;
}

/** The hyperparameters in force from `epoch` on (until the next change). */
export interface HyperparamChange {
  epoch: number;
  hyperparams: Hyperparams;
}

function hyperparamsOf({ lr, batchSize, optimiser, l2, dropout }: Hyperparams): Hyperparams {
  return { lr, batchSize, optimiser, l2, dropout };
}

const sameHyperparams = (a: Hyperparams, b: Hyperparams) =>
  JSON.stringify(hyperparamsOf(a)) === JSON.stringify(hyperparamsOf(b));

export interface SessionConfig extends Hyperparams {
  dataset: DatasetSpec;
  network: NetworkSpec;
  /** Seeds weight initialisation, per-epoch shuffling and dropout masks. */
  seed: number;
  /** Fraction of the dataset held out for testing. Default 0.3. */
  testFraction?: number;
  /** Resolution of the heatmap grid. Default 50. */
  gridSize?: number;
  /**
   * Continue from this checkpoint (taken from a session with the same data,
   * network and seed) instead of starting at epoch 0.
   */
  resume?: Checkpoint;
}

/** A grid of G×G points covering the input domain, row 0 at the top. */
function makeGrid(g: number): Tensor {
  const grid = Tensor.zeros([g * g, INPUTS]);
  const cell = (2 * DOMAIN) / g;
  for (let r = 0; r < g; r++) {
    for (let c = 0; c < g; c++) {
      const i = r * g + c;
      grid.data[2 * i] = -DOMAIN + (c + 0.5) * cell;
      grid.data[2 * i + 1] = DOMAIN - (r + 0.5) * cell;
    }
  }
  return grid;
}

/** Copies a `[points, units]` activation matrix into neuron-major order. */
function toNeuronMajor(h: Tensor): Float32Array {
  const [points, units] = [h.rows, h.cols];
  const out = new Float32Array(points * units);
  for (let p = 0; p < points; p++) {
    for (let u = 0; u < units; u++) out[u * points + p] = h.data[p * units + u]!;
  }
  return out;
}

/** Per column of `h` ([points, units]): 1 if it is 0 at every point. */
function deadUnits(h: Tensor): Uint8Array {
  const [points, units] = [h.rows, h.cols];
  const dead = new Uint8Array(units).fill(1);
  for (let p = 0; p < points; p++) {
    for (let u = 0; u < units; u++) if (h.data[p * units + u] !== 0) dead[u] = 0;
  }
  return dead;
}

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

function sigmoidInPlace(a: Float32Array): Float32Array {
  for (let i = 0; i < a.length; i++) a[i] = sigmoid(a[i]!);
  return a;
}

/**
 * One training run of an MLP on a 2D toy dataset with BCE-with-logits.
 * DOM-free, so it runs unchanged in a Worker (or in tests).
 *
 * Only `trainEpoch` changes the model or consumes randomness; `snapshot` and
 * its probe run in evaluation mode, so observing a run never changes it.
 */
export class TrainingSession {
  readonly train: Dataset;
  readonly test: Dataset;
  readonly gridSize: number;
  private readonly network: NetworkSpec;
  private readonly model: Sequential;
  private readonly dense: Dense[];
  private readonly dropouts: Dropout[];
  private readonly trainer: Trainer;
  /**
   * For each column after the input: the index into model.layers of its dense
   * layer and of the layer whose output is the column's value (the layer
   * before the next dense, or the last layer).
   */
  private readonly columnLayers: { dense: number; end: number }[];
  private readonly shuffleRng: Rng;
  /** Drew the initial weights; the dropout layers keep drawing their masks from it. */
  private readonly initRng: Rng;
  private readonly order: Uint32Array;
  private readonly grid: Tensor;
  private readonly evalLoss = new BCEWithLogitsLoss();
  private batchSize: number;
  private batch: { x: Tensor; y: Tensor } | null = null;
  private readonly losses = new LossHistory();
  private readonly timeline: HistogramTimeline;
  private readonly paramHistory: ParamHistory;
  /** Every hyperparameter setting used, starting with epoch 0's. */
  private hyperparamLog: HyperparamChange[];
  private trainAccuracy = 0;
  private testAccuracy = 0;
  epoch = 0;
  step = 0;

  constructor(config: SessionConfig) {
    const { dataset, network, seed, testFraction = 0.3, gridSize = 50 } = config;
    const all = generateToy2D(dataset.kind, dataset);
    ({ train: this.train, test: this.test } = splitTrainTest(
      all,
      testFraction,
      new Rng(dataset.seed + 1),
    ));
    this.network = network;
    this.gridSize = gridSize;
    this.grid = makeGrid(gridSize);

    // Dropout layers share the init rng for their masks; they draw nothing until training.
    const rng = new Rng(seed);
    this.initRng = rng;
    const model = layerFromConfig(networkToLayerConfig(network, config.dropout), rng);
    if (!(model instanceof Sequential)) throw new Error('TrainingSession: expected a Sequential');
    this.model = model;
    this.dense = model.layers.filter((l): l is Dense => l instanceof Dense);
    this.dropouts = model.layers.filter((l): l is Dropout => l instanceof Dropout);
    this.columnLayers = [];
    model.layers.forEach((layer, i) => {
      if (!(layer instanceof Dense)) return;
      let end = i;
      while (end + 1 < model.layers.length && !(model.layers[end + 1] instanceof Dense)) end++;
      this.columnLayers.push({ dense: i, end });
    });

    this.trainer = new Trainer({
      model,
      loss: new BCEWithLogitsLoss(),
      optimiser: makeOptimiser(config.optimiser, config.lr),
      l2: config.l2,
    });
    this.shuffleRng = new Rng(seed + 1);
    this.order = range(this.train.x.rows);
    this.batchSize = this.resolveBatchSize(config.batchSize);
    this.hyperparamLog = [{ epoch: 0, hyperparams: hyperparamsOf(config) }];
    this.timeline = new HistogramTimeline(this.dense.length);
    this.paramHistory = new ParamHistory(
      this.dense.map((d) => ({ W: d.W.value.size, b: d.b.value.size })),
    );
    if (config.resume) this.restore(config.resume, config);
    else this.recordMetrics();
  }

  /** Everything needed to continue this run exactly (see `Checkpoint`). */
  checkpoint(): Checkpoint {
    return {
      epoch: this.epoch,
      step: this.step,
      losses: this.losses.exportState(),
      params: exportParams(this.model),
      optimiser: this.trainer.optimiser.saveState(this.model.params()),
      rng: { shuffle: this.shuffleRng.getState(), dropout: this.initRng.getState() },
      order: this.order.slice(),
      hyperparamLog: this.hyperparamLogCopy(),
      observations: {
        timeline: this.timeline.exportState(),
        params: this.paramHistory.exportState(),
      },
    };
  }

  private hyperparamLogCopy(): HyperparamChange[] {
    return this.hyperparamLog.map(({ epoch, hyperparams }) => ({
      epoch,
      hyperparams: { ...hyperparams },
    }));
  }

  /**
   * Picks up from a checkpoint. Optimiser state is restored only when it is
   * for the configured optimiser (otherwise that optimiser starts fresh, as
   * when switching mid-run). The charts' history continues if the checkpoint
   * has it, or restarts at the checkpoint's epoch if not. The
   * settings log continues; if the configured settings differ from the last
   * ones logged, the change is logged at the checkpoint's epoch.
   */
  private restore(c: Checkpoint, configured: Hyperparams): void {
    const { optimiser } = configured;
    const n = this.train.x.rows;
    if (c.order.length !== n) {
      throw new Error(`Checkpoint is for ${c.order.length} training points, not ${n}`);
    }
    const seen = new Uint8Array(n);
    for (const i of c.order) {
      if (i >= n || seen[i]) throw new Error('Checkpoint order is not a permutation');
      seen[i] = 1;
    }
    if (c.losses.count !== c.epoch + 1) {
      throw new Error(`Checkpoint loss history does not cover epochs 0–${c.epoch}`);
    }
    importParams(this.model, c.params);
    if (c.optimiser.kind === optimiser) {
      this.trainer.optimiser.loadState(this.model.params(), c.optimiser);
    }
    this.shuffleRng.setState(c.rng.shuffle);
    this.initRng.setState(c.rng.dropout);
    this.order.set(c.order);
    this.epoch = c.epoch;
    this.step = c.step;
    this.losses.importState(c.losses);
    const log = c.hyperparamLog;
    if (
      log.length === 0 ||
      log[0]!.epoch !== 0 ||
      log.some((e, i) => i > 0 && e.epoch <= log[i - 1]!.epoch) ||
      log.at(-1)!.epoch > c.epoch
    ) {
      throw new Error('Checkpoint settings log is out of order');
    }
    this.hyperparamLog = c.hyperparamLog.map(({ epoch, hyperparams }) => ({
      epoch,
      hyperparams: hyperparamsOf(hyperparams),
    }));
    this.logHyperparams(configured);
    this.trainAccuracy = this.evaluate(this.train).accuracy;
    this.testAccuracy = this.evaluate(this.test).accuracy;
    if (c.observations) {
      this.timeline.importState(c.observations.timeline);
      this.paramHistory.importState(c.observations.params);
    } else {
      this.observe();
    }
  }

  private resolveBatchSize(b: Hyperparams['batchSize']): number {
    const n = this.train.x.rows;
    return b === 'full' ? n : Math.max(1, Math.min(n, Math.floor(b)));
  }

  /**
   * Logs a change of settings at the current epoch. Unchanged settings (the UI
   * re-sends them) log nothing; several changes within one epoch log only the
   * last, and a change back to the previous settings removes the entry.
   */
  private logHyperparams(h: Hyperparams): void {
    const log = this.hyperparamLog;
    const last = log[log.length - 1]!;
    if (sameHyperparams(last.hyperparams, h)) return;
    if (last.epoch === this.epoch) {
      log.pop(); // replaced (at epoch 0, the starting settings themselves change)
      const before = log[log.length - 1];
      if (before && sameHyperparams(before.hyperparams, h)) return;
    }
    log.push({ epoch: this.epoch, hyperparams: hyperparamsOf(h) });
  }

  /** Applied from the next mini-batch on; does not reset training. */
  setHyperparams(hyperparams: Hyperparams): void {
    this.logHyperparams(hyperparams);
    const { lr, batchSize, optimiser, l2, dropout } = hyperparams;
    if (optimiser !== this.trainer.optimiser.kind) {
      this.trainer.optimiser = makeOptimiser(optimiser, lr);
    }
    this.trainer.optimiser.lr = lr;
    this.trainer.l2 = l2;
    for (const d of this.dropouts) d.rate = dropout;
    this.batchSize = this.resolveBatchSize(batchSize);
  }

  private batchBuffers(rows: number): { x: Tensor; y: Tensor } {
    if (this.batch?.x.rows !== rows) {
      this.batch = { x: Tensor.zeros([rows, INPUTS]), y: Tensor.zeros([rows, 1]) };
    }
    return this.batch;
  }

  /** One pass over the shuffled training set in mini-batches, then records metrics. */
  trainEpoch(): void {
    const n = this.train.x.rows;
    shuffleInPlace(this.order, this.shuffleRng);
    for (let start = 0; start < n; start += this.batchSize) {
      const { x, y } = this.batchBuffers(Math.min(this.batchSize, n - start));
      gatherRows(this.train.x, this.order, start, x);
      gatherRows(this.train.y, this.order, start, y);
      this.trainer.trainStep(x, y);
      this.step++;
    }
    this.epoch++;
    this.recordMetrics();
  }

  private evaluate({ x, y }: Dataset): { loss: number; accuracy: number } {
    const logits = this.model.forward(x, false);
    const loss = this.evalLoss.forward(logits, y);
    let correct = 0;
    for (let i = 0; i < logits.size; i++) if (logits.data[i]! > 0 === (y.data[i] === 1)) correct++;
    return { loss, accuracy: correct / logits.size };
  }

  private recordMetrics(): void {
    const tr = this.evaluate(this.train);
    const te = this.evaluate(this.test);
    // Rounded to float32, as the losses have always been kept (and older files hold them).
    this.losses.push(Math.fround(tr.loss), Math.fround(te.loss));
    this.trainAccuracy = tr.accuracy;
    this.testAccuracy = te.accuracy;
    this.observe();
  }

  /** Records the parameter history and (when due) the histogram timeline. Observation only. */
  private observe(): void {
    this.paramHistory.record(
      this.epoch,
      this.dense.map((d) => ({ W: d.W.value.data, b: d.b.value.data })),
    );
    if (this.timeline.due(this.epoch)) {
      this.measureGradients();
      this.timeline.record(
        this.epoch,
        this.dense.map((d) => ({ W: d.W.value.data, gradW: d.W.grad.data })),
      );
    }
  }

  /**
   * Full-batch gradient of (data loss + L2 penalty) over the training set in
   * evaluation mode, left in each param's `grad`. Safe between steps: the next
   * `trainStep` overwrites every gradient before the optimiser reads it.
   * Also returns, per ReLU layer (by index in the model), which units are dead:
   * 0 for every training point.
   */
  private measureGradients(): Map<number, Uint8Array> {
    // Layer by layer (as Sequential.forward does) so the ReLU outputs can be read on the way.
    const dead = new Map<number, Uint8Array>();
    let h = this.train.x;
    this.model.layers.forEach((layer, i) => {
      h = layer.forward(h, false);
      if (layer instanceof ReLU) dead.set(i, deadUnits(h));
    });
    this.evalLoss.forward(h, this.train.y);
    this.model.backward(this.evalLoss.backward());
    addL2(this.model.params(), this.trainer.l2);
    return dead;
  }

  /** The point sets; fixed for the life of the session. */
  data(): SessionData {
    return {
      train: { x: this.train.x.data.slice(), y: this.train.y.data.slice() },
      test: { x: this.test.x.data.slice(), y: this.test.y.data.slice() },
    };
  }

  /** Traces one data point forward and backward (evaluation mode, data loss only). */
  probe(ref: ProbeRef): ProbeTrace | undefined {
    const set = ref.set === 'train' ? this.train : this.test;
    if (!Number.isInteger(ref.index) || ref.index < 0 || ref.index >= set.x.rows) return undefined;
    const x = new Tensor(set.x.data.slice(2 * ref.index, 2 * ref.index + 2), [1, INPUTS]);
    const label = set.y.data[ref.index]!;

    const { layers } = this.model;
    const outputs: Float32Array[] = [];
    let h = x;
    for (const layer of layers) {
      h = layer.forward(h, false);
      outputs.push(h.data.slice());
    }
    const logit = h.data[0]!;
    const p = sigmoid(logit);
    const loss = this.evalLoss.forward(h, new Tensor(Float32Array.of(label), [1, 1]));

    // gradOut[i] = ∂L/∂(output of layer i).
    const gradOut: Float32Array[] = new Array<Float32Array>(layers.length);
    let g = this.evalLoss.backward();
    for (let i = layers.length - 1; i >= 0; i--) {
      gradOut[i] = g.data.slice();
      g = layers[i]!.backward(g);
    }
    const dW = this.dense.map((d) => d.W.grad.data.slice());

    const columns: ProbeColumn[] = [
      { z: x.data.slice(), a: x.data.slice(), dZ: g.data.slice(), dA: g.data.slice() },
    ];
    this.columnLayers.forEach(({ dense, end }, c) => {
      const isOutput = c === this.columnLayers.length - 1;
      if (!isOutput) {
        columns.push({
          z: outputs[dense]!,
          a: outputs[end]!,
          dZ: gradOut[dense]!,
          dA: gradOut[end]!,
        });
        return;
      }
      const pc = Math.min(Math.max(p, 1e-7), 1 - 1e-7);
      columns.push({
        z: Float32Array.of(logit),
        a: Float32Array.of(p),
        dZ: Float32Array.of(p - label),
        dA: Float32Array.of((pc - label) / (pc * (1 - pc))),
      });
    });
    return { ref, x: x.data.slice(), label, p, loss, columns, dW };
  }

  snapshot(opts: { probe?: ProbeRef } = {}): Snapshot {
    const dead = this.measureGradients();
    const columns: NeuronColumn[] = [
      { kind: 'input', units: INPUTS, values: toNeuronMajor(this.grid) },
    ];
    let h = this.grid;
    let col = 0;
    this.model.layers.forEach((layer, i) => {
      h = layer.forward(h, false);
      if (i !== this.columnLayers[col]?.end) return;
      const hidden = this.network.hidden[col];
      const values = toNeuronMajor(h);
      const flags = dead.get(this.columnLayers[col]!.dense + 1);
      columns.push(
        hidden
          ? { kind: hidden.activation, units: hidden.units, values, ...(flags && { dead: flags }) }
          : { kind: 'output', units: 1, values: sigmoidInPlace(values) },
      );
      col++;
    });

    const weights = this.dense.map((d) => ({
      inFeatures: d.inFeatures,
      outFeatures: d.outFeatures,
      W: d.W.value.data.slice(),
      b: d.b.value.data.slice(),
      gradW: d.W.grad.data.slice(),
      gradB: d.b.grad.data.slice(),
    }));
    const probe = opts.probe && this.probe(opts.probe);

    return {
      epoch: this.epoch,
      step: this.step,
      losses: this.losses.snapshot(),
      trainAccuracy: this.trainAccuracy,
      testAccuracy: this.testAccuracy,
      domain: DOMAIN,
      gridSize: this.gridSize,
      columns,
      weights,
      l2: this.trainer.l2,
      timeline: this.timeline.snapshot(),
      paramHistory: this.paramHistory.snapshot(),
      hyperparamLog: this.hyperparamLogCopy(),
      ...(probe && { probe }),
    };
  }
}
