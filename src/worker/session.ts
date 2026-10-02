import {
  BCEWithLogitsLoss,
  Dense,
  Rng,
  SGD,
  Sequential,
  Tensor,
  Trainer,
  layerFromConfig,
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
import { INPUTS, type NetworkSpec, networkToLayerConfig } from './network';
import type { NeuronColumn, Snapshot } from './snapshot';

export interface DatasetSpec {
  kind: Toy2DKind;
  n: number;
  noise: number;
  seed: number;
}

export interface Hyperparams {
  lr: number;
  /** Mini-batch size; 'full' (or anything ≥ the training set) means full-batch. */
  batchSize: number | 'full';
}

export interface SessionConfig extends Hyperparams {
  dataset: DatasetSpec;
  network: NetworkSpec;
  /** Seeds weight initialisation and per-epoch shuffling. */
  seed: number;
  /** Fraction of the dataset held out for testing. Default 0.3. */
  testFraction?: number;
  /** Resolution of the heatmap grid. Default 50. */
  gridSize?: number;
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

function sigmoidInPlace(a: Float32Array): Float32Array {
  for (let i = 0; i < a.length; i++) a[i] = 1 / (1 + Math.exp(-a[i]!));
  return a;
}

/** Growable float history. */
class History {
  private data = new Float32Array(256);
  length = 0;

  push(v: number): void {
    if (this.length === this.data.length) {
      const next = new Float32Array(this.data.length * 2);
      next.set(this.data);
      this.data = next;
    }
    this.data[this.length++] = v;
  }

  copy(): Float32Array {
    return this.data.slice(0, this.length);
  }
}

/**
 * One training run of an MLP on a 2D toy dataset: SGD on BCE-with-logits.
 * DOM-free, so it can run on the main thread or, unchanged, in a Worker.
 */
export class TrainingSession {
  readonly train: Dataset;
  readonly test: Dataset;
  readonly gridSize: number;
  private readonly network: NetworkSpec;
  private readonly model: Sequential;
  private readonly trainer: Trainer;
  private readonly optimiser: SGD;
  /** Index into model.layers of the last layer of each column after the input. */
  private readonly columnEnds: number[];
  private readonly shuffleRng: Rng;
  private readonly order: Uint32Array;
  private readonly grid: Tensor;
  private readonly evalLoss = new BCEWithLogitsLoss();
  private batchSize: number;
  private batch: { x: Tensor; y: Tensor } | null = null;
  private readonly trainLoss = new History();
  private readonly testLoss = new History();
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

    const rng = new Rng(seed);
    const model = layerFromConfig(networkToLayerConfig(network), rng);
    if (!(model instanceof Sequential)) throw new Error('TrainingSession: expected a Sequential');
    this.model = model;
    this.columnEnds = [];
    model.layers.forEach((layer, i) => {
      const next = model.layers[i + 1];
      // A column ends at a dense layer with no activation after it, or at an activation.
      if (!(layer instanceof Dense) || !next || next instanceof Dense) this.columnEnds.push(i);
    });

    this.optimiser = new SGD(config.lr);
    this.trainer = new Trainer({ model, loss: new BCEWithLogitsLoss(), optimiser: this.optimiser });
    this.shuffleRng = new Rng(seed + 1);
    this.order = range(this.train.x.rows);
    this.batchSize = this.resolveBatchSize(config.batchSize);
    this.recordMetrics();
  }

  private resolveBatchSize(b: Hyperparams['batchSize']): number {
    const n = this.train.x.rows;
    return b === 'full' ? n : Math.max(1, Math.min(n, Math.floor(b)));
  }

  /** Applied from the next mini-batch on; does not reset training. */
  setHyperparams({ lr, batchSize }: Hyperparams): void {
    this.optimiser.lr = lr;
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
    this.trainLoss.push(tr.loss);
    this.testLoss.push(te.loss);
    this.trainAccuracy = tr.accuracy;
    this.testAccuracy = te.accuracy;
  }

  snapshot(): Snapshot {
    const columns: NeuronColumn[] = [
      { kind: 'input', units: INPUTS, values: toNeuronMajor(this.grid) },
    ];
    let h = this.grid;
    let col = 0;
    this.model.layers.forEach((layer, i) => {
      h = layer.forward(h, false);
      if (i !== this.columnEnds[col]) return;
      const hidden = this.network.hidden[col];
      const values = toNeuronMajor(h);
      columns.push(
        hidden
          ? { kind: hidden.activation, units: hidden.units, values }
          : { kind: 'output', units: 1, values: sigmoidInPlace(values) },
      );
      col++;
    });

    const weights = this.model.layers
      .filter((l): l is Dense => l instanceof Dense)
      .map((d) => ({
        inFeatures: d.inFeatures,
        outFeatures: d.outFeatures,
        W: d.W.value.data.slice(),
        b: d.b.value.data.slice(),
      }));

    return {
      epoch: this.epoch,
      step: this.step,
      trainLoss: this.trainLoss.copy(),
      testLoss: this.testLoss.copy(),
      trainAccuracy: this.trainAccuracy,
      testAccuracy: this.testAccuracy,
      domain: DOMAIN,
      gridSize: this.gridSize,
      columns,
      weights,
      train: { x: this.train.x.data.slice(), y: this.train.y.data.slice() },
      test: { x: this.test.x.data.slice(), y: this.test.y.data.slice() },
    };
  }
}
