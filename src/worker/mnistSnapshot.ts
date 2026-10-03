import type { OptimiserState, Padding, ParamValues, RngState } from '../engine';
import type { TimelineSnapshot, TimelineState } from './history';
import type { LossHistorySnapshot, LossHistoryState } from './lossHistory';
import type { MnistNetworkSpec } from './network';

/** What the network makes of the test set at the latest evaluation. */
export interface MnistEvaluation {
  /** `[test N]`: the predicted class (argmax) of each test image. */
  predicted: Uint8Array;
  /** `[test N]`: the softmax probability of that predicted class. */
  confidence: Float32Array;
}

/** One conv row's output for one input: `[C, H, W]` row-major, after its activation. */
export interface FeatureMap {
  /** Index of the row in the network's `conv` list. */
  row: number;
  shape: [channels: number, height: number, width: number];
  data: Float32Array;
}

/** The network's view of one input (a drawn digit). Evaluation mode. */
export interface MnistPrediction {
  /** Softmax probabilities of the 10 classes. */
  probs: Float32Array;
  /** MLPs: post-activation values of the first hidden layer (absent with no hidden layer). */
  hidden?: Float32Array;
  /** CNNs: every conv and pool row's output. */
  maps?: FeatureMap[];
  /**
   * CNNs: the class the gradients below are for (the requested one, or the
   * network's answer). L is the cross-entropy loss of this one input against it.
   */
  target?: number;
  /** CNNs: ∂L/∂(each row's output), the same shapes as `maps`. */
  gradMaps?: FeatureMap[];
  /** CNNs: ∂L/∂(input pixels), 784 values. */
  inputGrad?: Float32Array;
}

/**
 * ReLU units that output 0 for every image of the gradient subset ("dead"):
 * a dense layer's units, or a conv layer's channels. `layer` indexes the
 * parameterised layers (as in `layerNames` and the timeline).
 */
export interface DeadFlags {
  layer: number;
  /** 1 = dead, per unit or channel. */
  flags: Uint8Array;
}

/** A dense layer's weights `[inFeatures, outFeatures]` and biases. */
export interface FirstLayerWeights {
  inFeatures: number;
  outFeatures: number;
  W: Float32Array;
  b: Float32Array;
}

/**
 * One conv layer's filters. `W` is `[inChannels·k·k, outChannels]`: filter f's
 * weight for input channel c at kernel row ky, column kx is
 * `W[(c·k² + ky·k + kx)·outChannels + f]`.
 */
export interface ConvFilters {
  /** Index of the row in the network's `conv` list. */
  row: number;
  inChannels: number;
  outChannels: number;
  kernel: number;
  stride: number;
  padding: Padding;
  W: Float32Array;
  b: Float32Array;
}

/**
 * One moment of an MNIST run, for the UI. Plain data and fresh typed arrays,
 * so it is structured-cloneable and its buffers can be transferred.
 *
 * Metrics are recorded every `recordEvery` training examples (record 0 is
 * before training): after the first mini-batch that reaches each multiple.
 * Losses and error rates (1 − accuracy) are kept in the same bounded history
 * as the playground's per-epoch losses.
 */
export interface MnistSnapshot {
  task: 'mnist';
  /** The network this run trains, so views match the weights even while the builder is being edited. */
  network: MnistNetworkSpec;
  /** Training examples seen so far. */
  examples: number;
  /** examples / training-set size. */
  epoch: number;
  /** Optimiser steps (mini-batches). */
  step: number;
  trainSize: number;
  testSize: number;
  recordEvery: number;
  /** Train loss (on a fixed evaluation subset of the training set) and test loss, per record. */
  losses: LossHistorySnapshot;
  /** Train and test error rate per record, same records. */
  errors: LossHistorySnapshot;
  trainAccuracy: number;
  testAccuracy: number;
  /**
   * MLPs only: the first dense layer, `[784, units]` row-major, so unit j's
   * 28×28 weight image is `W[p·units + j]` for pixel p. With no hidden layer
   * this is the output layer, and the units are the 10 classes.
   */
  firstLayer?: FirstLayerWeights;
  /** CNNs only: every conv layer's filters, in order. */
  filters?: ConvFilters[];
  /** At the latest record. */
  evaluation: MnistEvaluation;
  /**
   * Weight and gradient histograms and gradient RMS per parameterised layer
   * (conv and dense, in order), one column per record: its `epochs` are
   * record indices. Gradients are of the mean loss (plus L2) over the
   * gradient subset.
   */
  timeline: TimelineSnapshot;
  /** One name per parameterised layer: "Conv 1", "Dense 1", "Output" or "Input → H1"-style. */
  layerNames: string[];
  /** Dead ReLU units or channels at the latest record (layers with none dead are left out). */
  dead: DeadFlags[];
  /** Images in the fixed gradient subset (from the training set). */
  gradEvalSize: number;
  /** Present when the snapshot was requested with a drawn digit. */
  prediction?: MnistPrediction;
}

/** Sent once per session, in `ready`: the UI already has the images. */
export interface MnistSessionData {
  task: 'mnist';
  trainSize: number;
  testSize: number;
}

/**
 * Everything needed to continue an MNIST run exactly, even mid-epoch:
 * resuming and training on is bitwise identical to never having stopped.
 * Structured-cloneable. The architecture, data and hyperparameters come from
 * the session config it is resumed with.
 */
export interface MnistCheckpoint {
  task: 'mnist';
  examples: number;
  step: number;
  /** Position of the next mini-batch in `order` (0 = the next batch starts an epoch). */
  cursor: number;
  /** The current epoch's training order (the next epoch shuffles it in place). */
  order: Uint32Array;
  losses: LossHistoryState;
  errors: LossHistoryState;
  params: ParamValues;
  optimiser: OptimiserState;
  rng: { shuffle: RngState; dropout: RngState };
  /** The latest test-set evaluation, so the confusion matrix and gallery carry on unchanged. */
  evaluation: MnistEvaluation;
  /** The charts' history and dead flags (absent in files saved before they existed: they restart). */
  timeline?: TimelineState;
  dead?: DeadFlags[];
}

export function isMnistSnapshot<T extends object>(s: T | MnistSnapshot): s is MnistSnapshot {
  return 'task' in s && s.task === 'mnist';
}

function historyArrays(...hs: (LossHistorySnapshot | LossHistoryState)[]): Float32Array[] {
  return hs.flatMap(({ train, test }) => [
    train.mean,
    train.min,
    train.max,
    test.mean,
    test.min,
    test.max,
  ]);
}

/** Every typed-array buffer in a snapshot, for a zero-copy `postMessage` transfer list. */
export function mnistSnapshotBuffers(s: MnistSnapshot): ArrayBuffer[] {
  const arrays: (Float32Array | Uint8Array)[] = [
    ...historyArrays(s.losses, s.errors),
    s.evaluation.predicted,
    s.evaluation.confidence,
  ];
  if (s.firstLayer) arrays.push(s.firstLayer.W, s.firstLayer.b);
  arrays.push(...timelineArrays(s.timeline), ...s.dead.map((d) => d.flags));
  for (const f of s.filters ?? []) arrays.push(f.W, f.b);
  if (s.prediction) {
    arrays.push(s.prediction.probs);
    if (s.prediction.hidden) arrays.push(s.prediction.hidden);
    for (const m of [...(s.prediction.maps ?? []), ...(s.prediction.gradMaps ?? [])]) {
      arrays.push(m.data);
    }
    if (s.prediction.inputGrad) arrays.push(s.prediction.inputGrad);
  }
  return [...new Set(arrays.map((a) => a.buffer as ArrayBuffer))];
}

export function mnistCheckpointBuffers(c: MnistCheckpoint): ArrayBuffer[] {
  const arrays: (Float32Array | Uint32Array | Uint8Array)[] = [
    ...historyArrays(c.losses, c.errors),
    c.order,
    ...Object.values(c.params),
    ...Object.values(c.optimiser.slots).flat(),
    c.evaluation.predicted,
    c.evaluation.confidence,
    ...(c.timeline ? timelineArrays(c.timeline) : []),
    ...(c.dead ?? []).map((d) => d.flags),
  ];
  return [...new Set(arrays.map((a) => a.buffer as ArrayBuffer))];
}

function timelineArrays(t: TimelineSnapshot | TimelineState): Float32Array[] {
  return [t.epochs, ...t.layers.flatMap((l) => [l.weightHist, l.gradHist, l.weightRms, l.gradRms])];
}
