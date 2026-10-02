import type { ParamHistorySnapshot, TimelineSnapshot } from './history';
import type { LossHistorySnapshot } from './lossHistory';
import type { Activation } from './network';
import type { HyperparamChange } from './session';

/** What a column of neurons in the network graph is. */
export type ColumnKind = 'input' | Activation | 'output';

export interface NeuronColumn {
  kind: ColumnKind;
  units: number;
  /**
   * Each neuron's value over the G×G grid, neuron-major: neuron u's map is
   * `values.subarray(u·G², (u+1)·G²)`. Hidden columns hold post-activation
   * values; the output column holds σ(logit), the probability of label 1.
   */
  values: Float32Array;
  /**
   * ReLU columns only: 1 for each neuron that outputs 0 for every training
   * point (evaluation mode). No gradient reaches a dead neuron's incoming
   * weights or bias, so it stays dead.
   */
  dead?: Uint8Array;
}

export interface DenseWeights {
  inFeatures: number;
  outFeatures: number;
  /** Row-major `[inFeatures, outFeatures]`: weight from input i to unit j is `W[i·out + j]`. */
  W: Float32Array;
  b: Float32Array;
  /**
   * ∂(loss + L2 penalty)/∂W over the full training set (evaluation mode, so no
   * dropout): the gradient the optimiser would see from a full batch. Same layout as W.
   */
  gradW: Float32Array;
  gradB: Float32Array;
}

export interface PointSet {
  /** `[N, 2]` row-major. */
  x: Float32Array;
  /** `[N]`, each 0 or 1. */
  y: Float32Array;
}

/** Fixed for the life of a session; sent to the UI once. */
export interface SessionData {
  train: PointSet;
  test: PointSet;
}

/** Identifies one data point to trace through the network. */
export interface ProbeRef {
  set: 'train' | 'test';
  index: number;
}

/**
 * One column's values for the probe input. For hidden columns z is the
 * pre-activation and a the activation (equal for linear layers). For the input
 * column z = a = x. For the output column z is the logit and a = σ(z).
 */
export interface ProbeColumn {
  z: Float32Array;
  a: Float32Array;
  /** ∂L/∂z. For the output this is p − y. */
  dZ: Float32Array;
  /** ∂L/∂a. For the output, ∂L/∂p (p clamped away from 0 and 1). */
  dA: Float32Array;
}

/** A single example's forward and backward pass (evaluation mode, data loss only). */
export interface ProbeTrace {
  ref: ProbeRef;
  /** The input point (x₁, x₂). */
  x: Float32Array;
  label: number;
  /** σ(logit): the predicted probability of label 1. */
  p: number;
  /** Binary cross-entropy for this example. */
  loss: number;
  /** Aligned with `Snapshot.columns`. */
  columns: ProbeColumn[];
  /** Per dense layer, ∂L/∂W for this example (= a_prevᵀ · dZ); layout as `DenseWeights.W`. */
  dW: Float32Array[];
}

/**
 * Everything the UI needs to draw one moment of training. Plain data and typed
 * arrays only, so it is structured-cloneable (and can cross a Worker boundary).
 * Every array is a fresh copy: the UI may keep it, and the worker may transfer it.
 */
export interface Snapshot {
  epoch: number;
  /** Total number of optimiser steps (mini-batches) so far. */
  step: number;
  /**
   * Full-dataset train and test loss per epoch (epoch 0 = before training).
   * Exact for the first 4,096 epochs, then merged into buckets with their
   * mean, min and max; the latest and lowest values are always exact.
   */
  losses: LossHistorySnapshot;
  trainAccuracy: number;
  testAccuracy: number;
  /** Inputs span [−domain, domain]². */
  domain: number;
  /** Grid resolution G. Grid cell (row r, col c) is index r·G + c; row 0 is the top (y = +domain). */
  gridSize: number;
  /** Input column first, then one per hidden layer, then the output. */
  columns: NeuronColumn[];
  /** One per dense layer: `weights[k]` connects `columns[k]` to `columns[k + 1]`. */
  weights: DenseWeights[];
  /** L2 strength λ when the snapshot was taken; when > 0, `gradW` and the timeline's gradients include λ·W. */
  l2: number;
  /** Weight / gradient histograms and RMS per dense layer over the whole run. */
  timeline: TimelineSnapshot;
  /** Recent per-parameter values, for sparklines. */
  paramHistory: ParamHistorySnapshot;
  /** The settings used over the run: epoch 0's, then each mid-run change. */
  hyperparamLog: HyperparamChange[];
  /** Present when the snapshot was requested with a probe. */
  probe?: ProbeTrace;
}

/** The output probability σ(logit) over the grid: what the decision boundary shows. */
export interface OutputGrid {
  /** G×G values, row 0 at the top (y = +domain). */
  values: Float32Array;
  gridSize: number;
  domain: number;
}

export function outputGrid(s: Snapshot): OutputGrid {
  return {
    values: s.columns[s.columns.length - 1]!.values,
    gridSize: s.gridSize,
    domain: s.domain,
  };
}

/** Every typed-array buffer in a snapshot, for a zero-copy `postMessage` transfer list. */
export function snapshotBuffers(s: Snapshot): ArrayBuffer[] {
  const arrays: Float32Array[] = [
    s.losses.train.mean,
    s.losses.train.min,
    s.losses.train.max,
    s.losses.test.mean,
    s.losses.test.min,
    s.losses.test.max,
    s.timeline.epochs,
    s.paramHistory.epochs,
  ];
  const bytes: Uint8Array[] = [];
  for (const c of s.columns) {
    arrays.push(c.values);
    if (c.dead) bytes.push(c.dead);
  }
  for (const w of s.weights) arrays.push(w.W, w.b, w.gradW, w.gradB);
  for (const l of s.timeline.layers) arrays.push(l.weightHist, l.gradHist, l.weightRms, l.gradRms);
  for (const l of s.paramHistory.layers) arrays.push(l.W, l.b);
  if (s.probe) {
    arrays.push(s.probe.x, ...s.probe.dW);
    for (const c of s.probe.columns) arrays.push(c.z, c.a, c.dZ, c.dA);
  }
  return [...new Set([...arrays, ...bytes].map((a) => a.buffer as ArrayBuffer))];
}
