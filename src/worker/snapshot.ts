import type { Activation } from './network';

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
}

export interface DenseWeights {
  inFeatures: number;
  outFeatures: number;
  /** Row-major `[inFeatures, outFeatures]`: weight from input i to unit j is `W[i·out + j]`. */
  W: Float32Array;
  b: Float32Array;
}

export interface PointSet {
  /** `[N, 2]` row-major. */
  x: Float32Array;
  /** `[N]`, each 0 or 1. */
  y: Float32Array;
}

/**
 * Everything the UI needs to draw one moment of training. Plain data and typed
 * arrays only, so it is structured-cloneable (and can cross a Worker boundary).
 * Every array is a copy: the UI may keep it after the session moves on.
 */
export interface Snapshot {
  epoch: number;
  /** Total number of optimiser steps (mini-batches) so far. */
  step: number;
  /** Full-dataset losses after each epoch; index e is the loss after epoch e (0 = before training). */
  trainLoss: Float32Array;
  testLoss: Float32Array;
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
  train: PointSet;
  test: PointSet;
}
