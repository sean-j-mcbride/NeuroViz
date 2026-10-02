import type { OptimiserState, ParamValues, RngState } from '../engine';
import type { ParamHistoryState, TimelineState } from './history';
import type { LossHistoryState } from './lossHistory';
import type { HyperparamChange } from './session';

/**
 * Everything needed to continue a training session exactly where it stopped:
 * resuming from a checkpoint and training on is bitwise identical to never
 * having stopped. Taken between epochs. Plain data and typed arrays only, so
 * it is structured-cloneable. The architecture, data and hyperparameters are
 * not included; they come from the session config it is resumed with.
 */
export interface Checkpoint {
  epoch: number;
  /** Optimiser steps (mini-batches) so far. */
  step: number;
  /** Full-dataset losses per epoch, bounded (see `LossHistory`); covers epochs 0 … epoch. */
  losses: LossHistoryState;
  /** Parameter values by name ("0.W", "0.b", …). */
  params: ParamValues;
  optimiser: OptimiserState;
  rng: {
    /** Drives the per-epoch shuffle. */
    shuffle: RngState;
    /** Drew the initial weights; now drives the dropout masks. */
    dropout: RngState;
  };
  /**
   * The training-set order after the last shuffle. Each epoch shuffles the
   * previous order in place, so the next order depends on it.
   */
  order: Uint32Array;
  /** The settings used over the run so far: epoch 0's, then each change. */
  hyperparamLog: HyperparamChange[];
  /**
   * The charts' history (histogram timeline and hover sparklines), so they
   * continue after a resume. Display only; null when unknown (older files),
   * and they then restart at the checkpoint's epoch.
   */
  observations: { timeline: TimelineState; params: ParamHistoryState } | null;
}

/** Every typed-array buffer in a checkpoint, for a zero-copy `postMessage` transfer list. */
export function checkpointBuffers(c: Checkpoint): ArrayBuffer[] {
  const arrays: (Float32Array | Uint32Array)[] = [
    ...lossArrays(c.losses),
    c.order,
    ...Object.values(c.params),
    ...Object.values(c.optimiser.slots).flat(),
  ];
  if (c.observations) {
    const { timeline, params } = c.observations;
    arrays.push(timeline.epochs, params.epochs, params.ring);
    for (const l of timeline.layers) arrays.push(l.weightHist, l.gradHist, l.weightRms, l.gradRms);
  }
  return [...new Set(arrays.map((a) => a.buffer as ArrayBuffer))];
}

function lossArrays({ train, test }: LossHistoryState): Float32Array[] {
  return [train.mean, train.min, train.max, test.mean, test.min, test.max];
}
