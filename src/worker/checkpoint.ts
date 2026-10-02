import type { OptimiserState, ParamValues, RngState } from '../engine';

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
  /** Full-dataset losses after each epoch, as in `Snapshot` (length epoch + 1). */
  trainLoss: Float32Array;
  testLoss: Float32Array;
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
}

/** Every typed-array buffer in a checkpoint, for a zero-copy `postMessage` transfer list. */
export function checkpointBuffers(c: Checkpoint): ArrayBuffer[] {
  const arrays: (Float32Array | Uint32Array)[] = [
    c.trainLoss,
    c.testLoss,
    c.order,
    ...Object.values(c.params),
    ...Object.values(c.optimiser.slots).flat(),
  ];
  return [...new Set(arrays.map((a) => a.buffer as ArrayBuffer))];
}
