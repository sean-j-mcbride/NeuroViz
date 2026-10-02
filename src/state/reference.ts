import {
  type OutputGrid,
  type SessionData,
  type Snapshot,
  TrainingSession,
  outputGrid,
} from '../worker';
import { type PlaygroundConfig, toSessionConfig } from './config';
import type { ModelFile } from './modelFile';

/** A frozen run to compare the live one against. Snapshot arrays are never mutated, so they are kept as-is. */
export interface ReferenceRun {
  label: string;
  config: PlaygroundConfig;
  epoch: number;
  trainLoss: Float32Array;
  testLoss: Float32Array;
  trainAccuracy: number;
  testAccuracy: number;
  boundary: OutputGrid;
  /** The run's data points; null if they never arrived. */
  data: SessionData | null;
}

export function referenceFromSnapshot(
  label: string,
  config: PlaygroundConfig,
  s: Snapshot,
  data: SessionData | null,
): ReferenceRun {
  return {
    label,
    config,
    epoch: s.epoch,
    trainLoss: s.trainLoss,
    testLoss: s.testLoss,
    trainAccuracy: s.trainAccuracy,
    testAccuracy: s.testAccuracy,
    boundary: outputGrid(s),
    data,
  };
}

/**
 * A saved model as a reference. Rebuilds its session on the calling thread
 * just long enough to take one snapshot (milliseconds at playground scale).
 */
export function referenceFromFile(label: string, { config, checkpoint }: ModelFile): ReferenceRun {
  const session = new TrainingSession(toSessionConfig(config, checkpoint));
  return referenceFromSnapshot(label, config, session.snapshot(), session.data());
}
