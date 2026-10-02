import {
  type HyperparamChange,
  type LossHistorySnapshot,
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
  losses: LossHistorySnapshot;
  trainAccuracy: number;
  testAccuracy: number;
  boundary: OutputGrid;
  /** The settings used over the run: epoch 0's, then each mid-run change. */
  hyperparamLog: HyperparamChange[];
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
    losses: s.losses,
    trainAccuracy: s.trainAccuracy,
    testAccuracy: s.testAccuracy,
    boundary: outputGrid(s),
    hyperparamLog: s.hyperparamLog,
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
