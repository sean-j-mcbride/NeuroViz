import { type SessionConfig, TrainingSession } from '../worker';

/** A small seeded session (2-3-1 tanh on the circle, 4×4 grid), trained `epochs` epochs. */
export function smallSession(epochs = 5, overrides: Partial<SessionConfig> = {}): TrainingSession {
  const s = new TrainingSession({
    dataset: { kind: 'circle', n: 100, noise: 0, seed: 1 },
    network: { hidden: [{ units: 3, activation: 'tanh' }] },
    lr: 0.03,
    batchSize: 10,
    optimiser: 'sgd',
    l2: 0,
    dropout: 0,
    seed: 1,
    gridSize: 4,
    ...overrides,
  });
  for (let e = 0; e < epochs; e++) s.trainEpoch();
  return s;
}
