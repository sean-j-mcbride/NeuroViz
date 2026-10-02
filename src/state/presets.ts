import type { HiddenLayerSpec } from '../worker';
import { DEFAULT_CONFIG, type PlaygroundConfig, sameConfig } from './config';

export type PresetId = 'underfitting' | 'overfitting' | 'dead-relus' | 'too-high-lr';

/** A classic failure mode, set up to show itself, plus the change that cures it. */
export interface Preset {
  id: PresetId;
  name: string;
  /** What is set up, and why it goes wrong. */
  summary: string;
  /** What to watch for while it trains. */
  lookFor: string;
  /** The cure, in words; `fix` is the same change as a config. */
  tryThis: string;
  config: PlaygroundConfig;
  fix: PlaygroundConfig;
}

const layers = (n: number, units: number, activation: HiddenLayerSpec['activation']) =>
  Array.from({ length: n }, () => ({ units, activation }));

function preset(p: Omit<Preset, 'fix'>, fix: (c: PlaygroundConfig) => PlaygroundConfig): Preset {
  return { ...p, fix: fix(p.config) };
}

/*
 * Each preset was picked from a seeded sweep (seeds 1–5 all show the effect and
 * respond to the fix; ranges are quoted in presets.test.ts, which checks seed 1).
 */
export const PRESETS: readonly Preset[] = [
  preset(
    {
      id: 'underfitting',
      name: 'Underfitting',
      summary:
        'One hidden neuron can only draw a straight line, and no straight line separates a circle.',
      lookFor:
        'Train and test loss level off together, well above zero, at about 65 % accuracy. ' +
        'The model isn’t memorising anything; it simply can’t express the answer.',
      tryThis:
        'Give the hidden layer three neurons: the boundary closes into a ring and accuracy ' +
        'passes 99 %.',
      config: {
        ...DEFAULT_CONFIG,
        dataset: { kind: 'circle', n: 400, noise: 0, seed: 1 },
        network: { hidden: layers(1, 1, 'tanh') },
      },
    },
    (c) => ({ ...c, network: { hidden: layers(1, 3, 'tanh') } }),
  ),
  preset(
    {
      id: 'overfitting',
      name: 'Overfitting',
      summary:
        'A six-layer network trained on 105 noisy points has room to learn the noise as well ' +
        'as the pattern.',
      lookFor:
        'Training loss keeps falling towards zero, but test loss bottoms out within a few dozen ' +
        'epochs and then climbs. The boundary grows islands around individual points.',
      tryThis:
        'Set L2 regularisation to 0.03: training loss stops a little higher, but test loss stays ' +
        'low and test accuracy improves.',
      config: {
        dataset: { kind: 'circle', n: 150, noise: 0.4, seed: 1 },
        network: { hidden: layers(6, 8, 'relu') },
        training: { lr: 0.01, batchSize: 10, optimiser: 'adam', l2: 0, dropout: 0 },
        seed: 1,
      },
    },
    (c) => ({ ...c, training: { ...c.training, l2: 0.03 } }),
  ),
  preset(
    {
      id: 'dead-relus',
      name: 'Dead ReLUs',
      summary:
        'Adam with a learning rate of 0.1 takes steps so large that many ReLU neurons are pushed ' +
        'to output 0 for every input. Their gradient is then 0 too, so they never recover.',
      lookFor:
        'Blank neuron tiles in the network, and gradients piling up in the bottom bin of the ' +
        'histograms under Inside training. Accuracy stays close to chance.',
      tryThis:
        'Lower the learning rate to 0.01: almost every neuron stays alive and the spirals ' +
        'untangle within 200 epochs.',
      config: {
        ...DEFAULT_CONFIG,
        network: { hidden: layers(4, 8, 'relu') },
        training: { ...DEFAULT_CONFIG.training, optimiser: 'adam', lr: 0.1 },
      },
    },
    (c) => ({ ...c, training: { ...c.training, lr: 0.01 } }),
  ),
  preset(
    {
      id: 'too-high-lr',
      name: 'Too high LR',
      summary:
        'The default spirals set-up with a learning rate of 3, so every step overshoots the ' +
        'minimum it is aiming for.',
      lookFor:
        'A jagged loss curve that never settles and often ends higher than it started. The ' +
        'boundary flickers from one frame to the next.',
      tryThis: 'Set the learning rate back to 0.03: the loss falls smoothly.',
      config: { ...DEFAULT_CONFIG, training: { ...DEFAULT_CONFIG.training, lr: 3 } },
    },
    (c) => ({ ...c, training: { ...c.training, lr: 0.03 } }),
  ),
];

export function findPreset(id: string): Preset | undefined {
  return PRESETS.find((p) => p.id === id);
}

/** The preset whose set-up (or fix) `config` is exactly, if any. */
export function presetFor(config: PlaygroundConfig): Preset | undefined {
  return PRESETS.find((p) => sameConfig(config, p.config) || sameConfig(config, p.fix));
}
