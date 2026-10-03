import type { InitKind, LayerConfig } from '../engine';

export type Activation = 'tanh' | 'relu' | 'sigmoid' | 'linear';

export const ACTIVATIONS: readonly Activation[] = ['tanh', 'relu', 'sigmoid', 'linear'];

export interface HiddenLayerSpec {
  units: number;
  activation: Activation;
}

/**
 * Compact, serialisable description of an MLP's hidden layers. The inputs and
 * outputs come from the task: the 2D playground has 2 inputs and a single
 * linear output unit (a logit for BCE); MNIST has 784 inputs and 10 logits.
 */
export interface NetworkSpec {
  hidden: HiddenLayerSpec[];
}

export const INPUTS = 2;

function initFor(activation: Activation): InitKind {
  return activation === 'relu' ? 'he' : 'xavier';
}

/**
 * Expands a spec to an engine model config: [dense, activation?, dropout]* then
 * a linear dense(→outputs). The dropout layers are always present (rate 0 is
 * the identity) so the rate can change mid-run without rebuilding the model.
 */
export function networkToLayerConfig(
  spec: NetworkSpec,
  dropout = 0,
  inputs = INPUTS,
  outputs = 1,
): LayerConfig {
  const layers: LayerConfig[] = [];
  let inFeatures = inputs;
  for (const { units, activation } of spec.hidden) {
    layers.push({ kind: 'dense', inFeatures, outFeatures: units, init: initFor(activation) });
    if (activation !== 'linear') layers.push({ kind: activation });
    layers.push({ kind: 'dropout', rate: dropout });
    inFeatures = units;
  }
  layers.push({ kind: 'dense', inFeatures, outFeatures: outputs, init: 'xavier' });
  return { kind: 'sequential', layers };
}
