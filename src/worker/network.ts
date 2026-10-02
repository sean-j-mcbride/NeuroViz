import type { InitKind, LayerConfig } from '../engine';

export type Activation = 'tanh' | 'relu' | 'sigmoid' | 'linear';

export const ACTIVATIONS: readonly Activation[] = ['tanh', 'relu', 'sigmoid', 'linear'];

export interface HiddenLayerSpec {
  units: number;
  activation: Activation;
}

/**
 * Compact, serialisable description of a 2D-playground MLP: 2 inputs, the
 * hidden layers, then a single linear output unit (a logit for BCE).
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
 * dense(→1). The dropout layers are always present (rate 0 is the identity) so
 * the rate can change mid-run without rebuilding the model.
 */
export function networkToLayerConfig(spec: NetworkSpec, dropout = 0): LayerConfig {
  const layers: LayerConfig[] = [];
  let inFeatures = INPUTS;
  for (const { units, activation } of spec.hidden) {
    layers.push({ kind: 'dense', inFeatures, outFeatures: units, init: initFor(activation) });
    if (activation !== 'linear') layers.push({ kind: activation });
    layers.push({ kind: 'dropout', rate: dropout });
    inFeatures = units;
  }
  layers.push({ kind: 'dense', inFeatures, outFeatures: 1, init: 'xavier' });
  return { kind: 'sequential', layers };
}
