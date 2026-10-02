import type { NeuronColumn } from '../worker';

export function columnLabel(c: Pick<NeuronColumn, 'kind'>, i: number): string {
  return c.kind === 'input' ? 'Input' : c.kind === 'output' ? 'Output' : `Hidden ${i}`;
}

export function neuronLabel(c: Pick<NeuronColumn, 'kind'>, column: number, unit: number): string {
  if (c.kind === 'input') return `x${unit + 1}`;
  if (c.kind === 'output') return 'output';
  return `h${column}.${unit + 1}`;
}

/** Short name for dense layer k, which connects column k to column k + 1. */
export function layerLabel(k: number, layerCount: number): string {
  const from = k === 0 ? 'Input' : `H${k}`;
  const to = k === layerCount - 1 ? 'Output' : `H${k + 1}`;
  return `${from} → ${to}`;
}
