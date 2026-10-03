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

/**
 * How the Inside-training charts name layers and label the x-axis. The
 * playground's columns are epochs; MNIST's are records (shown as epochs).
 */
export interface TimelineAxis {
  layerName(k: number, layerCount: number): string;
  /** "epoch 12" for a column's x value, as a phrase. */
  at(x: number): string;
  /** The x column's heading in tables, and one value in it. */
  column: string;
  cell(x: number): string;
}

export const EPOCH_AXIS: TimelineAxis = {
  layerName: layerLabel,
  at: (x) => `epoch ${x}`,
  column: 'Epoch',
  cell: (x) => String(x),
};

/** "epoch 12" → "Epoch 12". */
export const capitalise = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
