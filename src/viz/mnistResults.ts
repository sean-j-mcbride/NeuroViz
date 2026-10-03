import { MNIST_CLASSES, MNIST_PIXELS } from '../data';
import type { FirstLayerWeights } from '../worker';

/** `[10 × 10]` counts, row = true label, column = predicted class. */
export function confusionCounts(labels: Uint8Array, predicted: Uint8Array): Uint32Array {
  const counts = new Uint32Array(MNIST_CLASSES * MNIST_CLASSES);
  for (let i = 0; i < labels.length; i++) counts[labels[i]! * MNIST_CLASSES + predicted[i]!]!++;
  return counts;
}

/**
 * Indices of the misclassified images, most confident first: all of them, or
 * those with true `label` predicted as `predicted`. Ties keep index order.
 */
export function confidentMistakes(
  labels: Uint8Array,
  predicted: Uint8Array,
  confidence: Float32Array,
  filter?: { label: number; predicted: number } | null,
): number[] {
  const out: number[] = [];
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] === predicted[i]) continue;
    if (filter && (labels[i] !== filter.label || predicted[i] !== filter.predicted)) continue;
    out.push(i);
  }
  return out.sort((a, b) => confidence[b]! - confidence[a]! || a - b);
}

/** Per class: correct / total (recall), NaN for a class with no images. */
export function perClassAccuracy(counts: Uint32Array): number[] {
  return Array.from({ length: MNIST_CLASSES }, (_, r) => {
    let total = 0;
    for (let c = 0; c < MNIST_CLASSES; c++) total += counts[r * MNIST_CLASSES + c]!;
    return total === 0 ? Number.NaN : counts[r * MNIST_CLASSES + r]! / total;
  });
}

/** "2.5" epochs from a record index, for chart axes and tooltips. */
export function recordEpoch(index: number, recordEvery: number, trainSize: number): string {
  const e = (index * recordEvery) / trainSize;
  return Number.isInteger(e)
    ? e.toLocaleString('en-GB')
    : e.toLocaleString('en-GB', { maximumFractionDigits: 2 });
}

export interface UnitStats {
  min: number;
  max: number;
  rms: number;
  bias: number;
}

/** Each unit's weight range, RMS and bias. W is `[784, units]`. */
export function unitStats({ W, b, outFeatures: units }: FirstLayerWeights): UnitStats[] {
  return Array.from({ length: units }, (_, j) => {
    let min = Infinity;
    let max = -Infinity;
    let sq = 0;
    for (let p = 0; p < MNIST_PIXELS; p++) {
      const w = W[p * units + j]!;
      min = Math.min(min, w);
      max = Math.max(max, w);
      sq += w * w;
    }
    return { min, max, rms: Math.sqrt(sq / MNIST_PIXELS), bias: b[j]! };
  });
}

/** The `k` units with the largest activations, largest first. */
export function topUnits(hidden: Float32Array, k: number): number[] {
  return Array.from(hidden.keys())
    .filter((j) => hidden[j]! > 0)
    .sort((a, b) => hidden[b]! - hidden[a]! || a - b)
    .slice(0, k);
}
