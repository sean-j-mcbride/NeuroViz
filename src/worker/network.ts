import { type InitKind, type LayerConfig, type Padding, type Shape, inferShapes } from '../engine';
import { MNIST_CLASSES, MNIST_PIXELS, MNIST_SIDE } from '../data';

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

/** One row of a CNN's feature extractor: a convolution (with its activation) or a 2×2-style max pool. */
export type ConvLayerSpec =
  | {
      kind: 'conv';
      filters: number;
      kernel: number;
      stride: number;
      padding: Padding;
      activation: Activation;
    }
  | { kind: 'pool'; size: number };

/**
 * An MNIST network: optional conv/pool rows over the 1 × 28 × 28 image, then
 * (flattened) the dense hidden layers, then 10 logits. No conv rows is an MLP.
 */
export interface MnistNetworkSpec extends NetworkSpec {
  conv: ConvLayerSpec[];
}

/** A deep copy (the spec is plain data; the worker has no `structuredClone` type). */
export function copyMnistNetwork({ conv, hidden }: MnistNetworkSpec): MnistNetworkSpec {
  return { conv: conv.map((r) => ({ ...r })), hidden: hidden.map((l) => ({ ...l })) };
}

/** The image as a CNN sees it: channels × height × width. */
export const MNIST_IMAGE_SHAPE: Shape = [1, MNIST_SIDE, MNIST_SIDE];

/** Engine layers for the conv rows, and for each row the index of its last layer. */
function convLayers(conv: readonly ConvLayerSpec[]): { layers: LayerConfig[]; rowEnds: number[] } {
  const layers: LayerConfig[] = [];
  const rowEnds: number[] = [];
  let channels = MNIST_IMAGE_SHAPE[0]!;
  for (const row of conv) {
    if (row.kind === 'conv') {
      const { filters, kernel, stride, padding, activation } = row;
      layers.push({
        kind: 'conv2d',
        inChannels: channels,
        outChannels: filters,
        kernel,
        stride,
        padding,
        init: initFor(activation),
      });
      if (activation !== 'linear') layers.push({ kind: activation });
      channels = filters;
    } else {
      layers.push({ kind: 'maxpool2d', size: row.size, stride: row.size });
    }
    rowEnds.push(layers.length - 1);
  }
  return { layers, rowEnds };
}

export type ConvStackCheck =
  | { ok: true; shapes: Shape[]; flat: number }
  | { ok: false; shapes: Shape[]; row: number; message: string };

/**
 * The output shape of every conv row and the flattened size, or the first
 * row whose input is too small for it (with the shapes before it).
 */
export function checkConvStack(conv: readonly ConvLayerSpec[]): ConvStackCheck {
  const { layers, rowEnds } = convLayers(conv);
  const r = inferShapes(layers, MNIST_IMAGE_SHAPE);
  const rowOf = (index: number) => rowEnds.findIndex((end) => end >= index);
  const shapes = rowEnds.filter((end) => end < r.shapes.length).map((end) => r.shapes[end]!);
  if ('error' in r)
    return { ok: false, shapes, row: rowOf(r.error.index), message: r.error.message };
  const last = shapes.at(-1) ?? MNIST_IMAGE_SHAPE;
  return { ok: true, shapes, flat: last.reduce((a, b) => a * b, 1) };
}

/**
 * Expands an MNIST spec to an engine model config. With no conv rows this is
 * exactly `networkToLayerConfig(spec, dropout, 784, 10)` (the Phase 5 MLP).
 * Otherwise: conv2d + activation per conv row and maxpool2d per pool row (no
 * dropout on feature maps), flatten, then the dense layers as for an MLP.
 * Throws if the conv rows don't fit the image.
 */
export function mnistNetworkToLayerConfig(spec: MnistNetworkSpec, dropout = 0): LayerConfig {
  if (spec.conv.length === 0)
    return networkToLayerConfig(spec, dropout, MNIST_PIXELS, MNIST_CLASSES);
  const check = checkConvStack(spec.conv);
  if (!check.ok) throw new Error(`Conv layer ${check.row + 1}: ${check.message}`);
  const dense = networkToLayerConfig(spec, dropout, check.flat, MNIST_CLASSES);
  if (dense.kind !== 'sequential') throw new Error('expected a sequential config');
  return {
    kind: 'sequential',
    layers: [...convLayers(spec.conv).layers, { kind: 'flatten' }, ...dense.layers],
  };
}

/** For each conv row, the index of its last layer in the model `mnistNetworkToLayerConfig` builds. */
export function convRowEnds(conv: readonly ConvLayerSpec[]): number[] {
  return convLayers(conv).rowEnds;
}
