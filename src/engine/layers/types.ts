import type { Tensor } from '../tensor';

export type InitKind = 'xavier' | 'he';

/** 'valid': no padding. 'same': pad (k − 1) / 2 zeros, so stride 1 keeps the size (odd k). */
export type Padding = 'valid' | 'same';

/** Serialisable description of a layer (architecture only, no weights). */
export type LayerConfig =
  | { kind: 'dense'; inFeatures: number; outFeatures: number; init: InitKind }
  | { kind: 'relu' }
  | { kind: 'tanh' }
  | { kind: 'sigmoid' }
  | { kind: 'dropout'; rate: number }
  | {
      kind: 'conv2d';
      inChannels: number;
      outChannels: number;
      kernel: number;
      stride: number;
      padding: Padding;
      init: InitKind;
    }
  | { kind: 'maxpool2d'; size: number; stride: number }
  | { kind: 'flatten' }
  | { kind: 'sequential'; layers: LayerConfig[] };

export interface Param {
  name: string;
  value: Tensor;
  grad: Tensor;
}

/**
 * A differentiable layer. `forward` caches what `backward` needs.
 *
 * Tensors returned by `forward`/`backward` are buffers owned by the layer and
 * are overwritten by the next call — clone them to keep a copy. `backward`
 * overwrites (does not accumulate) parameter gradients.
 */
export interface Layer {
  readonly kind: string;
  forward(x: Tensor, train: boolean): Tensor;
  backward(gradOut: Tensor): Tensor;
  params(): Param[];
  toJSON(): LayerConfig;
}
