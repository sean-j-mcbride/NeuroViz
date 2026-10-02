import type { Tensor } from './tensor';
import type { Layer, LayerConfig, Param } from './layers/types';

/** Layers applied in order. Itself a `Layer`, so models can nest. */
export class Sequential implements Layer {
  readonly kind = 'sequential';

  constructor(readonly layers: readonly Layer[]) {}

  forward(x: Tensor, train: boolean): Tensor {
    let h = x;
    for (const layer of this.layers) h = layer.forward(h, train);
    return h;
  }

  backward(gradOut: Tensor): Tensor {
    let g = gradOut;
    for (let i = this.layers.length - 1; i >= 0; i--) g = this.layers[i]!.backward(g);
    return g;
  }

  /** Params named by layer index, e.g. "0.W", "2.b". */
  params(): Param[] {
    return this.layers.flatMap((layer, i) =>
      layer.params().map((p) => ({ ...p, name: `${i}.${p.name}` })),
    );
  }

  toJSON(): LayerConfig {
    return { kind: 'sequential', layers: this.layers.map((l) => l.toJSON()) };
  }
}
