import { type Shape, type Tensor, reshape } from '../tensor';
import type { Layer, LayerConfig, Param } from './types';

/**
 * `[N, ...]` → `[N, product of the rest]`, e.g. feature maps `[N, C, H, W]`
 * into rows of C·H·W values (channel by channel, each row by row) for a dense
 * layer. Both directions are views: nothing is copied.
 */
export class Flatten implements Layer {
  readonly kind = 'flatten';
  private inShape: Shape | null = null;

  forward(x: Tensor, _train: boolean): Tensor {
    this.inShape = x.shape;
    const n = x.shape[0] ?? 0;
    return reshape(x, [n, n === 0 ? 0 : x.size / n]);
  }

  backward(gradOut: Tensor): Tensor {
    if (!this.inShape) throw new Error('Flatten: backward called before forward');
    return reshape(gradOut, this.inShape);
  }

  params(): Param[] {
    return [];
  }

  toJSON(): LayerConfig {
    return { kind: 'flatten' };
  }
}
