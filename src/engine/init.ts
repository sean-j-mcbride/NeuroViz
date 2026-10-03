import type { Rng } from './random';
import { type Shape, Tensor } from './tensor';

/**
 * Glorot/Xavier normal: std = √(2 / (fanIn + fanOut)). Suits tanh/sigmoid.
 * The tensor is `[fanIn, fanOut]` unless `shape` says otherwise (a conv layer's
 * fans count every kernel tap, but its weights are `[inC·k·k, outC]`).
 */
export function xavierNormal(
  fanIn: number,
  fanOut: number,
  rng: Rng,
  shape: Shape = [fanIn, fanOut],
): Tensor {
  return Tensor.randn(shape, rng, Math.sqrt(2 / (fanIn + fanOut)));
}

/** He/Kaiming normal: std = √(2 / fanIn). Suits ReLU. Shape as for `xavierNormal`. */
export function heNormal(
  fanIn: number,
  fanOut: number,
  rng: Rng,
  shape: Shape = [fanIn, fanOut],
): Tensor {
  return Tensor.randn(shape, rng, Math.sqrt(2 / fanIn));
}
