import type { Rng } from './random';
import { Tensor } from './tensor';

/** Glorot/Xavier normal: std = √(2 / (fanIn + fanOut)). Suits tanh/sigmoid. */
export function xavierNormal(fanIn: number, fanOut: number, rng: Rng): Tensor {
  return Tensor.randn([fanIn, fanOut], rng, Math.sqrt(2 / (fanIn + fanOut)));
}

/** He/Kaiming normal: std = √(2 / fanIn). Suits ReLU. */
export function heNormal(fanIn: number, fanOut: number, rng: Rng): Tensor {
  return Tensor.randn([fanIn, fanOut], rng, Math.sqrt(2 / fanIn));
}
