import type { Layer } from './layers/types';
import type { Loss } from './losses';
import type { Optimiser } from './optim';
import { addL2 } from './regularise';
import type { Tensor } from './tensor';

export interface TrainerOptions {
  model: Layer;
  loss: Loss;
  optimiser: Optimiser;
  /** L2 regularisation strength λ (weights only). Default 0. */
  l2?: number;
}

export class Trainer {
  readonly model: Layer;
  readonly loss: Loss;
  /** May be replaced between steps (e.g. switching optimiser mid-run). */
  optimiser: Optimiser;
  /** May be changed between steps. */
  l2: number;

  constructor({ model, loss, optimiser, l2 = 0 }: TrainerOptions) {
    this.model = model;
    this.loss = loss;
    this.optimiser = optimiser;
    this.l2 = l2;
  }

  /**
   * One forward/backward/update on the batch (x, y). Returns the data loss
   * before the update (excluding the L2 penalty).
   */
  trainStep(x: Tensor, y: Tensor): number {
    const pred = this.model.forward(x, true);
    const loss = this.loss.forward(pred, y);
    this.model.backward(this.loss.backward());
    const params = this.model.params();
    addL2(params, this.l2);
    this.optimiser.step(params);
    return loss;
  }
}
