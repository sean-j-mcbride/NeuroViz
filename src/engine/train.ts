import type { Layer } from './layers/types';
import type { Loss } from './losses';
import type { Optimiser } from './optim';
import type { Tensor } from './tensor';

export interface TrainerOptions {
  model: Layer;
  loss: Loss;
  optimiser: Optimiser;
}

export class Trainer {
  readonly model: Layer;
  readonly loss: Loss;
  readonly optimiser: Optimiser;

  constructor({ model, loss, optimiser }: TrainerOptions) {
    this.model = model;
    this.loss = loss;
    this.optimiser = optimiser;
  }

  /** One forward/backward/update on the batch (x, y). Returns the loss before the update. */
  trainStep(x: Tensor, y: Tensor): number {
    const pred = this.model.forward(x, true);
    const loss = this.loss.forward(pred, y);
    this.model.backward(this.loss.backward());
    this.optimiser.step(this.model.params());
    return loss;
  }
}
