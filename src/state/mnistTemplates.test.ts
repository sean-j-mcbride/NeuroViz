import { describe, expect, it } from 'vitest';
import { loadMnistSubset } from '../test/mnist';
import { MnistSession } from '../worker';
import { MNIST_DEFAULT_CONFIG, MNIST_TEMPLATES, toMnistSessionConfig } from './mnistConfig';

const data = loadMnistSubset();

describe('CNN done-when: ≥ 98 % test accuracy in about a minute', () => {
  /*
   * The Small CNN template (conv 8 × 5×5 → pool → conv 16 × 5×5 → pool → 10,
   * ReLU; Adam lr 0.003, batch 32). Seeded sweep, records every 5,000 images:
   * seeds 1–5 first reach ≥ 98 % on the 2,000 test images after 5.5, 3, 4, 3
   * and 8.5 epochs, then hover just below and above it (about 97–98.6 % up to
   * epoch 10). The MLP default plateaus near 96 %. With lr 0.001 the CNN was
   * still at 96.6–97.3 % after 6 epochs; batch 64 at lr 0.003 took 4–7.5
   * epochs; a wider 16/32-filter net took 2.3× as long per epoch for no faster
   * climb. About 5 s an epoch in Node, 6 s in the browser (seed 1: 36 s).
   */
  it('the Small CNN template, seed 1: ≥ 98 % within 6 epochs', () => {
    const { network, training } = MNIST_TEMPLATES.cnn;
    const s = new MnistSession(
      toMnistSessionConfig(
        {
          ...MNIST_DEFAULT_CONFIG,
          network,
          training: { ...MNIST_DEFAULT_CONFIG.training, ...training },
        },
        data,
      ),
    );
    // The record that covers 6 epochs comes after the batch that crosses 60,000 images.
    const limit = 60_000 + Number(training.batchSize);
    let first98 = Infinity;
    let records = s.snapshot().losses.count;
    while (s.examples < limit && first98 === Infinity) {
      s.advance();
      if (s.snapshot().losses.count === records) continue;
      records++;
      if (s.snapshot().testAccuracy >= 0.98) first98 = s.examples;
    }
    expect(first98).toBeLessThanOrEqual(limit);
  }, 120_000);
});
