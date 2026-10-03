import { type Tensor, TensorBuffer } from './tensor';

/** A scalar loss averaged over the batch. `backward` returns dLoss/dPred for the last `forward`. */
export interface Loss {
  readonly kind: string;
  forward(pred: Tensor, target: Tensor): number;
  backward(): Tensor;
}

function assertSameShape(kind: string, pred: Tensor, target: Tensor): void {
  if (
    pred.shape.length !== target.shape.length ||
    pred.shape.some((d, i) => d !== target.shape[i])
  ) {
    throw new Error(
      `${kind}: pred [${pred.shape.join(', ')}] and target [${target.shape.join(', ')}] differ`,
    );
  }
}

/** Mean squared error over every element: mean((p − y)²). */
export class MSELoss implements Loss {
  readonly kind = 'mse';
  private grad: Tensor | null = null;
  private readonly gradBuf = new TensorBuffer();

  forward(pred: Tensor, target: Tensor): number {
    assertSameShape(this.kind, pred, target);
    this.grad = this.gradBuf.take(pred.shape);
    const n = pred.size;
    const g = this.grad.data;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const d = pred.data[i]! - target.data[i]!;
      sum += d * d;
      g[i] = (2 * d) / n;
    }
    return sum / n;
  }

  backward(): Tensor {
    if (!this.grad) throw new Error('mse: backward called before forward');
    return this.grad;
  }
}

/**
 * Softmax followed by cross-entropy, fused for numerical stability.
 * `pred` holds logits `[N, C]`; `target` holds one-hot (or soft) labels `[N, C]`.
 * Loss = mean over rows of −Σ_c y_c · log softmax(z)_c, computed via log-sum-exp
 * with the row max subtracted. The gradient is (softmax(z) − y) / N.
 */
export class SoftmaxCrossEntropyLoss implements Loss {
  readonly kind = 'softmax-cross-entropy';
  private grad: Tensor | null = null;
  private readonly gradBuf = new TensorBuffer();

  forward(pred: Tensor, target: Tensor): number {
    assertSameShape(this.kind, pred, target);
    const [n, c] = [pred.rows, pred.cols];
    this.grad = this.gradBuf.take(pred.shape);
    const z = pred.data;
    const y = target.data;
    const g = this.grad.data;
    let total = 0;
    for (let r = 0; r < n; r++) {
      const o = r * c;
      let max = -Infinity;
      for (let j = 0; j < c; j++) max = Math.max(max, z[o + j]!);
      let sumExp = 0;
      for (let j = 0; j < c; j++) sumExp += Math.exp(z[o + j]! - max);
      const lse = max + Math.log(sumExp);
      for (let j = 0; j < c; j++) {
        const yj = y[o + j]!;
        total += yj * (lse - z[o + j]!);
        g[o + j] = (Math.exp(z[o + j]! - lse) - yj) / n;
      }
    }
    return total / n;
  }

  backward(): Tensor {
    if (!this.grad) throw new Error('softmax-cross-entropy: backward called before forward');
    return this.grad;
  }
}

/**
 * Sigmoid followed by binary cross-entropy, fused for numerical stability.
 * `pred` holds logits z; `target` holds labels y ∈ [0, 1] of the same shape.
 * Per element: max(z, 0) − z·y + log(1 + e^{−|z|}), averaged over every element.
 * The gradient is (σ(z) − y) / N.
 */
export class BCEWithLogitsLoss implements Loss {
  readonly kind = 'bce-with-logits';
  private grad: Tensor | null = null;
  private readonly gradBuf = new TensorBuffer();

  forward(pred: Tensor, target: Tensor): number {
    assertSameShape(this.kind, pred, target);
    this.grad = this.gradBuf.take(pred.shape);
    const n = pred.size;
    const z = pred.data;
    const y = target.data;
    const g = this.grad.data;
    let total = 0;
    for (let i = 0; i < n; i++) {
      const zi = z[i]!;
      const yi = y[i]!;
      const e = Math.exp(-Math.abs(zi));
      total += Math.max(zi, 0) - zi * yi + Math.log1p(e);
      // σ(z) computed from e = e^{−|z|} so it never overflows.
      const sig = zi >= 0 ? 1 / (1 + e) : e / (1 + e);
      g[i] = (sig - yi) / n;
    }
    return total / n;
  }

  backward(): Tensor {
    if (!this.grad) throw new Error('bce-with-logits: backward called before forward');
    return this.grad;
  }
}
