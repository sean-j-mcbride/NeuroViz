import type { Layer } from './layers/types';
import type { Loss } from './losses';
import { Tensor } from './tensor';
import type { Rng } from './random';

export interface GradCheckOptions {
  /**
   * Central-difference step. 5e-3 balances truncation error (∝ eps²) against
   * float32 rounding noise (∝ 1/eps); measured optimum, ≈ (3·ε_f32)^(1/3).
   */
  eps?: number;
  /** Max allowed normwise relative error. */
  tol?: number;
}

export interface GradCheckResult {
  name: string;
  /** ‖a − n‖₂ / (‖a‖₂ + ‖n‖₂) over the whole tensor. */
  relError: number;
  /** Element with the largest absolute disagreement, for readable failures. */
  worst: { index: number; analytic: number; numeric: number };
  pass: boolean;
}

const DEFAULT_EPS = 5e-3;
const DEFAULT_TOL = 1e-4;

/**
 * Central-difference gradient of scalar `f` with respect to every element of
 * `target`, perturbing it in place and restoring it afterwards. Divides by the
 * perturbation actually stored in float32, not by the nominal 2·eps.
 */
export function numericalGrad(f: () => number, target: Tensor, eps = DEFAULT_EPS): Float64Array {
  const d = target.data;
  const grad = new Float64Array(d.length);
  for (let i = 0; i < d.length; i++) {
    const orig = d[i]!;
    d[i] = orig + eps;
    const xp = d[i]!;
    const fp = f();
    d[i] = orig - eps;
    const xm = d[i]!;
    const fm = f();
    d[i] = orig;
    grad[i] = (fp - fm) / (xp - xm);
  }
  return grad;
}

export function compareGrads(
  name: string,
  analytic: ArrayLike<number>,
  numeric: ArrayLike<number>,
  tol = DEFAULT_TOL,
): GradCheckResult {
  let diff2 = 0;
  let a2 = 0;
  let n2 = 0;
  const worst = { index: -1, analytic: 0, numeric: 0 };
  let worstAbs = -1;
  for (let i = 0; i < analytic.length; i++) {
    const a = analytic[i]!;
    const n = numeric[i]!;
    diff2 += (a - n) ** 2;
    a2 += a * a;
    n2 += n * n;
    if (Math.abs(a - n) > worstAbs) {
      worstAbs = Math.abs(a - n);
      Object.assign(worst, { index: i, analytic: a, numeric: n });
    }
  }
  const relError = Math.sqrt(diff2) / Math.max(Math.sqrt(a2) + Math.sqrt(n2), 1e-12);
  return { name, relError, worst, pass: relError < tol };
}

/** Float64 dot product, used to reduce a layer output to a scalar. */
function dot(a: Tensor, b: Tensor): number {
  let s = 0;
  for (let i = 0; i < a.size; i++) s += a.data[i]! * b.data[i]!;
  return s;
}

/**
 * Checks a layer's input and parameter gradients using the scalar objective
 * L = Σ forward(x) ⊙ R, with R a fixed random tensor (so dL/dy = R).
 */
export function checkLayer(
  layer: Layer,
  x: Tensor,
  rng: Rng,
  opts: GradCheckOptions = {},
): GradCheckResult[] {
  const { eps = DEFAULT_EPS, tol = DEFAULT_TOL } = opts;
  const R = Tensor.randn(layer.forward(x, true).shape, rng);
  const objective = () => dot(layer.forward(x, true), R);

  // Analytic grads first, copied out before numeric forwards overwrite the caches.
  layer.forward(x, true);
  const dx = layer.backward(R).data.slice();
  const params = layer.params().map((p) => ({ ...p, analytic: p.grad.data.slice() }));

  return [
    compareGrads('input', dx, numericalGrad(objective, x, eps), tol),
    ...params.map((p) =>
      compareGrads(p.name, p.analytic, numericalGrad(objective, p.value, eps), tol),
    ),
  ];
}

/** Checks dLoss/dPred. */
export function checkLoss(
  loss: Loss,
  pred: Tensor,
  target: Tensor,
  opts: GradCheckOptions = {},
): GradCheckResult[] {
  const { eps = DEFAULT_EPS, tol = DEFAULT_TOL } = opts;
  loss.forward(pred, target);
  const analytic = loss.backward().data.slice();
  const numeric = numericalGrad(() => loss.forward(pred, target), pred, eps);
  return [compareGrads('pred', analytic, numeric, tol)];
}

/** End-to-end check of a model + loss: input grad and every parameter. */
export function checkModel(
  model: Layer,
  loss: Loss,
  x: Tensor,
  y: Tensor,
  opts: GradCheckOptions = {},
): GradCheckResult[] {
  const { eps = DEFAULT_EPS, tol = DEFAULT_TOL } = opts;
  const objective = () => loss.forward(model.forward(x, true), y);

  objective();
  const dx = model.backward(loss.backward()).data.slice();
  const params = model.params().map((p) => ({ ...p, analytic: p.grad.data.slice() }));

  return [
    compareGrads('input', dx, numericalGrad(objective, x, eps), tol),
    ...params.map((p) =>
      compareGrads(p.name, p.analytic, numericalGrad(objective, p.value, eps), tol),
    ),
  ];
}

/** Throws a readable error listing every failing tensor. Framework-agnostic (works in Vitest). */
export function assertGradsOk(results: GradCheckResult[]): void {
  const failed = results.filter((r) => !r.pass);
  if (failed.length === 0) return;
  const lines = failed.map(
    (r) =>
      `  ${r.name}: relError=${r.relError.toExponential(2)}; worst [${r.worst.index}] ` +
      `analytic=${r.worst.analytic.toPrecision(6)} numeric=${r.worst.numeric.toPrecision(6)}`,
  );
  throw new Error(`Gradient check failed:\n${lines.join('\n')}`);
}
