import type { Rng } from './random';

export type Shape = readonly number[];

function shapeSize(shape: Shape): number {
  let n = 1;
  for (const d of shape) {
    if (!Number.isInteger(d) || d < 0)
      throw new Error(`Invalid dimension ${d} in shape [${shape.join(', ')}]`);
    n *= d;
  }
  return n;
}

function fmt(shape: Shape): string {
  return `[${shape.join(', ')}]`;
}

/** Dense row-major tensor backed by a Float32Array. */
export class Tensor {
  readonly data: Float32Array;
  readonly shape: Shape;
  readonly size: number;

  constructor(data: Float32Array, shape: Shape) {
    const size = shapeSize(shape);
    if (size !== data.length) {
      throw new Error(`Shape ${fmt(shape)} needs ${size} elements, got ${data.length}`);
    }
    this.data = data;
    this.shape = [...shape];
    this.size = size;
  }

  static zeros(shape: Shape): Tensor {
    return new Tensor(new Float32Array(shapeSize(shape)), shape);
  }

  /** Builds a rank-1 tensor from `number[]` or a rank-2 tensor from `number[][]`. */
  static from(values: number[] | number[][]): Tensor {
    if (values.length === 0 || typeof values[0] === 'number') {
      return new Tensor(Float32Array.from(values as number[]), [values.length]);
    }
    const rows = values as number[][];
    const cols = rows[0]!.length;
    const data = new Float32Array(rows.length * cols);
    rows.forEach((row, r) => {
      if (row.length !== cols)
        throw new Error(`Ragged input: row ${r} has ${row.length} values, expected ${cols}`);
      data.set(row, r * cols);
    });
    return new Tensor(data, [rows.length, cols]);
  }

  static randn(shape: Shape, rng: Rng, std = 1): Tensor {
    const t = Tensor.zeros(shape);
    for (let i = 0; i < t.size; i++) t.data[i] = rng.normal(0, std);
    return t;
  }

  get rows(): number {
    this.assertRank2('rows');
    return this.shape[0]!;
  }

  get cols(): number {
    this.assertRank2('cols');
    return this.shape[1]!;
  }

  clone(): Tensor {
    return new Tensor(this.data.slice(), this.shape);
  }

  toArray(): number[] | number[][] {
    if (this.shape.length === 2) {
      const [r, c] = this.shape as [number, number];
      return Array.from({ length: r }, (_, i) =>
        Array.from(this.data.subarray(i * c, (i + 1) * c)),
      );
    }
    return Array.from(this.data);
  }

  private assertRank2(what: string): void {
    if (this.shape.length !== 2)
      throw new Error(`${what} needs a rank-2 tensor, got ${fmt(this.shape)}`);
  }
}

/**
 * Backing store for a result whose shape changes now and then (layer outputs
 * when the batch size changes). It grows to the largest size asked for and
 * hands out views, so alternating between training batches and evaluation
 * chunks does not allocate new arrays. A view is only valid until the next
 * `take`.
 */
export class TensorBuffer {
  private data = new Float32Array(0);
  private view: Tensor | null = null;

  take(shape: Shape): Tensor {
    if (this.view && sameShape(this.view.shape, shape)) return this.view;
    const size = shapeSize(shape);
    if (this.data.length < size) this.data = new Float32Array(size);
    this.view = new Tensor(this.data.subarray(0, size), shape);
    return this.view;
  }
}

function sameShape(a: Shape, b: Shape): boolean {
  return a.length === b.length && a.every((d, i) => d === b[i]);
}

/** Returns `out` after checking its shape, or a fresh zero tensor of `shape`. */
function target(shape: Shape, out: Tensor | undefined, op: string): Tensor {
  if (!out) return Tensor.zeros(shape);
  if (!sameShape(out.shape, shape)) {
    throw new Error(`${op}: out has shape ${fmt(out.shape)}, expected ${fmt(shape)}`);
  }
  return out;
}

export interface MatmulOptions {
  transA?: boolean;
  transB?: boolean;
}

/** Float64 accumulators reused across `matmul` calls (grown, never shrunk). */
let scratch = new Float64Array(0);

function accumulators(n: number): Float64Array {
  if (scratch.length < n) scratch = new Float64Array(n);
  return scratch;
}

/** Whether two arrays share any memory. */
function overlaps(x: Float32Array, y: Float32Array): boolean {
  return (
    x.buffer === y.buffer &&
    x.byteOffset < y.byteOffset + y.byteLength &&
    y.byteOffset < x.byteOffset + x.byteLength
  );
}

function allFinite(d: Float32Array): boolean {
  for (let i = 0; i < d.length; i++) if (!Number.isFinite(d[i]!)) return false;
  return true;
}

/**
 * op(a) · op(b), where op transposes when the matching flag is set. Transposes
 * are folded into the indexing, so no transposed copies are allocated.
 *
 * Every output element is Σ_p op(a)[i,p] · op(b)[p,j], accumulated in float64
 * over p in ascending order and rounded to float32 once. The kernels below
 * reorder the loops so the inner loop runs over contiguous memory, but keep
 * that per-element order, so the result is bitwise identical to the plain
 * triple loop. Where an a-value is 0 its products are skipped (inputs and
 * ReLU outputs are often mostly zeros); that only changes anything when b
 * holds Inf or NaN (0 · Inf = NaN), so it is done only when b is finite.
 * Small products use the plain loop, which is the reference.
 */
export function matmul(a: Tensor, b: Tensor, opts: MatmulOptions = {}, out?: Tensor): Tensor {
  const { transA = false, transB = false } = opts;
  const [ar, ac] = [a.rows, a.cols];
  const [br, bc] = [b.rows, b.cols];
  const m = transA ? ac : ar;
  const k = transA ? ar : ac;
  const kb = transB ? bc : br;
  const n = transB ? br : bc;
  if (k !== kb) {
    throw new Error(
      `matmul: inner dimensions differ (${fmt(a.shape)}${transA ? 'ᵀ' : ''} · ${fmt(b.shape)}${transB ? 'ᵀ' : ''})`,
    );
  }
  const res = target([m, n], out, 'matmul');
  if (overlaps(res.data, a.data) || overlaps(res.data, b.data))
    throw new Error('matmul: out must not alias an input');
  const A = a.data;
  const B = b.data;
  const C = res.data;
  if (k * n < SMALL) matmulPlain(A, B, C, m, k, n, transA, transB);
  else if (!transA && !transB) matmulNN(A, B, C, m, k, n, allFinite(B));
  else if (transA && !transB) matmulTN(A, B, C, m, k, n, allFinite(B));
  else if (!transA && transB) matmulNT(A, B, C, m, k, n);
  else matmulPlain(A, B, C, m, k, n, true, true);
  return res;
}

/**
 * Below this many elements in op(b), the reordered kernels' set-up (the
 * finiteness scan, clearing and copying the accumulators) costs more than it
 * saves, so the plain loop is used (the 2D playground's 8-unit layers).
 */
const SMALL = 256;

/** The plain triple loop over strided indices; handles every transpose combination. */
function matmulPlain(
  A: Float32Array,
  B: Float32Array,
  C: Float32Array,
  m: number,
  k: number,
  n: number,
  transA: boolean,
  transB: boolean,
): void {
  // Strides for element (i, p) of op(a) and (p, j) of op(b).
  const aI = transA ? 1 : k;
  const aP = transA ? m : 1;
  const bP = transB ? 1 : n;
  const bJ = transB ? k : 1;
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let p = 0; p < k; p++) s += A[i * aI + p * aP]! * B[p * bP + j * bJ]!;
      C[i * n + j] = s;
    }
  }
}

/** C[m,n] = A[m,k] · B[k,n]: one row of float64 accumulators per output row. */
function matmulNN(
  A: Float32Array,
  B: Float32Array,
  C: Float32Array,
  m: number,
  k: number,
  n: number,
  skipZeros: boolean,
): void {
  const acc = accumulators(n);
  for (let i = 0; i < m; i++) {
    acc.fill(0, 0, n);
    const ao = i * k;
    for (let p = 0; p < k; p++) {
      const av = A[ao + p]!;
      if (av === 0 && skipZeros) continue;
      const bo = p * n;
      for (let j = 0; j < n; j++) acc[j]! += av * B[bo + j]!;
    }
    const co = i * n;
    for (let j = 0; j < n; j++) C[co + j] = acc[j]!;
  }
}

/** C[m,n] = Aᵀ · B with A [k,m], B [k,n] (e.g. dW = xᵀ·g): all m·n accumulators at once. */
function matmulTN(
  A: Float32Array,
  B: Float32Array,
  C: Float32Array,
  m: number,
  k: number,
  n: number,
  skipZeros: boolean,
): void {
  const acc = accumulators(m * n);
  acc.fill(0, 0, m * n);
  for (let p = 0; p < k; p++) {
    const ao = p * m;
    const bo = p * n;
    for (let i = 0; i < m; i++) {
      const av = A[ao + i]!;
      if (av === 0 && skipZeros) continue;
      const co = i * n;
      for (let j = 0; j < n; j++) acc[co + j]! += av * B[bo + j]!;
    }
  }
  for (let i = 0; i < m * n; i++) C[i] = acc[i]!;
}

/** C[m,n] = A · Bᵀ with A [m,k], B [n,k] (e.g. dx = g·Wᵀ): contiguous dot products. */
function matmulNT(
  A: Float32Array,
  B: Float32Array,
  C: Float32Array,
  m: number,
  k: number,
  n: number,
): void {
  for (let i = 0; i < m; i++) {
    const ao = i * k;
    for (let j = 0; j < n; j++) {
      const bo = j * k;
      let s = 0;
      for (let p = 0; p < k; p++) s += A[ao + p]! * B[bo + p]!;
      C[i * n + j] = s;
    }
  }
}

export function transpose(a: Tensor, out?: Tensor): Tensor {
  const [r, c] = [a.rows, a.cols];
  const res = target([c, r], out, 'transpose');
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) res.data[j * r + i] = a.data[i * c + j]!;
  }
  return res;
}

/** a + b. `b` has a's shape, or is `[cols]` / `[1, cols]` and is broadcast across rows of a rank-2 `a`. */
export function add(a: Tensor, b: Tensor, out?: Tensor): Tensor {
  const res = target(a.shape, out, 'add');
  if (sameShape(a.shape, b.shape)) {
    for (let i = 0; i < a.size; i++) res.data[i] = a.data[i]! + b.data[i]!;
    return res;
  }
  const cols = a.shape.length === 2 ? a.cols : -1;
  const isRow =
    (b.shape.length === 1 && b.shape[0] === cols) ||
    (b.shape.length === 2 && b.shape[0] === 1 && b.shape[1] === cols);
  if (!isRow) throw new Error(`add: cannot broadcast ${fmt(b.shape)} onto ${fmt(a.shape)}`);
  for (let i = 0; i < a.size; i++) res.data[i] = a.data[i]! + b.data[i % cols]!;
  return res;
}

export function map(a: Tensor, fn: (v: number) => number, out?: Tensor): Tensor {
  const res = target(a.shape, out, 'map');
  for (let i = 0; i < a.size; i++) res.data[i] = fn(a.data[i]!);
  return res;
}

export function map2(
  a: Tensor,
  b: Tensor,
  fn: (x: number, y: number) => number,
  out?: Tensor,
): Tensor {
  if (!sameShape(a.shape, b.shape))
    throw new Error(`map2: shapes differ (${fmt(a.shape)} vs ${fmt(b.shape)})`);
  const res = target(a.shape, out, 'map2');
  for (let i = 0; i < a.size; i++) res.data[i] = fn(a.data[i]!, b.data[i]!);
  return res;
}

/** Sums a rank-2 tensor over `axis`: 0 → column sums `[cols]`, 1 → row sums `[rows]`. */
export function sumAxis(a: Tensor, axis: 0 | 1, out?: Tensor): Tensor {
  const [r, c] = [a.rows, a.cols];
  const res = target([axis === 0 ? c : r], out, 'sumAxis');
  res.data.fill(0);
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) res.data[axis === 0 ? j : i]! += a.data[i * c + j]!;
  }
  return res;
}

/** Sum of every element, accumulated in float64. */
export function sumAll(a: Tensor): number {
  let s = 0;
  for (let i = 0; i < a.size; i++) s += a.data[i]!;
  return s;
}
