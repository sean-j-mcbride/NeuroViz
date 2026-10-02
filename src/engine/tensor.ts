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

/**
 * op(a) · op(b), where op transposes when the matching flag is set. Transposes
 * are folded into the indexing, so no transposed copies are allocated.
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
  if (res.data === a.data || res.data === b.data)
    throw new Error('matmul: out must not alias an input');
  const A = a.data;
  const B = b.data;
  const C = res.data;
  // Strides for element (i, p) of op(a) and (p, j) of op(b).
  const aI = transA ? 1 : ac;
  const aP = transA ? ac : 1;
  const bP = transB ? 1 : bc;
  const bJ = transB ? bc : 1;
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let p = 0; p < k; p++) s += A[i * aI + p * aP]! * B[p * bP + j * bJ]!;
      C[i * n + j] = s;
    }
  }
  return res;
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
