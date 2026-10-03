import { describe, expect, it } from 'vitest';
import { Rng } from './random';
import { Tensor, TensorBuffer, add, map, map2, matmul, sumAll, sumAxis, transpose } from './tensor';

const A = Tensor.from([
  [1, 2, 3],
  [4, 5, 6],
]); // [2,3]
const B = Tensor.from([
  [7, 8],
  [9, 10],
  [11, 12],
]); // [3,2]
const AB = [
  [58, 64],
  [139, 154],
];

describe('Tensor construction', () => {
  it('from() infers rank-1 and rank-2 shapes', () => {
    expect(Tensor.from([1, 2, 3]).shape).toEqual([3]);
    expect(A.shape).toEqual([2, 3]);
    expect(Array.from(A.data)).toEqual([1, 2, 3, 4, 5, 6]);
    expect(A.toArray()).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ]);
  });

  it('rejects ragged input and size mismatches', () => {
    expect(() => Tensor.from([[1, 2], [3]])).toThrow(/Ragged/);
    expect(() => new Tensor(new Float32Array(5), [2, 3])).toThrow(/needs 6 elements/);
  });

  it('zeros() and clone()', () => {
    const z = Tensor.zeros([2, 2]);
    expect(Array.from(z.data)).toEqual([0, 0, 0, 0]);
    const c = A.clone();
    c.data[0] = 99;
    expect(A.data[0]).toBe(1);
  });

  it('randn() is seeded', () => {
    const a = Tensor.randn([3, 4], new Rng(5), 0.5);
    const b = Tensor.randn([3, 4], new Rng(5), 0.5);
    expect(a.data).toEqual(b.data);
    expect(a.shape).toEqual([3, 4]);
  });

  it('rows/cols require rank 2', () => {
    expect(A.rows).toBe(2);
    expect(A.cols).toBe(3);
    expect(() => Tensor.from([1, 2]).rows).toThrow(/rank-2/);
  });
});

describe('matmul', () => {
  const At = transpose(A); // [3,2]
  const Bt = transpose(B); // [2,3]

  it('a · b', () => expect(matmul(A, B).toArray()).toEqual(AB));
  it('aᵀ · b', () => expect(matmul(At, B, { transA: true }).toArray()).toEqual(AB));
  it('a · bᵀ', () => expect(matmul(A, Bt, { transB: true }).toArray()).toEqual(AB));
  it('aᵀ · bᵀ', () => expect(matmul(At, Bt, { transA: true, transB: true }).toArray()).toEqual(AB));

  it('writes into out', () => {
    const out = Tensor.zeros([2, 2]);
    expect(matmul(A, B, {}, out)).toBe(out);
    expect(out.toArray()).toEqual(AB);
  });

  it('rejects bad shapes', () => {
    expect(() => matmul(A, A)).toThrow(/inner dimensions/);
    expect(() => matmul(A, B, {}, Tensor.zeros([3, 3]))).toThrow(/out has shape/);
    expect(() => matmul(A, B, {}, A)).toThrow();
  });
});

/** The original kernel: the plain triple loop with strided indexing (the reference). */
function referenceMatmul(a: Tensor, b: Tensor, transA: boolean, transB: boolean): Float32Array {
  const [ar, ac] = [a.rows, a.cols];
  const bc = b.cols;
  const m = transA ? ac : ar;
  const k = transA ? ar : ac;
  const n = transB ? b.rows : bc;
  const C = new Float32Array(m * n);
  const [aI, aP] = transA ? [1, ac] : [ac, 1];
  const [bP, bJ] = transB ? [1, bc] : [bc, 1];
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let p = 0; p < k; p++) s += a.data[i * aI + p * aP]! * b.data[p * bP + j * bJ]!;
      C[i * n + j] = s;
    }
  }
  return C;
}

/** Random values with about `zeros` of them exactly 0 (like MNIST pixels or ReLU outputs). */
function sparse(shape: [number, number], rng: Rng, zeros: number): Tensor {
  const t = Tensor.randn(shape, rng);
  for (let i = 0; i < t.size; i++) if (rng.next() < zeros) t.data[i] = 0;
  return t;
}

function bitwiseEqual(x: Float32Array, y: Float32Array): boolean {
  return x.length === y.length && x.every((v, i) => Object.is(v, y[i]));
}

describe('matmul kernels', () => {
  const rng = new Rng(11);
  const cases: [number, number, number][] = [
    [1, 1, 1],
    [3, 5, 2],
    [64, 784, 128],
    [17, 128, 10],
    [2500, 8, 8],
  ];

  for (const transA of [false, true]) {
    for (const transB of [false, true]) {
      it(`are bitwise identical to the plain loop (transA ${transA}, transB ${transB})`, () => {
        for (const [m, k, n] of cases) {
          for (const zeros of [0, 0.5, 0.9]) {
            const a = sparse(transA ? [k, m] : [m, k], rng, zeros);
            const b = sparse(transB ? [n, k] : [k, n], rng, zeros);
            const got = matmul(a, b, { transA, transB }).data;
            expect(bitwiseEqual(got, referenceMatmul(a, b, transA, transB))).toBe(true);
          }
        }
      });
    }
  }

  it('still turn 0 · Inf into NaN, as the plain loop does', () => {
    for (const transA of [false, true]) {
      const a = Tensor.from([[0, 1]]);
      const b = Tensor.from([
        [Infinity, 1],
        [2, 3],
      ]);
      const at = transA ? transpose(a) : a;
      const got = matmul(at, b, { transA }).data;
      expect(bitwiseEqual(got, referenceMatmul(at, b, transA, false))).toBe(true);
      expect(Number.isNaN(got[0])).toBe(true);
    }
  });

  it('allow out to share a buffer with an input when the memory does not overlap', () => {
    const buf = new Float32Array(8);
    const x = new Tensor(buf.subarray(0, 4), [2, 2]);
    x.data.set([1, 2, 3, 4]);
    const out = new Tensor(buf.subarray(4, 8), [2, 2]);
    expect(Array.from(matmul(x, x, {}, out).data)).toEqual([7, 10, 15, 22]);
    expect(() => matmul(x, x, {}, new Tensor(buf.subarray(2, 6), [2, 2]))).toThrow(/alias/);
  });
});

describe('TensorBuffer', () => {
  it('grows to the largest shape and hands out views, reusing memory', () => {
    const buf = new TensorBuffer();
    const big = buf.take([10, 4]);
    expect(big.shape).toEqual([10, 4]);
    expect(buf.take([10, 4])).toBe(big);
    const small = buf.take([3, 4]);
    expect(small.shape).toEqual([3, 4]);
    expect(small.data.buffer).toBe(big.data.buffer);
    expect(buf.take([10, 4]).data.buffer).toBe(big.data.buffer);
    expect(buf.take([20, 4]).size).toBe(80);
  });
});

describe('elementwise ops', () => {
  it('transpose', () => {
    expect(transpose(A).toArray()).toEqual([
      [1, 4],
      [2, 5],
      [3, 6],
    ]);
  });

  it('add with same shape', () => {
    expect(add(A, A).toArray()).toEqual([
      [2, 4, 6],
      [8, 10, 12],
    ]);
  });

  it('add with row broadcast ([cols] and [1,cols])', () => {
    const expected = [
      [11, 22, 33],
      [14, 25, 36],
    ];
    expect(add(A, Tensor.from([10, 20, 30])).toArray()).toEqual(expected);
    expect(add(A, Tensor.from([[10, 20, 30]])).toArray()).toEqual(expected);
    expect(() => add(A, Tensor.from([1, 2]))).toThrow(/broadcast/);
  });

  it('add in place via out', () => {
    const t = A.clone();
    add(t, Tensor.from([1, 1, 1]), t);
    expect(t.toArray()).toEqual([
      [2, 3, 4],
      [5, 6, 7],
    ]);
  });

  it('map and map2', () => {
    expect(map(A, (v) => v * v).toArray()).toEqual([
      [1, 4, 9],
      [16, 25, 36],
    ]);
    expect(map2(A, A, (x, y) => x - y).toArray()).toEqual([
      [0, 0, 0],
      [0, 0, 0],
    ]);
    expect(() => map2(A, B, (x) => x)).toThrow(/shapes differ/);
  });
});

describe('reductions', () => {
  it('sumAxis', () => {
    expect(sumAxis(A, 0).toArray()).toEqual([5, 7, 9]);
    expect(sumAxis(A, 1).toArray()).toEqual([6, 15]);
  });

  it('sumAxis overwrites a reused out buffer', () => {
    const out = Tensor.from([100, 100, 100]);
    sumAxis(A, 0, out);
    expect(out.toArray()).toEqual([5, 7, 9]);
  });

  it('sumAll', () => expect(sumAll(A)).toBe(21));
});
