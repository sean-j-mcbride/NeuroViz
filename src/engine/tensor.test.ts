import { describe, expect, it } from 'vitest';
import { Rng } from './random';
import { Tensor, add, map, map2, matmul, sumAll, sumAxis, transpose } from './tensor';

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
