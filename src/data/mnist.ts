import type { Tensor } from '../engine';

/** Pixels per image (28 × 28). */
export const MNIST_PIXELS = 784;
export const MNIST_SIDE = 28;
export const MNIST_CLASSES = 10;

/** Identifies the bundled subset (see scripts/build-mnist.ts); model files name it. */
export const MNIST_SUBSET_ID = 'mnist-subset-v1';
/** Served from public/, relative to the page. */
export const MNIST_SUBSET_FILE = 'mnist-subset-v1.bin.gz';

/** One split of MNIST, kept as bytes (a quarter of the size of float32 pixels). */
export interface MnistSet {
  /** `[N · 784]`, row-major 28×28, 0 = background, 255 = ink. */
  images: Uint8Array;
  /** `[N]`, each 0–9. */
  labels: Uint8Array;
  /** `[N]`, the index of each image in its official MNIST set. */
  source: Uint16Array;
}

export interface MnistSubset {
  train: MnistSet;
  test: MnistSet;
  /** FNV-1a of the decompressed file, so a saved model can check it has the same data. */
  checksum: number;
}

export function mnistCount(set: MnistSet): number {
  return set.labels.length;
}

/** 32-bit FNV-1a hash. */
export function fnv1a(bytes: Uint8Array): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) {
    h ^= bytes[i]!;
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** Parses the decompressed subset file written by scripts/build-mnist.ts. */
export function parseMnistSubset(bytes: Uint8Array): MnistSubset {
  const fail = (why: string): never => {
    throw new Error(`Not a NeuroViz MNIST file: ${why}`);
  };
  if (bytes.length < 20) fail('too short');
  if (String.fromCharCode(...bytes.subarray(0, 4)) !== 'NVMN') fail('bad header');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = view.getUint16(4, true);
  if (version !== 1) fail(`version ${version} is not supported`);
  if (view.getUint16(6, true) !== MNIST_SIDE || view.getUint16(8, true) !== MNIST_SIDE) {
    fail('images are not 28×28');
  }
  const counts = [view.getUint32(12, true), view.getUint32(16, true)] as const;
  const expected = 20 + (counts[0] + counts[1]) * (2 + 1 + MNIST_PIXELS);
  if (bytes.length !== expected) fail(`${bytes.length} bytes, expected ${expected}`);

  let o = 20;
  const [train, test] = counts.map((n) => {
    const source = new Uint16Array(n);
    for (let i = 0; i < n; i++, o += 2) source[i] = view.getUint16(o, true);
    const labels = bytes.slice(o, o + n);
    o += n;
    const images = bytes.slice(o, o + n * MNIST_PIXELS);
    o += n * MNIST_PIXELS;
    if (labels.some((l) => l >= MNIST_CLASSES)) fail('label out of range');
    return { images, labels, source };
  }) as [MnistSet, MnistSet];
  return { train, test, checksum: fnv1a(bytes) };
}

/**
 * Copies images `indices[start .. start + out.rows)` into `out` (`[rows, 784]`),
 * scaled to [0, 1]. Used to assemble mini-batches without allocating.
 */
export function gatherImages(
  set: MnistSet,
  indices: Uint32Array,
  start: number,
  out: Tensor,
): Tensor {
  if (out.cols !== MNIST_PIXELS) throw new Error(`gatherImages: out has ${out.cols} cols`);
  const src = set.images;
  const dst = out.data;
  for (let r = 0; r < out.rows; r++) {
    const s = indices[start + r]! * MNIST_PIXELS;
    const d = r * MNIST_PIXELS;
    for (let p = 0; p < MNIST_PIXELS; p++) dst[d + p] = src[s + p]! / 255;
  }
  return out;
}

/** One-hot labels for `indices[start .. start + out.rows)` into `out` (`[rows, 10]`). */
export function oneHotInto(
  labels: Uint8Array,
  indices: Uint32Array,
  start: number,
  out: Tensor,
): Tensor {
  if (out.cols !== MNIST_CLASSES) throw new Error(`oneHotInto: out has ${out.cols} cols`);
  out.data.fill(0);
  for (let r = 0; r < out.rows; r++) out.data[r * MNIST_CLASSES + labels[indices[start + r]!]!] = 1;
  return out;
}
