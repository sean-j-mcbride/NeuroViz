// Builds public/mnist-subset-v1.bin.gz: a seeded, digit-stratified subset of MNIST.
//
//   npm run data:mnist
//
// Downloads the four official IDX files, picks TRAIN_PER_DIGIT images of each digit
// from the official training set and TEST_PER_DIGIT from the official test set,
// and writes them in the small format `src/data/mnist.ts` parses:
//
//   header (20 bytes, little-endian)
//     0  "NVMN"   magic
//     4  u16      version (1)
//     6  u16      rows (28)
//     8  u16      cols (28)
//    10  u16      reserved (0)
//    12  u32      train count
//    16  u32      test count
//   then for train, then test:
//     u16[n]      index of each image in its official set
//     u8[n]       label
//     u8[n·784]   pixels, row-major, 0 = background
//
// MNIST: Yann LeCun and Corinna Cortes, CC BY-SA 3.0.
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync, gzipSync } from 'node:zlib';
import { Rng } from '../src/engine/random.ts';

const MIRROR = 'https://storage.googleapis.com/cvdf-datasets/mnist/';
const TRAIN_PER_DIGIT = 1000;
const TEST_PER_DIGIT = 200;
const SEED = 1;
const OUT = resolve(dirname(fileURLToPath(import.meta.url)), '../public/mnist-subset-v1.bin.gz');

async function download(name: string): Promise<Uint8Array> {
  const res = await fetch(MIRROR + name);
  if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`);
  return gunzipSync(new Uint8Array(await res.arrayBuffer()));
}

function readIdx(bytes: Uint8Array, magic: number): { count: number; body: Uint8Array } {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(0) !== magic) throw new Error(`Bad IDX magic ${view.getUint32(0)}`);
  const count = view.getUint32(4);
  const header = magic === 2051 ? 16 : 8;
  if (magic === 2051 && (view.getUint32(8) !== 28 || view.getUint32(12) !== 28)) {
    throw new Error('Expected 28×28 images');
  }
  return { count, body: bytes.subarray(header) };
}

interface Split {
  source: number[];
  labels: Uint8Array;
  images: Uint8Array;
}

/** `perDigit` images of each digit, chosen by a seeded shuffle, kept in official order. */
function pick(images: Uint8Array, labels: Uint8Array, perDigit: number, rng: Rng): Split {
  const chosen: number[] = [];
  for (let d = 0; d < 10; d++) {
    const idx: number[] = [];
    labels.forEach((l, i) => l === d && idx.push(i));
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(rng.next() * (i + 1));
      [idx[i], idx[j]] = [idx[j]!, idx[i]!];
    }
    if (idx.length < perDigit) throw new Error(`Only ${idx.length} images of ${d}`);
    chosen.push(...idx.slice(0, perDigit));
  }
  chosen.sort((a, b) => a - b);
  const out: Split = {
    source: chosen,
    labels: new Uint8Array(chosen.length),
    images: new Uint8Array(chosen.length * 784),
  };
  chosen.forEach((src, i) => {
    out.labels[i] = labels[src]!;
    out.images.set(images.subarray(src * 784, (src + 1) * 784), i * 784);
  });
  return out;
}

function encode(train: Split, test: Split): Uint8Array {
  const size = (s: Split) => s.source.length * (2 + 1 + 784);
  const bytes = new Uint8Array(20 + size(train) + size(test));
  const view = new DataView(bytes.buffer);
  bytes.set([0x4e, 0x56, 0x4d, 0x4e]); // "NVMN"
  view.setUint16(4, 1, true);
  view.setUint16(6, 28, true);
  view.setUint16(8, 28, true);
  view.setUint32(12, train.source.length, true);
  view.setUint32(16, test.source.length, true);
  let o = 20;
  for (const s of [train, test]) {
    for (const src of s.source) {
      view.setUint16(o, src, true);
      o += 2;
    }
    bytes.set(s.labels, o);
    o += s.labels.length;
    bytes.set(s.images, o);
    o += s.images.length;
  }
  return bytes;
}

const [trainImg, trainLab, testImg, testLab] = await Promise.all(
  [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz',
  ].map(download),
);
const ti = readIdx(trainImg!, 2051);
const tl = readIdx(trainLab!, 2049);
const vi = readIdx(testImg!, 2051);
const vl = readIdx(testLab!, 2049);
if (ti.count !== 60000 || tl.count !== 60000 || vi.count !== 10000 || vl.count !== 10000) {
  throw new Error('Unexpected MNIST sizes');
}

const rng = new Rng(SEED);
const train = pick(ti.body, tl.body, TRAIN_PER_DIGIT, rng);
const test = pick(vi.body, vl.body, TEST_PER_DIGIT, rng);
const raw = encode(train, test);
const gz = gzipSync(raw, { level: 9 });
mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, gz);
console.log(
  `Wrote ${OUT}: ${train.source.length} train + ${test.source.length} test images, ` +
    `${(raw.length / 1e6).toFixed(2)} MB raw, ${(gz.length / 1e6).toFixed(2)} MB gzipped`,
);
