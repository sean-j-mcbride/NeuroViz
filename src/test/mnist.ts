/// <reference types="node" />
// Loads the committed MNIST subset for tests (which run under Node).
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { MNIST_SUBSET_FILE, type MnistSubset, parseMnistSubset } from '../data';

let cached: { bytes: Uint8Array; subset: MnistSubset } | null = null;

export function loadMnistBytes(): Uint8Array {
  return loadMnist().bytes;
}

export function loadMnistSubset(): MnistSubset {
  return loadMnist().subset;
}

function loadMnist() {
  if (!cached) {
    const bytes = new Uint8Array(gunzipSync(readFileSync(`public/${MNIST_SUBSET_FILE}`)));
    cached = { bytes, subset: parseMnistSubset(bytes) };
  }
  return cached;
}
