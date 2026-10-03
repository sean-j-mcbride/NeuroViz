import { MNIST_SUBSET_FILE, type MnistSubset, parseMnistSubset } from '../data';

let pending: Promise<MnistSubset> | null = null;

async function gunzip(bytes: Uint8Array): Promise<Uint8Array> {
  // A host may already have decoded it (Content-Encoding: gzip); then there is no gzip header.
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes;
  const stream = new Blob([bytes as BlobPart])
    .stream()
    .pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/**
 * Fetches and parses the bundled MNIST subset (~2 MB), once. It is served
 * next to the page, so this works from a GitHub Pages subpath too. A failed
 * load can be retried.
 */
export function loadMnist(): Promise<MnistSubset> {
  pending ??= (async () => {
    const res = await fetch(new URL(MNIST_SUBSET_FILE, document.baseURI));
    if (!res.ok) throw new Error(`the server answered ${res.status} ${res.statusText}`);
    return parseMnistSubset(await gunzip(new Uint8Array(await res.arrayBuffer())));
  })();
  pending.catch(() => (pending = null));
  return pending;
}
