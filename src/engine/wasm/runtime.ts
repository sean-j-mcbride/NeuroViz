import { KERNELS_WASM } from './kernels';

/*
 * The WebAssembly kernels' runtime: one shared instance and memory, a small
 * allocator, and the choice between the Wasm and JS kernels. Both give
 * bitwise-identical results (see kernels.wat), so the choice only affects
 * speed. Typed through a minimal interface on `globalThis` so the engine
 * needs no DOM or Wasm lib types.
 */

interface WasmMemory {
  readonly buffer: ArrayBuffer;
  grow(pages: number): number;
}

interface WasmGlobal {
  Memory: new (d: { initial: number; maximum?: number }) => WasmMemory;
  Module: new (bytes: Uint8Array) => object;
  Instance: new (module: object, imports: object) => { exports: Record<string, unknown> };
}

/** The exported kernels (byte offsets into memory; see kernels.wat). */
export interface KernelExports {
  nn(
    A: number,
    B: number,
    C: number,
    acc: number,
    m: number,
    k: number,
    n: number,
    skip: number,
  ): void;
  tn(
    A: number,
    B: number,
    C: number,
    acc: number,
    m: number,
    k: number,
    n: number,
    skip: number,
  ): void;
  patches(
    x: number,
    col: number,
    b0: number,
    m: number,
    c: number,
    h: number,
    w: number,
    k: number,
    s: number,
    pad: number,
    oh: number,
    ow: number,
    add: number,
  ): void;
}

const PAGE = 65536;
const ALIGN = 16;

/**
 * First-fit allocator over the Wasm memory, growing it as needed. Growing
 * detaches every existing view of the old buffer, so callers make views
 * (`f32`, `f64`) after allocating, and never keep them across calls.
 */
export class WasmHeap {
  /** Free blocks, sorted by offset, never adjacent (they are merged). */
  private free: { at: number; size: number }[] = [];
  private top = ALIGN; // offset 0 is never handed out
  constructor(readonly memory: WasmMemory) {}

  alloc(bytes: number): number {
    const size = Math.max(ALIGN, Math.ceil(bytes / ALIGN) * ALIGN);
    const i = this.free.findIndex((b) => b.size >= size);
    if (i >= 0) {
      const b = this.free[i]!;
      const at = b.at;
      if (b.size === size) this.free.splice(i, 1);
      else this.free[i] = { at: b.at + size, size: b.size - size };
      return at;
    }
    const at = this.top;
    const end = at + size;
    const have = this.memory.buffer.byteLength;
    if (end > have) this.memory.grow(Math.ceil((end - have) / PAGE));
    this.top = end;
    return at;
  }

  release(at: number, bytes: number): void {
    const size = Math.max(ALIGN, Math.ceil(bytes / ALIGN) * ALIGN);
    let i = this.free.findIndex((b) => b.at > at);
    if (i < 0) i = this.free.length;
    this.free.splice(i, 0, { at, size });
    // Merge with the following and preceding blocks.
    const next = this.free[i + 1];
    if (next && at + size === next.at) {
      this.free[i]!.size += next.size;
      this.free.splice(i + 1, 1);
    }
    const prev = this.free[i - 1];
    if (prev && prev.at + prev.size === at) {
      prev.size += this.free[i]!.size;
      this.free.splice(i, 1);
    }
    // A free block at the top goes back to the top.
    const last = this.free.at(-1);
    if (last && last.at + last.size === this.top) {
      this.top = last.at;
      this.free.pop();
    }
  }

  /** Bytes handed out and not released (for tests). */
  get inUse(): number {
    return this.top - ALIGN - this.free.reduce((s, b) => s + b.size, 0);
  }

  f32(at: number, length: number): Float32Array {
    return new Float32Array(this.memory.buffer, at, length);
  }
}

export type { WasmGlobal };

export interface WasmKernels {
  heap: WasmHeap;
  k: KernelExports;
}

export type KernelBackend = 'auto' | 'wasm' | 'js';

let backend: KernelBackend = 'auto';
let loaded: WasmKernels | null | undefined;

/** Instantiates the kernels, or null where WebAssembly (with SIMD) isn't available. */
export function loadKernels(
  wasm: WasmGlobal | null = (globalThis as unknown as { WebAssembly?: WasmGlobal }).WebAssembly ??
    null,
): WasmKernels | null {
  if (!wasm) return null;
  try {
    const memory = new wasm.Memory({ initial: 16 });
    const instance = new wasm.Instance(new wasm.Module(KERNELS_WASM), { env: { memory } });
    return { heap: new WasmHeap(memory), k: instance.exports as unknown as KernelExports };
  } catch {
    // No SIMD, or a synchronous compile refused: the JS kernels give the same results.
    return null;
  }
}

/**
 * Chooses the kernels: 'auto' (Wasm when available), 'wasm' (throws if it
 * isn't) or 'js'. For tests and benchmarks; results are identical.
 */
export function setKernelBackend(b: KernelBackend): void {
  backend = b;
}

/** The Wasm kernels to use now, or null to use the JS ones. */
export function wasmKernels(): WasmKernels | null {
  if (backend === 'js') return null;
  if (loaded === undefined) loaded = loadKernels();
  if (!loaded && backend === 'wasm') throw new Error('WebAssembly kernels are not available here');
  return loaded;
}

/**
 * Named, grow-only blocks of Wasm memory owned by one object (a layer). They
 * are released when the owner is garbage-collected, so rebuilding models
 * doesn't leak Wasm memory.
 */
export class WasmBlocks {
  private readonly blocks = new Map<string, { at: number; bytes: number }>();
  private static readonly registry = new FinalizationRegistry(
    ({ heap, blocks }: { heap: WasmHeap; blocks: Map<string, { at: number; bytes: number }> }) => {
      for (const b of blocks.values()) heap.release(b.at, b.bytes);
    },
  );

  constructor(
    owner: object,
    private readonly heap: WasmHeap,
  ) {
    WasmBlocks.registry.register(owner, { heap, blocks: this.blocks });
  }

  /** Offset of block `name`, (re)allocated to hold at least `bytes`. */
  get(name: string, bytes: number): number {
    const b = this.blocks.get(name);
    if (b && b.bytes >= bytes) return b.at;
    if (b) this.heap.release(b.at, b.bytes);
    const at = this.heap.alloc(bytes);
    this.blocks.set(name, { at, bytes });
    return at;
  }
}
