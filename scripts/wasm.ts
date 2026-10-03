import wabtInit from 'wabt';

/** kernels.wat → WebAssembly bytes (SIMD and bulk memory enabled). */
export async function compileKernels(wat: string): Promise<Uint8Array> {
  const wabt = await wabtInit();
  const mod = wabt.parseWat('kernels.wat', wat, { simd: true, bulk_memory: true });
  try {
    mod.validate();
    return new Uint8Array(mod.toBinary({}).buffer);
  } finally {
    mod.destroy();
  }
}
