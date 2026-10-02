/*
 * Typed arrays ↔ base64, little-endian whatever the platform, for the compact
 * parts of a saved model. `atob` / `btoa` exist in browsers and in Node ≥ 16.
 */

function bytesToBase64(bytes: Uint8Array): string {
  let s = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    s += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(s);
}

/** Throws a plain Error if `text` is not base64. */
function base64ToBytes(text: string): Uint8Array {
  const s = atob(text);
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
}

export function float32ToBase64(a: Float32Array): string {
  const bytes = new Uint8Array(a.length * 4);
  const view = new DataView(bytes.buffer);
  a.forEach((v, i) => view.setFloat32(i * 4, v, true));
  return bytesToBase64(bytes);
}

export function base64ToFloat32(text: string): Float32Array {
  const bytes = base64ToBytes(text);
  if (bytes.length % 4 !== 0) throw new Error('length is not a whole number of float32 values');
  const view = new DataView(bytes.buffer);
  return Float32Array.from({ length: bytes.length / 4 }, (_, i) => view.getFloat32(i * 4, true));
}

export type UintWidth = 1 | 2 | 4;

/** The narrowest unsigned width that holds `max`. */
export function uintWidthFor(max: number): UintWidth {
  return max < 2 ** 8 ? 1 : max < 2 ** 16 ? 2 : 4;
}

export function uintsToBase64(values: ArrayLike<number>, width: UintWidth): string {
  const bytes = new Uint8Array(values.length * width);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < values.length; i++) {
    const v = values[i]!;
    if (width === 1) view.setUint8(i, v);
    else if (width === 2) view.setUint16(i * 2, v, true);
    else view.setUint32(i * 4, v, true);
  }
  return bytesToBase64(bytes);
}

export function base64ToUints(text: string, width: UintWidth): Uint32Array {
  const bytes = base64ToBytes(text);
  if (bytes.length % width !== 0)
    throw new Error(`length is not a whole number of ${width}-byte values`);
  const view = new DataView(bytes.buffer);
  return Uint32Array.from({ length: bytes.length / width }, (_, i) =>
    width === 1
      ? view.getUint8(i)
      : width === 2
        ? view.getUint16(i * 2, true)
        : view.getUint32(i * 4, true),
  );
}
