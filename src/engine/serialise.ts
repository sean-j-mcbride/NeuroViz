import type { Layer } from './layers/types';

/** A model's parameter values by name (as `Layer.params()` names them, e.g. "0.W"). */
export type ParamValues = Record<string, Float32Array>;

/** Copies of every parameter value, keyed by name. */
export function exportParams(model: Layer): ParamValues {
  const out: ParamValues = {};
  for (const p of model.params()) out[p.name] = p.value.data.slice();
  return out;
}

/**
 * Overwrites the model's parameters with `values`. Every parameter must be
 * present with the right size, and no others; otherwise nothing is written.
 */
export function importParams(model: Layer, values: ParamValues): void {
  const params = model.params();
  const names = new Set(params.map((p) => p.name));
  for (const name of Object.keys(values)) {
    if (!names.has(name)) throw new Error(`Unknown parameter "${name}"`);
  }
  for (const p of params) {
    const v = values[p.name];
    if (!v) throw new Error(`Missing parameter "${p.name}"`);
    if (v.length !== p.value.size) {
      throw new Error(`Parameter "${p.name}" has ${v.length} values, expected ${p.value.size}`);
    }
  }
  for (const p of params) p.value.data.set(values[p.name]!);
}
