import { describe, expect, it } from 'vitest';
import { Tanh } from './layers/activations';
import { Dense } from './layers/dense';
import { Rng } from './random';
import { Sequential } from './sequential';
import { exportParams, importParams } from './serialise';

function model(seed: number, hidden = 3): Sequential {
  const rng = new Rng(seed);
  return new Sequential([
    new Dense(2, hidden, { init: 'xavier', rng }),
    new Tanh(),
    new Dense(hidden, 1, { init: 'xavier', rng }),
  ]);
}

describe('exportParams / importParams', () => {
  it('names params by layer index and round-trips exactly', () => {
    const a = model(1);
    const values = exportParams(a);
    expect(Object.keys(values)).toEqual(['0.W', '0.b', '2.W', '2.b']);
    const b = model(2);
    importParams(b, values);
    expect(exportParams(b)).toEqual(values);
  });

  it('exports copies', () => {
    const a = model(1);
    const values = exportParams(a);
    a.params()[0]!.value.data[0] = 99;
    expect(values['0.W']![0]).not.toBe(99);
  });

  it('rejects missing, unknown or wrong-size params and then writes nothing', () => {
    const target = model(3);
    const before = exportParams(target);
    const values = exportParams(model(1));

    const missing = { ...values };
    delete missing['2.b'];
    expect(() => importParams(target, missing)).toThrow(/Missing parameter "2.b"/);
    expect(() => importParams(target, { ...values, '9.W': new Float32Array(1) })).toThrow(
      /Unknown parameter "9.W"/,
    );
    expect(() => importParams(model(3, 4), values)).toThrow(/"0.W" has 6 values, expected 8/);
    expect(exportParams(target)).toEqual(before);
  });
});
