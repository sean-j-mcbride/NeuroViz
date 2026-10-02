import type { HiddenLayerSpec } from '../worker';
import type { PlaygroundConfig } from './config';
import { type ParsedConfig, parseConfig } from './validate';

/*
 * A run's setup as a readable URL hash, e.g.
 *   #data=spirals&points=400&noise=0&dataSeed=1&layers=8tanh,8tanh
 *    &lr=0.03&batch=10&optimiser=sgd&l2=0&dropout=0&seed=1
 * Everything is seeded, so the link reproduces the whole run, not just the settings.
 */

const NO_LAYERS = 'none';
const KEYS = [
  'data',
  'points',
  'noise',
  'dataSeed',
  'layers',
  'lr',
  'batch',
  'optimiser',
  'l2',
  'dropout',
  'seed',
] as const;

function encodeLayers(hidden: HiddenLayerSpec[]): string {
  return hidden.length === 0 ? NO_LAYERS : hidden.map((l) => `${l.units}${l.activation}`).join(',');
}

/** "8tanh,4relu" → layer specs; anything unrecognised is returned as-is for the validator to report. */
function decodeLayers(s: string): unknown {
  if (s === NO_LAYERS) return [];
  const layers = s.split(',').map((token) => /^(\d+)([a-z]+)$/.exec(token.trim()));
  if (layers.some((m) => !m)) return s;
  return layers.map((m) => ({ units: Number(m![1]), activation: m![2] }));
}

/** A number if the text is one, else the text (so a warning can quote it). */
function num(s: string | null): unknown {
  if (s === null) return undefined;
  const v = Number(s);
  return s.trim() !== '' && Number.isFinite(v) ? v : s;
}

export function encodeConfig(c: PlaygroundConfig): string {
  const p = new URLSearchParams({
    data: c.dataset.kind,
    points: String(c.dataset.n),
    noise: String(c.dataset.noise),
    dataSeed: String(c.dataset.seed),
    layers: encodeLayers(c.network.hidden),
    lr: String(c.training.lr),
    batch: String(c.training.batchSize),
    optimiser: c.training.optimiser,
    l2: String(c.training.l2),
    dropout: String(c.training.dropout),
    seed: String(c.seed),
  });
  // Commas are safe in a fragment and much easier to read unescaped.
  return `#${p.toString().replaceAll('%2C', ',')}`;
}

/**
 * Reads a hash written by `encodeConfig`. Missing values take their defaults;
 * invalid ones do too, with a warning. Returns null when the hash has none of
 * the keys (so an empty or unrelated hash leaves the app alone).
 */
export function decodeConfig(hash: string): ParsedConfig | null {
  const p = new URLSearchParams(hash.replace(/^#/, ''));
  if (!KEYS.some((k) => p.has(k))) return null;
  const str = (k: string) => p.get(k) ?? undefined;
  const batch = p.get('batch');
  const layers = p.get('layers');
  return parseConfig(
    {
      dataset: {
        kind: str('data'),
        n: num(p.get('points')),
        noise: num(p.get('noise')),
        seed: num(p.get('dataSeed')),
      },
      network: { hidden: layers === null ? undefined : decodeLayers(layers) },
      training: {
        lr: num(p.get('lr')),
        batchSize: batch === 'full' ? 'full' : num(batch),
        optimiser: str('optimiser'),
        l2: num(p.get('l2')),
        dropout: num(p.get('dropout')),
      },
      seed: num(p.get('seed')),
    },
    { partial: true },
  );
}
