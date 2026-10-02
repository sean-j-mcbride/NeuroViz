import { type HiddenLayerSpec, SPEEDS, type Speed } from '../worker';
import type { PlaygroundConfig } from './config';
import { type ParsedConfig, parseConfig } from './validate';

/*
 * A run's setup as a readable URL hash, e.g.
 *   #data=spirals&points=400&noise=0&dataSeed=1&layers=8tanh,8tanh
 *    &lr=0.03&batch=10&optimiser=sgd&l2=0&dropout=0&seed=1&speed=300&showTest=0
 * Everything is seeded, so the link reproduces the whole run, not just the settings.
 */

/** How the run is shown, as opposed to what is trained. */
export interface LinkView {
  speed: Speed;
  showTestData: boolean;
}

export interface ParsedLink extends ParsedConfig {
  /** The view settings the link carries; absent or invalid ones are left out. */
  view: Partial<LinkView>;
}

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
  'speed',
  'showTest',
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

export function encodeLink(c: PlaygroundConfig, view: LinkView): string {
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
    speed: String(view.speed),
    showTest: view.showTestData ? '1' : '0',
  });
  // Commas are safe in a fragment and much easier to read unescaped.
  return `#${p.toString().replaceAll('%2C', ',')}`;
}

/** Reads the view keys; invalid values are reported and left out. */
function decodeView(p: URLSearchParams, warnings: string[]): Partial<LinkView> {
  const view: Partial<LinkView> = {};
  const speed = p.get('speed');
  if (speed !== null) {
    const match = SPEEDS.find((s) => String(s) === speed);
    if (match !== undefined) view.speed = match;
    else warnings.push(`Ignored speed “${speed}” (choices: ${SPEEDS.join(', ')}); left unchanged`);
  }
  const showTest = p.get('showTest');
  if (showTest !== null) {
    if (showTest === '0' || showTest === '1') view.showTestData = showTest === '1';
    else warnings.push(`Ignored showTest “${showTest}” (choices: 0, 1); left unchanged`);
  }
  return view;
}

/**
 * Reads a hash written by `encodeLink`. Missing settings take their defaults;
 * invalid ones do too, with a warning. Returns null when the hash has none of
 * the keys (so an empty or unrelated hash leaves the app alone).
 */
export function decodeLink(hash: string): ParsedLink | null {
  const p = new URLSearchParams(hash.replace(/^#/, ''));
  if (!KEYS.some((k) => p.has(k))) return null;
  const str = (k: string) => p.get(k) ?? undefined;
  const batch = p.get('batch');
  const layers = p.get('layers');
  const parsed = parseConfig(
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
  const view = decodeView(p, parsed.warnings);
  return { ...parsed, view };
}
