import { type ReactNode, useRef, useState } from 'react';
import type { NeuronColumn, ProbeTrace, Snapshot } from '../worker';
import {
  NEGATIVE,
  NORMALISERS,
  POSITIVE,
  formatCompact,
  formatPrecise,
  lutIndex,
  rgbCss,
} from './colour';
import { Heatmap } from './Heatmap';
import { columnLabel, neuronLabel } from './labels';
import { useElementWidth, useHeatmapLut } from './hooks';
import { Sparkline } from './Sparkline';
import { type StageView, forwardLayer, gradLayer, stageView } from './stepThrough';
import { Tooltip } from './Tooltip';

const TILE = 40;
const GAP = 10;
const HEADER = 76;
const PAD_X = 28;
const MIN_COLUMN_SPACING = 90;

interface NetworkGraphProps {
  snapshot: Snapshot;
  /** Optional controls rendered above each column (e.g. the architecture builder). */
  renderHeader?: (column: number, kind: NeuronColumn['kind']) => ReactNode;
  /** Step-through: the traced example and the current stage. */
  trace?: { probe: ProbeTrace; stage: number } | null;
}

type Hover =
  | { kind: 'edge'; layer: number; i: number; j: number; x: number; y: number }
  | { kind: 'neuron'; column: number; unit: number; x: number; y: number };

/** Edge width ∝ |value|, clamped so huge values don't swamp the picture. */
function strokeWidth(w: number): number {
  return Math.min(6, 0.3 + Math.abs(w) * 1.2);
}

/** Decades of gradient magnitude shown, from thinnest to thickest edge. */
const GRAD_DECADES = 4;

/**
 * Gradients span orders of magnitude, so their width is logarithmic, relative
 * to the largest gradient anywhere in the trace (so layers stay comparable and
 * vanishing gradients show as thin edges): max → 6 px, max·10⁻⁴ or less → 0.3 px.
 */
function gradStrokeWidth(g: number, logMax: number): number {
  const a = Math.abs(g);
  if (a === 0) return 0.3;
  const t = (Math.log10(a) - (logMax - GRAD_DECADES)) / GRAD_DECADES;
  return 0.3 + 5.7 * Math.min(1, Math.max(0, t));
}

function maxAbs(a: Float32Array): number {
  let m = 0;
  for (const v of a) m = Math.max(m, Math.abs(v));
  return m;
}

/**
 * The network as columns of neurons. Each neuron is drawn as a mini heatmap of
 * its value across the input plane; edges (SVG) show weights — thickness =
 * |w|, colour = sign. Hovering an edge or neuron shows its values, gradients
 * and recent history. In step-through mode the graph instead shows one
 * example's activations (forward) or gradients (backward).
 */
export function NetworkGraph({ snapshot, renderHeader, trace }: NetworkGraphProps) {
  const ref = useRef<HTMLDivElement>(null);
  const measured = useElementWidth(ref);
  const lut = useHeatmapLut();
  const [hover, setHover] = useState<Hover | null>(null);
  const { columns, weights, gridSize: g, domain } = snapshot;
  const probe = trace?.probe ?? null;
  const view: StageView | null = trace ? stageView(trace.stage, columns.length) : null;
  const gradLogMax = probe ? Math.log10(Math.max(...probe.dW.map(maxAbs)) || 1) : 0;

  const tallest = Math.max(...columns.map((c) => c.units));
  const height = HEADER + tallest * (TILE + GAP);
  const width = Math.max(measured, 2 * PAD_X + TILE + (columns.length - 1) * MIN_COLUMN_SPACING);
  const spacing = (width - 2 * PAD_X - TILE) / (columns.length - 1);
  const colX = (i: number) => PAD_X + i * spacing;
  const nodeY = (c: NeuronColumn, u: number) =>
    HEADER + ((tallest - c.units) * (TILE + GAP)) / 2 + u * (TILE + GAP);

  const edges: ReactNode[] = [];
  weights.forEach((layer, k) => {
    const from = columns[k]!;
    const to = columns[k + 1]!;
    const x1 = colX(k) + TILE;
    const x2 = colX(k + 1);
    const xm = (x1 + x2) / 2;
    const showGrad = probe && view && gradLayer(view, k);
    const showForward = probe && view && forwardLayer(view, k);
    for (let i = 0; i < layer.inFeatures; i++) {
      const y1 = nodeY(from, i) + TILE / 2;
      for (let j = 0; j < layer.outFeatures; j++) {
        const idx = i * layer.outFeatures + j;
        const w = layer.W[idx]!;
        let value = w;
        let widthPx = strokeWidth(w);
        let dim = false;
        if (showGrad) {
          value = probe.dW[k]![idx]!;
          widthPx = gradStrokeWidth(value, gradLogMax);
        } else if (showForward) {
          value = w * probe.columns[k]!.a[i]!;
          widthPx = strokeWidth(value);
        } else if (probe) {
          dim = true;
        }
        const y2 = nodeY(to, j) + TILE / 2;
        const d = `M${x1},${y1} C${xm},${y1} ${xm},${y2} ${x2},${y2}`;
        const active =
          hover?.kind === 'edge' && hover.layer === k && hover.i === i && hover.j === j;
        edges.push(
          <g key={`${k}-${i}-${j}`}>
            <path
              d={d}
              className={`edge ${value >= 0 ? 'pos' : 'neg'}${dim ? ' dim' : ''}${active ? ' active' : ''}`}
              strokeWidth={widthPx}
            />
            <path
              d={d}
              className="edge-hit"
              onMouseMove={(e) =>
                setHover({ kind: 'edge', layer: k, i, j, x: e.clientX, y: e.clientY })
              }
              onMouseLeave={() => setHover(null)}
            />
          </g>,
        );
      }
    }
  });

  return (
    <div ref={ref} className="network-graph">
      <div className="network-canvas" style={{ width, height }}>
        <svg width={width} height={height} className="network-edges">
          {edges}
        </svg>
        {columns.map((c, ci) => {
          const start = (u: number) => g * g * u;
          const pc = probe?.columns[ci];
          const showForward = pc && view && view.phase !== 'backward' && ci <= view.forwardThrough;
          const showGrad = pc && view && view.phase === 'backward' && ci >= view.gradFrom;
          const gradValues = pc && (c.kind === 'output' ? pc.dZ : pc.dA);
          const gradScale = gradValues ? maxAbs(gradValues) || 1 : 1;
          return (
            <div key={ci}>
              <div
                className="column-header"
                style={{ left: colX(ci) + TILE / 2, width: Math.min(spacing, 140) }}
              >
                <div className="column-label">{columnLabel(c, ci)}</div>
                {renderHeader?.(ci, c.kind)}
              </div>
              {Array.from({ length: c.units }, (_, u) => {
                const values = c.values.subarray(start(u), start(u + 1));
                const norm = NORMALISERS[c.kind](values, domain);
                let badge: string | null = null;
                let ring: string | undefined;
                if (showForward) {
                  const a = pc.a[u]!;
                  badge =
                    c.kind === 'output'
                      ? view.phase === 'loss'
                        ? `L ${formatCompact(probe.loss)}`
                        : `p ${a.toFixed(2)}`
                      : formatCompact(a);
                  const k = lutIndex(a * norm.scale + norm.offset) * 3;
                  ring = `rgb(${lut[k]} ${lut[k + 1]} ${lut[k + 2]})`;
                } else if (showGrad) {
                  const gv = gradValues![u]!;
                  badge = formatCompact(gv);
                  ring = rgbCss(
                    gv >= 0 ? POSITIVE : NEGATIVE,
                    0.25 + 0.75 * (Math.abs(gv) / gradScale),
                  );
                }
                const dim = probe && !showForward && !showGrad;
                return (
                  <div
                    key={u}
                    className={`neuron${c.kind === 'output' ? ' output' : ''}${dim ? ' dim' : ''}${ring ? ' probed' : ''}`}
                    style={{
                      left: colX(ci),
                      top: nodeY(c, u),
                      width: TILE,
                      height: TILE,
                      ...(ring && { boxShadow: `0 0 0 3px ${ring}` }),
                    }}
                    onMouseMove={(e) =>
                      setHover({ kind: 'neuron', column: ci, unit: u, x: e.clientX, y: e.clientY })
                    }
                    onMouseLeave={() => setHover(null)}
                  >
                    <Heatmap values={values} gridSize={g} norm={norm} />
                    {probe && (
                      <span
                        className="probe-marker"
                        style={{
                          left: ((probe.x[0]! + domain) / (2 * domain)) * TILE,
                          top: ((domain - probe.x[1]!) / (2 * domain)) * TILE,
                        }}
                      />
                    )}
                    {badge && <span className="probe-value">{badge}</span>}
                    {c.kind === 'input' && <span className="neuron-tag">{`x${u + 1}`}</span>}
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
      {hover && <HoverCard hover={hover} snapshot={snapshot} probe={probe} />}
    </div>
  );
}

function HoverCard({
  hover,
  snapshot,
  probe,
}: {
  hover: Hover;
  snapshot: Snapshot;
  probe: ProbeTrace | null;
}) {
  const { columns, weights, paramHistory: ph } = snapshot;
  const n = ph.epochs.length;

  if (hover.kind === 'edge') {
    const { layer: k, i, j } = hover;
    const layer = weights[k];
    if (!layer || j >= layer.outFeatures || i >= layer.inFeatures) return null;
    const idx = i * layer.outFeatures + j;
    const title = `${neuronLabel(columns[k]!, k, i)} → ${neuronLabel(columns[k + 1]!, k + 1, j)}`;
    const rows: [string, string][] = [
      ['Weight', formatPrecise(layer.W[idx]!)],
      ['∂L/∂w (all training data)', formatPrecise(layer.gradW[idx]!)],
    ];
    if (probe) {
      rows.push(['w · a (this point)', formatPrecise(layer.W[idx]! * probe.columns[k]!.a[i]!)]);
      rows.push(['∂L/∂w (this point)', formatPrecise(probe.dW[k]![idx]!)]);
    }
    const history = ph.layers[k]?.W.subarray(idx * n, (idx + 1) * n);
    return (
      <Tooltip x={hover.x} y={hover.y} title={title} rows={rows}>
        {history && <Sparkline values={history} epochs={ph.epochs} />}
      </Tooltip>
    );
  }

  const { column: ci, unit: u } = hover;
  const c = columns[ci];
  if (!c || u >= c.units) return null;
  const pc = probe?.columns[ci];
  const rows: [string, string][] = [];
  let history: Float32Array | undefined;
  if (ci > 0) {
    const layer = weights[ci - 1]!;
    rows.push(['Bias', formatPrecise(layer.b[u]!)]);
    rows.push(['∂L/∂b (all training data)', formatPrecise(layer.gradB[u]!)]);
    history = ph.layers[ci - 1]?.b.subarray(u * n, (u + 1) * n);
  }
  if (pc) {
    if (c.kind === 'input') {
      rows.push(['Value (this point)', formatPrecise(pc.a[u]!)]);
      rows.push(['∂L/∂x (this point)', formatPrecise(pc.dA[u]!)]);
    } else if (c.kind === 'output') {
      rows.push(['Logit z', formatPrecise(pc.z[u]!)]);
      rows.push(['p = σ(z)', formatPrecise(pc.a[u]!)]);
      rows.push(['Label', String(probe!.label)]);
      rows.push(['Loss', formatPrecise(probe!.loss)]);
      rows.push(['∂L/∂z = p − y', formatPrecise(pc.dZ[u]!)]);
    } else {
      rows.push(['z (pre-activation)', formatPrecise(pc.z[u]!)]);
      rows.push(['a (activation)', formatPrecise(pc.a[u]!)]);
      rows.push(['∂L/∂a', formatPrecise(pc.dA[u]!)]);
      rows.push(['∂L/∂z', formatPrecise(pc.dZ[u]!)]);
    }
  }
  const title =
    c.kind === 'input' || c.kind === 'output'
      ? neuronLabel(c, ci, u)
      : `${neuronLabel(c, ci, u)} (${c.kind})`;
  if (rows.length === 0)
    rows.push(['Input', `the ${u === 0 ? 'horizontal' : 'vertical'} coordinate`]);
  return (
    <Tooltip x={hover.x} y={hover.y} title={title} rows={rows}>
      {history && <Sparkline values={history} epochs={ph.epochs} />}
    </Tooltip>
  );
}
