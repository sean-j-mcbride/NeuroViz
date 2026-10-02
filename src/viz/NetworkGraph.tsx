import { type ReactNode, useRef } from 'react';
import type { NeuronColumn, Snapshot } from '../worker';
import { NORMALISERS } from './colour';
import { Heatmap } from './Heatmap';
import { useElementWidth } from './hooks';

const TILE = 40;
const GAP = 10;
const HEADER = 76;
const PAD_X = 28;
const MIN_COLUMN_SPACING = 90;

interface NetworkGraphProps {
  snapshot: Snapshot;
  /** Optional controls rendered above each column (e.g. the architecture builder). */
  renderHeader?: (column: number, kind: NeuronColumn['kind']) => ReactNode;
}

function columnLabel(c: NeuronColumn, i: number): string {
  return c.kind === 'input' ? 'Input' : c.kind === 'output' ? 'Output' : `Hidden ${i}`;
}

function neuronLabel(c: NeuronColumn, column: number, unit: number): string {
  if (c.kind === 'input') return `x${unit + 1}`;
  if (c.kind === 'output') return 'output';
  return `h${column}.${unit + 1}`;
}

/** Edge width ∝ |weight|, clamped so huge weights don't swamp the picture. */
function strokeWidth(w: number): number {
  return Math.min(6, 0.3 + Math.abs(w) * 1.2);
}

/**
 * The network as columns of neurons. Each neuron is drawn as a mini heatmap of
 * its value across the input plane; edges (SVG) show weights — thickness =
 * |w|, colour = sign.
 */
export function NetworkGraph({ snapshot, renderHeader }: NetworkGraphProps) {
  const ref = useRef<HTMLDivElement>(null);
  const measured = useElementWidth(ref);
  const { columns, weights, gridSize: g, domain } = snapshot;

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
    for (let i = 0; i < layer.inFeatures; i++) {
      const y1 = nodeY(from, i) + TILE / 2;
      for (let j = 0; j < layer.outFeatures; j++) {
        const w = layer.W[i * layer.outFeatures + j]!;
        const y2 = nodeY(to, j) + TILE / 2;
        edges.push(
          <path
            key={`${k}-${i}-${j}`}
            d={`M${x1},${y1} C${xm},${y1} ${xm},${y2} ${x2},${y2}`}
            className={w >= 0 ? 'edge pos' : 'edge neg'}
            strokeWidth={strokeWidth(w)}
          >
            <title>
              {`${neuronLabel(from, k, i)} → ${neuronLabel(to, k + 1, j)}: weight ${w.toFixed(3)}`}
            </title>
          </path>,
        );
      }
    }
  });

  return (
    <div ref={ref} className="network-graph">
      <div className="network-canvas" style={{ width, height }}>
        <svg width={width} height={height} className="network-edges" aria-hidden>
          {edges}
        </svg>
        {columns.map((c, ci) => {
          const start = (u: number) => g * g * u;
          const bias = ci > 0 ? weights[ci - 1]!.b : null;
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
                const b = bias?.[u];
                return (
                  <div
                    key={u}
                    className={`neuron ${c.kind === 'output' ? 'output' : ''}`}
                    style={{ left: colX(ci), top: nodeY(c, u), width: TILE, height: TILE }}
                  >
                    <Heatmap
                      values={values}
                      gridSize={g}
                      norm={NORMALISERS[c.kind](values, domain)}
                      title={
                        b === undefined
                          ? neuronLabel(c, ci, u)
                          : `${neuronLabel(c, ci, u)} (${c.kind}), bias ${b.toFixed(3)}`
                      }
                    />
                    {c.kind === 'input' && <span className="neuron-tag">{`x${u + 1}`}</span>}
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
}
