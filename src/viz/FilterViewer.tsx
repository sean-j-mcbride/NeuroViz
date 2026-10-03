import { type MouseEvent, useEffect, useMemo, useRef, useState } from 'react';
import type { ConvFilters, ConvLayerSpec } from '../worker';
import { formatPrecise, lutIndex } from './colour';
import { useHeatmapLut } from './hooks';
import { type FilterStats, filterStats, kernelOf, rowTitle } from './layerViews';
import { Tooltip } from './Tooltip';

const GAP = 1;
/** Largest CSS size of one kernel tile. */
const TILE_PX = 56;

export type FilterScale = 'tile' | 'shared';

interface Pick {
  filter: number;
  channel: number;
}

/** Tile layout: a first layer's filters wrap in rows; deeper ones are filters (rows) × input channels (columns). */
function layoutOf({ inChannels, outChannels }: ConvFilters) {
  const wrap = inChannels === 1;
  const cols = wrap ? Math.min(outChannels, 8) : inChannels;
  const rows = wrap ? Math.ceil(outChannels / cols) : outChannels;
  const at = (filter: number, channel: number) =>
    wrap ? { col: filter % cols, row: Math.floor(filter / cols) } : { col: channel, row: filter };
  const pickAt = (col: number, row: number): Pick | null => {
    const filter = wrap ? row * cols + col : row;
    const channel = wrap ? 0 : col;
    return filter < outChannels && channel < inChannels ? { filter, channel } : null;
  };
  return { wrap, cols, rows, at, pickAt };
}

/**
 * One conv layer's learned kernels, as k×k tiles in the sign colours (blue
 * positive, orange negative). Every filter has one kernel per input channel.
 */
export function FilterGrid({
  layer,
  scale,
  selected,
  onSelect,
}: {
  layer: ConvFilters;
  scale: FilterScale;
  selected: Pick | null;
  onSelect(p: Pick | null): void;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const lut = useHeatmapLut();
  const { kernel: k, inChannels, outChannels } = layer;
  const layout = useMemo(() => layoutOf(layer), [layer]);
  const { wrap, cols, rows, at, pickAt } = layout;
  const cell = k + GAP;
  const width = cols * cell - GAP;
  const height = rows * cell - GAP;
  const stats = useMemo(() => filterStats(layer), [layer]);
  const [hover, setHover] = useState<(Pick & { x: number; y: number }) | null>(null);

  useEffect(() => {
    const ctx = ref.current?.getContext('2d');
    if (!ctx) return;
    const img = ctx.createImageData(width, height);
    const shared = Math.max(...stats.map((s) => Math.max(-s.min, s.max))) || 1;
    for (let f = 0; f < outChannels; f++) {
      const range = scale === 'shared' ? shared : Math.max(-stats[f]!.min, stats[f]!.max) || 1;
      for (let c = 0; c < inChannels; c++) {
        const { col, row } = layout.at(f, c);
        const kern = kernelOf(layer, f, c);
        for (let i = 0; i < k * k; i++) {
          const li = lutIndex(kern[i]! / range) * 3;
          const o = ((row * cell + Math.floor(i / k)) * width + col * cell + (i % k)) * 4;
          img.data[o] = lut[li]!;
          img.data[o + 1] = lut[li + 1]!;
          img.data[o + 2] = lut[li + 2]!;
          img.data[o + 3] = 255;
        }
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [layer, layout, scale, lut, stats, width, height, cell, k, inChannels, outChannels]);

  const pickOf = (e: MouseEvent<HTMLCanvasElement>): Pick | null => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * width;
    const y = ((e.clientY - r.top) / r.height) * height;
    if (x % cell >= k || y % cell >= k) return null;
    return pickAt(Math.floor(x / cell), Math.floor(y / cell));
  };

  const box = (p: Pick) => {
    const { col, row } = at(p.filter, p.channel);
    // A deeper layer's selection outlines the whole filter (its row of kernels).
    const span = wrap ? 1 : inChannels;
    const left = wrap ? col : 0;
    return {
      left: `${((left * cell) / width) * 100}%`,
      top: `${((row * cell) / height) * 100}%`,
      width: `${((span * cell - GAP) / width) * 100}%`,
      height: `${(k / height) * 100}%`,
    };
  };

  const h = hover && stats[hover.filter];
  return (
    <div className="filter-grid-wrap">
      {!wrap && (
        <div className="filter-axes hint">
          Rows: the {outChannels} filters. Columns: the {inChannels} input channels each one reads.
        </div>
      )}
      <div
        className="weight-images filter-grid"
        style={{ aspectRatio: `${width} / ${height}`, maxWidth: cols * TILE_PX }}
      >
        <canvas
          ref={ref}
          width={width}
          height={height}
          role="img"
          aria-label={`${outChannels} filters of ${k} by ${k} weights over ${inChannels} input channel${inChannels > 1 ? 's' : ''}`}
          onMouseMove={(e) => {
            const p = pickOf(e);
            setHover(p && { ...p, x: e.clientX, y: e.clientY });
          }}
          onMouseLeave={() => setHover(null)}
          onClick={(e) => {
            const p = pickOf(e);
            onSelect(
              p && selected?.filter === p.filter && selected.channel === p.channel ? null : p,
            );
          }}
        />
        {selected && selected.filter < outChannels && (
          <div className="weight-selected" style={box(selected)} aria-hidden />
        )}
      </div>
      {hover && h && (
        <Tooltip
          x={hover.x}
          y={hover.y}
          title={`Filter ${hover.filter + 1}${wrap ? '' : `, input channel ${hover.channel + 1}`}`}
          rows={statRows(h)}
        />
      )}
    </div>
  );
}

function statRows(s: FilterStats): [string, string][] {
  return [
    ['Weights', `${formatPrecise(s.min)} to ${formatPrecise(s.max)}`],
    ['RMS', formatPrecise(s.rms)],
    ['Bias', formatPrecise(s.bias)],
  ];
}

/** One kernel enlarged, with its weights written in. */
function KernelDetail({ layer, pick }: { layer: ConvFilters; pick: Pick }) {
  const kern = kernelOf(layer, pick.filter, pick.channel);
  const k = layer.kernel;
  const range = Math.max(...kern.map(Math.abs)) || 1;
  const lut = useHeatmapLut();
  return (
    <table className="kernel-table" aria-label={`Weights of filter ${pick.filter + 1}`}>
      <tbody>
        {Array.from({ length: k }, (_, y) => (
          <tr key={y}>
            {Array.from({ length: k }, (_, x) => {
              const v = kern[y * k + x]!;
              const li = lutIndex(v / range) * 3;
              const strong = Math.abs(v / range) > 0.55;
              return (
                <td
                  key={x}
                  style={{ background: `rgb(${lut[li]} ${lut[li + 1]} ${lut[li + 2]})` }}
                  className={strong ? 'strong' : undefined}
                >
                  {v.toFixed(2)}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** The Weights panel for a CNN: pick a conv layer, see its filters as images or a table. */
export function FilterViewer({
  filters,
  rows,
}: {
  filters: ConvFilters[];
  rows: readonly ConvLayerSpec[];
}) {
  const [layerIndex, setLayerIndex] = useState(0);
  const [scale, setScale] = useState<FilterScale>('tile');
  const [asTable, setAsTable] = useState(false);
  const [selected, setSelected] = useState<Pick | null>(null);
  const li = Math.min(layerIndex, filters.length - 1);
  const layer = filters[li]!;
  const stats = useMemo(() => (asTable ? filterStats(layer) : []), [asTable, layer]);
  const sel =
    selected && selected.filter < layer.outChannels && selected.channel < layer.inChannels
      ? selected
      : null;
  const title = (i: number) => rowTitle(rows, filters[i]!.row);

  return (
    <>
      <div className="weights-options">
        {filters.length > 1 && (
          <div className="segmented" role="group" aria-label="Conv layer">
            {filters.map((_, i) => (
              <button
                key={i}
                type="button"
                aria-pressed={i === li}
                onClick={() => {
                  setLayerIndex(i);
                  setSelected(null);
                }}
              >
                {title(i)}
              </button>
            ))}
          </div>
        )}
        <div className="segmented" role="group" aria-label="Show as">
          <button type="button" aria-pressed={!asTable} onClick={() => setAsTable(false)}>
            Images
          </button>
          <button type="button" aria-pressed={asTable} onClick={() => setAsTable(true)}>
            Table
          </button>
        </div>
        {!asTable && (
          <div className="segmented" role="group" aria-label="Colour scale">
            <button type="button" aria-pressed={scale === 'tile'} onClick={() => setScale('tile')}>
              Each filter
            </button>
            <button
              type="button"
              aria-pressed={scale === 'shared'}
              onClick={() => setScale('shared')}
            >
              Shared scale
            </button>
          </div>
        )}
      </div>
      {asTable ? (
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">Filter</th>
                <th scope="col">Lowest weight</th>
                <th scope="col">Highest weight</th>
                <th scope="col">RMS</th>
                <th scope="col">Bias</th>
              </tr>
            </thead>
            <tbody>
              {stats.map((s, f) => (
                <tr key={f}>
                  <th scope="row">{f + 1}</th>
                  <td>{formatPrecise(s.min)}</td>
                  <td>{formatPrecise(s.max)}</td>
                  <td>{formatPrecise(s.rms)}</td>
                  <td>{formatPrecise(s.bias)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <>
          <FilterGrid layer={layer} scale={scale} selected={sel} onSelect={setSelected} />
          {sel && (
            <div className="weight-detail">
              <KernelDetail layer={layer} pick={sel} />
              <div>
                <h3>
                  {title(li)}, filter {sel.filter + 1}
                  {layer.inChannels > 1 && `, input channel ${sel.channel + 1}`}
                </h3>
                <p className="hint">
                  This {layer.kernel} × {layer.kernel} kernel slides over{' '}
                  {layer.inChannels > 1
                    ? `channel ${sel.channel + 1} of the layer below`
                    : 'the image'}
                  : at each position the weights multiply the values under them and add up
                  {layer.inChannels > 1 ? ' (with the other channels’ kernels)' : ''}, plus the bias{' '}
                  {formatPrecise(layer.b[sel.filter]!)}.{' '}
                  {layer.inChannels > 1
                    ? 'Blue weights respond to that channel firing there, orange ones are held back by it.'
                    : 'Blue weights reward ink there, orange ones penalise it.'}
                </p>
                <button type="button" onClick={() => setSelected(null)}>
                  Close
                </button>
              </div>
            </div>
          )}
        </>
      )}
    </>
  );
}
