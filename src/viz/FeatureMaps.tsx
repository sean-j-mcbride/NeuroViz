import { type MouseEvent, useEffect, useMemo, useRef, useState } from 'react';
import type { Shape } from '../engine';
import { type ConvLayerSpec, type FeatureMap, MNIST_IMAGE_SHAPE } from '../worker';
import { type Normalisation, formatPrecise, lutIndex } from './colour';
import { useHeatmapLut } from './hooks';
import { channelStats, mapNormalisation, rowTitle } from './layerViews';
import { type Rect, describeRect, receptiveField } from './receptiveField';
import { Tooltip } from './Tooltip';

/** Target CSS size of one channel's map; small maps get bigger pixels. */
const TILE_PX = 48;
const GAP = 4;
const MAX_COLS = 8;

export type MapScale = 'layer' | 'map';

/** A pixel of a feature map: row index in the network, channel, and position. */
export interface MapPick {
  row: number;
  channel: number;
  y: number;
  x: number;
}

function geometry([c, h, w]: Shape) {
  const cell = Math.max(1, Math.ceil(TILE_PX / h!));
  const cols = Math.min(c!, MAX_COLS);
  const tileW = w! * cell;
  const tileH = h! * cell;
  return {
    cell,
    cols,
    tileW,
    tileH,
    width: cols * (tileW + GAP) - GAP,
    height: Math.ceil(c! / cols) * (tileH + GAP) - GAP,
    origin: (ch: number) => ({
      left: (ch % cols) * (tileW + GAP),
      top: Math.floor(ch / cols) * (tileH + GAP),
    }),
  };
}

/** One layer's output: every channel's map, tiled, with optional outlines. */
function MapLayer({
  title,
  shape,
  data,
  normalise,
  field,
  picked,
  onPick,
  onHover,
}: {
  title: string;
  shape: Shape;
  data: Float32Array;
  normalise: (values: Float32Array) => Normalisation;
  /** Outlined on every channel: the part of this layer the picked pixel sees. */
  field: Rect | null;
  /** The picked pixel, if it is in this layer. */
  picked: { channel: number; y: number; x: number } | null;
  onPick?: (channel: number, y: number, x: number) => void;
  onHover(h: { channel: number; y: number; x: number; px: number; py: number } | null): void;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const lut = useHeatmapLut();
  const g = useMemo(() => geometry(shape), [shape]);
  const [c, h, w] = shape as [number, number, number];

  useEffect(() => {
    const ctx = ref.current?.getContext('2d');
    if (!ctx) return;
    const img = ctx.createImageData(g.width, g.height);
    const area = h * w;
    for (let ch = 0; ch < c; ch++) {
      const values = data.subarray(ch * area, (ch + 1) * area);
      const { scale, offset } = normalise(values);
      const { left, top } = g.origin(ch);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const li = lutIndex(values[y * w + x]! * scale + offset) * 3;
          for (let dy = 0; dy < g.cell; dy++) {
            let o = ((top + y * g.cell + dy) * g.width + left + x * g.cell) * 4;
            for (let dx = 0; dx < g.cell; dx++, o += 4) {
              img.data[o] = lut[li]!;
              img.data[o + 1] = lut[li + 1]!;
              img.data[o + 2] = lut[li + 2]!;
              img.data[o + 3] = 255;
            }
          }
        }
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [data, normalise, lut, g, c, h, w]);

  const at = (e: MouseEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * g.width;
    const py = ((e.clientY - r.top) / r.height) * g.height;
    const col = Math.floor(px / (g.tileW + GAP));
    const row = Math.floor(py / (g.tileH + GAP));
    const channel = row * g.cols + col;
    const { left, top } = g.origin(channel);
    const x = Math.floor((px - left) / g.cell);
    const y = Math.floor((py - top) / g.cell);
    return channel < c && x >= 0 && x < w && y >= 0 && y < h ? { channel, y, x } : null;
  };

  const box = (ch: number, r: Rect) => {
    const { left, top } = g.origin(ch);
    return {
      left: `${((left + r.x0 * g.cell) / g.width) * 100}%`,
      top: `${((top + r.y0 * g.cell) / g.height) * 100}%`,
      width: `${(((r.x1 - r.x0 + 1) * g.cell) / g.width) * 100}%`,
      height: `${(((r.y1 - r.y0 + 1) * g.cell) / g.height) * 100}%`,
    };
  };

  return (
    <figure className="map-layer">
      <figcaption>
        {title} <span className="hint">{shape.join(' × ')}</span>
      </figcaption>
      <div
        className="map-canvas"
        style={{ width: g.width, aspectRatio: `${g.width} / ${g.height}` }}
      >
        <canvas
          ref={ref}
          width={g.width}
          height={g.height}
          className={onPick ? 'pickable' : undefined}
          role="img"
          aria-label={`${title}: ${c} map${c > 1 ? 's' : ''} of ${h} by ${w}`}
          onMouseMove={(e) => {
            const p = at(e);
            onHover(p && { ...p, px: e.clientX, py: e.clientY });
          }}
          onMouseLeave={() => onHover(null)}
          onClick={(e) => {
            const p = at(e);
            if (p && onPick) onPick(p.channel, p.y, p.x);
          }}
        />
        {field &&
          Array.from({ length: c }, (_, ch) => (
            <div key={ch} className="map-field" style={box(ch, field)} aria-hidden />
          ))}
        {picked && (
          <div
            className="map-picked"
            style={box(picked.channel, { y0: picked.y, y1: picked.y, x0: picked.x, x1: picked.x })}
            aria-hidden
          />
        )}
      </div>
    </figure>
  );
}

const INPUT_NORMALISATION: Normalisation = { scale: 1, offset: 0 };

/**
 * What each image layer makes of one input: every channel's feature map, layer
 * by layer. Click a pixel to see its receptive field: the part of the input
 * (and of every layer before it) that it is computed from.
 */
export function FeatureMaps({
  rows,
  maps,
  input,
}: {
  rows: readonly ConvLayerSpec[];
  maps: FeatureMap[];
  /** The 28×28 input in [0, 1]. */
  input: Float32Array;
}) {
  const [asTable, setAsTable] = useState(false);
  const [scale, setScale] = useState<MapScale>('layer');
  const [pick, setPick] = useState<MapPick | null>(null);
  const [hover, setHover] = useState<
    (MapPick & { px: number; py: number; title: string; value: number }) | null
  >(null);
  // A pick only makes sense for the network it was made on.
  const valid = pick && pick.row < maps.length ? pick : null;

  useEffect(() => {
    if (!valid) return;
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setPick(null);
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [valid]);

  const shapes = useMemo(() => maps.map((m) => m.shape), [maps]);
  const field =
    valid && receptiveField(rows, shapes, MNIST_IMAGE_SHAPE, valid.row, valid.y, valid.x);
  const normalisers = useMemo(
    () =>
      maps.map((m) => {
        if (scale === 'map') return (v: Float32Array) => mapNormalisation(rows, m.row, v);
        const whole = mapNormalisation(rows, m.row, m.data);
        return () => whole;
      }),
    [maps, rows, scale],
  );
  const inputNormalise = useMemo(() => () => INPUT_NORMALISATION, []);
  const stats = useMemo(
    () => (asTable ? maps.map((m) => channelStats(m.data, m.shape)) : []),
    [asTable, maps],
  );

  const valueAt = (m: FeatureMap, ch: number, y: number, x: number) =>
    m.data[(ch * m.shape[1] + y) * m.shape[2] + x]!;
  const describePick = (p: MapPick, f: { input: Rect }) =>
    `${rowTitle(rows, p.row)}, channel ${p.channel + 1}, at row ${p.y}, column ${p.x} ` +
    `(value ${formatPrecise(valueAt(maps[p.row]!, p.channel, p.y, p.x))}) is computed from input ` +
    describeRect(f.input);

  return (
    <>
      <div className="panel-toolbar">
        <h2>Feature maps</h2>
        <div className="weights-options">
          <div className="segmented" role="group" aria-label="Show maps as">
            <button type="button" aria-pressed={!asTable} onClick={() => setAsTable(false)}>
              Images
            </button>
            <button type="button" aria-pressed={asTable} onClick={() => setAsTable(true)}>
              Table
            </button>
          </div>
          {!asTable && (
            <div className="segmented" role="group" aria-label="Map colour scale">
              <button
                type="button"
                aria-pressed={scale === 'layer'}
                onClick={() => setScale('layer')}
              >
                Each layer
              </button>
              <button type="button" aria-pressed={scale === 'map'} onClick={() => setScale('map')}>
                Each map
              </button>
            </div>
          )}
        </div>
      </div>
      {asTable ? (
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">Layer</th>
                <th scope="col">Channel</th>
                <th scope="col">Mean</th>
                <th scope="col">Max</th>
                <th scope="col">Zero</th>
              </tr>
            </thead>
            <tbody>
              {stats.flatMap((layer, i) =>
                layer.map((s, ch) => (
                  <tr key={`${i}.${ch}`}>
                    <th scope="row">{ch === 0 ? rowTitle(rows, maps[i]!.row) : ''}</th>
                    <td>{ch + 1}</td>
                    <td>{formatPrecise(s.mean)}</td>
                    <td>{formatPrecise(s.max)}</td>
                    <td>{(s.zeros * 100).toFixed(0)} %</td>
                  </tr>
                )),
              )}
            </tbody>
          </table>
        </div>
      ) : (
        <>
          <p className="hint map-readout" aria-live="polite">
            {valid && field ? (
              <>
                {describePick(valid, field)}.{' '}
                <button type="button" className="link" onClick={() => setPick(null)}>
                  Clear
                </button>{' '}
                (or press Escape)
              </>
            ) : (
              'Click any pixel of a feature map to outline the part of the input it is computed from: its receptive field.'
            )}
          </p>
          <div className="feature-maps">
            <MapLayer
              title="Input"
              shape={MNIST_IMAGE_SHAPE}
              data={input}
              normalise={inputNormalise}
              field={field ? field.input : null}
              picked={null}
              onHover={(hv) =>
                setHover(
                  hv && {
                    ...hv,
                    row: -1,
                    title: 'Input',
                    value: input[hv.y * 28 + hv.x]!,
                  },
                )
              }
            />
            {maps.map((m, i) => (
              <MapLayer
                key={i}
                title={rowTitle(rows, m.row)}
                shape={m.shape}
                data={m.data}
                normalise={normalisers[i]!}
                field={field && valid && i < valid.row ? field.maps[i]! : null}
                picked={valid && valid.row === i ? valid : null}
                onPick={(channel, y, x) =>
                  setPick((p) =>
                    p && p.row === i && p.channel === channel && p.y === y && p.x === x
                      ? null
                      : { row: i, channel, y, x },
                  )
                }
                onHover={(hv) =>
                  setHover(
                    hv && {
                      ...hv,
                      row: i,
                      title: `${rowTitle(rows, m.row)}, channel ${hv.channel + 1}`,
                      value: valueAt(m, hv.channel, hv.y, hv.x),
                    },
                  )
                }
              />
            ))}
          </div>
        </>
      )}
      {hover && (
        <Tooltip
          x={hover.px}
          y={hover.py}
          title={hover.title}
          rows={[
            ['Position', `row ${hover.y}, column ${hover.x}`],
            ['Value', formatPrecise(hover.value)],
          ]}
        />
      )}
    </>
  );
}
