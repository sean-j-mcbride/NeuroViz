import { type MouseEvent, useEffect, useMemo, useRef, useState } from 'react';
import { MNIST_PIXELS, MNIST_SIDE } from '../data';
import type { MnistSnapshot } from '../worker';
import { formatPrecise, lutIndex } from './colour';
import { topUnits, unitStats } from './mnistResults';
import { useHeatmapLut } from './hooks';
import { Tooltip } from './Tooltip';

/** How many of the most active units to outline for a drawn digit. */
const TOP_ACTIVE = 8;
const GAP = 2;

export type WeightScale = 'tile' | 'shared';

interface WeightImagesProps {
  layer: MnistSnapshot['firstLayer'];
  scale: WeightScale;
  /** First-hidden-layer activations for a drawn digit, to outline the most active units. */
  hidden?: Float32Array;
  /** With no hidden layer the units are the 10 classes. */
  unitName: (j: number) => string;
  selected: number | null;
  onSelect: (j: number | null) => void;
}

/**
 * The first layer's weights, one 28×28 tile per unit: what each unit looks for
 * in the image. Blue pixels add to the unit's input when inked, orange ones
 * subtract. All tiles go on one canvas (drawn at native size, scaled up crisply).
 */
export function WeightImages({
  layer,
  scale,
  hidden,
  unitName,
  selected,
  onSelect,
}: WeightImagesProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const lut = useHeatmapLut();
  const units = layer.outFeatures;
  const cols = Math.min(units, units > 64 ? 16 : units > 16 ? 8 : units <= 10 ? 5 : 4);
  const rows = Math.ceil(units / cols);
  const cell = MNIST_SIDE + GAP;
  const width = cols * cell - GAP;
  const height = rows * cell - GAP;
  const stats = useMemo(() => unitStats(layer), [layer]);
  const active = useMemo(() => (hidden ? topUnits(hidden, TOP_ACTIVE) : []), [hidden]);
  const [hover, setHover] = useState<{ unit: number; x: number; y: number } | null>(null);

  useEffect(() => {
    const ctx = ref.current?.getContext('2d');
    if (!ctx) return;
    const img = ctx.createImageData(width, height);
    const shared = Math.max(...stats.map((s) => Math.max(-s.min, s.max))) || 1;
    for (let j = 0; j < units; j++) {
      const range = scale === 'shared' ? shared : Math.max(-stats[j]!.min, stats[j]!.max) || 1;
      const ox = (j % cols) * cell;
      const oy = Math.floor(j / cols) * cell;
      for (let p = 0; p < MNIST_PIXELS; p++) {
        const k = lutIndex(layer.W[p * units + j]! / range) * 3;
        const o = ((oy + Math.floor(p / MNIST_SIDE)) * width + ox + (p % MNIST_SIDE)) * 4;
        img.data[o] = lut[k]!;
        img.data[o + 1] = lut[k + 1]!;
        img.data[o + 2] = lut[k + 2]!;
        img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [layer, scale, lut, stats, units, cols, cell, width, height]);

  const unitAt = (e: MouseEvent<HTMLCanvasElement>): number | null => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * width;
    const y = ((e.clientY - r.top) / r.height) * height;
    const j = Math.floor(y / cell) * cols + Math.floor(x / cell);
    return j >= 0 && j < units && x % cell < MNIST_SIDE && y % cell < MNIST_SIDE ? j : null;
  };

  const box = (j: number) => ({
    left: `${(((j % cols) * cell) / width) * 100}%`,
    top: `${((Math.floor(j / cols) * cell) / height) * 100}%`,
    width: `${(MNIST_SIDE / width) * 100}%`,
    height: `${(MNIST_SIDE / height) * 100}%`,
  });

  const h = hover && stats[hover.unit];
  return (
    <div className="weight-images" style={{ aspectRatio: `${width} / ${height}` }}>
      <canvas
        ref={ref}
        width={width}
        height={height}
        role="img"
        aria-label={`First-layer weights of ${units} units as 28 by 28 images`}
        onMouseMove={(e) => {
          const unit = unitAt(e);
          setHover(unit === null ? null : { unit, x: e.clientX, y: e.clientY });
        }}
        onMouseLeave={() => setHover(null)}
        onClick={(e) => {
          const unit = unitAt(e);
          onSelect(unit === selected ? null : unit);
        }}
      />
      {active.map((j, rank) => (
        <div key={j} className="weight-active" style={box(j)} aria-hidden>
          {rank === 0 && <span>most active</span>}
        </div>
      ))}
      {selected !== null && selected < units && (
        <div className="weight-selected" style={box(selected)} aria-hidden />
      )}
      {hover && h && (
        <Tooltip
          x={hover.x}
          y={hover.y}
          title={unitName(hover.unit)}
          rows={[
            ['Weights', `${formatPrecise(h.min)} to ${formatPrecise(h.max)}`],
            ['RMS', formatPrecise(h.rms)],
            ['Bias', formatPrecise(h.bias)],
            ...(hidden
              ? [['For your digit', formatPrecise(hidden[hover.unit]!)] as [string, string]]
              : []),
          ]}
        />
      )}
    </div>
  );
}

/** One unit's weights as a single large tile (scaled to its own range). */
export function WeightTile({
  layer,
  unit,
  size = 168,
}: {
  layer: MnistSnapshot['firstLayer'];
  unit: number;
  size?: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const lut = useHeatmapLut();
  useEffect(() => {
    const ctx = ref.current?.getContext('2d');
    if (!ctx) return;
    const units = layer.outFeatures;
    let range = 0;
    for (let p = 0; p < MNIST_PIXELS; p++)
      range = Math.max(range, Math.abs(layer.W[p * units + unit]!));
    const img = ctx.createImageData(MNIST_SIDE, MNIST_SIDE);
    for (let p = 0; p < MNIST_PIXELS; p++) {
      const k = lutIndex(layer.W[p * units + unit]! / (range || 1)) * 3;
      img.data.set([lut[k]!, lut[k + 1]!, lut[k + 2]!, 255], p * 4);
    }
    ctx.putImageData(img, 0, 0);
  }, [layer, unit, lut]);
  return (
    <canvas
      ref={ref}
      width={MNIST_SIDE}
      height={MNIST_SIDE}
      className="weight-tile"
      style={{ width: size, height: size }}
      aria-hidden
    />
  );
}
