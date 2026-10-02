import { type MouseEvent, useEffect, useMemo, useRef, useState } from 'react';
import type { BinSpec } from '../worker';
import { cssVar, formatPow10, prepareCanvas } from './canvas';
import { LUT_SIZE, sequentialLut } from './colour';
import { useElementWidth, usePrefersDark } from './hooks';
import { Tooltip } from './Tooltip';

const HEIGHT = 64;
const PAD = { left: 40, right: 4, top: 4, bottom: 4 };

interface HistogramTimelineProps {
  /** `[T, bins.count]` row-major fractions: column t is the histogram at `epochs[t]`. */
  hist: Float32Array;
  epochs: Float32Array;
  bins: BinSpec;
  /** Bins are over log10|v| rather than v. */
  log: boolean;
  label: string;
}

/**
 * A distribution over time: x = epoch, y = value bin, stronger colour = more
 * of the layer's weights in that bin. The square root of the fraction is shown so
 * thin tails stay visible.
 */
export function HistogramTimeline({ hist, epochs, bins, log, label }: HistogramTimelineProps) {
  const wrap = useRef<HTMLDivElement>(null);
  const ref = useRef<HTMLCanvasElement>(null);
  const width = useElementWidth(wrap);
  const dark = usePrefersDark();
  const lut = useMemo(() => sequentialLut(dark), [dark]);
  const [hover, setHover] = useState<{ t: number; b: number; x: number; y: number } | null>(null);
  const T = epochs.length;
  const B = bins.count;
  const plotW = width - PAD.left - PAD.right;
  const plotH = HEIGHT - PAD.top - PAD.bottom;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || width === 0 || T === 0) return;
    const ctx = prepareCanvas(canvas, width, HEIGHT);
    if (!ctx) return;

    let max = 0;
    for (const v of hist) max = Math.max(max, v);
    const img = new ImageData(T, B);
    for (let t = 0; t < T; t++) {
      for (let b = 0; b < B; b++) {
        const f = hist[t * B + b]!;
        const k = Math.round(Math.sqrt(max > 0 ? f / max : 0) * (LUT_SIZE - 1)) * 3;
        const p = ((B - 1 - b) * T + t) * 4; // highest bin at the top
        img.data[p] = lut[k]!;
        img.data[p + 1] = lut[k + 1]!;
        img.data[p + 2] = lut[k + 2]!;
        img.data[p + 3] = 255;
      }
    }
    const cells = document.createElement('canvas');
    cells.width = T;
    cells.height = B;
    cells.getContext('2d')?.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(cells, PAD.left, PAD.top, plotW, plotH);

    ctx.strokeStyle = cssVar(canvas, '--border');
    ctx.strokeRect(PAD.left - 0.5, PAD.top - 0.5, plotW + 1, plotH + 1);
    ctx.font = '10px system-ui, sans-serif';
    ctx.fillStyle = cssVar(canvas, '--muted');
    ctx.textAlign = 'right';
    const yOf = (v: number) => PAD.top + ((bins.hi - v) / (bins.hi - bins.lo)) * plotH;
    const ticks = log
      ? [bins.hi - 1, Math.round((bins.lo + bins.hi) / 2), bins.lo]
      : [bins.hi, 0, bins.lo];
    ticks.forEach((v, i) => {
      ctx.textBaseline = i === 0 ? 'top' : i === ticks.length - 1 ? 'bottom' : 'middle';
      ctx.fillText(log ? formatPow10(v) : String(v), PAD.left - 4, yOf(v));
    });

    if (hover) {
      ctx.strokeStyle = cssVar(canvas, '--fg');
      ctx.lineWidth = 1;
      const cw = plotW / T;
      const ch = plotH / B;
      ctx.strokeRect(PAD.left + hover.t * cw, PAD.top + (B - 1 - hover.b) * ch, cw, ch);
    }
  }, [hist, T, B, bins, log, width, plotW, plotH, lut, hover]);

  const onMove = (e: MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const t = Math.floor(((e.clientX - rect.left - PAD.left) / plotW) * T);
    const b = B - 1 - Math.floor(((e.clientY - rect.top - PAD.top) / plotH) * B);
    if (t < 0 || t >= T || b < 0 || b >= B) return setHover(null);
    setHover({ t, b, x: e.clientX, y: e.clientY });
  };

  const binRange = (b: number) => {
    const w = (bins.hi - bins.lo) / B;
    const [a, z] = [bins.lo + b * w, bins.lo + (b + 1) * w];
    if (!log)
      return `${a.toFixed(2)} … ${z.toFixed(2)}${b === 0 || b === B - 1 ? ' (and beyond)' : ''}`;
    return `|v| ${(10 ** a).toExponential(1)} … ${(10 ** z).toExponential(1)}${b === 0 ? ' (and below)' : ''}`;
  };

  return (
    <div ref={wrap} className="histogram-timeline">
      <canvas
        ref={ref}
        style={{ width: '100%', height: HEIGHT }}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        role="img"
        aria-label={label}
      />
      {hover && (
        <Tooltip
          x={hover.x}
          y={hover.y}
          title={`${label} · epoch ${epochs[hover.t]}`}
          rows={[
            ['Range', binRange(hover.b)],
            ['Share of weights', `${((hist[hover.t * B + hover.b] ?? 0) * 100).toFixed(1)} %`],
          ]}
        />
      )}
    </div>
  );
}
