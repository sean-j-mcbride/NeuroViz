import { type MouseEvent, useEffect, useMemo, useRef, useState } from 'react';
import type { TimelineSnapshot } from '../worker';
import { cssVar, formatPow10, prepareCanvas } from './canvas';
import { depthColour, formatPrecise } from './colour';
import { useElementWidth, usePrefersDark } from './hooks';
import { layerLabel } from './labels';
import { Tooltip } from './Tooltip';

const HEIGHT = 200;
const PAD = { left: 44, right: 10, top: 8, bottom: 20 };

interface GradientNormsProps {
  timeline: TimelineSnapshot;
}

/** Decade range [lo, hi] (as exponents) covering every positive value. */
function decadeRange(series: Float32Array[]): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const s of series) {
    for (const v of s) {
      if (!(v > 0) || !Number.isFinite(v)) continue;
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
  }
  if (!Number.isFinite(lo)) return [-6, 0];
  const [a, b] = [Math.floor(Math.log10(lo)), Math.ceil(Math.log10(hi))];
  return a === b ? [a - 1, b] : [a, b];
}

interface Geometry {
  width: number;
  lo: number;
  hi: number;
  lastEpoch: number;
  xOf: (epoch: number) => number;
  yOf: (v: number) => number;
}

function geometry({ epochs, layers }: TimelineSnapshot, width: number): Geometry {
  const [lo, hi] = decadeRange(layers.map((l) => l.gradRms));
  const plotW = width - PAD.left - PAD.right;
  const plotH = HEIGHT - PAD.top - PAD.bottom;
  const lastEpoch = epochs.length > 0 ? epochs[epochs.length - 1]! : 0;
  return {
    width,
    lo,
    hi,
    lastEpoch,
    xOf: (e) => PAD.left + (lastEpoch === 0 ? 0 : (e / lastEpoch) * plotW),
    yOf: (v) => PAD.top + ((hi - Math.log10(v)) / (hi - lo)) * plotH,
  };
}

function draw(
  canvas: HTMLCanvasElement,
  timeline: TimelineSnapshot,
  geo: Geometry,
  dark: boolean,
  hoverT: number | null,
): void {
  const { width, lo, hi, lastEpoch, xOf, yOf } = geo;
  const ctx = prepareCanvas(canvas, width, HEIGHT);
  if (!ctx) return;
  const { epochs, layers } = timeline;
  const n = epochs.length;
  const series = layers.map((l) => l.gradRms);

  ctx.font = '10px system-ui, sans-serif';
  ctx.fillStyle = cssVar(canvas, '--muted');
  ctx.strokeStyle = cssVar(canvas, '--border');
  ctx.lineWidth = 1;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const stepDecades = Math.max(1, Math.ceil((hi - lo) / 5));
  for (let e = hi; e >= lo; e -= stepDecades) {
    const y = yOf(10 ** e);
    ctx.beginPath();
    ctx.moveTo(PAD.left, y);
    ctx.lineTo(width - PAD.right, y);
    ctx.stroke();
    ctx.fillText(formatPow10(e), PAD.left - 4, y);
  }
  ctx.textBaseline = 'top';
  ctx.textAlign = 'left';
  ctx.fillText('0', PAD.left, HEIGHT - PAD.bottom + 5);
  ctx.textAlign = 'right';
  ctx.fillText(`epoch ${lastEpoch}`, width - PAD.right, HEIGHT - PAD.bottom + 5);

  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  series.forEach((s, k) => {
    ctx.strokeStyle = depthColour(k, series.length, dark);
    ctx.beginPath();
    let started = false;
    for (let t = 0; t < n; t++) {
      const v = s[t]!;
      if (!(v > 0)) continue;
      if (started) ctx.lineTo(xOf(epochs[t]!), yOf(v));
      else ctx.moveTo(xOf(epochs[t]!), yOf(v));
      started = true;
    }
    ctx.stroke();
  });

  if (hoverT === null || hoverT >= n) return;
  const x = xOf(epochs[hoverT]!);
  ctx.strokeStyle = cssVar(canvas, '--muted');
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x, PAD.top);
  ctx.lineTo(x, HEIGHT - PAD.bottom);
  ctx.stroke();
  ctx.strokeStyle = cssVar(canvas, '--surface');
  ctx.lineWidth = 2;
  series.forEach((s, k) => {
    const v = s[hoverT]!;
    if (!(v > 0)) return;
    ctx.fillStyle = depthColour(k, series.length, dark);
    ctx.beginPath();
    ctx.arc(x, yOf(v), 4, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  });
}

/**
 * Gradient size (RMS of ∂L/∂W over the training set) per dense layer over the
 * run, on a log axis. Early layers are lighter, later layers darker.
 */
export function GradientNorms({ timeline }: GradientNormsProps) {
  const wrap = useRef<HTMLDivElement>(null);
  const ref = useRef<HTMLCanvasElement>(null);
  const width = useElementWidth(wrap);
  const dark = usePrefersDark();
  const [hover, setHover] = useState<{ t: number; x: number; y: number } | null>(null);
  const { epochs, layers } = timeline;
  const n = epochs.length;
  const geo = useMemo(() => geometry(timeline, width), [timeline, width]);
  const hoverT = hover?.t ?? null;

  useEffect(() => {
    if (ref.current && width > 0) draw(ref.current, timeline, geo, dark, hoverT);
  }, [timeline, geo, width, dark, hoverT]);

  const onMove = (e: MouseEvent<HTMLCanvasElement>) => {
    if (n === 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const plotW = width - PAD.left - PAD.right;
    const epoch = ((e.clientX - rect.left - PAD.left) / plotW) * geo.lastEpoch;
    let t = 0;
    for (let i = 1; i < n; i++) {
      if (Math.abs(epochs[i]! - epoch) < Math.abs(epochs[t]! - epoch)) t = i;
    }
    setHover({ t, x: e.clientX, y: e.clientY });
  };

  return (
    <div ref={wrap} className="gradient-norms">
      <canvas
        ref={ref}
        style={{ width: '100%', height: HEIGHT }}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        role="img"
        aria-label="Gradient size per layer over training, log scale"
      />
      {hover && hover.t < n && (
        <Tooltip
          x={hover.x}
          y={hover.y}
          title={`Epoch ${epochs[hover.t]}`}
          rows={layers.map((l, k) => [
            layerLabel(k, layers.length),
            formatPrecise(l.gradRms[hover.t]!),
          ])}
        />
      )}
    </div>
  );
}
