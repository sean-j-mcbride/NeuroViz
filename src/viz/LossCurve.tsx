import { type MouseEvent, useEffect, useRef, useState } from 'react';
import type { LossCurveSummary, LossHistorySnapshot } from '../worker';
import { cssVar } from './canvas';
import { useElementWidth, usePrefersDark } from './hooks';
import { lossAxis } from './lossAxis';
import { type LossAt, lossAt, lossPoints } from './lossPoints';
import { Tooltip } from './Tooltip';

const HEIGHT = 150;
const PAD = { left: 44, right: 8, top: 8, bottom: 20 };

export interface LossSeries {
  losses: LossHistorySnapshot;
  /** Mid-run setting changes, marked with ticks on the x-axis. */
  changes?: readonly { epoch: number; text: string }[];
}

/** How near (in pixels) the pointer must be to a change tick for the tooltip to describe it. */
const TICK_REACH_PX = 4;

interface LossCurveProps extends LossSeries {
  logScale: boolean;
  /** A pinned run, drawn behind in the reference colour (same dashes: solid train, dashed test). */
  reference?: LossSeries | null;
  /**
   * How to name a point on the x-axis (the history's record index), e.g.
   * "epoch 12" (the default: one record per epoch) or "epoch 2.5".
   */
  xName?: (index: number) => string;
  /** A merged bucket's span in the tooltip, e.g. "epochs 4,096–4,103" (the default). */
  xRange?: (from: number, to: number) => string;
  /** Values in the tooltip (default: 4 decimals) and on the axis. */
  formatValue?: (v: number) => string;
  formatTick?: (v: number) => string;
  /** What the chart shows, for screen readers. */
  label?: string;
}

const epochName = (i: number) => `epoch ${i.toLocaleString('en-GB')}`;
const epochRange = (from: number, to: number) =>
  `epochs ${from.toLocaleString('en-GB')}–${to.toLocaleString('en-GB')}`;

function formatAxisTick(v: number): string {
  if (v === 0) return '0';
  return Math.abs(v) >= 0.01 && Math.abs(v) < 1000 ? v.toPrecision(2) : v.toExponential(0);
}

function formatLoss(v: number): string {
  return Number.isFinite(v) ? v.toFixed(4) : String(v);
}

/** "0.3142", or for a merged bucket "0.3142 (0.2901–0.3550, epochs 4,096–4,103)". */
function formatAt(
  at: LossAt | null,
  format: (v: number) => string,
  xRange: (from: number, to: number) => string,
): string {
  if (!at) return '–';
  if (at.from === at.to) return format(at.mean);
  return `${format(at.mean)} (${format(at.min)}–${format(at.max)}, ${xRange(at.from, at.to)})`;
}

const capitalise = (t: string) => t.charAt(0).toUpperCase() + t.slice(1);

/** Plot geometry shared by drawing and hover. */
function layout(width: number, epochs: number) {
  const plotW = width - PAD.left - PAD.right;
  const xOf = (i: number) => PAD.left + (epochs <= 1 ? 0 : (i / (epochs - 1)) * plotW);
  const epochAt = (x: number) =>
    Math.round(Math.min(1, Math.max(0, (x - PAD.left) / plotW)) * Math.max(0, epochs - 1));
  return { plotW, xOf, epochAt };
}

/**
 * Train and test loss per epoch, hand-drawn on a canvas, optionally over a
 * pinned reference run. The x-axis spans the longer of the two runs. Where a
 * drawn point stands for several epochs (long runs), a faint band shows their
 * range, so spikes stay visible. Hover shows every value at that epoch.
 */
export function LossCurve({
  losses,
  changes,
  logScale,
  reference,
  xName = epochName,
  xRange = epochRange,
  formatValue = formatLoss,
  formatTick = formatAxisTick,
  label = 'Loss per epoch',
}: LossCurveProps) {
  const wrap = useRef<HTMLDivElement>(null);
  const ref = useRef<HTMLCanvasElement>(null);
  const width = useElementWidth(wrap);
  const dark = usePrefersDark();
  const [hover, setHover] = useState<{ epoch: number; x: number; y: number } | null>(null);
  const epochs = Math.max(losses.count, reference?.losses.count ?? 0);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx || width === 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, HEIGHT);

    const n = epochs;
    const { plotW, xOf } = layout(width, n);
    const plotH = HEIGHT - PAD.top - PAD.bottom;
    const runs = reference ? [losses, reference.losses] : [losses];
    const axis = lossAxis(
      runs.flatMap((l) => [l.train.mean, l.test.mean, l.train.max, l.test.max]),
      logScale,
    );
    if (!axis) return;
    const { lo, hi, clippedAbove } = axis;
    const f = logScale ? Math.log10 : (v: number) => v;
    const [flo, fhi] = [f(lo), f(hi) === f(lo) ? f(lo) + 1 : f(hi)];
    // Values above a clipped top are drawn along it; the tooltip still shows them.
    const yOf = (v: number) => PAD.top + plotH - ((f(Math.min(v, hi)) - flo) / (fhi - flo)) * plotH;

    const muted = cssVar(canvas, '--muted');
    const grid = cssVar(canvas, '--border');
    ctx.font = '10px system-ui, sans-serif';
    ctx.fillStyle = muted;
    ctx.strokeStyle = grid;
    ctx.lineWidth = 1;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let k = 0; k <= 3; k++) {
      const fv = flo + ((fhi - flo) * k) / 3;
      const v = logScale ? 10 ** fv : fv;
      const y = yOf(v);
      ctx.beginPath();
      ctx.moveTo(PAD.left, y);
      ctx.lineTo(width - PAD.right, y);
      ctx.stroke();
      ctx.fillText(formatTick(v), PAD.left - 4, y);
    }
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('0', PAD.left, HEIGHT - PAD.bottom + 5);
    ctx.textAlign = 'right';
    ctx.fillText(xName(n - 1), width - PAD.right, HEIGHT - PAD.bottom + 5);

    // At most ~2 points per pixel; each stands for its epochs' mean, with their range as a band.
    const usable = (v: number) => Number.isFinite(v) && (!logScale || v > 0);
    const curve = (
      c: LossCurveSummary,
      l: LossHistorySnapshot,
      colour: string,
      dash: number[],
      lineWidth: number,
      bandAlpha: number,
    ) => {
      const p = lossPoints(c, l, Math.max(1, plotW * 2));
      if (p.banded) {
        ctx.fillStyle = colour;
        ctx.globalAlpha = bandAlpha;
        // One polygon per run of finite points: along the maxima, back along the minima.
        let start = 0;
        for (let k = 0; k <= p.x.length; k++) {
          if (k < p.x.length && usable(p.lo[k]!) && usable(p.hi[k]!)) continue;
          if (k - start > 1) {
            ctx.beginPath();
            for (let q = start; q < k; q++) ctx.lineTo(xOf(p.x[q]!), yOf(p.hi[q]!));
            for (let q = k - 1; q >= start; q--) ctx.lineTo(xOf(p.x[q]!), yOf(p.lo[q]!));
            ctx.closePath();
            ctx.fill();
          }
          start = k + 1;
        }
        ctx.globalAlpha = 1;
      }
      ctx.strokeStyle = colour;
      ctx.lineWidth = lineWidth;
      ctx.setLineDash(dash);
      ctx.beginPath();
      let started = false;
      for (let k = 0; k < p.x.length; k++) {
        const v = p.mean[k]!;
        if (!usable(v)) {
          started = false; // a gap, not a line through the missing value
          continue;
        }
        if (started) ctx.lineTo(xOf(p.x[k]!), yOf(v));
        else ctx.moveTo(xOf(p.x[k]!), yOf(v));
        started = true;
      }
      ctx.stroke();
      ctx.setLineDash([]);
    };
    if (reference) {
      const colour = cssVar(canvas, '--reference-line');
      // Thinner than the live run's lines, which stay in front.
      const l = reference.losses;
      curve(l.test, l, colour, [4, 3], 1, 0.1);
      curve(l.train, l, colour, [], 1, 0.12);
    }
    curve(losses.test, losses, cssVar(canvas, '--test-line'), [4, 3], 1.5, 0.15);
    curve(losses.train, losses, cssVar(canvas, '--train-line'), [], 1.5, 0.15);

    // Mid-run setting changes: short ticks up from the x-axis, in each run's colour.
    const ticks = (list: LossSeries['changes'], colour: string, lift: number) => {
      ctx.strokeStyle = colour;
      ctx.lineWidth = 2;
      for (const { epoch } of list ?? []) {
        const x = xOf(epoch);
        ctx.beginPath();
        ctx.moveTo(x, PAD.top + plotH - lift);
        ctx.lineTo(x, PAD.top + plotH - lift - 6);
        ctx.stroke();
      }
    };
    ticks(reference?.changes, cssVar(canvas, '--reference-line'), 7);
    ticks(changes, cssVar(canvas, '--train-line'), 0);

    if (clippedAbove !== null) {
      const label = `Clipped above ${formatTick(hi)} (peak ${formatTick(clippedAbove)})`;
      const [x, y] = [PAD.left + 4, PAD.top + 3];
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.globalAlpha = 0.85;
      ctx.fillStyle = cssVar(canvas, '--surface');
      ctx.fillRect(x - 3, y - 2, ctx.measureText(label).width + 6, 14);
      ctx.globalAlpha = 1;
      ctx.fillStyle = muted;
      ctx.fillText(label, x, y);
    }
  }, [losses, changes, reference, epochs, logScale, width, dark, xName, formatTick]);

  const onMove = (e: MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const epoch = layout(rect.width, epochs).epochAt(e.clientX - rect.left);
    setHover({ epoch, x: e.clientX, y: e.clientY });
  };

  const rows: [string, string][] = [];
  if (hover) {
    const at = (c: LossCurveSummary, l: LossHistorySnapshot) =>
      formatAt(lossAt(c, l, hover.epoch), formatValue, xRange);
    const prefix = reference ? 'This run, ' : '';
    rows.push(
      [`${prefix}train`, at(losses.train, losses)],
      [`${prefix}test`, at(losses.test, losses)],
    );
    if (reference) {
      const l = reference.losses;
      rows.push(['Reference, train', at(l.train, l)], ['Reference, test', at(l.test, l)]);
    }
    // Describe any change whose tick is within reach of the pointer.
    const plotW = Math.max(1, width - PAD.left - PAD.right);
    const reach = Math.round((TICK_REACH_PX * Math.max(0, epochs - 1)) / plotW);
    const near = (list: LossSeries['changes'], who: string) => {
      for (const c of list ?? []) {
        if (Math.abs(c.epoch - hover.epoch) <= reach) {
          rows.push([`${who}changed at epoch ${c.epoch}`, c.text]);
        }
      }
    };
    near(changes, reference ? 'This run: ' : 'Settings ');
    near(reference?.changes, 'Reference: ');
  }
  const crosshairX = hover && width > 0 ? layout(width, epochs).xOf(hover.epoch) : null;

  return (
    <div ref={wrap} className="loss-curve">
      <canvas
        ref={ref}
        style={{ width: '100%', height: HEIGHT }}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        aria-label={label}
        role="img"
      />
      {crosshairX !== null && (
        <div
          className="crosshair"
          style={{ left: crosshairX, top: PAD.top, height: HEIGHT - PAD.top - PAD.bottom }}
        />
      )}
      {hover && (
        <Tooltip x={hover.x} y={hover.y} title={capitalise(xName(hover.epoch))} rows={rows} />
      )}
    </div>
  );
}
