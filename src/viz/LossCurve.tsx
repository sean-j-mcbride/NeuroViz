import { type MouseEvent, useEffect, useRef, useState } from 'react';
import { cssVar } from './canvas';
import { useElementWidth, usePrefersDark } from './hooks';
import { Tooltip } from './Tooltip';

const HEIGHT = 150;
const PAD = { left: 44, right: 8, top: 8, bottom: 20 };

export interface LossSeries {
  train: Float32Array;
  test: Float32Array;
}

interface LossCurveProps extends LossSeries {
  logScale: boolean;
  /** A pinned run, drawn behind in the reference colour (same dashes: solid train, dashed test). */
  reference?: LossSeries | null;
}

function formatTick(v: number): string {
  if (v === 0) return '0';
  return Math.abs(v) >= 0.01 && Math.abs(v) < 1000 ? v.toPrecision(2) : v.toExponential(0);
}

function formatLoss(v: number | undefined): string {
  return v === undefined ? '–' : Number.isFinite(v) ? v.toFixed(4) : String(v);
}

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
 * pinned reference run. The x-axis spans the longer of the two runs. Hover
 * shows every value at that epoch.
 */
export function LossCurve({ train, test, logScale, reference }: LossCurveProps) {
  const wrap = useRef<HTMLDivElement>(null);
  const ref = useRef<HTMLCanvasElement>(null);
  const width = useElementWidth(wrap);
  const dark = usePrefersDark();
  const [hover, setHover] = useState<{ epoch: number; x: number; y: number } | null>(null);
  const epochs = Math.max(train.length, reference?.train.length ?? 0);

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
    const all = reference ? [train, test, reference.train, reference.test] : [train, test];
    let lo = Infinity;
    let hi = -Infinity;
    for (const series of all) {
      for (const v of series) {
        if (!Number.isFinite(v) || (logScale && v <= 0)) continue;
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }
    }
    if (!Number.isFinite(lo)) return;
    if (!logScale) lo = 0;
    const f = logScale ? Math.log10 : (v: number) => v;
    const [flo, fhi] = [f(lo), f(hi) === f(lo) ? f(lo) + 1 : f(hi)];
    const yOf = (v: number) => PAD.top + plotH - ((f(v) - flo) / (fhi - flo)) * plotH;

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
    ctx.fillText(`epoch ${n - 1}`, width - PAD.right, HEIGHT - PAD.bottom + 5);

    // At most ~2 samples per pixel; long runs are thinned by striding.
    const stride = Math.max(1, Math.floor(n / (plotW * 2)));
    const line = (series: Float32Array, colour: string, dash: number[], width = 1.5) => {
      const len = series.length;
      ctx.strokeStyle = colour;
      ctx.lineWidth = width;
      ctx.setLineDash(dash);
      ctx.beginPath();
      let started = false;
      const point = (i: number) => {
        const v = series[i]!;
        if (!Number.isFinite(v) || (logScale && v <= 0)) {
          started = false; // a gap, not a line through the missing value
          return;
        }
        if (started) ctx.lineTo(xOf(i), yOf(v));
        else ctx.moveTo(xOf(i), yOf(v));
        started = true;
      };
      for (let i = 0; i < len; i += stride) point(i);
      if (len > 0 && (len - 1) % stride !== 0) point(len - 1);
      ctx.stroke();
      ctx.setLineDash([]);
    };
    if (reference) {
      const colour = cssVar(canvas, '--reference-line');
      // Thinner than the live run's lines, which stay in front.
      line(reference.test, colour, [4, 3], 1);
      line(reference.train, colour, [], 1);
    }
    line(test, cssVar(canvas, '--test-line'), [4, 3]);
    line(train, cssVar(canvas, '--train-line'), []);
  }, [train, test, reference, epochs, logScale, width, dark]);

  const onMove = (e: MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const epoch = layout(rect.width, epochs).epochAt(e.clientX - rect.left);
    setHover({ epoch, x: e.clientX, y: e.clientY });
  };

  const rows: [string, string][] = [];
  if (hover) {
    const at = (s: Float32Array) => formatLoss(s[hover.epoch]);
    const prefix = reference ? 'This run, ' : '';
    rows.push([`${prefix}train`, at(train)], [`${prefix}test`, at(test)]);
    if (reference) {
      rows.push(['Reference, train', at(reference.train)], ['Reference, test', at(reference.test)]);
    }
  }
  const crosshairX = hover && width > 0 ? layout(width, epochs).xOf(hover.epoch) : null;

  return (
    <div ref={wrap} className="loss-curve">
      <canvas
        ref={ref}
        style={{ width: '100%', height: HEIGHT }}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        aria-label="Loss per epoch"
        role="img"
      />
      {crosshairX !== null && (
        <div
          className="crosshair"
          style={{ left: crosshairX, top: PAD.top, height: HEIGHT - PAD.top - PAD.bottom }}
        />
      )}
      {hover && <Tooltip x={hover.x} y={hover.y} title={`Epoch ${hover.epoch}`} rows={rows} />}
    </div>
  );
}
