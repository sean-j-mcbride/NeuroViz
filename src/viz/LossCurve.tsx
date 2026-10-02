import { useEffect, useRef } from 'react';
import { cssVar } from './canvas';
import { useElementWidth, usePrefersDark } from './hooks';

const HEIGHT = 150;
const PAD = { left: 44, right: 8, top: 8, bottom: 20 };

interface LossCurveProps {
  train: Float32Array;
  test: Float32Array;
  logScale: boolean;
}

function formatTick(v: number): string {
  if (v === 0) return '0';
  return Math.abs(v) >= 0.01 && Math.abs(v) < 1000 ? v.toPrecision(2) : v.toExponential(0);
}

/** Train and test loss per epoch, hand-drawn on a canvas. */
export function LossCurve({ train, test, logScale }: LossCurveProps) {
  const wrap = useRef<HTMLDivElement>(null);
  const ref = useRef<HTMLCanvasElement>(null);
  const width = useElementWidth(wrap);
  const dark = usePrefersDark();

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx || width === 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = HEIGHT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, HEIGHT);

    const n = train.length;
    const plotW = width - PAD.left - PAD.right;
    const plotH = HEIGHT - PAD.top - PAD.bottom;
    let lo = Infinity;
    let hi = -Infinity;
    for (const series of [train, test]) {
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
    const xOf = (i: number) => PAD.left + (n <= 1 ? 0 : (i / (n - 1)) * plotW);
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
    const line = (series: Float32Array, colour: string, dash: number[]) => {
      ctx.strokeStyle = colour;
      ctx.lineWidth = 1.5;
      ctx.setLineDash(dash);
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < n; i += stride) {
        const v = series[i]!;
        if (!Number.isFinite(v) || (logScale && v <= 0)) continue;
        if (started) ctx.lineTo(xOf(i), yOf(v));
        else ctx.moveTo(xOf(i), yOf(v));
        started = true;
      }
      if (n > 0 && (n - 1) % stride !== 0) ctx.lineTo(xOf(n - 1), yOf(series[n - 1]!));
      ctx.stroke();
      ctx.setLineDash([]);
    };
    line(test, cssVar(canvas, '--test-line'), [4, 3]);
    line(train, cssVar(canvas, '--train-line'), []);
  }, [train, test, logScale, width, dark]);

  return (
    <div ref={wrap} className="loss-curve">
      <canvas ref={ref} style={{ width: '100%', height: HEIGHT }} />
    </div>
  );
}
