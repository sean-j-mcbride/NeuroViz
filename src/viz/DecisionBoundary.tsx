import { type MouseEvent, useEffect, useRef } from 'react';
import type { OutputGrid, PointSet, ProbeRef, SessionData } from '../worker';
import { NEGATIVE, NORMALISERS, POSITIVE, rgbCss } from './colour';
import { Heatmap } from './Heatmap';

/** Internal resolution of the points overlay, in CSS pixels (scaled by devicePixelRatio). */
const OVERLAY = 320;
const RADIUS = 3;
/** How close (in overlay pixels) a click must be to pick a point. */
const PICK_RADIUS = 10;

interface DecisionBoundaryProps {
  boundary: OutputGrid;
  data: SessionData | null;
  showTestData: boolean;
  /** When set, clicking near a point picks it. */
  onPick?: (ref: ProbeRef) => void;
  /** The picked point, drawn with a ring. */
  selected?: ProbeRef | null;
  /** Hover text when not picking; also the accessible name. */
  title?: string;
}

function drawPoints(
  ctx: CanvasRenderingContext2D,
  { x, y }: PointSet,
  toPx: (v: number) => number,
  outline: string,
): void {
  ctx.strokeStyle = outline;
  for (let i = 0; i < y.length; i++) {
    ctx.beginPath();
    ctx.arc(toPx(x[2 * i]!), OVERLAY - toPx(x[2 * i + 1]!), RADIUS, 0, 2 * Math.PI);
    ctx.fillStyle = rgbCss(y[i] === 1 ? POSITIVE : NEGATIVE);
    ctx.fill();
    ctx.stroke();
  }
}

/** The output probability over the input plane, with the data drawn on top. */
export function DecisionBoundary({
  boundary,
  data,
  showTestData,
  onPick,
  selected,
  title = 'Probability of the blue class',
}: DecisionBoundaryProps) {
  const overlay = useRef<HTMLCanvasElement>(null);
  const { values: output, gridSize, domain } = boundary;

  useEffect(() => {
    const canvas = overlay.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx || !data) return;
    const toPx = (v: number) => ((v + domain) / (2 * domain)) * OVERLAY;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = OVERLAY * dpr;
    canvas.height = OVERLAY * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.lineWidth = 1;
    drawPoints(ctx, data.train, toPx, 'rgb(255 255 255 / 0.9)');
    if (showTestData) drawPoints(ctx, data.test, toPx, 'rgb(0 0 0 / 0.85)');
    if (selected) {
      const set = data[selected.set];
      const [px, py] = [set.x[2 * selected.index], set.x[2 * selected.index + 1]];
      if (px !== undefined && py !== undefined) {
        ctx.lineWidth = 2;
        for (const [colour, r] of [
          ['rgb(255 255 255)', 8],
          ['rgb(0 0 0)', 10],
        ] as const) {
          ctx.strokeStyle = colour;
          ctx.beginPath();
          ctx.arc(toPx(px), OVERLAY - toPx(py), r, 0, 2 * Math.PI);
          ctx.stroke();
        }
      }
    }
  }, [data, domain, showTestData, selected]);

  const pick = (e: MouseEvent<HTMLCanvasElement>) => {
    if (!onPick || !data) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const toPx = (v: number) => ((v + domain) / (2 * domain)) * OVERLAY;
    const cx = ((e.clientX - rect.left) / rect.width) * OVERLAY;
    const cy = ((e.clientY - rect.top) / rect.height) * OVERLAY;
    let best: ProbeRef | null = null;
    let bestD = PICK_RADIUS ** 2;
    for (const set of showTestData ? (['train', 'test'] as const) : (['train'] as const)) {
      const { x } = data[set];
      for (let i = 0; i < x.length / 2; i++) {
        const d = (toPx(x[2 * i]!) - cx) ** 2 + (OVERLAY - toPx(x[2 * i + 1]!) - cy) ** 2;
        if (d < bestD) [best, bestD] = [{ set, index: i }, d];
      }
    }
    if (best) onPick(best);
  };

  return (
    <div className={`decision-boundary${onPick ? ' picking' : ''}`}>
      <Heatmap
        values={output}
        gridSize={gridSize}
        norm={NORMALISERS.output(output, domain)}
        title={onPick ? undefined : title}
      />
      <canvas ref={overlay} className="points" onClick={pick} />
    </div>
  );
}
