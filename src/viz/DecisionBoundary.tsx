import { useEffect, useRef } from 'react';
import type { PointSet, Snapshot } from '../worker';
import { NEGATIVE, NORMALISERS, POSITIVE, rgbCss } from './colour';
import { Heatmap } from './Heatmap';

/** Internal resolution of the points overlay, in CSS pixels (scaled by devicePixelRatio). */
const OVERLAY = 320;
const RADIUS = 3;

interface DecisionBoundaryProps {
  snapshot: Snapshot;
  showTestData: boolean;
}

function drawPoints(
  ctx: CanvasRenderingContext2D,
  { x, y }: PointSet,
  domain: number,
  outline: string,
): void {
  const toPx = (v: number) => ((v + domain) / (2 * domain)) * OVERLAY;
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
export function DecisionBoundary({ snapshot, showTestData }: DecisionBoundaryProps) {
  const overlay = useRef<HTMLCanvasElement>(null);
  const { columns, gridSize, domain, train, test } = snapshot;
  const output = columns[columns.length - 1]!.values;

  useEffect(() => {
    const canvas = overlay.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = OVERLAY * dpr;
    canvas.height = OVERLAY * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.lineWidth = 1;
    drawPoints(ctx, train, domain, 'rgb(255 255 255 / 0.9)');
    if (showTestData) drawPoints(ctx, test, domain, 'rgb(0 0 0 / 0.85)');
  }, [train, test, domain, showTestData]);

  return (
    <div className="decision-boundary">
      <Heatmap
        values={output}
        gridSize={gridSize}
        norm={NORMALISERS.output(output, domain)}
        title="Probability of the blue class"
      />
      <canvas ref={overlay} className="points" />
    </div>
  );
}
