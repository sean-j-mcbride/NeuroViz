import { memo, useEffect, useRef } from 'react';
import { type Normalisation, lutIndex } from './colour';
import { useHeatmapLut } from './hooks';

interface HeatmapProps {
  /** G×G values, row-major, row 0 at the top. */
  values: Float32Array;
  gridSize: number;
  norm: Normalisation;
  className?: string;
  title?: string;
}

/**
 * Draws a G×G grid at its native resolution; CSS scales the canvas up and the
 * browser's bilinear smoothing makes it look continuous.
 */
export const Heatmap = memo(function Heatmap({
  values,
  gridSize: g,
  norm,
  className,
  title,
}: HeatmapProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const lut = useHeatmapLut();
  const { scale, offset } = norm;

  useEffect(() => {
    const ctx = ref.current?.getContext('2d');
    if (!ctx) return;
    const img = ctx.createImageData(g, g);
    const px = img.data;
    for (let i = 0; i < g * g; i++) {
      const k = lutIndex(values[i]! * scale + offset) * 3;
      px[i * 4] = lut[k]!;
      px[i * 4 + 1] = lut[k + 1]!;
      px[i * 4 + 2] = lut[k + 2]!;
      px[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [values, g, scale, offset, lut]);

  return (
    <canvas ref={ref} width={g} height={g} className={`heatmap ${className ?? ''}`} title={title} />
  );
});
