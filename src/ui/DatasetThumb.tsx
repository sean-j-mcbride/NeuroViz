import { useEffect, useRef } from 'react';
import { DOMAIN, type Toy2DKind, generateToy2D } from '../data';
import { NEGATIVE, POSITIVE, rgbCss } from '../viz';

const SIZE = 56;

/** A tiny scatter plot of a clean sample of the dataset, for the picker buttons. */
export function DatasetThumb({ kind }: { kind: Toy2DKind }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = SIZE * dpr;
    canvas.height = SIZE * dpr;
    ctx.scale(dpr, dpr);
    const { x, y } = generateToy2D(kind, { n: 160, noise: 0, seed: 7 });
    const toPx = (v: number) => ((v + DOMAIN) / (2 * DOMAIN)) * SIZE;
    for (let i = 0; i < y.size; i++) {
      ctx.fillStyle = rgbCss(y.data[i] === 1 ? POSITIVE : NEGATIVE);
      ctx.fillRect(toPx(x.data[2 * i]!) - 1, SIZE - toPx(x.data[2 * i + 1]!) - 1, 2, 2);
    }
  }, [kind]);
  return <canvas ref={ref} style={{ width: SIZE, height: SIZE }} aria-hidden />;
}
