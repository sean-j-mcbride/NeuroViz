import { memo, useEffect, useRef } from 'react';
import { MNIST_PIXELS, MNIST_SIDE } from '../data';
import { cssVar } from './canvas';
import { usePrefersDark } from './hooks';

function rgbOf(css: string): [number, number, number] {
  const hex = /^#([0-9a-f]{6})$/i.exec(css)?.[1];
  if (!hex) return [0, 0, 0];
  return [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16)) as [number, number, number];
}

interface DigitImageProps {
  /** 784 pixels: bytes (0–255) or floats in [0, 1]. */
  pixels: Uint8Array | Float32Array;
  /** CSS size in pixels. */
  size?: number;
  label?: string;
  className?: string;
}

/** One 28×28 digit, ink in the text colour on the panel surface, with crisp (pixelated) scaling. */
export const DigitImage = memo(function DigitImage({
  pixels,
  size = 56,
  label,
  className,
}: DigitImageProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const dark = usePrefersDark();

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const ink = rgbOf(cssVar(canvas, '--fg'));
    const paper = rgbOf(cssVar(canvas, '--surface'));
    const scale = pixels instanceof Uint8Array ? 1 / 255 : 1;
    const img = ctx.createImageData(MNIST_SIDE, MNIST_SIDE);
    for (let p = 0; p < MNIST_PIXELS; p++) {
      const t = Math.min(1, Math.max(0, pixels[p]! * scale));
      for (let c = 0; c < 3; c++) img.data[p * 4 + c] = paper[c]! + (ink[c]! - paper[c]!) * t;
      img.data[p * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [pixels, dark]);

  return (
    <canvas
      ref={ref}
      width={MNIST_SIDE}
      height={MNIST_SIDE}
      className={`digit-image ${className ?? ''}`}
      style={{ width: size, height: size }}
      role="img"
      aria-label={label}
      title={label}
    />
  );
});
