import { type PointerEvent, useEffect, useImperativeHandle, useRef } from 'react';
import { MNIST_PIXELS, MNIST_SIDE, preprocessDrawing } from '../data';
import { cssVar } from './canvas';

/** The pad's drawing resolution: 10 pad pixels per MNIST pixel. */
const PAD = 280;
/** MNIST strokes are about 2–3 pixels wide in a 20-pixel box. */
const BRUSH = 22;

export interface DigitPadHandle {
  clear(): void;
  /**
   * Shows a 28×28 image (bytes) on the pad, to draw over. It does not report
   * a change: the caller passes on the exact image (already MNIST-shaped);
   * drawing over it then reports as usual.
   */
  show(image: Uint8Array): void;
}

interface DigitPadProps {
  /** The drawing as an MNIST-style input (cropped, scaled, centred), or null when blank. */
  onChange: (pixels: Float32Array | null) => void;
  ref?: React.Ref<DigitPadHandle>;
}

/**
 * A canvas to draw a digit on with mouse, pen or touch. Each change is
 * preprocessed the way MNIST was made (see `preprocessDrawing`), at most once
 * per animation frame.
 */
export function DigitPad({ onChange, ref }: DigitPadProps) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const last = useRef<{ x: number; y: number } | null>(null);
  const frame = useRef(0);

  useEffect(() => () => cancelAnimationFrame(frame.current), []);

  const context = () => canvas.current?.getContext('2d', { willReadFrequently: true }) ?? null;

  /** Reads the ink back (the alpha channel) and reports the preprocessed digit. */
  const report = () => {
    if (frame.current) return;
    frame.current = requestAnimationFrame(() => {
      frame.current = 0;
      const ctx = context();
      if (!ctx) return;
      const rgba = ctx.getImageData(0, 0, PAD, PAD).data;
      const alpha = new Float32Array(PAD * PAD);
      for (let i = 0; i < alpha.length; i++) alpha[i] = rgba[i * 4 + 3]! / 255;
      onChange(preprocessDrawing(alpha, PAD, PAD));
    });
  };

  useImperativeHandle(ref, () => ({
    clear: () => {
      context()?.clearRect(0, 0, PAD, PAD);
      onChange(null);
    },
    show: (image) => {
      const ctx = context();
      if (!ctx || !canvas.current) return;
      const small = new ImageData(MNIST_SIDE, MNIST_SIDE);
      for (let p = 0; p < MNIST_PIXELS; p++) small.data[p * 4 + 3] = image[p]!;
      const tmp = document.createElement('canvas');
      tmp.width = tmp.height = MNIST_SIDE;
      tmp.getContext('2d')?.putImageData(small, 0, 0);
      ctx.clearRect(0, 0, PAD, PAD);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(tmp, 0, 0, PAD, PAD);
      // Ink colour: keep the alpha, recolour to the pen colour.
      ctx.globalCompositeOperation = 'source-in';
      ctx.fillStyle = cssVar(canvas.current, '--fg');
      ctx.fillRect(0, 0, PAD, PAD);
      ctx.globalCompositeOperation = 'source-over';
    },
  }));

  const point = (e: PointerEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * PAD, y: ((e.clientY - r.top) / r.height) * PAD };
  };

  const stroke = (to: { x: number; y: number }) => {
    const ctx = context();
    if (!ctx || !canvas.current) return;
    const from = last.current ?? to;
    ctx.strokeStyle = cssVar(canvas.current, '--fg');
    ctx.lineWidth = BRUSH;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
    last.current = to;
    report();
  };

  return (
    <canvas
      ref={canvas}
      className="digit-pad"
      width={PAD}
      height={PAD}
      aria-label="Drawing pad: draw a digit from 0 to 9"
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        last.current = null;
        stroke(point(e));
      }}
      onPointerMove={(e) => {
        if (e.buttons & 1 || e.pointerType === 'touch') {
          if (last.current) stroke(point(e));
        }
      }}
      onPointerUp={() => (last.current = null)}
      onPointerCancel={() => (last.current = null)}
    />
  );
}
