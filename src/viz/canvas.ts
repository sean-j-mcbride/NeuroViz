/** Reads a CSS custom property (e.g. a colour token) as resolved on `el`. */
export function cssVar(el: Element, name: string): string {
  return getComputedStyle(el).getPropertyValue(name).trim();
}

/** Sizes a canvas for crisp drawing at `width × height` CSS pixels and returns a cleared context. */
export function prepareCanvas(
  canvas: HTMLCanvasElement,
  width: number,
  height: number,
): CanvasRenderingContext2D | null {
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return ctx;
}

/** Log-axis tick label: 1, 0.1, 1e-3 … */
export function formatPow10(exp: number): string {
  if (exp === 0) return '1';
  if (exp === 1) return '10';
  if (exp === -1) return '0.1';
  if (exp === -2) return '0.01';
  return `1e${exp}`;
}
