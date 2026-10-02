import { type RefObject, useEffect, useMemo, useState, useSyncExternalStore } from 'react';
import { NEGATIVE, POSITIVE, type Rgb, divergingLut } from './colour';

const DARK_QUERY = '(prefers-color-scheme: dark)';

function subscribeDark(onChange: () => void): () => void {
  const mq = window.matchMedia(DARK_QUERY);
  mq.addEventListener('change', onChange);
  return () => mq.removeEventListener('change', onChange);
}

export function usePrefersDark(): boolean {
  return useSyncExternalStore(subscribeDark, () => window.matchMedia(DARK_QUERY).matches);
}

const MID_LIGHT: Rgb = [250, 250, 250];
const MID_DARK: Rgb = [44, 48, 56];

/** The heatmap colour table, with a neutral midpoint that suits the colour scheme. */
export function useHeatmapLut(): Uint8Array {
  const dark = usePrefersDark();
  return useMemo(() => divergingLut(NEGATIVE, dark ? MID_DARK : MID_LIGHT, POSITIVE), [dark]);
}

/** Tracks an element's content-box width (0 until first measured). */
export function useElementWidth(ref: RefObject<HTMLElement | null>): number {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      if (entry) setWidth(Math.floor(entry.contentRect.width));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return width;
}
