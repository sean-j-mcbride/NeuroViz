import { useEffect } from 'react';
import { decodeConfig, encodeConfig } from '../state/shareUrl';
import { sameConfig, useAppStore } from '../state/store';

/**
 * Coalesces hash updates while a slider is dragged: Safari throws after 100
 * `history.replaceState` calls in 30 seconds.
 */
const DEBOUNCE_MS = 300;

/**
 * Applies a share-link hash to the store (a fresh, paused run). Ignored values
 * are reported as a notice. Returns whether the hash held a config.
 */
export function applyHash(hash: string): boolean {
  const parsed = decodeConfig(hash);
  if (!parsed) return false;
  const { config, setConfig, setNotice } = useAppStore.getState();
  if (!sameConfig(config, parsed.config)) setConfig(parsed.config);
  if (parsed.warnings.length > 0) {
    setNotice({
      kind: 'error',
      text: `Some link settings were ignored. ${parsed.warnings.join('. ')}.`,
    });
  }
  return true;
}

/** Keeps the URL hash in step with the settings, and follows links pasted into this tab. */
export function useUrlSync(): void {
  const config = useAppStore((s) => s.config);

  useEffect(() => {
    const t = setTimeout(() => {
      const hash = encodeConfig(config);
      // replaceState: settings changes shouldn't fill the Back button's history.
      if (location.hash !== hash) history.replaceState(history.state, '', hash);
    }, DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [config]);

  useEffect(() => {
    const onHashChange = () => applyHash(location.hash);
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);
}
