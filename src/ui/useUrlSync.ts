import { useEffect } from 'react';
import { presetFor } from '../state/presets';
import { decodeLink, encodeLink } from '../state/shareUrl';
import { sameConfig, useAppStore } from '../state/store';

/**
 * Coalesces hash updates while a slider is dragged: Safari throws after 100
 * `history.replaceState` calls in 30 seconds.
 */
const DEBOUNCE_MS = 300;

/**
 * Applies a share-link hash to the store: a fresh, paused run (with its
 * preset's note if it is one), plus the speed and test-data view. Ignored
 * values are reported as a notice. Returns whether the hash held a config.
 */
export function applyHash(hash: string): boolean {
  const parsed = decodeLink(hash);
  if (!parsed) return false;
  const { config, setConfig, setSpeed, setShowTestData, setNotice } = useAppStore.getState();
  if (!sameConfig(config, parsed.config)) {
    setConfig(parsed.config, { presetId: presetFor(parsed.config)?.id });
  }
  const { speed, showTestData } = parsed.view;
  if (speed !== undefined) setSpeed(speed);
  if (showTestData !== undefined) setShowTestData(showTestData);
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
  const speed = useAppStore((s) => s.speed);
  const showTestData = useAppStore((s) => s.showTestData);

  useEffect(() => {
    const t = setTimeout(() => {
      const hash = encodeLink(config, { speed, showTestData });
      // replaceState: settings changes shouldn't fill the Back button's history.
      if (location.hash !== hash) history.replaceState(history.state, '', hash);
    }, DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [config, speed, showTestData]);

  useEffect(() => {
    const onHashChange = () => applyHash(location.hash);
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);
}
