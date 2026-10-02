// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { PRESETS } from '../state/presets';
import { encodeLink } from '../state/shareUrl';
import { DEFAULT_CONFIG, useAppStore } from '../state/store';
import { applyHash } from './useUrlSync';

const initial = useAppStore.getState();
const s = () => useAppStore.getState();

beforeEach(() => useAppStore.setState(initial, true));

describe('applyHash', () => {
  it('ignores a hash without settings', () => {
    expect(applyHash('#top')).toBe(false);
    expect(s().resetCount).toBe(initial.resetCount);
  });

  it('starts a fresh run with the link’s settings and view', () => {
    const config = { ...DEFAULT_CONFIG, seed: 5 };
    expect(applyHash(encodeLink(config, { speed: 'max', showTestData: true }))).toBe(true);
    expect(s()).toMatchObject({ config, speed: 'max', showTestData: true, presetId: null });
    expect(s().resetCount).toBe(initial.resetCount + 1);
    expect(s().notice).toBeNull();
  });

  it('recognises a preset (or its fix), so its note shows', () => {
    const p = PRESETS.find((q) => q.id === 'dead-relus')!;
    applyHash(encodeLink(p.config, { speed: 300, showTestData: false }));
    expect(s().presetId).toBe('dead-relus');
    applyHash(encodeLink(p.fix, { speed: 300, showTestData: false }));
    expect(s().presetId).toBe('dead-relus');
  });

  it('reports ignored values', () => {
    applyHash('#lr=7&speed=fast');
    expect(s().notice?.kind).toBe('error');
    expect(s().notice?.text).toMatch(/^Some link settings were ignored\. Ignored learning rate 7/);
    expect(s().speed).toBe(initial.speed);
  });
});
