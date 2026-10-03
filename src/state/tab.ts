import { create } from 'zustand';
import { useMnistStore } from './mnistStore';
import { useAppStore } from './store';

export type Tab = 'playground' | 'mnist';

/** The URL hash that opens the MNIST tab. */
export const MNIST_HASH = '#mnist';

export function tabForHash(hash: string): Tab {
  return hash === MNIST_HASH ? 'mnist' : 'playground';
}

interface TabState {
  tab: Tab;
  /** Leaving a tab pauses its run; the run is kept, ready to continue. */
  setTab(tab: Tab): void;
}

export const useTabStore = create<TabState>()((set, get) => ({
  tab: typeof location === 'undefined' ? 'playground' : tabForHash(location.hash),
  setTab: (tab) => {
    const left = get().tab;
    if (tab === left) return;
    if (left === 'playground') useAppStore.getState().setRunning(false);
    else useMnistStore.getState().setRunning(false);
    set({ tab });
  },
}));
