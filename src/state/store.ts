import { create } from 'zustand';

// App state. Model config must stay serialisable JSON. Populated in Phase 2.
export type AppState = Record<string, never>;

export const useAppStore = create<AppState>()(() => ({}));
