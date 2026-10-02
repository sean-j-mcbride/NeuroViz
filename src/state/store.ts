import { create } from 'zustand';
import type {
  Checkpoint,
  DatasetSpec,
  Hyperparams,
  NetworkSpec,
  ProbeRef,
  SessionData,
  Snapshot,
  Speed,
} from '../worker';
import { DEFAULT_CONFIG, type PlaygroundConfig, sameConfig } from './config';
import type { ModelFile } from './modelFile';
import { PRESETS, type PresetId } from './presets';
import { type ReferenceRun, referenceFromSnapshot } from './reference';

export * from './config';

/** Step-through mode: one data point traced forward then backward, one stage at a time. */
export interface StepThrough {
  /** The traced point; null until one is picked. */
  probe: ProbeRef | null;
  /**
   * 0 = input; 1 … C−1 = forward through column k; C = loss; then C+1 … 2C−1 =
   * backward through columns C−1 … 1 (C = number of columns).
   */
  stage: number;
}

export interface AppState {
  config: PlaygroundConfig;
  running: boolean;
  speed: Speed;
  showTestData: boolean;
  /** Bumped to rebuild the session from the same config (Reset). */
  resetCount: number;
  /** The current session's point sets; null until the worker reports them. */
  sessionData: SessionData | null;
  /** Latest snapshot from the training worker; null before the first one. */
  snapshot: Snapshot | null;
  /** Non-null while step-through mode is open. */
  stepThrough: StepThrough | null;
  /**
   * A loaded model's checkpoint: the session starts from it, and Reset returns
   * to it. Changing the data, network or seed drops it.
   */
  resume: Checkpoint | null;
  /** The last preset loaded, while its note should show; null after any other load. */
  presetId: PresetId | null;
  /** A frozen run whose curves and boundary are drawn alongside the live run. */
  reference: ReferenceRun | null;
  /** A message for the user (bad file, ignored link values, link copied, …). */
  notice: Notice | null;

  setDataset(patch: Partial<DatasetSpec>): void;
  setNetwork(network: NetworkSpec): void;
  setTraining(patch: Partial<Hyperparams>): void;
  setSeed(seed: number): void;
  /** Starting training closes step-through. */
  setRunning(running: boolean): void;
  setSpeed(speed: Speed): void;
  setShowTestData(show: boolean): void;
  reset(): void;
  setSessionData(data: SessionData): void;
  setSnapshot(snapshot: Snapshot): void;
  /** Opens step-through (pausing training) or closes it. */
  setStepThroughOpen(open: boolean): void;
  setProbe(probe: ProbeRef): void;
  setStage(stage: number): void;
  /** Starts a fresh, paused run from `config` (optionally resuming a checkpoint). */
  setConfig(config: PlaygroundConfig, opts?: { resume?: Checkpoint; presetId?: PresetId }): void;
  loadModel(file: ModelFile): void;
  loadPreset(id: PresetId): void;
  /** Pins the current run, then starts its preset's fix so the two can be compared. */
  tryPresetFix(): void;
  /** Freezes the current run as the reference. */
  pinReference(label?: string): void;
  setReference(reference: ReferenceRun | null): void;
  setNotice(notice: Notice | null): void;
  /** Hides the preset note. */
  dismissPreset(): void;
}

export interface Notice {
  kind: 'error' | 'info';
  text: string;
}

/** The preset whose config (or fix) `config` is, if any. */
export function presetLabel(config: PlaygroundConfig, presetId: PresetId | null): string | null {
  const p = PRESETS.find((q) => q.id === presetId);
  if (!p) return null;
  if (sameConfig(config, p.config)) return p.name;
  if (sameConfig(config, p.fix)) return `${p.name}, fixed`;
  return null;
}

export const useAppStore = create<AppState>()((set, get) => ({
  config: DEFAULT_CONFIG,
  running: false,
  speed: 300,
  showTestData: false,
  resetCount: 0,
  sessionData: null,
  snapshot: null,
  stepThrough: null,
  resume: null,
  presetId: null,
  reference: null,
  notice: null,

  setDataset: (patch) =>
    set(({ config }) => ({
      config: { ...config, dataset: { ...config.dataset, ...patch } },
      // Point indices refer to the old data.
      stepThrough: null,
      resume: null,
    })),
  setNetwork: (network) =>
    set(({ config, stepThrough }) => ({
      config: { ...config, network },
      // Stage numbers depend on the column count.
      stepThrough: stepThrough && { ...stepThrough, stage: 0 },
      resume: null,
    })),
  setTraining: (patch) =>
    set(({ config }) => ({ config: { ...config, training: { ...config.training, ...patch } } })),
  setSeed: (seed) => set(({ config }) => ({ config: { ...config, seed }, resume: null })),
  setRunning: (running) =>
    set(({ stepThrough }) => ({ running, stepThrough: running ? null : stepThrough })),
  setSpeed: (speed) => set({ speed }),
  setShowTestData: (showTestData) =>
    set(({ stepThrough }) => ({
      showTestData,
      // A hidden test point can't stay selected.
      stepThrough:
        !showTestData && stepThrough?.probe?.set === 'test'
          ? { probe: null, stage: 0 }
          : stepThrough,
    })),
  reset: () => set(({ resetCount }) => ({ resetCount: resetCount + 1 })),
  setSessionData: (sessionData) => set({ sessionData }),
  setSnapshot: (snapshot) => set({ snapshot }),
  setStepThroughOpen: (open) =>
    set(({ running }) => ({
      stepThrough: open ? { probe: null, stage: 0 } : null,
      running: open ? false : running,
    })),
  setProbe: (probe) => set({ stepThrough: { probe, stage: 0 } }),
  setStage: (stage) =>
    set(({ stepThrough }) => ({ stepThrough: stepThrough && { ...stepThrough, stage } })),
  setConfig: (config, { resume, presetId } = {}) =>
    set(({ resetCount }) => ({
      config,
      resume: resume ?? null,
      presetId: presetId ?? null,
      running: false,
      stepThrough: null,
      // Always a fresh session, even when only hyperparameters changed.
      resetCount: resetCount + 1,
    })),
  loadModel: ({ config, checkpoint }) => get().setConfig(config, { resume: checkpoint }),
  loadPreset: (id) => {
    const p = PRESETS.find((q) => q.id === id);
    if (p) get().setConfig(p.config, { presetId: id });
  },
  tryPresetFix: () => {
    const { presetId, config } = get();
    const p = PRESETS.find((q) => q.id === presetId);
    if (!p || !sameConfig(config, p.config)) return;
    get().pinReference(p.name);
    get().setConfig(p.fix, { presetId: p.id });
  },
  pinReference: (label) => {
    const { config, presetId, snapshot, sessionData } = get();
    if (!snapshot) return;
    const name = label ?? presetLabel(config, presetId) ?? 'Pinned run';
    set({ reference: referenceFromSnapshot(name, config, snapshot, sessionData) });
  },
  setReference: (reference) => set({ reference }),
  setNotice: (notice) => set({ notice }),
  dismissPreset: () => set({ presetId: null }),
}));

/** A fresh random seed for "Regenerate" / "New weights" (UI-only randomness). */
export function randomSeed(): number {
  return Math.floor(Math.random() * 1e9);
}
