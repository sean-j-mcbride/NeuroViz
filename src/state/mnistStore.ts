import { create } from 'zustand';
import type { MnistSubset } from '../data';
import type {
  Hyperparams,
  MnistCheckpoint,
  MnistNetworkSpec,
  MnistSnapshot,
  Speed,
} from '../worker';
import { MNIST_DEFAULT_CONFIG, type MnistConfig, mnistNetworkError } from './mnistConfig';
import type { MnistModelFile } from './mnistModelFile';

export * from './mnistConfig';

/** The bundled images, fetched the first time the MNIST tab opens. */
export type MnistData =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'ready'; subset: MnistSubset }
  | { status: 'error'; message: string };

/** Which test images the gallery shows: every mistake, or one confusion-matrix cell. */
export interface GalleryFilter {
  label: number;
  predicted: number;
}

export interface MnistState {
  config: MnistConfig;
  running: boolean;
  speed: Speed;
  /** Bumped to rebuild the session from the same config (Reset). */
  resetCount: number;
  data: MnistData;
  snapshot: MnistSnapshot | null;
  /** A loaded model's checkpoint: the session starts from it and Reset returns to it. */
  resume: MnistCheckpoint | null;
  /** The digit pad's drawing as a 28×28 MNIST-style input; null when blank. */
  drawn: Float32Array | null;
  /** null = the most confidently wrong test images overall. */
  galleryFilter: GalleryFilter | null;
  /** The class the drawn digit's gradient maps are for; null = the network's answer. */
  gradTarget: number | null;

  setNetwork(network: MnistNetworkSpec): void;
  setTraining(patch: Partial<Hyperparams>): void;
  setSeed(seed: number): void;
  setRunning(running: boolean): void;
  setSpeed(speed: Speed): void;
  reset(): void;
  setData(data: MnistData): void;
  setSnapshot(snapshot: MnistSnapshot): void;
  /** Starts a fresh, paused run from `config` (optionally resuming a checkpoint). */
  setConfig(config: MnistConfig, opts?: { resume?: MnistCheckpoint }): void;
  loadModel(file: MnistModelFile): void;
  setDrawn(drawn: Float32Array | null): void;
  setGalleryFilter(filter: GalleryFilter | null): void;
  setGradTarget(target: number | null): void;
}

export const useMnistStore = create<MnistState>()((set, get) => ({
  config: MNIST_DEFAULT_CONFIG,
  running: false,
  speed: 'max',
  resetCount: 0,
  data: { status: 'idle' },
  snapshot: null,
  resume: null,
  drawn: null,
  galleryFilter: null,
  gradTarget: null,

  // A network that doesn't fit the image is kept (so the builder can show why) but can't run.
  setNetwork: (network) =>
    set(({ config, running }) => ({
      config: { ...config, network },
      resume: null,
      running: running && !mnistNetworkError(network),
    })),
  setTraining: (patch) =>
    set(({ config }) => ({ config: { ...config, training: { ...config.training, ...patch } } })),
  setSeed: (seed) => set(({ config }) => ({ config: { ...config, seed }, resume: null })),
  setRunning: (running) => set({ running }),
  setSpeed: (speed) => set({ speed }),
  reset: () => set(({ resetCount }) => ({ resetCount: resetCount + 1 })),
  setData: (data) => set({ data }),
  setSnapshot: (snapshot) => set({ snapshot }),
  setConfig: (config, { resume } = {}) =>
    set(({ resetCount }) => ({
      config,
      resume: resume ?? null,
      running: false,
      resetCount: resetCount + 1,
    })),
  loadModel: ({ config, checkpoint }) => get().setConfig(config, { resume: checkpoint }),
  setDrawn: (drawn) => set({ drawn }),
  setGalleryFilter: (galleryFilter) => set({ galleryFilter }),
  setGradTarget: (gradTarget) => set({ gradTarget }),
}));
