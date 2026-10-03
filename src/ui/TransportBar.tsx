import {
  BATCH_SIZES,
  LEARNING_RATES,
  describeBatchSize,
  randomSeed,
  useAppStore,
} from '../state/store';
import { SPEEDS, type Hyperparams, type Speed } from '../worker';

export interface TransportControlsProps {
  running: boolean;
  onRunning(running: boolean): void;
  onReset(): void;
  onStep(): void;
  /** What one Step trains, e.g. "one epoch". */
  stepUnit: string;
  /** Plural unit for the speed choices, e.g. "epochs". */
  speedUnit: string;
  readout: { label: string; value: string };
  lr: number;
  batchSize: Hyperparams['batchSize'];
  batchSizes: readonly Hyperparams['batchSize'][];
  onTraining(patch: Partial<Hyperparams>): void;
  speed: Speed;
  speeds?: readonly Speed[];
  onSpeed(speed: Speed): void;
  seed: number;
  onSeed(seed: number): void;
}

/** Play / pause / step / reset, learning rate, batch size, speed and weight seed. */
export function TransportControls(p: TransportControlsProps) {
  const speedLabel = (s: Speed) =>
    s === 'max' ? 'Max' : `${s.toLocaleString('en-GB')} ${p.speedUnit}/s`;
  return (
    <div className="transport">
      <div className="transport-buttons">
        <button type="button" onClick={p.onReset} title="Reset to the initial weights (same seed)">
          ↺ Reset
        </button>
        <button
          type="button"
          className="primary"
          onClick={() => p.onRunning(!p.running)}
          title={p.running ? 'Pause' : 'Train'}
        >
          {p.running ? '❚❚ Pause' : '▶ Play'}
        </button>
        <button
          type="button"
          onClick={p.onStep}
          disabled={p.running}
          title={`Train for ${p.stepUnit}`}
        >
          ⏭ Step
        </button>
      </div>

      <div className="readout">
        <span className="readout-label">{p.readout.label}</span>
        <span className="readout-value">{p.readout.value}</span>
      </div>

      <label className="field">
        <span>Learning rate</span>
        <select value={p.lr} onChange={(e) => p.onTraining({ lr: Number(e.target.value) })}>
          {LEARNING_RATES.map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </select>
      </label>

      <label className="field">
        <span>Batch size</span>
        <select
          value={p.batchSize}
          onChange={(e) =>
            p.onTraining({
              batchSize: e.target.value === 'full' ? 'full' : Number(e.target.value),
            })
          }
        >
          {p.batchSizes.map((v) => (
            <option key={v} value={v}>
              {describeBatchSize(v)}
            </option>
          ))}
        </select>
      </label>

      <label className="field">
        <span>Speed</span>
        <select
          value={p.speed}
          onChange={(e) =>
            p.onSpeed(e.target.value === 'max' ? 'max' : (Number(e.target.value) as Speed))
          }
        >
          {(p.speeds ?? SPEEDS).map((v) => (
            <option key={v} value={v}>
              {speedLabel(v)}
            </option>
          ))}
        </select>
      </label>

      <div className="field seed">
        <span>Weight seed</span>
        <div className="seed-row">
          <code>{p.seed}</code>
          <button
            type="button"
            onClick={() => p.onSeed(randomSeed())}
            title="Initialise new random weights"
          >
            New weights
          </button>
        </div>
      </div>
    </div>
  );
}

/** The playground's transport, wired to the app store. */
export function TransportBar({ onStep }: { onStep: () => void }) {
  const running = useAppStore((s) => s.running);
  const speed = useAppStore((s) => s.speed);
  const { lr, batchSize } = useAppStore((s) => s.config.training);
  const seed = useAppStore((s) => s.config.seed);
  const epoch = useAppStore((s) => s.snapshot?.epoch ?? 0);
  const { setRunning, setSpeed, setTraining, setSeed, reset } = useAppStore.getState();

  return (
    <TransportControls
      running={running}
      onRunning={setRunning}
      onReset={reset}
      onStep={onStep}
      stepUnit="one epoch"
      speedUnit="epochs"
      readout={{ label: 'Epoch', value: epoch.toLocaleString('en-GB') }}
      lr={lr}
      batchSize={batchSize}
      batchSizes={BATCH_SIZES}
      onTraining={setTraining}
      speed={speed}
      onSpeed={setSpeed}
      seed={seed}
      onSeed={setSeed}
    />
  );
}
