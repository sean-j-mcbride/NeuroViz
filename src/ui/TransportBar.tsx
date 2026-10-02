import {
  BATCH_SIZES,
  LEARNING_RATES,
  describeBatchSize,
  randomSeed,
  useAppStore,
} from '../state/store';
import { SPEEDS, type Speed } from '../worker';

function speedLabel(s: Speed): string {
  return s === 'max' ? 'Max' : `${s.toLocaleString('en-GB')} epochs/s`;
}

export function TransportBar({ onStep }: { onStep: () => void }) {
  const running = useAppStore((s) => s.running);
  const speed = useAppStore((s) => s.speed);
  const { lr, batchSize } = useAppStore((s) => s.config.training);
  const seed = useAppStore((s) => s.config.seed);
  const epoch = useAppStore((s) => s.snapshot?.epoch ?? 0);
  const { setRunning, setSpeed, setTraining, setSeed, reset } = useAppStore.getState();

  return (
    <div className="transport">
      <div className="transport-buttons">
        <button type="button" onClick={reset} title="Reset to the initial weights (same seed)">
          ↺ Reset
        </button>
        <button
          type="button"
          className="primary"
          onClick={() => setRunning(!running)}
          title={running ? 'Pause' : 'Train'}
        >
          {running ? '❚❚ Pause' : '▶ Play'}
        </button>
        <button type="button" onClick={onStep} disabled={running} title="Train for one epoch">
          ⏭ Step
        </button>
      </div>

      <div className="readout">
        <span className="readout-label">Epoch</span>
        <span className="readout-value">{epoch.toLocaleString('en-GB')}</span>
      </div>

      <label className="field">
        <span>Learning rate</span>
        <select value={lr} onChange={(e) => setTraining({ lr: Number(e.target.value) })}>
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
          value={batchSize}
          onChange={(e) =>
            setTraining({ batchSize: e.target.value === 'full' ? 'full' : Number(e.target.value) })
          }
        >
          {BATCH_SIZES.map((v) => (
            <option key={v} value={v}>
              {describeBatchSize(v)}
            </option>
          ))}
        </select>
      </label>

      <label className="field">
        <span>Speed</span>
        <select
          value={speed}
          onChange={(e) =>
            setSpeed(e.target.value === 'max' ? 'max' : (Number(e.target.value) as Speed))
          }
        >
          {SPEEDS.map((v) => (
            <option key={v} value={v}>
              {speedLabel(v)}
            </option>
          ))}
        </select>
      </label>

      <div className="field seed">
        <span>Weight seed</span>
        <div className="seed-row">
          <code>{seed}</code>
          <button
            type="button"
            onClick={() => setSeed(randomSeed())}
            title="Initialise new random weights"
          >
            New weights
          </button>
        </div>
      </div>
    </div>
  );
}
