import { TOY2D_KINDS, type Toy2DKind } from '../data';
import { randomSeed, useAppStore } from '../state/store';
import { DatasetThumb } from './DatasetThumb';

const NAMES: Record<Toy2DKind, string> = {
  circle: 'Circle',
  xor: 'XOR',
  spirals: 'Spirals',
  gaussians: 'Gaussians',
};

export function DatasetPanel() {
  const { kind, n, noise } = useAppStore((s) => s.config.dataset);
  const showTestData = useAppStore((s) => s.showTestData);
  const { setDataset, setShowTestData } = useAppStore.getState();

  return (
    <section className="panel dataset-panel">
      <h2>Data</h2>
      <div className="dataset-picker" role="radiogroup" aria-label="Dataset">
        {TOY2D_KINDS.map((k) => (
          <button
            key={k}
            type="button"
            role="radio"
            aria-checked={k === kind}
            className={k === kind ? 'selected' : ''}
            onClick={() => setDataset({ kind: k })}
          >
            <DatasetThumb kind={k} />
            <span>{NAMES[k]}</span>
          </button>
        ))}
      </div>

      <label className="slider">
        <span>
          Noise <output>{Math.round(noise * 100)} %</output>
        </span>
        <input
          type="range"
          min={0}
          max={0.5}
          step={0.05}
          value={noise}
          onChange={(e) => setDataset({ noise: Number(e.target.value) })}
        />
      </label>

      <label className="slider">
        <span>
          Points <output>{n}</output>
        </span>
        <input
          type="range"
          min={100}
          max={1000}
          step={50}
          value={n}
          onChange={(e) => setDataset({ n: Number(e.target.value) })}
        />
      </label>

      <p className="hint">30 % of the points are held out for testing.</p>

      <label className="checkbox">
        <input
          type="checkbox"
          checked={showTestData}
          onChange={(e) => setShowTestData(e.target.checked)}
        />
        Show test data
      </label>

      <button type="button" onClick={() => setDataset({ seed: randomSeed() })}>
        Regenerate data
      </button>
    </section>
  );
}
