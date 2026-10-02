import { TOY2D_KINDS } from '../data';
import { DATASET_NAMES, NOISE, POINTS, randomSeed, useAppStore } from '../state/store';
import { DatasetThumb } from './DatasetThumb';

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
            <span>{DATASET_NAMES[k]}</span>
          </button>
        ))}
      </div>

      <label className="slider">
        <span>
          Noise <output>{Math.round(noise * 100)} %</output>
        </span>
        <input
          type="range"
          min={NOISE.min}
          max={NOISE.max}
          step={NOISE.step}
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
          min={POINTS.min}
          max={POINTS.max}
          step={POINTS.step}
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
