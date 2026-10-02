import { useState } from 'react';
import { useAppStore } from '../state/store';
import { DecisionBoundary, LossCurve, Stats } from '../viz';
import { outputGrid } from '../worker';

export function OutputPanel() {
  const snapshot = useAppStore((s) => s.snapshot);
  const showTestData = useAppStore((s) => s.showTestData);
  const sessionData = useAppStore((s) => s.sessionData);
  const stepThrough = useAppStore((s) => s.stepThrough);
  const reference = useAppStore((s) => s.reference);
  const { setProbe, pinReference } = useAppStore.getState();
  const [logScale, setLogScale] = useState(false);
  if (!snapshot) return null;

  return (
    <section className="panel output-panel">
      <h2>Output</h2>
      <DecisionBoundary
        boundary={outputGrid(snapshot)}
        data={sessionData}
        showTestData={showTestData}
        onPick={stepThrough ? setProbe : undefined}
        selected={stepThrough?.probe}
      />
      <Stats snapshot={snapshot} />
      <div className="loss-header">
        <h3>Loss</h3>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={logScale}
            onChange={(e) => setLogScale(e.target.checked)}
          />
          Log scale
        </label>
      </div>
      {/* Each run's pair stays on one line; hue = run, dashes = test. */}
      <div className="legend loss-legend">
        <span className="legend-item">
          {reference && <span className="legend-run">This run</span>}
          <span className="swatch train" /> Train
          <span className="swatch test" /> Test
        </span>
        {reference && (
          <span className="legend-item">
            <span className="legend-run">Reference</span>
            <span className="swatch reference" /> Train
            <span className="swatch reference test" /> Test
          </span>
        )}
      </div>
      <LossCurve
        train={snapshot.trainLoss}
        test={snapshot.testLoss}
        logScale={logScale}
        reference={reference && { train: reference.trainLoss, test: reference.testLoss }}
      />
      <button
        type="button"
        className="pin"
        onClick={() => pinReference()}
        title="Freeze this run's curves and boundary to compare the next run against"
      >
        {reference ? 'Pin this run instead' : 'Pin as reference'}
      </button>
    </section>
  );
}
