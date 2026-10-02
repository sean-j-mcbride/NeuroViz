import { useState } from 'react';
import { useAppStore } from '../state/store';
import { DecisionBoundary, LossCurve, Stats } from '../viz';

export function OutputPanel() {
  const snapshot = useAppStore((s) => s.snapshot);
  const showTestData = useAppStore((s) => s.showTestData);
  const [logScale, setLogScale] = useState(false);
  if (!snapshot) return null;

  return (
    <section className="panel output-panel">
      <h2>Output</h2>
      <DecisionBoundary snapshot={snapshot} showTestData={showTestData} />
      <Stats snapshot={snapshot} />
      <div className="loss-header">
        <h3>Loss</h3>
        <span className="legend">
          <span className="swatch train" /> Train
          <span className="swatch test" /> Test
        </span>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={logScale}
            onChange={(e) => setLogScale(e.target.checked)}
          />
          Log scale
        </label>
      </div>
      <LossCurve train={snapshot.trainLoss} test={snapshot.testLoss} logScale={logScale} />
    </section>
  );
}
