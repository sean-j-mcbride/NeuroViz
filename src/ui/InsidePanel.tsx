import { useState } from 'react';
import { useAppStore } from '../state/store';
import { GradientNorms, HistogramTimeline, InsideTables, depthColour, layerLabel } from '../viz';
import { usePrefersDark } from '../viz/hooks';

/** Per-layer weight and gradient distributions over the run: spot vanishing or exploding gradients. */
export function InsidePanel() {
  const timeline = useAppStore((s) => s.snapshot?.timeline);
  const dark = usePrefersDark();
  const [asTable, setAsTable] = useState(false);
  if (!timeline) return null;
  const n = timeline.layers.length;

  return (
    <section className="panel inside-panel">
      <div className="panel-toolbar">
        <h2>Inside training</h2>
        <div className="segmented" role="group" aria-label="Show as">
          <button type="button" aria-pressed={!asTable} onClick={() => setAsTable(false)}>
            Charts
          </button>
          <button type="button" aria-pressed={asTable} onClick={() => setAsTable(true)}>
            Table
          </button>
        </div>
      </div>

      {asTable ? (
        <InsideTables timeline={timeline} />
      ) : (
        <div className="inside-grid">
          <div className="inside-norms">
            <div className="chart-header">
              <h3>Gradient size per layer</h3>
              <span className="legend">
                {timeline.layers.map((_, k) => (
                  <span key={k} className="legend-item">
                    <span className="swatch" style={{ borderTopColor: depthColour(k, n, dark) }} />
                    {layerLabel(k, n)}
                  </span>
                ))}
              </span>
            </div>
            <GradientNorms timeline={timeline} />
            <p className="hint">
              Root-mean-square of ∂L/∂W over the training set (including the L2 term when L2 is on),
              on a log scale. If the early layers (fainter lines) sit orders of magnitude below the
              later ones (stronger lines), they barely learn — vanishing gradients. Try 6 sigmoid
              layers, then ReLU.
            </p>
          </div>

          <div className="inside-hists">
            <div className="hist-row hist-head">
              <span />
              <h3>Weights (signed log scale)</h3>
              <h3>Gradients |∂L/∂W| (log scale)</h3>
            </div>
            {timeline.layers.map((l, k) => (
              <div key={k} className="hist-row">
                <span className="hist-label">
                  <span className="swatch" style={{ borderTopColor: depthColour(k, n, dark) }} />
                  {layerLabel(k, n)}
                </span>
                <HistogramTimeline
                  hist={l.weightHist}
                  epochs={timeline.epochs}
                  bins={timeline.weightBins}
                  label={`${layerLabel(k, n)} weights`}
                />
                <HistogramTimeline
                  hist={l.gradHist}
                  epochs={timeline.epochs}
                  bins={timeline.gradBins}
                  label={`${layerLabel(k, n)} gradients`}
                />
              </div>
            ))}
            <p className="hint">
              Each strip is a distribution over time (left = start of training, right = now);
              stronger colour means more of the layer’s weights are in that range. Dotted lines mark
              w = ±1.
            </p>
          </div>
        </div>
      )}
    </section>
  );
}
