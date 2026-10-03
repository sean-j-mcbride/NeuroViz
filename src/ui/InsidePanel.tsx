import { type ReactNode, useState } from 'react';
import { useAppStore } from '../state/store';
import {
  EPOCH_AXIS,
  GradientNorms,
  HistogramTimeline,
  InsideTables,
  type TimelineAxis,
  depthColour,
} from '../viz';
import { usePrefersDark } from '../viz/hooks';
import type { TimelineSnapshot } from '../worker';

/**
 * Per-layer weight and gradient distributions over a run, as charts or a table:
 * spot vanishing or exploding gradients. Shared by the playground and MNIST.
 */
export function InsideTrainingPanel({
  timeline,
  axis = EPOCH_AXIS,
  table,
  normsHint,
  className,
}: {
  timeline: TimelineSnapshot;
  axis?: TimelineAxis;
  /** The table view (an `InsideTables` with the page's dead counts). */
  table: ReactNode;
  /** What the gradient-size chart measures, under it. */
  normsHint: ReactNode;
  className?: string;
}) {
  const dark = usePrefersDark();
  const [asTable, setAsTable] = useState(false);
  const n = timeline.layers.length;

  return (
    <section className={`panel inside-panel ${className ?? ''}`}>
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
        table
      ) : (
        <div className="inside-grid">
          <div className="inside-norms">
            <div className="chart-header">
              <h3>Gradient size per layer</h3>
              <span className="legend">
                {timeline.layers.map((_, k) => (
                  <span key={k} className="legend-item">
                    <span className="swatch" style={{ borderTopColor: depthColour(k, n, dark) }} />
                    {axis.layerName(k, n)}
                  </span>
                ))}
              </span>
            </div>
            <GradientNorms timeline={timeline} axis={axis} />
            <p className="hint">{normsHint}</p>
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
                  {axis.layerName(k, n)}
                </span>
                <HistogramTimeline
                  hist={l.weightHist}
                  epochs={timeline.epochs}
                  bins={timeline.weightBins}
                  label={`${axis.layerName(k, n)} weights`}
                  at={axis.at}
                />
                <HistogramTimeline
                  hist={l.gradHist}
                  epochs={timeline.epochs}
                  bins={timeline.gradBins}
                  label={`${axis.layerName(k, n)} gradients`}
                  at={axis.at}
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

/** The playground's Inside training panel. */
export function InsidePanel() {
  const timeline = useAppStore((s) => s.snapshot?.timeline);
  const columns = useAppStore((s) => s.snapshot?.columns);
  if (!timeline) return null;
  return (
    <InsideTrainingPanel
      timeline={timeline}
      table={<InsideTables timeline={timeline} columns={columns} />}
      normsHint={
        <>
          Root-mean-square of ∂L/∂W over the training set (including the L2 term when L2 is on), on
          a log scale. If the early layers (fainter lines) sit orders of magnitude below the later
          ones (stronger lines), they barely learn — vanishing gradients. Try 6 sigmoid layers, then
          ReLU.
        </>
      }
    />
  );
}
